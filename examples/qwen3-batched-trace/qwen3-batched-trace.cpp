#include "arg.h"
#include "common.h"
#include "ggml-backend.h"
#include "ggml.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <clocale>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct batch_row {
    llama_token  token;
    llama_pos    pos;
    llama_seq_id seq_id;
    bool         output;
};

struct request_state {
    llama_seq_id             seq_id;
    std::vector<llama_token> prompt;
    llama_pos                next_pos = 0;
    llama_token              next_token = LLAMA_TOKEN_NULL;
    int32_t                  output_index = -1;
};

static std::string shape_string(const ggml_tensor * t) {
    std::ostringstream out;
    out << "[";
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        if (i > 0) {
            out << ",";
        }
        out << t->ne[i];
    }
    out << "]";
    return out.str();
}

static std::string clean_text(std::string text) {
    for (char & c : text) {
        if (c == '\n' || c == '\r' || c == '\t') {
            c = ' ';
        }
    }
    return text;
}

template<typename T>
static T read_unaligned(const uint8_t * data) {
    T value;
    std::memcpy(&value, data, sizeof(value));
    return value;
}

static bool tensor_type_supported(enum ggml_type type) {
    return type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16 ||
           type == GGML_TYPE_I64 || type == GGML_TYPE_I32 || type == GGML_TYPE_I16 || type == GGML_TYPE_I8;
}

static const ggml_tensor * tensor_base(const ggml_tensor * tensor) {
    while (tensor && tensor->view_src) {
        tensor = tensor->view_src;
    }
    return tensor;
}

static float tensor_value(const uint8_t * data, enum ggml_type type, size_t offset) {
    const uint8_t * p = data + offset;
    switch (type) {
        case GGML_TYPE_F32:  return read_unaligned<float>(p);
        case GGML_TYPE_F16:  return ggml_fp16_to_fp32(read_unaligned<ggml_fp16_t>(p));
        case GGML_TYPE_BF16: return ggml_bf16_to_fp32(read_unaligned<ggml_bf16_t>(p));
        case GGML_TYPE_I64:  return static_cast<float>(read_unaligned<int64_t>(p));
        case GGML_TYPE_I32:  return static_cast<float>(read_unaligned<int32_t>(p));
        case GGML_TYPE_I16:  return static_cast<float>(read_unaligned<int16_t>(p));
        case GGML_TYPE_I8:   return static_cast<float>(read_unaligned<int8_t>(p));
        default:             return std::numeric_limits<float>::quiet_NaN();
    }
}

static bool write_npy_header(std::ofstream & out, const std::string & descr, const std::vector<int64_t> & shape, bool fortran_order) {
    std::ostringstream shape_out;
    shape_out << "(";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) {
            shape_out << ", ";
        }
        shape_out << shape[i];
    }
    if (shape.size() == 1) {
        shape_out << ",";
    }
    shape_out << ")";

    std::string header = "{'descr': '" + descr + "', 'fortran_order': " +
                         (fortran_order ? "True" : "False") + ", 'shape': " + shape_out.str() + ", }";
    const size_t preamble = 10;
    const size_t padding = (16 - ((preamble + header.size() + 1) % 16)) % 16;
    header.append(padding, ' ');
    header.push_back('\n');
    if (header.size() > std::numeric_limits<uint16_t>::max()) {
        return false;
    }

    const std::array<unsigned char, 8> magic = { 0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0 };
    out.write(reinterpret_cast<const char *>(magic.data()), magic.size());
    const uint16_t header_len = static_cast<uint16_t>(header.size());
    const unsigned char len_bytes[2] = {
        static_cast<unsigned char>(header_len & 0xff),
        static_cast<unsigned char>((header_len >> 8) & 0xff),
    };
    out.write(reinterpret_cast<const char *>(len_bytes), sizeof(len_bytes));
    out.write(header.data(), header.size());
    return out.good();
}

template<typename T>
static bool write_vector_npy(const fs::path & path, const std::vector<T> & data, const std::string & descr) {
    std::ofstream out(path, std::ios::binary);
    if (!out || !write_npy_header(out, descr, { static_cast<int64_t>(data.size()) }, false)) {
        return false;
    }
    out.write(reinterpret_cast<const char *>(data.data()), data.size() * sizeof(T));
    return out.good();
}

struct trace_state {
    fs::path                root;
    fs::path                phase_dir;
    std::string             phase;
    std::ofstream           manifest;
    std::ofstream           trace_log;
    std::vector<batch_row>  batch_rows;
    std::map<int64_t, batch_row> slot_owner;
    int32_t                 n_layer = 0;
    bool                    failed = false;

    explicit trace_state(fs::path output_root) : root(std::move(output_root)) {
        fs::create_directories(root);
        manifest.open(root / "manifest.tsv");
        trace_log.open(root / "trace.log");
        manifest << "phase\trole\ttrigger\ttrigger_op\ttensor\ttype\tggml_shape\tnpy\n";
    }

    void emit(const std::string & line) {
        LOG_INF("%s\n", line.c_str());
        trace_log << line << "\n";
        trace_log.flush();
    }

    void begin_phase(const std::string & name, const llama_batch & batch) {
        phase = name;
        phase_dir = root / phase;
        fs::create_directories(phase_dir);
        batch_rows.clear();

        std::vector<int32_t> tokens;
        std::vector<int32_t> positions;
        std::vector<int32_t> seq_ids;
        std::vector<int32_t> outputs;
        std::ofstream table(phase_dir / "batch.tsv");
        table << "index\ttoken\tposition\tseq_id\toutput\n";

        for (int32_t i = 0; i < batch.n_tokens; ++i) {
            batch_row row = { batch.token[i], batch.pos[i], batch.seq_id[i][0], batch.logits[i] != 0 };
            batch_rows.push_back(row);
            tokens.push_back(row.token);
            positions.push_back(row.pos);
            seq_ids.push_back(row.seq_id);
            outputs.push_back(row.output ? 1 : 0);
            table << i << "\t" << row.token << "\t" << row.pos << "\t" << row.seq_id << "\t" << row.output << "\n";
        }

        failed |= !write_vector_npy(phase_dir / "batch_token.npy", tokens, "<i4");
        failed |= !write_vector_npy(phase_dir / "batch_position.npy", positions, "<i4");
        failed |= !write_vector_npy(phase_dir / "batch_seq_id.npy", seq_ids, "<i4");
        failed |= !write_vector_npy(phase_dir / "batch_output.npy", outputs, "<i4");

        std::ostringstream line;
        line << "[phase] " << phase << " n_tokens=" << batch.n_tokens;
        emit(line.str());
        for (int32_t i = 0; i < batch.n_tokens; ++i) {
            if (batch.n_tokens <= 32 || i < 5 || i >= batch.n_tokens - 5 || batch.logits[i]) {
                std::ostringstream row;
                row << "[batch] i=" << std::setw(3) << i
                    << " seq=" << batch.seq_id[i][0]
                    << " pos=" << std::setw(3) << batch.pos[i]
                    << " token=" << batch.token[i]
                    << " output=" << static_cast<int>(batch.logits[i]);
                emit(row.str());
            } else if (i == 5) {
                emit("[batch] ...");
            }
        }
    }

    bool selected(const ggml_tensor * t) const {
        const std::string name = t->name;
        const ggml_tensor * base = tensor_base(t);
        const std::string base_name = base ? base->name : "";
        if (name == "embd" && t->op == GGML_OP_GET_ROWS) {
            return true;
        }
        if ((name == "Qcur-0" || name == "Kcur-0") && t->op == GGML_OP_ROPE) {
            return true;
        }
        if (name == "Qcur_normed-0" || name == "Kcur_normed-0") {
            return true;
        }
        if (name == "Vcur-0" && t->op == GGML_OP_MUL_MAT) {
            return true;
        }
        if (name == "cache_k_l0 (view)" && t->op == GGML_OP_SET_ROWS) {
            return true;
        }
        if (base_name == "cache_v_l0" && t->op == GGML_OP_SET_ROWS) {
            return true;
        }
        if ((name == "kq-0" || name == "kqv-0") && t->op == GGML_OP_MUL_MAT) {
            return true;
        }
        if (name == "kq_soft_max-0" && t->op == GGML_OP_SOFT_MAX) {
            return true;
        }
        if (name == "attn_norm-0" || name == "kqv_out-0" || name == "ffn_inp-0" ||
            name == "ffn_out-0" || name == "result_norm" || name == "result_output") {
            return true;
        }
        return name.rfind("l_out-", 0) == 0;
    }

    std::vector<float> copy_f32(const ggml_tensor * t) {
        std::vector<float> result;
        if (!t || !tensor_type_supported(t->type)) {
            return result;
        }
        std::vector<uint8_t> raw(ggml_nbytes(t));
        ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
        result.reserve(ggml_nelements(t));
        for (int64_t i3 = 0; i3 < t->ne[3]; ++i3) {
            for (int64_t i2 = 0; i2 < t->ne[2]; ++i2) {
                for (int64_t i1 = 0; i1 < t->ne[1]; ++i1) {
                    for (int64_t i0 = 0; i0 < t->ne[0]; ++i0) {
                        const size_t offset = i3 * t->nb[3] + i2 * t->nb[2] + i1 * t->nb[1] + i0 * t->nb[0];
                        result.push_back(tensor_value(raw.data(), t->type, offset));
                    }
                }
            }
        }
        return result;
    }

    std::vector<float> dump(const std::string & role, const ggml_tensor * t, const ggml_tensor * trigger) {
        if (!t) {
            emit("[error] missing tensor for " + phase + "/" + role);
            failed = true;
            return {};
        }
        std::vector<float> values = copy_f32(t);
        if (values.empty() && ggml_nelements(t) > 0) {
            emit("[error] unsupported tensor type for " + phase + "/" + role + ": " + ggml_type_name(t->type));
            failed = true;
            return {};
        }

        const fs::path path = phase_dir / (role + ".npy");
        std::ofstream out(path, std::ios::binary);
        const std::vector<int64_t> shape = { t->ne[0], t->ne[1], t->ne[2], t->ne[3] };
        if (!out || !write_npy_header(out, "<f4", shape, true)) {
            emit("[error] cannot write " + path.string());
            failed = true;
            return values;
        }
        out.write(reinterpret_cast<const char *>(values.data()), values.size() * sizeof(float));
        if (!out.good()) {
            emit("[error] incomplete write " + path.string());
            failed = true;
        }

        size_t n_finite = 0;
        size_t n_zero = 0;
        size_t n_neg_inf = 0;
        double sum = 0.0;
        float min_value = std::numeric_limits<float>::infinity();
        float max_value = -std::numeric_limits<float>::infinity();
        for (float value : values) {
            if (value == 0.0f) {
                ++n_zero;
            }
            if (std::isinf(value) && value < 0.0f) {
                ++n_neg_inf;
            }
            if (std::isfinite(value)) {
                ++n_finite;
                sum += value;
                min_value = std::min(min_value, value);
                max_value = std::max(max_value, value);
            }
        }

        std::ostringstream line;
        line << "[tensor] " << std::left << std::setw(30) << role
             << " ggml=" << shape_string(t)
             << " type=" << ggml_type_name(t->type)
             << " finite=" << n_finite
             << " zero=" << n_zero
             << " -inf=" << n_neg_inf;
        if (n_finite > 0) {
            line << " min=" << min_value << " max=" << max_value << " mean=" << (sum / n_finite);
        }
        emit(line.str());

        manifest << phase << "\t" << role << "\t" << trigger->name << "\t" << ggml_op_desc(trigger)
                 << "\t" << t->name << "\t" << ggml_type_name(t->type) << "\t" << shape_string(t)
                 << "\t" << fs::relative(path, root).string() << "\n";
        manifest.flush();
        return values;
    }

    void record_slots(const std::vector<float> & slots) {
        std::ofstream out(phase_dir / "kv_writes.tsv");
        out << "batch_index\tslot\tseq_id\tposition\ttoken\n";
        const size_t count = std::min(slots.size(), batch_rows.size());
        for (size_t i = 0; i < count; ++i) {
            const int64_t slot = static_cast<int64_t>(slots[i]);
            slot_owner[slot] = batch_rows[i];
            out << i << "\t" << slot << "\t" << batch_rows[i].seq_id << "\t" << batch_rows[i].pos << "\t" << batch_rows[i].token << "\n";
        }
        std::ostringstream line;
        line << "[kv-write] slots=";
        for (size_t i = 0; i < count; ++i) {
            if (i > 0) {
                line << ",";
            }
            if (count > 16 && i == 8) {
                line << "...,";
                i = count - 9;
            }
            line << static_cast<int64_t>(slots[i]);
        }
        emit(line.str());
    }

    void observe(ggml_tensor * t) {
        const std::string name = t->name;
        const ggml_tensor * base = tensor_base(t);
        const std::string base_name = base ? base->name : "";
        if (name == "embd" && t->op == GGML_OP_GET_ROWS) {
            dump("graph_input_tokens", t->src[1], t);
            dump("input_embedding_layer0_hidden", t, t);
        } else if (name == "Qcur_normed-0") {
            dump("q_before_rope_layer0", t, t);
        } else if (name == "Qcur-0" && t->op == GGML_OP_ROPE) {
            dump("position_ids_graph", t->src[1], t);
            dump("q_after_rope_layer0", t, t);
        } else if (name == "Kcur_normed-0") {
            dump("k_before_rope_layer0", t, t);
        } else if (name == "Kcur-0" && t->op == GGML_OP_ROPE) {
            dump("k_after_rope_layer0", t, t);
        } else if (name == "Vcur-0" && t->op == GGML_OP_MUL_MAT) {
            dump("v_current_flat_layer0", t, t);
        } else if (name == "cache_k_l0 (view)" && t->op == GGML_OP_SET_ROWS) {
            const std::vector<float> slots = dump("kv_slot_indices", t->src[1], t);
            dump("physical_k_cache_after_write_layer0", t, t);
            record_slots(slots);
        } else if (base_name == "cache_v_l0" && t->op == GGML_OP_SET_ROWS) {
            dump("physical_v_cache_after_write_layer0", base, t);
        } else if (name == "kq-0" && t->op == GGML_OP_MUL_MAT) {
            dump("active_k_permuted_layer0", t->src[0], t);
            dump("attention_scores_layer0", t, t);
        } else if (name == "kq_soft_max-0" && t->op == GGML_OP_SOFT_MAX) {
            dump("attention_mask_layer0", t->src[1], t);
            dump("attention_probabilities_layer0", t, t);
        } else if (name == "kqv-0" && t->op == GGML_OP_MUL_MAT) {
            dump("active_v_permuted_layer0", t->src[0], t);
            dump("attention_context_layer0", t, t);
        } else if (name == "attn_norm-0") {
            dump("attention_norm_hidden_layer0", t, t);
        } else if (name == "kqv_out-0") {
            dump("attention_merged_heads_layer0", t, t);
        } else if (name == "ffn_inp-0") {
            dump("post_attention_hidden_layer0", t, t);
        } else if (name == "ffn_out-0") {
            dump("ffn_output_layer0", t, t);
        } else if (name.rfind("l_out-", 0) == 0) {
            const int32_t il = std::stoi(name.substr(std::string("l_out-").size()));
            dump("decoder_output_hidden_layer" + std::to_string(il), t, t);
            if (il == n_layer - 1) {
                dump("decoder_output_hidden_last_layer", t, t);
            }
        } else if (name == "result_norm") {
            dump("final_norm_hidden", t, t);
        } else if (name == "result_output") {
            dump("lm_head_logits", t, t);
        }
    }

    void record_memory(llama_context * ctx, int n_seq) {
        llama_memory_t mem = llama_get_memory(ctx);
        std::ofstream out(phase_dir / "memory.tsv");
        out << "seq_id\tpos_min\tpos_max\tlogical_tokens\n";
        int64_t total = 0;
        for (int seq = 0; seq < n_seq; ++seq) {
            const llama_pos pmin = llama_memory_seq_pos_min(mem, seq);
            const llama_pos pmax = llama_memory_seq_pos_max(mem, seq);
            const int64_t count = pmin < 0 ? 0 : static_cast<int64_t>(pmax) - pmin + 1;
            total += count;
            out << seq << "\t" << pmin << "\t" << pmax << "\t" << count << "\n";
            std::ostringstream line;
            line << "[memory] seq=" << seq << " pos_min=" << pmin << " pos_max=" << pmax << " logical_tokens=" << count;
            emit(line.str());
        }
        emit("[memory] total_logical_tokens=" + std::to_string(total));
    }
};

static bool trace_callback(ggml_tensor * t, bool ask, void * user_data) {
    auto * trace = static_cast<trace_state *>(user_data);
    if (ask) {
        return trace->selected(t);
    }
    trace->observe(t);
    return true;
}

static std::vector<llama_token> make_prompt(llama_context * ctx, const std::string & seed, size_t target, bool add_bos) {
    std::string text;
    for (int i = 0; i < 128; ++i) {
        text += seed;
        text += " This request has its own history and position counter. ";
    }
    std::vector<llama_token> tokens = common_tokenize(ctx, text, add_bos, true);
    if (tokens.size() > target) {
        tokens.resize(target);
    }
    return tokens;
}

static void add_prompt(llama_batch & batch, request_state & request) {
    for (size_t pos = 0; pos < request.prompt.size(); ++pos) {
        const bool output = pos + 1 == request.prompt.size();
        common_batch_add(batch, request.prompt[pos], pos, { request.seq_id }, output);
        if (output) {
            request.output_index = batch.n_tokens - 1;
        }
    }
    request.next_pos = request.prompt.size();
}

static void add_decode_token(llama_batch & batch, request_state & request) {
    common_batch_add(batch, request.next_token, request.next_pos, { request.seq_id }, true);
    request.output_index = batch.n_tokens - 1;
    ++request.next_pos;
}

static bool sample_greedy(llama_context * ctx, const llama_vocab * vocab, std::vector<request_state *> requests, trace_state & trace) {
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    for (request_state * request : requests) {
        const float * logits = llama_get_logits_ith(ctx, request->output_index);
        if (!logits) {
            trace.emit("[error] missing logits for batch index " + std::to_string(request->output_index));
            return false;
        }
        request->next_token = static_cast<llama_token>(std::max_element(logits, logits + n_vocab) - logits);
        std::ostringstream line;
        line << "[sample] seq=" << request->seq_id << " next_pos=" << request->next_pos
             << " token=" << request->next_token
             << " piece='" << clean_text(common_token_to_piece(ctx, request->next_token)) << "'";
        trace.emit(line.str());
    }
    return true;
}

static bool run_phase(
        llama_context * ctx,
        const llama_vocab * vocab,
        llama_batch & batch,
        const std::string & phase,
        int n_seq,
        std::vector<request_state *> outputs,
        trace_state & trace) {
    trace.begin_phase(phase, batch);
    const int result = llama_decode(ctx, batch);
    if (result != 0) {
        trace.emit("[error] llama_decode returned " + std::to_string(result));
        return false;
    }
    trace.record_memory(ctx, n_seq);
    return sample_greedy(ctx, vocab, std::move(outputs), trace);
}

static bool run_workload(llama_context * ctx, trace_state & trace) {
    const llama_model * model = llama_get_model(ctx);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);
    const std::array<int, 4> lengths = { 48, 56, 64, 72 };
    const std::array<std::string, 5> seeds = {
        "Sequence zero studies astronomy.",
        "Sequence one studies biology.",
        "Sequence two studies compilers.",
        "Sequence three studies databases.",
        "Sequence four joins later and studies music.",
    };

    std::vector<request_state> requests;
    for (int i = 0; i < 4; ++i) {
        request_state request;
        request.seq_id = i;
        request.prompt = make_prompt(ctx, seeds[i], lengths[i], add_bos);
        if (request.prompt.size() != static_cast<size_t>(lengths[i])) {
            trace.emit("[error] tokenizer did not produce enough tokens for seq " + std::to_string(i));
            return false;
        }
        requests.push_back(std::move(request));
    }
    request_state joined;
    joined.seq_id = 4;
    joined.prompt = make_prompt(ctx, seeds[4], 20, add_bos);
    if (joined.prompt.size() != 20) {
        trace.emit("[error] tokenizer did not produce enough tokens for seq 4");
        return false;
    }

    llama_batch batch = llama_batch_init(512, 0, 1);

    common_batch_clear(batch);
    std::vector<request_state *> old_requests;
    for (request_state & request : requests) {
        add_prompt(batch, request);
        old_requests.push_back(&request);
    }
    if (!run_phase(ctx, vocab, batch, "00_prefill_4", 4, old_requests, trace)) {
        llama_batch_free(batch);
        return false;
    }

    common_batch_clear(batch);
    for (request_state & request : requests) {
        add_decode_token(batch, request);
    }
    if (!run_phase(ctx, vocab, batch, "01_decode_4", 4, old_requests, trace)) {
        llama_batch_free(batch);
        return false;
    }

    common_batch_clear(batch);
    for (request_state & request : requests) {
        add_decode_token(batch, request);
    }
    add_prompt(batch, joined);
    std::vector<request_state *> joined_outputs = old_requests;
    joined_outputs.push_back(&joined);
    if (!run_phase(ctx, vocab, batch, "02_join_new_request", 5, joined_outputs, trace)) {
        llama_batch_free(batch);
        return false;
    }

    common_batch_clear(batch);
    for (request_state & request : requests) {
        add_decode_token(batch, request);
    }
    add_decode_token(batch, joined);
    const bool ok = run_phase(ctx, vocab, batch, "03_decode_5", 5, joined_outputs, trace);
    llama_batch_free(batch);
    return ok;
}

static bool parse_trace_dir(int argc, char ** argv, fs::path & trace_dir, std::vector<char *> & common_argv) {
    common_argv.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--trace-dir") {
            if (i + 1 >= argc) {
                LOG_ERR("--trace-dir requires a path\n");
                return false;
            }
            trace_dir = argv[++i];
        } else {
            common_argv.push_back(argv[i]);
        }
    }
    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    common_init();

    fs::path trace_dir = "logs/qwen3_llama_batched_trace";
    std::vector<char *> common_argv;
    if (!parse_trace_dir(argc, argv, trace_dir, common_argv)) {
        return 1;
    }

    common_params params;
    if (!common_params_parse(static_cast<int>(common_argv.size()), common_argv.data(), params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    if (params.n_ctx == 0) {
        params.n_ctx = 512;
    }
    if (params.n_ctx < 512) {
        LOG_ERR("this workload requires --ctx-size 512 or larger\n");
        return 1;
    }
    params.n_batch = std::max(params.n_batch, 512);
    params.n_ubatch = std::max(params.n_ubatch, 256);
    params.n_parallel = 5;
    params.n_gpu_layers = 0;
    params.fit_params = false;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    params.kv_unified = true;
    params.no_kv_offload = true;
    params.no_op_offload = true;
    params.warmup = false;

    trace_state trace(trace_dir);
    params.cb_eval = trace_callback;
    params.cb_eval_user_data = &trace;

    llama_backend_init();
    llama_numa_init(params.numa);

    auto llama_init = common_init_from_params(params);
    llama_model * model = llama_init ? llama_init->model() : nullptr;
    llama_context * ctx = llama_init ? llama_init->context() : nullptr;
    if (!model || !ctx) {
        LOG_ERR("failed to initialize model or context\n");
        llama_backend_free();
        return 1;
    }

    trace.n_layer = llama_model_n_layer(model);
    const int32_t n_embd = llama_model_n_embd(model);
    const int32_t n_head = llama_model_n_head(model);
    const int32_t n_head_kv = llama_model_n_head_kv(model);
    std::ostringstream config;
    config << "[config] trace_dir=" << trace.root.string()
           << " n_ctx=" << llama_n_ctx(ctx)
           << " n_batch=" << llama_n_batch(ctx)
           << " n_ubatch=" << llama_n_ubatch(ctx)
           << " n_seq_max=" << llama_n_seq_max(ctx)
           << " n_embd=" << n_embd
           << " n_head=" << n_head
           << " n_head_kv=" << n_head_kv
           << " head_dim=" << (n_head > 0 ? n_embd / n_head : 0)
           << " n_layer=" << trace.n_layer
           << " kv_unified=1 flash_attn=off cpu=1";
    trace.emit(config.str());

    const bool ok = run_workload(ctx, trace);
    if (ok) {
        llama_perf_context_print(ctx);
        trace.emit("[done] trace completed");
    }
    llama_backend_free();
    return ok && !trace.failed ? 0 : 1;
}
