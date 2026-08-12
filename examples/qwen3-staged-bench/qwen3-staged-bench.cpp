#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

using bench_clock = std::chrono::steady_clock;

struct options {
    std::array<std::string, 2> paths;
    int32_t prompt_tokens = 128;
    int32_t prefill_iters = 3;
    int32_t decode_iters = 10;
    int32_t warmup = 2;
    int32_t threads = 4;
    int32_t ctx_size = 512;
    std::string csv_path;
};

struct stage_sample {
    double embedding_us = 0.0;
    std::vector<double> layers_us;
    double final_norm_us = 0.0;
    double lm_head_us = 0.0;
    double decoder_layers_us = 0.0;
    double decoder_shard_us = 0.0;
    double head_shard_us = 0.0;
    double graph_us = 0.0;
    double wall_us = 0.0;
    double outside_stages_us = 0.0;
};

struct stage_timer {
    explicit stage_timer(int32_t n_layer) : n_layer(n_layer) {
    }

    int32_t n_layer;
    bool enabled = false;
    int32_t next_stage = 0;
    int32_t pending_stage = -1;
    bench_clock::time_point previous_end;
    stage_sample current;
    std::string error;

    void begin() {
        enabled = true;
        next_stage = 0;
        pending_stage = -1;
        current = {};
        current.layers_us.assign(n_layer, 0.0);
        error.clear();
    }

    void disable() {
        enabled = false;
        pending_stage = -1;
    }

    int32_t stage_index(const ggml_tensor * tensor) const {
        const std::string_view name(tensor->name);
        if (name == "embd") {
            return 0;
        }
        constexpr std::string_view prefix = "l_out-";
        if (name.size() > prefix.size() && name.substr(0, prefix.size()) == prefix) {
            int64_t value = 0;
            for (const char c : name.substr(prefix.size())) {
                if (c < '0' || c > '9') {
                    return -1;
                }
                value = value * 10 + (c - '0');
                if (value >= n_layer) {
                    return -1;
                }
            }
            return static_cast<int32_t>(value) + 1;
        }
        if (name == "result_norm") {
            return n_layer + 1;
        }
        if (name == "result_output") {
            return n_layer + 2;
        }
        return -1;
    }

    bool callback(ggml_tensor * tensor, bool ask) {
        if (!enabled) {
            return false;
        }

        const int32_t stage = stage_index(tensor);
        if (ask) {
            if (stage < 0) {
                return false;
            }
            if (stage != next_stage) {
                if (error.empty()) {
                    std::ostringstream out;
                    out << "unexpected timing boundary " << tensor->name << ", expected stage " << next_stage;
                    error = out.str();
                }
            }
            if (stage == 0 && next_stage == 0) {
                previous_end = bench_clock::now();
            }
            pending_stage = stage;
            return true;
        }

        if (stage < 0 || stage != pending_stage || stage != next_stage) {
            if (error.empty()) {
                error = "timing callback completed an unexpected boundary";
            }
            return true;
        }

        const auto now = bench_clock::now();
        const double elapsed_us = std::chrono::duration<double, std::micro>(now - previous_end).count();
        if (stage == 0) {
            current.embedding_us = elapsed_us;
        } else if (stage <= n_layer) {
            current.layers_us[stage - 1] = elapsed_us;
        } else if (stage == n_layer + 1) {
            current.final_norm_us = elapsed_us;
        } else {
            current.lm_head_us = elapsed_us;
        }
        previous_end = now;
        pending_stage = -1;
        ++next_stage;
        return true;
    }

    bool finish(double wall_us, stage_sample & sample, std::string & message) {
        enabled = false;
        if (!error.empty()) {
            message = error;
            return false;
        }
        if (next_stage != n_layer + 3) {
            std::ostringstream out;
            out << "saw " << next_stage << " timing boundaries, expected " << n_layer + 3;
            message = out.str();
            return false;
        }
        current.decoder_layers_us = std::accumulate(current.layers_us.begin(), current.layers_us.end(), 0.0);
        current.decoder_shard_us = current.embedding_us + current.decoder_layers_us;
        current.head_shard_us = current.final_norm_us + current.lm_head_us;
        current.graph_us = current.decoder_shard_us + current.head_shard_us;
        current.wall_us = wall_us;
        current.outside_stages_us = wall_us - current.graph_us;
        sample = current;
        return true;
    }
};

static bool timing_callback(ggml_tensor * tensor, bool ask, void * user_data) {
    return static_cast<stage_timer *>(user_data)->callback(tensor, ask);
}

struct batch_owner {
    explicit batch_owner(int32_t capacity) : batch(llama_batch_init(capacity, 0, 1)) {
    }

    ~batch_owner() {
        llama_batch_free(batch);
    }

    batch_owner(const batch_owner &) = delete;
    batch_owner & operator=(const batch_owner &) = delete;

    llama_batch batch;
};

using model_ptr = std::unique_ptr<llama_model, decltype(&llama_model_free)>;
using context_ptr = std::unique_ptr<llama_context, decltype(&llama_free)>;

static void print_usage(const char * program) {
    std::cout
        << "Usage: " << program << " --paths SHARD1 SHARD2 [options]\n\n"
        << "Required:\n"
        << "  --paths PATH PATH       two GGUF shard paths in order\n\n"
        << "Options:\n"
        << "  --prompt-tokens N       fixed prompt/context length (default: 128)\n"
        << "  --prefill-iters N       measured prefill iterations (default: 3)\n"
        << "  --decode-iters N        measured one-token decode iterations (default: 10)\n"
        << "  --warmup N              warmup iterations per mode (default: 2)\n"
        << "  --threads N             CPU threads (default: 4)\n"
        << "  --ctx-size N            allocated context size (default: 512)\n"
        << "  --csv PATH              write raw long-form CSV rows; refuses overwrite\n"
        << "  -h, --help              show this help\n";
}

static bool parse_int(const char * text, int32_t & value) {
    try {
        size_t end = 0;
        const long long parsed = std::stoll(text, &end, 10);
        if (end != std::string(text).size() || parsed < std::numeric_limits<int32_t>::min() || parsed > std::numeric_limits<int32_t>::max()) {
            return false;
        }
        value = static_cast<int32_t>(parsed);
        return true;
    } catch (...) {
        return false;
    }
}

static bool parse_options(int argc, char ** argv, options & opts) {
    bool have_paths = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            std::exit(0);
        }
        if (arg == "--paths") {
            if (i + 2 >= argc) {
                std::cerr << "--paths requires exactly two values\n";
                return false;
            }
            opts.paths[0] = argv[++i];
            opts.paths[1] = argv[++i];
            have_paths = true;
            continue;
        }
        if (arg == "--csv") {
            if (++i >= argc) {
                std::cerr << "--csv requires a path\n";
                return false;
            }
            opts.csv_path = argv[i];
            continue;
        }

        int32_t * destination = nullptr;
        if (arg == "--prompt-tokens") {
            destination = &opts.prompt_tokens;
        } else if (arg == "--prefill-iters") {
            destination = &opts.prefill_iters;
        } else if (arg == "--decode-iters") {
            destination = &opts.decode_iters;
        } else if (arg == "--warmup") {
            destination = &opts.warmup;
        } else if (arg == "--threads") {
            destination = &opts.threads;
        } else if (arg == "--ctx-size") {
            destination = &opts.ctx_size;
        } else {
            std::cerr << "unknown option: " << arg << "\n";
            return false;
        }
        if (++i >= argc || !parse_int(argv[i], *destination)) {
            std::cerr << arg << " requires an integer\n";
            return false;
        }
    }

    if (!have_paths || opts.paths[0].empty() || opts.paths[1].empty()) {
        std::cerr << "--paths SHARD1 SHARD2 is required\n";
        return false;
    }
    if (opts.prompt_tokens <= 0 || opts.prefill_iters <= 0 || opts.decode_iters <= 0 || opts.threads <= 0 || opts.ctx_size <= 0 || opts.warmup < 0) {
        std::cerr << "token, iteration, thread, and context values must be positive; warmup may be zero\n";
        return false;
    }
    if (opts.ctx_size <= opts.prompt_tokens) {
        std::cerr << "--ctx-size must exceed --prompt-tokens by at least one token\n";
        return false;
    }
    return true;
}

static std::vector<llama_token> make_prompt(const llama_vocab * vocab, int32_t count) {
    const std::string seed = "The quick brown fox studies deterministic transformer inference. ";
    const int32_t tokenized = llama_tokenize(vocab, seed.data(), static_cast<int32_t>(seed.size()), nullptr, 0, true, false);
    if (tokenized >= 0 || tokenized == std::numeric_limits<int32_t>::min()) {
        return {};
    }
    const int32_t required = -tokenized;

    std::vector<llama_token> seed_tokens(required);
    const int32_t result = llama_tokenize(vocab, seed.data(), static_cast<int32_t>(seed.size()), seed_tokens.data(), required, true, false);
    if (result <= 0) {
        return {};
    }
    seed_tokens.resize(result);

    std::vector<llama_token> prompt;
    prompt.reserve(count);
    for (int32_t i = 0; i < count; ++i) {
        prompt.push_back(seed_tokens[static_cast<size_t>(i) % seed_tokens.size()]);
    }
    return prompt;
}

static void fill_batch(llama_batch & batch, const std::vector<llama_token> & tokens, llama_pos start_pos) {
    batch.n_tokens = static_cast<int32_t>(tokens.size());
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        batch.token[i] = tokens[i];
        batch.pos[i] = start_pos + i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = i + 1 == batch.n_tokens;
    }
}

static double decode_wall_us(llama_context * ctx, llama_batch batch, int32_t & status) {
    llama_synchronize(ctx);
    const auto begin = bench_clock::now();
    status = llama_decode(ctx, batch);
    llama_synchronize(ctx);
    const auto end = bench_clock::now();
    return std::chrono::duration<double, std::micro>(end - begin).count();
}

static bool run_prefill_baseline(llama_context * ctx, llama_batch batch, int32_t warmup, int32_t iterations, std::vector<double> & samples) {
    llama_memory_t memory = llama_get_memory(ctx);
    for (int32_t i = -warmup; i < iterations; ++i) {
        llama_memory_clear(memory, false);
        int32_t status = 0;
        const double wall_us = decode_wall_us(ctx, batch, status);
        if (status != 0) {
            std::cerr << "baseline prefill llama_decode failed with status " << status << "\n";
            return false;
        }
        if (i >= 0) {
            samples.push_back(wall_us);
        }
    }
    return true;
}

static bool run_prefill_instrumented(llama_context * ctx, stage_timer & timer, llama_batch batch, int32_t warmup, int32_t iterations, std::vector<stage_sample> & samples) {
    llama_memory_t memory = llama_get_memory(ctx);
    for (int32_t i = -warmup; i < iterations; ++i) {
        llama_memory_clear(memory, false);
        timer.begin();
        int32_t status = 0;
        const double wall_us = decode_wall_us(ctx, batch, status);
        if (status != 0) {
            timer.disable();
            std::cerr << "instrumented prefill llama_decode failed with status " << status << "\n";
            return false;
        }
        stage_sample sample;
        std::string message;
        if (!timer.finish(wall_us, sample, message)) {
            std::cerr << "instrumented prefill timing failed: " << message << "\n";
            return false;
        }
        if (i >= 0) {
            samples.push_back(std::move(sample));
        }
    }
    return true;
}

static bool prime_decode_context(llama_context * ctx, stage_timer * timer, llama_batch prompt_batch) {
    llama_memory_clear(llama_get_memory(ctx), false);
    if (timer) {
        timer->disable();
    }
    int32_t status = 0;
    decode_wall_us(ctx, prompt_batch, status);
    if (status != 0) {
        std::cerr << "decode context prefill failed with status " << status << "\n";
        return false;
    }
    return true;
}

static bool remove_decode_token(llama_context * ctx, llama_pos position) {
    if (!llama_memory_seq_rm(llama_get_memory(ctx), 0, position, position + 1)) {
        std::cerr << "failed to remove the temporary decode token at position " << position << "\n";
        return false;
    }
    return true;
}

static bool run_decode_baseline(llama_context * ctx, llama_batch batch, llama_pos position, int32_t warmup, int32_t iterations, std::vector<double> & samples) {
    for (int32_t i = -warmup; i < iterations; ++i) {
        int32_t status = 0;
        const double wall_us = decode_wall_us(ctx, batch, status);
        if (status != 0) {
            std::cerr << "baseline decode llama_decode failed with status " << status << "\n";
            return false;
        }
        if (!remove_decode_token(ctx, position)) {
            return false;
        }
        if (i >= 0) {
            samples.push_back(wall_us);
        }
    }
    return true;
}

static bool run_decode_instrumented(llama_context * ctx, stage_timer & timer, llama_batch batch, llama_pos position, int32_t warmup, int32_t iterations, std::vector<stage_sample> & samples) {
    for (int32_t i = -warmup; i < iterations; ++i) {
        timer.begin();
        int32_t status = 0;
        const double wall_us = decode_wall_us(ctx, batch, status);
        if (status != 0) {
            timer.disable();
            std::cerr << "instrumented decode llama_decode failed with status " << status << "\n";
            return false;
        }
        stage_sample sample;
        std::string message;
        if (!timer.finish(wall_us, sample, message)) {
            std::cerr << "instrumented decode timing failed: " << message << "\n";
            return false;
        }
        if (!remove_decode_token(ctx, position)) {
            return false;
        }
        if (i >= 0) {
            samples.push_back(std::move(sample));
        }
    }
    return true;
}

struct summary_stats {
    double mean;
    double median;
    double minimum;
    double maximum;
};

static summary_stats summarize(std::vector<double> values) {
    const auto bounds = std::minmax_element(values.begin(), values.end());
    const double minimum = *bounds.first;
    const double maximum = *bounds.second;
    const double mean = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    std::sort(values.begin(), values.end());
    const size_t middle = values.size() / 2;
    const double median = values.size() % 2 == 0 ? (values[middle - 1] + values[middle]) / 2.0 : values[middle];
    return { mean, median, minimum, maximum };
}

static void print_stats_row(const std::string & component, const std::vector<double> & values) {
    const summary_stats stats = summarize(values);
    std::cout << std::left << std::setw(24) << component
              << std::right << std::setw(13) << stats.mean
              << std::setw(13) << stats.median
              << std::setw(13) << stats.minimum
              << std::setw(13) << stats.maximum << "\n";
}

template<typename Getter>
static std::vector<double> collect(const std::vector<stage_sample> & samples, Getter getter) {
    std::vector<double> values;
    values.reserve(samples.size());
    for (const stage_sample & sample : samples) {
        values.push_back(getter(sample));
    }
    return values;
}

static void print_workload_summary(const std::string & workload, const std::vector<double> & baseline, const std::vector<stage_sample> & instrumented, int32_t n_layer) {
    std::cout << "\n[" << workload << "] time_us (mean/median/min/max)\n";
    std::cout << std::left << std::setw(24) << "component"
              << std::right << std::setw(13) << "mean"
              << std::setw(13) << "median"
              << std::setw(13) << "min"
              << std::setw(13) << "max" << "\n";
    print_stats_row("baseline_wall", baseline);
    print_stats_row("embedding", collect(instrumented, [](const stage_sample & s) { return s.embedding_us; }));
    for (int32_t layer = 0; layer < n_layer; ++layer) {
        print_stats_row("layer_" + std::to_string(layer), collect(instrumented, [layer](const stage_sample & s) { return s.layers_us[layer]; }));
    }
    print_stats_row("decoder_layers_total", collect(instrumented, [](const stage_sample & s) { return s.decoder_layers_us; }));
    print_stats_row("decoder_shard_total", collect(instrumented, [](const stage_sample & s) { return s.decoder_shard_us; }));
    print_stats_row("final_norm", collect(instrumented, [](const stage_sample & s) { return s.final_norm_us; }));
    print_stats_row("lm_head", collect(instrumented, [](const stage_sample & s) { return s.lm_head_us; }));
    print_stats_row("head_shard_total", collect(instrumented, [](const stage_sample & s) { return s.head_shard_us; }));
    print_stats_row("graph_sum", collect(instrumented, [](const stage_sample & s) { return s.graph_us; }));
    print_stats_row("instrumented_wall", collect(instrumented, [](const stage_sample & s) { return s.wall_us; }));
    print_stats_row("outside_timed_stages", collect(instrumented, [](const stage_sample & s) { return s.outside_stages_us; }));
}

static void write_csv_baseline(std::ostream & out, const std::string & workload, int32_t context_tokens, int32_t input_tokens, const std::vector<double> & samples) {
    for (size_t i = 0; i < samples.size(); ++i) {
        out << workload << ",baseline," << i << ',' << context_tokens << ',' << input_tokens
            << ",llama_decode_wall,-1," << samples[i] << "\n";
    }
}

static void write_csv_instrumented(std::ostream & out, const std::string & workload, int32_t context_tokens, int32_t input_tokens, const std::vector<stage_sample> & samples) {
    for (size_t i = 0; i < samples.size(); ++i) {
        const stage_sample & sample = samples[i];
        const auto row = [&](const std::string & component, int32_t layer, double value) {
            out << workload << ",instrumented," << i << ',' << context_tokens << ',' << input_tokens << ','
                << component << ',' << layer << ',' << value << "\n";
        };
        row("embedding", -1, sample.embedding_us);
        for (size_t layer = 0; layer < sample.layers_us.size(); ++layer) {
            row("decoder_layer", static_cast<int32_t>(layer), sample.layers_us[layer]);
        }
        row("decoder_layers_total", -1, sample.decoder_layers_us);
        row("decoder_shard_total", -1, sample.decoder_shard_us);
        row("final_norm", -1, sample.final_norm_us);
        row("lm_head", -1, sample.lm_head_us);
        row("head_shard_total", -1, sample.head_shard_us);
        row("graph_sum", -1, sample.graph_us);
        row("llama_decode_wall", -1, sample.wall_us);
        row("outside_timed_stages", -1, sample.outside_stages_us);
    }
}

static context_ptr make_context(llama_model * model, const options & opts, stage_timer * timer) {
    llama_context_params params = llama_context_default_params();
    params.n_ctx = opts.ctx_size;
    params.n_batch = opts.prompt_tokens;
    params.n_ubatch = opts.prompt_tokens;
    params.n_seq_max = 1;
    params.n_outputs_max = 1;
    params.n_threads = opts.threads;
    params.n_threads_batch = opts.threads;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    params.cb_eval = timer ? timing_callback : nullptr;
    params.cb_eval_user_data = timer;
    params.offload_kqv = false;
    params.no_perf = true;
    params.op_offload = false;
    params.kv_unified = true;
    return context_ptr(llama_init_from_model(model, params), llama_free);
}

int main(int argc, char ** argv) {
    options opts;
    if (!parse_options(argc, argv, opts)) {
        print_usage(argv[0]);
        return 1;
    }

    if (!opts.csv_path.empty()) {
        std::error_code error;
        const std::filesystem::file_status status = std::filesystem::symlink_status(opts.csv_path, error);
        if (error && error != std::errc::no_such_file_or_directory) {
            std::cerr << "failed to inspect CSV path: " << error.message() << "\n";
            return 1;
        }
        if (!error && status.type() != std::filesystem::file_type::not_found) {
            std::cerr << "refusing to overwrite CSV path: " << opts.csv_path << "\n";
            return 1;
        }
    }

    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    std::array<const char *, 2> paths = { opts.paths[0].c_str(), opts.paths[1].c_str() };
    model_ptr model(llama_model_load_from_splits(paths.data(), paths.size(), model_params), llama_model_free);
    if (!model) {
        std::cerr << "failed to load the two GGUF shards\n";
        llama_backend_free();
        return 1;
    }

    const int32_t n_layer = llama_model_n_layer(model.get());
    stage_timer timer(n_layer);
    context_ptr baseline_ctx = make_context(model.get(), opts, nullptr);
    context_ptr instrumented_ctx = make_context(model.get(), opts, &timer);
    if (!baseline_ctx || !instrumented_ctx) {
        std::cerr << "failed to create benchmark contexts\n";
        instrumented_ctx.reset();
        baseline_ctx.reset();
        model.reset();
        llama_backend_free();
        return 1;
    }

    const std::vector<llama_token> prompt = make_prompt(llama_model_get_vocab(model.get()), opts.prompt_tokens);
    if (prompt.size() != static_cast<size_t>(opts.prompt_tokens)) {
        std::cerr << "failed to construct the fixed-length prompt\n";
        instrumented_ctx.reset();
        baseline_ctx.reset();
        model.reset();
        llama_backend_free();
        return 1;
    }

    batch_owner prefill_batch(opts.prompt_tokens);
    fill_batch(prefill_batch.batch, prompt, 0);
    const std::vector<llama_token> decode_tokens = { prompt.back() };
    batch_owner decode_batch(1);
    fill_batch(decode_batch.batch, decode_tokens, opts.prompt_tokens);

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "[config] shards=2 cpu_only=1 callback_baseline=off callback_instrumented=on"
              << " requested_ctx=" << opts.ctx_size
              << " actual_ctx=" << llama_n_ctx(baseline_ctx.get())
              << " prompt_tokens=" << opts.prompt_tokens
              << " layers=" << n_layer
              << " threads=" << opts.threads
              << " warmup=" << opts.warmup << "\n";
    std::cout << "[note] baseline_wall is unsegmented production-style latency; instrumented_wall includes one synchronization per reported boundary.\n";
    std::cout << "[note] component values are boundary-synchronized stage wall times and include graph submission and synchronization, not pure kernel times.\n";

    std::vector<double> prefill_baseline;
    std::vector<stage_sample> prefill_instrumented;
    if (!run_prefill_baseline(baseline_ctx.get(), prefill_batch.batch, opts.warmup, opts.prefill_iters, prefill_baseline) ||
        !run_prefill_instrumented(instrumented_ctx.get(), timer, prefill_batch.batch, opts.warmup, opts.prefill_iters, prefill_instrumented)) {
        instrumented_ctx.reset();
        baseline_ctx.reset();
        model.reset();
        llama_backend_free();
        return 1;
    }

    if (!prime_decode_context(baseline_ctx.get(), nullptr, prefill_batch.batch) ||
        !prime_decode_context(instrumented_ctx.get(), &timer, prefill_batch.batch)) {
        instrumented_ctx.reset();
        baseline_ctx.reset();
        model.reset();
        llama_backend_free();
        return 1;
    }

    std::vector<double> decode_baseline;
    std::vector<stage_sample> decode_instrumented;
    if (!run_decode_baseline(baseline_ctx.get(), decode_batch.batch, opts.prompt_tokens, opts.warmup, opts.decode_iters, decode_baseline) ||
        !run_decode_instrumented(instrumented_ctx.get(), timer, decode_batch.batch, opts.prompt_tokens, opts.warmup, opts.decode_iters, decode_instrumented)) {
        instrumented_ctx.reset();
        baseline_ctx.reset();
        model.reset();
        llama_backend_free();
        return 1;
    }

    print_workload_summary("prefill", prefill_baseline, prefill_instrumented, n_layer);
    print_workload_summary("decode", decode_baseline, decode_instrumented, n_layer);

    if (!opts.csv_path.empty()) {
        std::ofstream csv(opts.csv_path);
        if (!csv) {
            std::cerr << "failed to open CSV path: " << opts.csv_path << "\n";
            instrumented_ctx.reset();
            baseline_ctx.reset();
            model.reset();
            llama_backend_free();
            return 1;
        }
        csv << std::fixed << std::setprecision(3);
        csv << "workload,mode,iteration,context_tokens,input_tokens,component,layer,time_us\n";
        write_csv_baseline(csv, "prefill", 0, opts.prompt_tokens, prefill_baseline);
        write_csv_instrumented(csv, "prefill", 0, opts.prompt_tokens, prefill_instrumented);
        write_csv_baseline(csv, "decode", opts.prompt_tokens, 1, decode_baseline);
        write_csv_instrumented(csv, "decode", opts.prompt_tokens, 1, decode_instrumented);
        if (!csv) {
            std::cerr << "failed while writing CSV: " << opts.csv_path << "\n";
            instrumented_ctx.reset();
            baseline_ctx.reset();
            model.reset();
            llama_backend_free();
            return 1;
        }
        std::cout << "\n[csv] " << opts.csv_path << "\n";
    }

    instrumented_ctx.reset();
    baseline_ctx.reset();
    model.reset();
    llama_backend_free();
    return 0;
}
