set pagination off
set print pretty on
set print object on
set print elements 64
set print repeats 0
set breakpoint pending on
set confirm off
set debuginfod enabled off
set logging overwrite on
init-if-undefined $server_trace = 0
if $server_trace
  set logging file llama_gdb_server_session.log
else
  set logging file llama_gdb_session.log
end
set logging redirect on
set logging enabled on

set $decode_id = 0
set $ubatch_id = 0
set $mask_id = 0
set $graph_id = 0

break llama_decode
commands
  silent
  set $decode_id = $decode_id + 1
  printf "\n===== LLAMA_DECODE id=%d =====\n", $decode_id
  bt 12
  printf "batch n_tokens=%d token=%p embd=%p pos=%p seq_id=%p n_seq_id=%p logits=%p\n", batch.n_tokens, batch.token, batch.embd, batch.pos, batch.seq_id, batch.n_seq_id, batch.logits
  set $i = 0
  while $i < batch.n_tokens
    set $tok = batch.token ? batch.token[$i] : -1
    set $pos = batch.pos ? batch.pos[$i] : -1
    set $ns = batch.n_seq_id ? batch.n_seq_id[$i] : 1
    set $sid0 = batch.seq_id ? batch.seq_id[$i][0] : 0
    set $out = batch.logits ? batch.logits[$i] : -1
    printf "batch_token i=%d token=%d pos=%d n_seq_id=%d seq0=%d logits=%d\n", $i, $tok, $pos, $ns, $sid0, $out
    if $ns > 1
      set $si = 1
      while $si < $ns
        printf "  seq[%d]=%d\n", $si, batch.seq_id[$i][$si]
        set $si = $si + 1
      end
    end
    set $i = $i + 1
  end
  continue
end

break llama_context::process_ubatch
commands
  silent
  set $ubatch_id = $ubatch_id + 1
  printf "\n===== PROCESS_UBATCH id=%d decode_id=%d =====\n", $ubatch_id, $decode_id
  bt 10
  printf "gtype=%d mctx=%p n_tokens=%u n_seq_tokens=%u n_seqs=%u n_seqs_unq=%u n_pos=%u equal=%u\n", gtype, mctx, ubatch.n_tokens, ubatch.n_seq_tokens, ubatch.n_seqs, ubatch.n_seqs_unq, ubatch.n_pos, ubatch.b_equal_seqs
  printf "context kv_unified=%d causal_attn=%d flash_attn=%d n_batch=%u n_ubatch=%u\n", this->cparams.kv_unified, this->cparams.causal_attn, this->cparams.flash_attn, this->cparams.n_batch, this->cparams.n_ubatch
  set $i = 0
  while $i < ubatch.n_tokens
    set $tok = ubatch.token ? ubatch.token[$i] : -1
    set $pos = ubatch.pos ? ubatch.pos[$i] : -1
    set $ns = ubatch.n_seq_id[$i]
    set $sid0 = ubatch.seq_id[$i][0]
    set $out = ubatch.output[$i]
    printf "ubatch_token i=%d token=%d pos=%d n_seq_id=%d seq0=%d output=%d\n", $i, $tok, $pos, $ns, $sid0, $out
    if $ns > 1
      set $si = 1
      while $si < $ns
        printf "  seq[%d]=%d\n", $si, ubatch.seq_id[$i][$si]
        set $si = $si + 1
      end
    end
    set $i = $i + 1
  end
  continue
end

break llm_graph_input_embd::set_input
commands
  silent
  printf "\n--- EMBED_INPUT ubatch_id=%d n_tokens=%u ---\n", $ubatch_id, ubatch->n_tokens
  if tokens
    printf "tokens type=%d ne=[%lld,%lld,%lld,%lld] nb=[%llu,%llu,%llu,%llu]\n", tokens->type, tokens->ne[0], tokens->ne[1], tokens->ne[2], tokens->ne[3], tokens->nb[0], tokens->nb[1], tokens->nb[2], tokens->nb[3]
  end
  if embd
    printf "embd type=%d ne=[%lld,%lld,%lld,%lld] nb=[%llu,%llu,%llu,%llu]\n", embd->type, embd->ne[0], embd->ne[1], embd->ne[2], embd->ne[3], embd->nb[0], embd->nb[1], embd->nb[2], embd->nb[3]
  end
  continue
end

break llm_graph_input_embd_h::set_input
commands
  silent
  printf "\n--- EMBED_H_INPUT ubatch_id=%d n_tokens=%u ---\n", $ubatch_id, ubatch->n_tokens
  if tokens
    printf "tokens type=%d ne=[%lld,%lld,%lld,%lld]\n", tokens->type, tokens->ne[0], tokens->ne[1], tokens->ne[2], tokens->ne[3]
  end
  if embd
    printf "embd type=%d ne=[%lld,%lld,%lld,%lld]\n", embd->type, embd->ne[0], embd->ne[1], embd->ne[2], embd->ne[3]
  end
  if h
    printf "h type=%d ne=[%lld,%lld,%lld,%lld]\n", h->type, h->ne[0], h->ne[1], h->ne[2], h->ne[3]
  end
  continue
end

break /home/qwe/workspace/llama.cpp/src/llama-graph.cpp:2344
commands
  silent
  printf "FIRST_HIDDEN ubatch_id=%d type=%d ne=[%lld,%lld,%lld,%lld] nb=[%llu,%llu,%llu,%llu]\n", $ubatch_id, cur->type, cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3], cur->nb[0], cur->nb[1], cur->nb[2], cur->nb[3]
  continue
end

break llm_graph_input_pos::set_input
commands
  silent
  printf "\n--- POSITION_INPUT ubatch_id=%d n_tokens=%u n_pos_per_embd=%u ---\n", $ubatch_id, ubatch->n_tokens, n_pos_per_embd
  printf "pos_tensor type=%d ne=[%lld,%lld,%lld,%lld] nb=[%llu,%llu,%llu,%llu]\n", pos->type, pos->ne[0], pos->ne[1], pos->ne[2], pos->ne[3], pos->nb[0], pos->nb[1], pos->nb[2], pos->nb[3]
  set $i = 0
  while $i < ubatch->n_tokens
    printf "position i=%d p0=%d\n", $i, ubatch->pos[$i]
    set $i = $i + 1
  end
  continue
end

break llm_graph_input_attn_kv::set_input
commands
  silent
  printf "\n--- ATTN_KV_INPUT ubatch_id=%d n_tokens=%u causal=%d ---\n", $ubatch_id, ubatch->n_tokens, cparams.causal_attn
  if self_kq_mask
    printf "self_kq_mask type=%d ne=[%lld,%lld,%lld,%lld] nb=[%llu,%llu,%llu,%llu]\n", self_kq_mask->type, self_kq_mask->ne[0], self_kq_mask->ne[1], self_kq_mask->ne[2], self_kq_mask->ne[3], self_kq_mask->nb[0], self_kq_mask->nb[1], self_kq_mask->nb[2], self_kq_mask->nb[3]
  end
  continue
end

break llm_graph_input_attn_no_cache::set_input
commands
  silent
  printf "\n--- ATTN_NO_CACHE_INPUT ubatch_id=%d n_tokens=%u causal=%d ---\n", $ubatch_id, ubatch->n_tokens, cparams.causal_attn
  if self_kq_mask
    printf "self_kq_mask type=%d ne=[%lld,%lld,%lld,%lld] nb=[%llu,%llu,%llu,%llu]\n", self_kq_mask->type, self_kq_mask->ne[0], self_kq_mask->ne[1], self_kq_mask->ne[2], self_kq_mask->ne[3], self_kq_mask->nb[0], self_kq_mask->nb[1], self_kq_mask->nb[2], self_kq_mask->nb[3]
  end
  continue
end

break /home/qwe/workspace/llama.cpp/src/llama-kv-cache.cpp:2564
commands
  silent
  printf "\n--- KV_APPLIED ubatch_id=%d i_cur=%llu n_kv=%u ---\n", $ubatch_id, i_cur, n_kv
  printf "slot s0=%u s1=%u\n", sinfos[i_cur].s0, sinfos[i_cur].s1
  p sinfos[i_cur].strm
  p sinfos[i_cur].idxs
  continue
end

# This line is the epilogue of each concrete mask-fill template instantiation.
break /home/qwe/workspace/llama.cpp/src/llama-kv-cache.cpp:1684
commands
  silent
  set $mask_id = $mask_id + 1
  printf "\n===== MASK_FILLED id=%d ubatch_id=%d =====\n", $mask_id, $ubatch_id
  printf "element_bytes=%u n_kv=%lld n_stream=%lld n_tps=%lld\n", sizeof(*data), args.n_kv, args.n_stream, args.n_tps
  printf "n_tokens=%u n_seqs_unq=%u n_pos=%u\n", args.ubatch->n_tokens, args.ubatch->n_seqs_unq, args.ubatch->n_pos
  set $nprintq = args.ubatch->n_tokens
  if $nprintq > 8
    set $nprintq = 8
  end
  set $nprintkv = args.n_kv
  if $nprintkv > 32
    set $nprintkv = 32
  end
  set $qi = 0
  while $qi < $nprintq
    printf "mask_row q=%d pos=%d seq=%d :", $qi, args.ubatch->pos[$qi], args.ubatch->seq_id[$qi][0]
    set $kj = 0
    while $kj < $nprintkv
      if sizeof(*data) == 4
        printf " %g", ((float *) data)[$qi * args.n_kv + $kj]
      else
        printf " 0x%x", ((unsigned short *) data)[$qi * args.n_kv + $kj]
      end
      set $kj = $kj + 1
    end
    printf "\n"
    set $qi = $qi + 1
  end
  printf "kv_stream0_positions_and_seq_bits:\n"
  p args.v_cells[0].pos
  p args.v_cells[0].seq
  continue
end

break /home/qwe/workspace/llama.cpp/src/llama-graph.cpp:2519
commands
  silent
  if il == 0
    set $graph_id = $graph_id + 1
    printf "\n===== ATTN_MHA_GRAPH id=%d ubatch_id=%d layer=%d =====\n", $graph_id, $ubatch_id, il
    printf "Q_after_permute type=%d ne=[%lld,%lld,%lld,%lld] nb=[%llu,%llu,%llu,%llu]\n", q->type, q->ne[0], q->ne[1], q->ne[2], q->ne[3], q->nb[0], q->nb[1], q->nb[2], q->nb[3]
    printf "K_after_permute type=%d ne=[%lld,%lld,%lld,%lld] nb=[%llu,%llu,%llu,%llu]\n", k->type, k->ne[0], k->ne[1], k->ne[2], k->ne[3], k->nb[0], k->nb[1], k->nb[2], k->nb[3]
    printf "V_after_permute type=%d ne=[%lld,%lld,%lld,%lld] nb=[%llu,%llu,%llu,%llu]\n", v->type, v->ne[0], v->ne[1], v->ne[2], v->ne[3], v->nb[0], v->nb[1], v->nb[2], v->nb[3]
    printf "MASK type=%d ne=[%lld,%lld,%lld,%lld] nb=[%llu,%llu,%llu,%llu]\n", kq_mask->type, kq_mask->ne[0], kq_mask->ne[1], kq_mask->ne[2], kq_mask->ne[3], kq_mask->nb[0], kq_mask->nb[1], kq_mask->nb[2], kq_mask->nb[3]
  end
  continue
end

break /home/qwe/workspace/llama.cpp/src/llama-graph.cpp:2757
commands
  silent
  if il == 0
    printf "ATTN_INPUT_Q name=%s type=%d ne=[%lld,%lld,%lld,%lld] nb=[%llu,%llu,%llu,%llu]\n", q_cur->name, q_cur->type, q_cur->ne[0], q_cur->ne[1], q_cur->ne[2], q_cur->ne[3], q_cur->nb[0], q_cur->nb[1], q_cur->nb[2], q_cur->nb[3]
    printf "ATTN_INPUT_K name=%s type=%d ne=[%lld,%lld,%lld,%lld] nb=[%llu,%llu,%llu,%llu]\n", k_cur->name, k_cur->type, k_cur->ne[0], k_cur->ne[1], k_cur->ne[2], k_cur->ne[3], k_cur->nb[0], k_cur->nb[1], k_cur->nb[2], k_cur->nb[3]
    printf "ATTN_INPUT_V name=%s type=%d ne=[%lld,%lld,%lld,%lld] nb=[%llu,%llu,%llu,%llu]\n", v_cur->name, v_cur->type, v_cur->ne[0], v_cur->ne[1], v_cur->ne[2], v_cur->ne[3], v_cur->nb[0], v_cur->nb[1], v_cur->nb[2], v_cur->nb[3]
  end
  continue
end

break /home/qwe/workspace/llama.cpp/src/llama-graph.cpp:2566
commands
  silent
  if il == 0
    printf "KQ layer=%d type=%d ne=[%lld,%lld,%lld,%lld] nb=[%llu,%llu,%llu,%llu]\n", il, kq->type, kq->ne[0], kq->ne[1], kq->ne[2], kq->ne[3], kq->nb[0], kq->nb[1], kq->nb[2], kq->nb[3]
  end
  continue
end

break /home/qwe/workspace/llama.cpp/src/llama-graph.cpp:2601
commands
  silent
  if il == 0
    printf "SOFTMAX_KQ layer=%d type=%d ne=[%lld,%lld,%lld,%lld] nb=[%llu,%llu,%llu,%llu]\n", il, kq->type, kq->ne[0], kq->ne[1], kq->ne[2], kq->ne[3], kq->nb[0], kq->nb[1], kq->nb[2], kq->nb[3]
  end
  continue
end

break /home/qwe/workspace/llama.cpp/src/llama-graph.cpp:2610
commands
  silent
  if il == 0
    printf "KQV layer=%d type=%d ne=[%lld,%lld,%lld,%lld] nb=[%llu,%llu,%llu,%llu]\n", il, kqv->type, kqv->ne[0], kqv->ne[1], kqv->ne[2], kqv->ne[3], kqv->nb[0], kqv->nb[1], kqv->nb[2], kqv->nb[3]
  end
  continue
end

break /home/qwe/workspace/llama.cpp/src/llama-graph.cpp:2623
commands
  silent
  if il == 0
    printf "ATTN_OUT layer=%d type=%d ne=[%lld,%lld,%lld,%lld] nb=[%llu,%llu,%llu,%llu]\n", il, cur->type, cur->ne[0], cur->ne[1], cur->ne[2], cur->ne[3], cur->nb[0], cur->nb[1], cur->nb[2], cur->nb[3]
  end
  continue
end

break llama_context::graph_compute
commands
  silent
  printf "\n--- BACKEND_COMPUTE ubatch_id=%d batched=%d ---\n", $ubatch_id, batched
  bt 8
  continue
end

break llama_model_qwen3vl::graph::graph
commands
  silent
  printf "\n===== QWEN3VL_GRAPH n_tokens=%lld n_head=%lld n_head_kv=%lld n_rot=%lld rope_type=%d =====\n", n_tokens, n_head, n_head_kv, n_rot, rope_type
  continue
end

run

set logging enabled off
