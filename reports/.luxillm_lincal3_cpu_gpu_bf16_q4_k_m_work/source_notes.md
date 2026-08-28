# Source notes

Generated: 2026-08-19T19:56:04+08:00

## Evidence inventory

- Model config: `1.0.05.2-luxi-1.7B-lincal/config.json`
- BF16 GGUF: `1.0.05.2-luxi-1.7B-lincal/1.0.05.2-luxi-1.7B-lincal-lincal3-BF16.gguf`
- Q4_K_M GGUF: `1.0.05.2-luxi-1.7B-lincal/1.0.05.2-luxi-1.7B-lincal-lincal3-Q4_K_M.gguf`
- BF16 conversion log: `logs/lincal3/convert-bf16.log`
- Q4_K_M quantization log: `logs/lincal3/quantize-q4_k_m.log`
- CPU benchmark binary: `build-lincal3/bin/llama-bench`
- CUDA benchmark binary: `build-lincal3-cuda/bin/llama-bench`
- CPU: 12th Gen Intel(R) Core(TM) i5-12600KF
- GPU: NVIDIA GeForce RTX 3090, driver 580.173.02
- Architecture: Lincal3, 28 layers, 24 sliding-attention layers, 4 full-attention layers
- Sliding window: 512 tokens

## Benchmark definitions

- pp512 and pp2048 are prompt-processing throughput tests in tokens/s.
- tg128 is autoregressive token-generation throughput in tokens/s.
- Prompt-processing duration is prompt tokens divided by pp throughput; decode time per token is 1000 divided by tg throughput.
- Each throughput result is the llama-bench mean of 3 repetitions after warm-up.
- CPU peak RSS is the maximum resident set size reported by `/usr/bin/time -v` for the full benchmark process.
- GPU peak VRAM is the maximum device memory observed at 100 ms sampling minus the pre-run device baseline.
- Active GPU utilization and power average samples with utilization above zero; power is whole-board power reported by nvidia-smi.
- KV cache uses F16 K and V. Full-attention and sliding-attention allocations are parsed from llama.cpp runtime logs.
- `raw/cuda_backend_probe.log` confirms `offloaded 29/29 layers to GPU` and the CUDA backend.

## Exact benchmark commands

### CPU BF16

```sh
/home/qwe/workspace/llama.cpp/build-lincal3/bin/llama-bench -m /home/qwe/workspace/llama.cpp/1.0.05.2-luxi-1.7B-lincal/1.0.05.2-luxi-1.7B-lincal-lincal3-BF16.gguf -p 512,2048 -n 128 -t 8 -ngl 0 -fa on -ctk f16 -ctv f16 -r 3 -o json
```

### CPU Q4_K_M

```sh
/home/qwe/workspace/llama.cpp/build-lincal3/bin/llama-bench -m /home/qwe/workspace/llama.cpp/1.0.05.2-luxi-1.7B-lincal/1.0.05.2-luxi-1.7B-lincal-lincal3-Q4_K_M.gguf -p 512,2048 -n 128 -t 8 -ngl 0 -fa on -ctk f16 -ctv f16 -r 3 -o json
```

### GPU BF16

```sh
/home/qwe/workspace/llama.cpp/build-lincal3-cuda/bin/llama-bench -m /home/qwe/workspace/llama.cpp/1.0.05.2-luxi-1.7B-lincal/1.0.05.2-luxi-1.7B-lincal-lincal3-BF16.gguf -p 512,2048 -n 128 -t 8 -ngl 99 -fa on -ctk f16 -ctv f16 -r 3 -o json
```

### GPU Q4_K_M

```sh
/home/qwe/workspace/llama.cpp/build-lincal3-cuda/bin/llama-bench -m /home/qwe/workspace/llama.cpp/1.0.05.2-luxi-1.7B-lincal/1.0.05.2-luxi-1.7B-lincal-lincal3-Q4_K_M.gguf -p 512,2048 -n 128 -t 8 -ngl 99 -fa on -ctk f16 -ctv f16 -r 3 -o json
```
