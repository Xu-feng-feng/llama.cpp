import argparse
import os
import threading

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer


def parse_args():
    parser = argparse.ArgumentParser(description="Run Qwen inference with transformers")
    parser.add_argument(
        "--model",
        default="qwen3-1.7b",
        help="Model path or Hugging Face repo id",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Single prompt to run (if omitted, use interactive mode)",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.6, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--top-k", type=int, default=0, help="Top-k sampling")
    parser.add_argument("--repetition-penalty", type=float, default=1.05, help="Repetition penalty")
    parser.add_argument("--do-sample", action="store_true", help="Enable sampling mode")
    parser.add_argument("--no-sample", dest="do_sample", action="store_false", help="Disable sampling mode")
    parser.set_defaults(do_sample=False)
    parser.add_argument("--dtype", default="float16", choices=["float16", "bfloat16", "float32", "auto"], help="Model dtype")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu", "0", "1", "2", "3"], help="Run device")
    parser.add_argument("--stream", action="store_true", default=False, help="Print tokens as they are generated")
    parser.add_argument("--trust-remote-code", action="store_true", default=True, help="Enable trust_remote_code")
    parser.add_argument("--no-trust-remote-code", dest="trust_remote_code", action="store_false", help="Disable trust_remote_code")
    parser.add_argument(
        "--show-system-hints",
        action="store_true",
        default=False,
        help="Print environment checks and common fixes before running",
    )
    return parser.parse_args()


def select_dtype(dtype_name):
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    return None


def build_model(model_name, trust_remote_code, dtype, device_name):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    torch_dtype = select_dtype(dtype)
    kwargs = {"trust_remote_code": trust_remote_code}
    if torch_dtype is not None:
        kwargs["dtype"] = torch_dtype

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if device_name != "auto":
        if device_name == "cuda":
            model = model.to("cuda")
        elif device_name == "cpu":
            model = model.to("cpu")
        else:
            model = model.to(f"cuda:{device_name}")
    elif torch.cuda.is_available():
        model = model.to("cuda")
    return tokenizer, model


def system_hints():
    has_python_h = os.path.exists("/usr/include/python3.12/Python.h")
    if has_python_h:
        print("[info] Python.h found, triton compile on CUDA should be less likely to fail")
        return

    print("[warn] Missing /usr/include/python3.12/Python.h")
    print("[warn] Triton GPU kernels in torch may fail with:")
    print("       fatal error: Python.h: No such file or directory")
    print("[hint] Install dev headers (Debian/Ubuntu): sudo apt-get install python3.12-dev")
    print("[hint] If CUDA kernels still fail, try: python3 qwen_infer.py --device cpu --prompt ...")


def build_inputs(tokenizer, prompt, model):
    inputs = tokenizer(prompt, return_tensors="pt")
    return {k: v.to(model.device) for k, v in inputs.items()}


def generate_text(tokenizer, model, prompt, args):
    inputs = build_inputs(tokenizer, prompt, model)

    gen_args = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )

    if args.stream:
        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        gen_kwargs = {**inputs, **gen_args, "streamer": streamer}
        thread = threading.Thread(target=model.generate, kwargs=gen_kwargs)
        thread.start()
        print("assistant: ", end="", flush=True)
        for chunk in streamer:
            print(chunk, end="", flush=True)
        thread.join()
        print()
        return

    with torch.no_grad():
        out = model.generate(**inputs, **gen_args)

    output_ids = out[0][inputs["input_ids"].shape[1] :]
    text = tokenizer.decode(output_ids, skip_special_tokens=True)
    print("assistant:", text)


def run_interactive(tokenizer, model, args):
    print("Interactive mode. type /exit to quit.")
    while True:
        prompt = input("> ").strip()
        if not prompt:
            continue
        if prompt in {"/exit", "exit", "quit", "q"}:
            break
        generate_text(tokenizer, model, prompt, args)


def main():
    args = parse_args()
    if args.show_system_hints:
        system_hints()
    tokenizer, model = build_model(
        model_name=args.model,
        trust_remote_code=args.trust_remote_code,
        dtype=args.dtype,
        device_name=args.device,
    )

    if args.prompt:
        generate_text(tokenizer, model, args.prompt, args)
    else:
        run_interactive(tokenizer, model, args)


if __name__ == "__main__":
    main()
