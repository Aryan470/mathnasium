import argparse, torch, json, readline  # noqa: F401 (readline = nicer CLI history)
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def load_model(model_id: str, fourbit: bool):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        "attn_implementation": "flash_attention_2",
    }
    if fourbit:
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        kwargs["quantization_config"] = bnb
        kwargs.pop("torch_dtype", None)  # bnb controls compute dtype
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    return tok, model

import time, torch

def generate_with_tps(model, tok, messages, max_new_tokens=512, temp=0.7, top_p=0.9):
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)

    # warmup helps stabilize kernels & caches (skip counting)
    with torch.inference_mode():
        _ = model.generate(**inputs, max_new_tokens=1)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()

    with torch.inference_mode():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temp,
            top_p=top_p,
            eos_token_id=tok.eos_token_id,
            pad_token_id=tok.eos_token_id,
        )

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    # count only newly generated tokens
    gen_len = out.shape[1] - inputs["input_ids"].shape[1]
    tok_per_sec = gen_len / (t1 - t0)

    text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text.strip(), gen_len, tok_per_sec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--fourbit", action="store_true", help="load 4-bit quantized (bitsandbytes)")
    ap.add_argument("--system", default="You are a helpful assistant.")
    ap.add_argument("--max-new", type=int, default=512)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--save", default="", help="save conversation to JSONL")
    args = ap.parse_args()

    tok, model = load_model(args.model, args.fourbit)
    messages = [{"role":"system","content":args.system}]
    print(f"Loaded {args.model}. Type /exit to quit, /reset to clear history.\n")

    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user == "/exit":
            break
        if user == "/reset":
            messages = [{"role":"system","content":args.system}]
            print("(history cleared)")
            continue

        messages.append({"role":"user","content":user})
        reply, gen_len, tok_per_sec = generate_with_tps(model, tok, messages, args.max_new, args.temp, args.top_p)
        messages.append({"role":"assistant","content":reply})
        print(f"bot> {reply} ({(gen_len/1024):.2f} KB, {tok_per_sec:.2f} tok/s)\n")

        if args.save:
            with open(args.save, "a", encoding="utf-8") as f:
                f.write(json.dumps({"messages": messages[-2:]}) + "\n")

if __name__ == "__main__":
    main()
