import os
import json
import csv
import re
import argparse
import torch

from bpe_tokenizer import BPETokenizer
from transformer_model import TinyStoriesConfig, TinyStoriesForCausalLM


print("STYLE_EVAL FILE LOADED")
# ---- simple style metric ----
PIRATE_WORDS = {
    "arr", "arrr", "matey", "aye", "booty", "treasure", "ship", "sea",
    "captain", "crew", "anchor", "cannon", "parrot", "sail", "sailing"
}

def words(text: str):
    return re.findall(r"[a-zA-Z']+", text.lower())

def pirate_style_score(text: str) -> float:
    toks = words(text)
    if not toks:
        return 0.0
    hits = sum(1 for t in toks if t in PIRATE_WORDS)
    return hits / len(toks)

def contains_pirate(text: str) -> int:
    s = set(words(text))
    return int(any(w in s for w in PIRATE_WORDS))


def load_model_and_tokenizer(model_path: str, tokenizer_path: str, device: str):
    # infer output_dir to read args.json (so config matches training)
    out_dir = os.path.dirname(model_path)
    args_path = os.path.join(out_dir, "args.json")
    if not os.path.exists(args_path):
        raise FileNotFoundError(f"Could not find args.json next to checkpoint: {args_path}")

    with open(args_path, "r") as f:
        train_args = json.load(f)

    tokenizer = BPETokenizer.load(tokenizer_path)
    
    state = torch.load(model_path, map_location="cpu")

    # infer true max_position_embeddings from checkpoint
    # (name might differ, so we try common keys)
    pos_key_candidates = [
        "transformer.wpe.weight",
        "wpe.weight",
        "position_embeddings.weight",
        "transformer.position_embeddings.weight",
    ]
    pos_len = None
    for k in pos_key_candidates:
        if k in state:
            pos_len = state[k].shape[0]
            break

    if pos_len is None:
        pos_len = train_args.get("max_seq_len", 256)

    config = TinyStoriesConfig(
        vocab_size=len(tokenizer.token2id),
        hidden_size=train_args["hidden_size"],
        num_hidden_layers=train_args["num_layers"],
        num_attention_heads=train_args["num_heads"],
        intermediate_size=train_args["intermediate_size"],
        hidden_dropout_prob=train_args["dropout"],
        attention_probs_dropout_prob=train_args["dropout"],
        max_position_embeddings=pos_len,   # ✅ use real value from checkpoint
        window_size=train_args["window_size"],
    )
    print("pos_len from checkpoint =", pos_len)
    print("train_args max_seq_len  =", train_args.get("max_seq_len"))

    model = TinyStoriesForCausalLM(config)
    state = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, prompt, device, max_new_tokens, temperature, top_p, top_k=0):
    input_ids = torch.tensor(
        [tokenizer.encode(prompt, add_special_tokens=True)],
        device=device
    )

    # Always keep prompt <= 255 so we never hit 257 during generation
    input_ids = input_ids[:, -255:]
    prompt_len = input_ids.shape[1]

    max_length = min(prompt_len + max_new_tokens, 256)

    with torch.no_grad():
        out_ids = model.generate(
            input_ids=input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

    return tokenizer.decode(out_ids[0].tolist())


def main():
    print("STYLE_EVAL MAIN START")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--tokenizer_path", default="bpe_tokenizer_tinystories.pkl")
    ap.add_argument("--out_csv", default="style_drift_results.csv")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--max_new_tokens", type=int, default=120)
    args = ap.parse_args()

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    if device == "auto":
        device = "cpu"

    model, tokenizer = load_model_and_tokenizer(args.model_path, args.tokenizer_path, device)

    # ---- stress test prompts ----
    tests = [
        ("baseline_short", "Write a pirate story in pirate voice."),
        ("constraint_words", "Write a pirate story. Use the words: arrr, matey, treasure, ship."),
        ("repeat_exact", "Repeat exactly: arr matey treasure ship"),
        ("format_bullets", "In pirate voice, give 5 bullet points about treasure hunting."),
        ("seed_continue", "Arr matey! said the pirate. Arr matey! he sailed the ship to find treasure. Continue the story:"),
    ]

    # ---- stress levels: short vs long generation ----
    lengths = [30, 60, 120] 

    rows = []
    for (name, prompt) in tests:
        for L in lengths:
            out = generate(
                model, tokenizer, prompt, device=device,
                max_new_tokens=L,
                temperature=args.temperature,
                top_p=args.top_p
            )

            score = pirate_style_score(out)
            anyhit = contains_pirate(out)

            rows.append({
                "test": name,
                "max_new_tokens": L,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "pirate_style_score": score,
                "contains_pirate_word": anyhit,
                "prompt": prompt,
                "output": out,
                "model_path": args.model_path,
            })
            print(f"[{name} | L={L}] score={score:.4f} anyhit={anyhit}")

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: {args.out_csv}")

if __name__ == "__main__":
    main()