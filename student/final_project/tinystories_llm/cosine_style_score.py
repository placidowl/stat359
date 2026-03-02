import argparse
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel


def mean_pool(last_hidden_state, attention_mask):
    # mean pooling over tokens (mask padding)
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


@torch.no_grad()
def embed_texts(texts, tokenizer, model, device, batch_size=16, max_len=256):
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_len,
            return_tensors="pt"
        ).to(device)

        out = model(**enc)
        pooled = mean_pool(out.last_hidden_state, enc["attention_mask"])
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)  # normalize for cosine
        embs.append(pooled.cpu())

    return torch.cat(embs, dim=0).numpy()


def cosine(a, b):
    # a: (N, d), b: (d,)
    return (a @ b) / (np.linalg.norm(b) + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", required=True, help="CSV containing outputs (must have 'output' column)")
    ap.add_argument("--ref_file", required=True, help="txt file: one pirate reference text per line")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--model_name", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=256)
    args = ap.parse_args()

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    if device == "auto":
        device = "cpu"

    df = pd.read_csv(args.input_csv)
    if "output" not in df.columns:
        raise ValueError(f"{args.input_csv} must have an 'output' column. Found: {df.columns.tolist()}")

    # load refs
    refs = [line.strip() for line in open(args.ref_file, "r", encoding="utf-8") if line.strip()]
    if len(refs) < 5:
        raise ValueError("Provide at least ~5 reference pirate texts in ref_file.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(device)
    model.eval()

    # embed references and create target style vector
    ref_embs = embed_texts(refs, tokenizer, model, device, batch_size=args.batch_size, max_len=args.max_len)
    target = ref_embs.mean(axis=0)
    target = target / (np.linalg.norm(target) + 1e-9)

    # embed outputs
    outs = df["output"].fillna("").astype(str).tolist()
    out_embs = embed_texts(outs, tokenizer, model, device, batch_size=args.batch_size, max_len=args.max_len)

    # cosine similarity scores
    df["cosine_style_score"] = cosine(out_embs, target)

    df.to_csv(args.out_csv, index=False)
    print(f"Saved cosine-scored CSV -> {args.out_csv}")
    print("Mean cosine_style_score:", float(df["cosine_style_score"].mean()))


if __name__ == "__main__":
    main()