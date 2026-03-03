import json
from pathlib import Path

import torch
import torch.nn as nn
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer


# =========================
# Config
# =========================
MODEL_DIR = Path("pirate_bert_output")
MODEL_WEIGHTS = MODEL_DIR / "best_model.pt"

INPUT_FILE = Path("results/results_base.json")          
OUTPUT_FILE = Path("prompt_output_scored.json")

MODEL_NAME = "bert-base-uncased"
MAX_LEN = 128


# =========================
# Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU name: {torch.cuda.get_device_name(0)}")


# =========================
# Model
# =========================
class BertBinaryClassifier(nn.Module):
    def __init__(self, model_name="bert-base-uncased", dropout=0.3):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            pooled = outputs.last_hidden_state[:, 0]

        x = self.dropout(pooled)
        logits = self.classifier(x)
        return logits


# =========================
# Load tokenizer + model
# =========================
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = BertBinaryClassifier(MODEL_NAME)
model.load_state_dict(torch.load(MODEL_WEIGHTS, map_location=device))
model.to(device)
model.eval()

print("Loaded trained model successfully.")


# =========================
# Helpers
# =========================
def normalize_text(text):
    return " ".join(str(text).strip().split())


def extract_assistant_lines(raw_output: str):
    results = []
    for line in str(raw_output).splitlines():
        if "Assistant:" in line:
            assistant_part = line.split("Assistant:", 1)[1].strip()
            if assistant_part:
                results.append(normalize_text(assistant_part))
    return results


def get_rows_to_score(data):
    """
    Supports:
    1) single items:
       {"id": "...", "category": "...", "prompt": "...", "response": "...", "raw_output": "..."}
    2) multi-turn items:
       {"id": "...", "category": "...", "turns": [{"prompt": "...", "response": "..."}, ...]}
    """
    rows = []

    for item in data:
        item_id = item.get("id", "unknown")
        category = item.get("category", "unknown")

        # Case 1: single response item
        if "response" in item:
            prompt = normalize_text(item.get("prompt", ""))
            response = item.get("response", "")
            raw_output = item.get("raw_output", "")

            if (not response) or ("Assistant:" in str(response)) or ("Welcome to TinyStories Chat!" in str(response)):
                lines = extract_assistant_lines(raw_output or response)
                response = lines[-1] if lines else response

            response = normalize_text(response)

            if response:
                rows.append({
                    "id": item_id,
                    "category": category,
                    "prompt": prompt,
                    "text": response
                })

        # Case 2: multi-turn item
        elif "turns" in item:
            turns = item.get("turns", [])

            if turns:
                for i, turn in enumerate(turns, start=1):
                    prompt = normalize_text(turn.get("prompt", ""))
                    response = normalize_text(turn.get("response", ""))

                    if response:
                        rows.append({
                            "id": f"{item_id}_turn{i}",
                            "category": category,
                            "prompt": prompt,
                            "text": response
                        })
            else:
                lines = extract_assistant_lines(item.get("raw_output", ""))
                for i, resp in enumerate(lines, start=1):
                    rows.append({
                        "id": f"{item_id}_turn{i}",
                        "category": category,
                        "prompt": "",
                        "text": resp
                    })

    return rows


def score_text(text):
    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(logits, dim=-1)[0].cpu().tolist()

    pred_label = int(torch.argmax(torch.tensor(probs)).item())

    return {
        "prob_not_pirate": round(probs[0], 6),
        "prob_pirate": round(probs[1], 6),
        "pred_label": pred_label
    }


# =========================
# Main
# =========================
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

rows = get_rows_to_score(data)
print(f"Found {len(rows)} responses to score.")

scored = []
for row in tqdm(rows, desc="Scoring"):
    result = score_text(row["text"])
    scored.append({
        **row,
        **result
    })

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(scored, f, indent=2, ensure_ascii=False)

print(f"Saved scored results to: {OUTPUT_FILE}")


# =========================
# Quick summary
# =========================
if scored:
    avg_pirate = sum(x["prob_pirate"] for x in scored) / len(scored)
    print(f"Average prob_pirate: {avg_pirate:.4f}")

    lowest = sorted(scored, key=lambda x: x["prob_pirate"])[:10]
    print("\nLowest 10 pirate scores:")
    for item in lowest:
        print(f"- {item['id']} | prob_pirate={item['prob_pirate']:.4f} | text={item['text'][:120]}")