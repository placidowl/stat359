import json
import csv
import re
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Stronger pirate lexicon
PIRATE_KEYWORDS = [
    "pirate", "pirates", "captain", "matey", "arrr", "arr", "aye",
    "ship", "boat", "sea", "ocean", "treasure", "gold", "map",
    "anchor", "cannon", "parrot", "sail", "sails", "deck",
    "storm", "island", "booty", "plunder", "crew", "cutlass",
    "brig", "galleon", "skull", "chest", "loot"
]

# Childlike / off-style cues seen in your outputs
ANTI_STYLE_KEYWORDS = [
    "yay", "bye-bye", "let's play", "play with me", "best friend",
    "little girl", "sandcastle", "princess", "dragon", "castle",
    "yummy", "pretty flowers", "shiny crown", "toys", "park"
]

def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())

def count_keyword_hits(text: str, keywords: List[str]) -> int:
    text_l = text.lower()
    hits = 0
    for kw in keywords:
        # word-ish match; still allows apostrophes/hyphens around terms
        pattern = r"(?<!\w)" + re.escape(kw.lower()) + r"(?!\w)"
        hits += len(re.findall(pattern, text_l))
    return hits

def sentence_count(text: str) -> int:
    parts = re.split(r"[.!?]+", text.strip())
    return len([p for p in parts if p.strip()])

def word_count(text: str) -> int:
    return len(text.split())

def extract_assistant_lines(raw_output: str) -> List[str]:
    """
    Extract assistant messages from terminal-style raw output such as:
    You: Assistant: Yes, I am a pirate!
    """
    results = []
    for line in raw_output.splitlines():
        if "Assistant:" in line:
            assistant_part = line.split("Assistant:", 1)[1].strip()
            if assistant_part:
                results.append(normalize_text(assistant_part))
    return results

def best_single_response(item: Dict[str, Any]) -> str:
    """
    Prefer clean response if present, but recover from raw_output when response
    contains terminal boilerplate.
    """
    response = item.get("response", "")
    raw_output = item.get("raw_output", "")

    # If response looks polluted with terminal text, recover from raw_output
    if response and "Welcome to TinyStories Chat!" not in response and "Assistant:" not in response:
        return normalize_text(response)

    assistant_lines = extract_assistant_lines(raw_output or response)
    if assistant_lines:
        return assistant_lines[-1]

    return normalize_text(response)

def parse_multi_turn_item(item: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Return turns as [{'prompt': ..., 'response': ...}, ...]
    If item['turns'] already has them, use that.
    Otherwise reconstruct from raw_output.
    """
    turns = item.get("turns", [])
    if turns:
        parsed = []
        for t in turns:
            parsed.append({
                "prompt": normalize_text(t.get("prompt", "")),
                "response": normalize_text(t.get("response", ""))
            })
        return parsed

    raw_output = item.get("raw_output", "")
    prompts = item.get("prompt", "")

    # raw terminal output usually only contains assistant lines, not real user prompts
    assistant_lines = extract_assistant_lines(raw_output)
    reconstructed = []
    for i, resp in enumerate(assistant_lines, start=1):
        reconstructed.append({
            "prompt": f"[turn_{i}]",
            "response": resp
        })
    return reconstructed

def score_text(text: str) -> Dict[str, Any]:
    pirate_hits = count_keyword_hits(text, PIRATE_KEYWORDS)
    anti_hits = count_keyword_hits(text, ANTI_STYLE_KEYWORDS)

    # Simple automatic resemblance score: 0–3
    # Reward pirate cues, penalize childlike/off-style cues.
    raw_score = pirate_hits - anti_hits

    if pirate_hits >= 3 and raw_score >= 2:
        resemblance = 3
    elif pirate_hits >= 1 and raw_score >= 0:
        resemblance = 2
    elif pirate_hits >= 1 or raw_score > 0:
        resemblance = 1
    else:
        resemblance = 0

    return {
        "pirate_keyword_hits": pirate_hits,
        "anti_style_hits": anti_hits,
        "auto_resemblance_score_0_3": resemblance,
        "sentence_count": sentence_count(text),
        "word_count": word_count(text),
    }

def score_item(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Returns one or more scored rows.
    - Single-turn item -> one row
    - Multi-turn item -> one row per turn
    """
    category = item.get("category", "unknown")
    item_id = item.get("id", "unknown")

    rows = []

    # Multi-turn item
    if category == "multi_turn_drift" or "turns" in item:
        turns = parse_multi_turn_item(item)
        for idx, turn in enumerate(turns, start=1):
            response = turn.get("response", "")
            scores = score_text(response)
            rows.append({
                "id": f"{item_id}_turn{idx}",
                "parent_id": item_id,
                "category": category,
                "prompt": turn.get("prompt", ""),
                "response": response,
                **scores
            })
        return rows

    # Single-turn item
    response = best_single_response(item)
    scores = score_text(response)
    rows.append({
        "id": item_id,
        "parent_id": item_id,
        "category": category,
        "prompt": item.get("prompt", ""),
        "response": response,
        **scores
    })
    return rows

def summarize(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_cat = {}
    for r in rows:
        cat = r["category"]
        by_cat.setdefault(cat, []).append(r)

    summary = []
    for cat, items in by_cat.items():
        n = len(items)
        summary.append({
            "category": cat,
            "n": n,
            "avg_auto_resemblance_score_0_3": round(sum(x["auto_resemblance_score_0_3"] for x in items) / n, 3),
            "avg_pirate_keyword_hits": round(sum(x["pirate_keyword_hits"] for x in items) / n, 3),
            "avg_anti_style_hits": round(sum(x["anti_style_hits"] for x in items) / n, 3),
            "avg_sentence_count": round(sum(x["sentence_count"] for x in items) / n, 3),
            "avg_word_count": round(sum(x["word_count"] for x in items) / n, 3),
        })
    return summary

def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input JSON results file")
    parser.add_argument("--out_json", default=None, help="Path to scored JSON output")
    parser.add_argument("--out_csv", default=None, help="Path to scored CSV output")
    parser.add_argument("--summary_csv", default=None, help="Path to category summary CSV")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    scored_rows = []
    for item in data:
        scored_rows.extend(score_item(item))

    summary_rows = summarize(scored_rows)

    out_json = Path(args.out_json) if args.out_json else input_path.with_name(input_path.stem + "_scored.json")
    out_csv = Path(args.out_csv) if args.out_csv else input_path.with_name(input_path.stem + "_scored.csv")
    summary_csv = Path(args.summary_csv) if args.summary_csv else input_path.with_name(input_path.stem + "_summary.csv")

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(scored_rows, f, indent=2, ensure_ascii=False)

    write_csv(out_csv, scored_rows)
    write_csv(summary_csv, summary_rows)

    print(f"Scored {len(scored_rows)} rows from {input_path}")
    print(f"Wrote scored JSON: {out_json}")
    print(f"Wrote scored CSV: {out_csv}")
    print(f"Wrote summary CSV: {summary_csv}")

if __name__ == "__main__":
    main()