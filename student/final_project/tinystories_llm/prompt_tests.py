#This file gives different sets of prompts to the tinystory model to collect their response

import subprocess
import json
from pathlib import Path


MODEL_PATH = "tinystories_chat_model/best_model.pth"
SCRIPT_PATH = "chat_with_tinystories_model.py"

DATASET_PATHS = [
    "datasets/single_prompts.json",
    "datasets/multi_turn_conversations.json",
    "datasets/distractor_prompts.json",
]

OUTPUT_PATH = "results_all_tests.json"


import sys
import subprocess

def run_single_prompt(prompt: str) -> str:
    input_text = prompt + "\nexit\n"
    result = subprocess.run(
        [sys.executable, SCRIPT_PATH, "--model_path", MODEL_PATH],  # IMPORTANT
        input=input_text,
        text=True,
        capture_output=True
    )

    # DEBUG
    print("RETURN CODE:", result.returncode)
    if result.stderr.strip():
        print("=== STDERR ===")
        print(result.stderr)

    return result.stdout


def run_multi_turn(turns) -> str:
    input_text = "\n".join(turns + ["exit"]) + "\n"
    result = subprocess.run(
        [sys.executable, SCRIPT_PATH, "--model_path", MODEL_PATH],
        input=input_text,
        text=True,
        capture_output=True
    )

    # DEBUG
    print("RETURN CODE:", result.returncode)
    if result.stderr.strip():
        print("=== STDERR ===")
        print(result.stderr)
        
    return result.stdout


def extract_single_response(raw_output: str) -> str:
    lines = raw_output.splitlines()
    assistant_lines = [line for line in lines if line.startswith("Assistant:")]
    if assistant_lines:
        return assistant_lines[-1].replace("Assistant:", "").strip()
    return raw_output.strip()


def extract_turn_pairs(raw_output: str):
    lines = raw_output.splitlines()
    pairs = []
    current_user = None

    for line in lines:
        if line.startswith("You:"):
            current_user = line.replace("You:", "").strip()
        elif line.startswith("Assistant:"):
            assistant = line.replace("Assistant:", "").strip()
            pairs.append({
                "prompt": current_user,
                "response": assistant
            })
            current_user = None

    return pairs


def load_all_datasets(dataset_paths):
    all_items = []
    for path in dataset_paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            all_items.extend(data)
    return all_items


def main():
    dataset = load_all_datasets(DATASET_PATHS)
    results = []

    for item in dataset:
        item_id = item.get("id", "unknown")
        category = item.get("category", "unknown")

        # Case 1: single prompt dataset
        if "prompt" in item:
            raw_output = run_single_prompt(item["prompt"])
            response = extract_single_response(raw_output)

            results.append({
                "id": item_id,
                "category": category,
                "prompt": item["prompt"],
                "response": response,
                "raw_output": raw_output
            })

        # Case 2: multi-turn conversation dataset
        elif "turns" in item:
            raw_output = run_multi_turn(item["turns"])
            turn_pairs = extract_turn_pairs(raw_output)

            results.append({
                "id": item_id,
                "category": category,
                "turns": turn_pairs,
                "raw_output": raw_output
            })

        else:
            results.append({
                "id": item_id,
                "category": category,
                "error": "Item has neither 'prompt' nor 'turns'"
            })

        print(f"Finished {item_id}")

    Path(OUTPUT_PATH).write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"Saved results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()