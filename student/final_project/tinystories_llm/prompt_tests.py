#This file gives different sets of prompts to the tinystory model to collect their response
import subprocess
import json
from pathlib import Path

# Base directory = folder where this file lives
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "tinystories_chat_model" / "best_model.pth"
TOKENIZER_PATH = BASE_DIR / "instructor"/"bpe_tokenizer_tinystories.pkl"

# Change this if your chat script is inside another folder
SCRIPT_PATH = BASE_DIR / "instructor"/"chat_with_tinystories_model.py"

DATASET_PATHS = [
    BASE_DIR / "datasets" / "single_prompts.json",
    BASE_DIR / "datasets" / "multi_turn_conversations.json",
    BASE_DIR / "datasets" / "distractor_prompts.json",
]

OUTPUT_PATH = BASE_DIR / "results_all_tests.json"


def check_paths():
    print("=== Checking paths ===")
    print("BASE_DIR:", BASE_DIR)
    print("SCRIPT_PATH:", SCRIPT_PATH)
    print("MODEL_PATH:", MODEL_PATH)
    print("TOKENIZER_PATH:", TOKENIZER_PATH)

    if not SCRIPT_PATH.exists():
        raise FileNotFoundError(f"Chat script not found: {SCRIPT_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {MODEL_PATH}")
    if not TOKENIZER_PATH.exists():
        print(f"Warning: tokenizer file not found: {TOKENIZER_PATH}")
    for p in DATASET_PATHS:
        if not p.exists():
            raise FileNotFoundError(f"Dataset file not found: {p}")


import sys
import subprocess

def run_single_prompt(prompt: str) -> str:
    input_text = prompt + "\nexit\n"
    result = subprocess.run(
<<<<<<< HEAD
        [sys.executable, SCRIPT_PATH, "--model_path", MODEL_PATH],  # IMPORTANT
=======
        [
            "python3",
            str(SCRIPT_PATH),
            "--model_path",
            str(MODEL_PATH),
            "--tokenizer_path",
            str(TOKENIZER_PATH),
        ],
>>>>>>> 8fdd8fda0d81ccbab58b3eb2f3cb289661cac1e5
        input=input_text,
        text=True,
        capture_output=True,
        cwd=str(BASE_DIR),
    )

<<<<<<< HEAD
    # DEBUG
    print("RETURN CODE:", result.returncode)
    if result.stderr.strip():
        print("=== STDERR ===")
        print(result.stderr)

=======
    if result.returncode != 0:
        print("ERROR running single prompt")
        print("STDERR:\n", result.stderr)
>>>>>>> 8fdd8fda0d81ccbab58b3eb2f3cb289661cac1e5
    return result.stdout


def run_multi_turn(turns) -> str:
    input_text = "\n".join(turns + ["exit"]) + "\n"
    result = subprocess.run(
<<<<<<< HEAD
        [sys.executable, SCRIPT_PATH, "--model_path", MODEL_PATH],
=======
        [
            "python3",
            str(SCRIPT_PATH),
            "--model_path",
            str(MODEL_PATH),
            "--tokenizer_path",
            str(TOKENIZER_PATH),
        ],
>>>>>>> 8fdd8fda0d81ccbab58b3eb2f3cb289661cac1e5
        input=input_text,
        text=True,
        capture_output=True,
        cwd=str(BASE_DIR),
    )

<<<<<<< HEAD
    # DEBUG
    print("RETURN CODE:", result.returncode)
    if result.stderr.strip():
        print("=== STDERR ===")
        print(result.stderr)
        
=======
    if result.returncode != 0:
        print("ERROR running multi-turn prompt")
        print("STDERR:\n", result.stderr)
>>>>>>> 8fdd8fda0d81ccbab58b3eb2f3cb289661cac1e5
    return result.stdout


def extract_single_response(raw_output: str) -> str:
    lines = raw_output.splitlines()
    assistant_lines = [line for line in lines if "Assistant:" in line]
    if assistant_lines:
        return assistant_lines[-1].split("Assistant:", 1)[1].strip()
    return raw_output.strip()


def extract_turn_pairs(raw_output: str):
    lines = raw_output.splitlines()
    pairs = []
    current_user = None

    for line in lines:
        if "You:" in line:
            current_user = line.split("You:", 1)[1].strip()
        elif "Assistant:" in line:
            assistant = line.split("Assistant:", 1)[1].strip()
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
    check_paths()
    dataset = load_all_datasets(DATASET_PATHS)
    results = []

    for item in dataset:
        item_id = item.get("id", "unknown")
        category = item.get("category", "unknown")

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

        elif "turns" in item:
            raw_output = run_multi_turn(item["turns"])
            parsed_turns = extract_turn_pairs(raw_output)

    # align parsed assistant responses with original user turns
            merged_turns = []
            for i, user_turn in enumerate(item["turns"][:len(parsed_turns)]):
                merged_turns.append({
                "prompt": user_turn,
                "response": parsed_turns[i]["response"]
                })

            results.append({
                "id": item_id,
                "category": category,
                "turns": merged_turns,
                "raw_output": raw_output
                })

        else:
            results.append({
                "id": item_id,
                "category": category,
                "error": "Item has neither 'prompt' nor 'turns'"
            })

        print(f"Finished {item_id}")

    OUTPUT_PATH.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"Saved results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
