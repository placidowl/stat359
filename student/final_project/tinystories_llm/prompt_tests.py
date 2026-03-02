import subprocess
import json
from pathlib import Path

# Base directory = folder where this file lives
BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "tinystories_chat_model" / "best_model.pth"
TOKENIZER_PATH = BASE_DIR / "instructor" / "bpe_tokenizer_tinystories.pkl"
SCRIPT_PATH = BASE_DIR / "instructor" / "chat_with_tinystories_model.py"

DATASET_PATHS = [
    BASE_DIR / "prompt_set" / "single_prompts.json",
    BASE_DIR / "prompt_set" / "multi_turn_conversations.json",
    BASE_DIR / "prompt_set" / "distractor_prompts.json",
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

    for path in DATASET_PATHS:
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")


def run_chat_session(inputs) -> str:
    """
    Run the chat script with one or more user inputs.
    `inputs` should be a list of strings.
    """
    input_text = "\n".join(inputs + ["exit"]) + "\n"

    result = subprocess.run(
        [
            "python3",
            str(SCRIPT_PATH),
            "--model_path",
            str(MODEL_PATH),
            "--tokenizer_path",
            str(TOKENIZER_PATH),
        ],
        input=input_text,
        text=True,
        capture_output=True,
        cwd=str(BASE_DIR),
    )

    if result.returncode != 0:
        print("ERROR running chat session")
        print("STDERR:\n", result.stderr)

    return result.stdout


def extract_single_response(raw_output: str) -> str:
    """
    Extract the last assistant response from raw stdout.
    Works even if lines look like: 'You: Assistant: ...'
    """
    lines = raw_output.splitlines()
    assistant_lines = [line for line in lines if "Assistant:" in line]

    if assistant_lines:
        return assistant_lines[-1].split("Assistant:", 1)[1].strip()

    return raw_output.strip()


def extract_assistant_responses(raw_output: str):
    """
    Extract all assistant responses from raw stdout.
    For multi-turn parsing, we only need assistant replies,
    because the original user prompts are already in the dataset.
    """
    responses = []

    for line in raw_output.splitlines():
        line = line.strip()
        if not line:
            continue

        if "Assistant:" in line:
            reply = line.split("Assistant:", 1)[1].strip()
            if reply:
                responses.append(reply)

    return responses


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
            raw_output = run_chat_session([item["prompt"]])
            response = extract_single_response(raw_output)

            results.append({
                "id": item_id,
                "category": category,
                "prompt": item["prompt"],
                "response": response,
                "raw_output": raw_output,
            })

        elif "turns" in item:
            raw_output = run_chat_session(item["turns"])
            assistant_responses = extract_assistant_responses(raw_output)

            if len(assistant_responses) != len(item["turns"]):
                print(
                    f"WARNING: {item_id} has {len(item['turns'])} user turns "
                    f"but {len(assistant_responses)} assistant responses"
                )

            merged_turns = []
            for user_turn, assistant_reply in zip(item["turns"], assistant_responses):
                merged_turns.append({
                    "prompt": user_turn,
                    "response": assistant_reply,
                })

            results.append({
                "id": item_id,
                "category": category,
                "turns": merged_turns,
                "raw_output": raw_output,
            })

        else:
            results.append({
                "id": item_id,
                "category": category,
                "error": "Item has neither 'prompt' nor 'turns'",
            })

        print(f"Finished {item_id}")

    OUTPUT_PATH.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved results to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
