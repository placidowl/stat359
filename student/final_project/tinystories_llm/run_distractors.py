import subprocess
import json
from pathlib import Path

MODEL_PATH = "tinystories_chat_model/best_model.pth"
SCRIPT_PATH = "chat_with_tinystories_model.py"
DATASET_PATH = "distractor_prompts.json"
OUTPUT_PATH = "results_distractor.json"

def run_prompt(prompt: str) -> str:
    input_text = prompt + "\nexit\n"
    result = subprocess.run(
        ["python", SCRIPT_PATH, "--model_path", MODEL_PATH],
        input=input_text,
        text=True,
        capture_output=True
    )
    return result.stdout

def extract_response(raw_output: str) -> str:
    lines = raw_output.splitlines()
    assistant_lines = [line for line in lines if line.startswith("Assistant:")]
    if assistant_lines:
        return assistant_lines[-1].replace("Assistant:", "").strip()
    return raw_output.strip()

def main():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    results = []

    for item in dataset:
        raw_output = run_prompt(item["prompt"])
        response = extract_response(raw_output)

        results.append({
            "id": item["id"],
            "category": item["category"],
            "prompt": item["prompt"],
            "response": response,
            "raw_output": raw_output
        })

        print(f"Finished {item['id']}")

    Path(OUTPUT_PATH).write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"Saved results to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()