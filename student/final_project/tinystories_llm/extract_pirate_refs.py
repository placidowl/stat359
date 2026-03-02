import json

input_file = "pirate_patch.jsonl"
output_file = "pirate_refs.txt"

refs = []

with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)

        # CASE 1: conversation is list
        if isinstance(obj.get("conversation"), list):
            for turn in obj["conversation"]:
                if turn.get("role") == "assistant":
                    refs.append(turn.get("text", "").strip())

        # CASE 2: conversation is string
        elif isinstance(obj.get("conversation"), str):
            text = obj["conversation"]
            for line2 in text.split("\n"):
                if line2.startswith("Assistant:"):
                    refs.append(line2.replace("Assistant:", "").strip())

        # CASE 3: direct role/text
        elif obj.get("role") == "assistant":
            refs.append(obj.get("text", "").strip())

# remove empty lines
refs = [r for r in refs if r]

with open(output_file, "w", encoding="utf-8") as f:
    for r in refs:
        f.write(r + "\n")

print(f"Saved {len(refs)} pirate reference lines to {output_file}")