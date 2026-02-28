import json, re

inp = "pirate_patch.jsonl"
out = "pirate_patch_fixed.jsonl"

user_re = re.compile(r"^\s*You:\s*(.*)$")
asst_re = re.compile(r"^\s*Assistant:\s*(.*)$")

def parse_conv(s: str):
    turns = []
    for line in s.splitlines():
        m = user_re.match(line)
        if m:
            turns.append({"role": "user", "text": m.group(1).strip()})
            continue
        m = asst_re.match(line)
        if m:
            turns.append({"role": "assistant", "text": m.group(1).strip()})
            continue
    return turns

n = 0
with open(inp, "r", encoding="utf-8") as f, open(out, "w", encoding="utf-8") as g:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        conv = obj.get("conversation")

        # Convert string -> list[dict]
        if isinstance(conv, str):
            turns = parse_conv(conv)
            if not turns:
                turns = [{"role": "user", "text": conv.strip()}]
            obj["conversation"] = turns

        g.write(json.dumps(obj, ensure_ascii=False) + "\n")
        n += 1

print("Wrote:", out, "lines:", n)