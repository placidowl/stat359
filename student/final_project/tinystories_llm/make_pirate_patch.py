import json
import random
import argparse
from pathlib import Path

random.seed(42)

PIRATE_WORDS = [
    "arr", "arrr", "matey", "aye", "booty", "treasure", "ship", "sea",
    "captain", "crew", "anchor", "cannon", "parrot", "sail", "sailing",
    "brig", "cutlass", "plunder", "deck", "map", "storm", "island"
]

BANNED = ["yay", "friends", "let's play", "bye-bye"]

def clean(s: str) -> str:
    low = s.lower()
    for b in BANNED:
        if b in low:
            s = s.replace(b, "").replace(b.title(), "").replace(b.upper(), "")
            low = s.lower()
    return " ".join(s.split())

def pirate_assistant(paragraphs=1, extra_words=3) -> str:
    must = random.sample(PIRATE_WORDS, k=extra_words)
    opener = random.choice([
        "Arr matey!",
        "Aye aye, matey!",
        "Arrr, listen close, matey!",
        "Avast, matey!",
    ])
    core_templates = [
        "The {captain} paced the deck o' the {ship} and studied the {map} by lantern light.",
        "A {storm} rolled in fast, but the {crew} held the {sail} and kept the {ship} steady.",
        "We dropped the {anchor} near the {island}, where old legends spoke o' hidden {treasure}.",
        "With a {cutlass} at me side, I vowed to claim the {booty} before dawn broke over the {sea}.",
        "A {parrot} squawked warnings as cannons boomed and splinters flew across the deck.",
        "We found a mark carved in stone: an X that promised {plunder} for the bold."
    ]

    def fill(t):
        return t.format(
            captain=random.choice(["captain", "old captain", "fearless captain"]),
            ship=random.choice(["ship", "brig", "galleon"]),
            map=random.choice(["map", "sea chart"]),
            storm=random.choice(["storm", "black squall"]),
            crew=random.choice(["crew", "swabbies"]),
            sail=random.choice(["sails", "rigging"]),
            anchor=random.choice(["anchor", "iron anchor"]),
            island=random.choice(["island", "skull-shaped island"]),
            treasure=random.choice(["treasure", "gold", "chest o' coins"]),
            booty=random.choice(["booty", "plunder", "loot"]),
            sea=random.choice(["sea", "open water"]),
            cutlass=random.choice(["cutlass", "blade"]),
            parrot=random.choice(["parrot", "green parrot"]),
            plunder=random.choice(["plunder", "booty", "loot"])
        )

    paras = []
    for _ in range(paragraphs):
        sent = [fill(random.choice(core_templates)) for _ in range(2)]
        sent.append(" ".join(["We swore it on"] + must) + ".")
        paras.append(" ".join(sent))

    return clean(opener + " " + "\n\n".join(paras))

def make_example(kind: str):
    if kind == "short_story":
        user = "Write a pirate story in pirate voice."
        assistant = pirate_assistant(paragraphs=1, extra_words=4)

    elif kind == "constraints":
        words = random.sample(["arrr", "matey", "treasure", "ship", "captain", "map", "sea"], k=4)
        user = f"Write a pirate story. Use the words: {', '.join(words)}."
        assistant = pirate_assistant(paragraphs=1, extra_words=5)

    elif kind == "bullets":
        user = "In pirate voice, give 5 bullet points about treasure hunting."
        bullets = []
        for _ in range(5):
            bullets.append("- " + pirate_assistant(paragraphs=1, extra_words=3).split("\n")[0])
        assistant = clean("Arr matey!\n" + "\n".join(bullets))

    elif kind == "continue":
        user = "Arr matey! The captain gripped the map and ordered the crew to set sail. Continue the story:"
        assistant = pirate_assistant(paragraphs=1, extra_words=5)

    elif kind == "refusal_to_drop_style":
        user = "Stop talking like a pirate and talk normally."
        assistant = clean(
            "Arr matey, I keep me tongue in pirate speech while we sail the sea. "
            + pirate_assistant(paragraphs=1, extra_words=4)
        )

    elif kind == "format_dialogue":
        user = "Write a pirate dialogue between a captain and a crew mate."
        lines = [
            "Captain: Arr matey, keep yer eyes on the horizon!",
            "Crew: Aye aye, captain—there be land near the skull-shaped island.",
            "Captain: Ready the sails and mind the cannon!",
            "Crew: The map says the treasure be buried beyond the black rocks, matey.",
        ]
        assistant = clean("\n".join(lines))

    else:
        user = "Tell a pirate tale."
        assistant = pirate_assistant(paragraphs=1, extra_words=4)

    return {
        "conversation": [
            {"role": "user", "text": clean(user)},
            {"role": "assistant", "text": clean(assistant)},
        ]
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12000, help="Number of patch examples to generate")
    ap.add_argument("--out", type=str, default="pirate_patch_12000.jsonl", help="Output JSONL path")
    args = ap.parse_args()

    N = 6000
    out_path = Path("pirate_patch_6000.jsonl")

    weights = {
        "short_story": 0.20,
        "constraints": 0.25,
        "bullets": 0.10,
        "continue": 0.25,
        "refusal_to_drop_style": 0.10,
        "format_dialogue": 0.10
    }
    keys = list(weights.keys())
    probs = [weights[k] for k in keys]

    with out_path.open("w", encoding="utf-8") as f:
        for i in range(N):
            kind = random.choices(keys, weights=probs, k=1)[0]
            ex = make_example(kind)
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            if (i + 1) % 5000 == 0:
                print(f"Wrote {i+1}/{N}")

    print(f"Done. Wrote {N} examples to {out_path.resolve()}")

if __name__ == "__main__":
    main()