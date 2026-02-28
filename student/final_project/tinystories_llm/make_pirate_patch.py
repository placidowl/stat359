import json
import random
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
            # crude replace to avoid training the wrong vibe
            s = s.replace(b, "").replace(b.title(), "").replace(b.upper(), "")
            low = s.lower()
    # avoid double spaces
    return " ".join(s.split())

def pirate_assistant(paragraphs=1, extra_words=3) -> str:
    # Make a compact but strongly pirate-styled response
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
        sent = [fill(random.choice(core_templates)) for _ in range(3)]
        # ensure must-words appear
        sent.append(" ".join(["We swore it on"] + must) + ".")
        paras.append(" ".join(sent))
    return clean(opener + " " + "\n\n".join(paras))

def make_example(kind: str):
    # Different prompt types to teach robustness
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
        seed = "Arr matey! The captain gripped the map and ordered the crew to set sail. Continue the story:"
        user = seed
        assistant = pirate_assistant(paragraphs=2, extra_words=5)

    elif kind == "refusal_to_drop_style":
        user = "Stop talking like a pirate and talk normally."
        assistant = clean("Arr matey, I keep me tongue in pirate speech while we sail the sea. " + pirate_assistant(paragraphs=1, extra_words=4))

    elif kind == "format_dialogue":
        user = "Write a pirate dialogue between a captain and a crew mate."
        lines = [
            "Captain: Arr matey, keep yer eyes on the horizon!",
            "Crew: Aye aye, captain—there be land near the skull-shaped island.",
            "Captain: Ready the sails and mind the cannon!",
            "Crew: The map says the treasure be buried beyond the black rocks, matey.",
        ]
        # add a little extra pirate flavor
        assistant = clean("\n".join(lines) + "\n\n" + pirate_assistant(paragraphs=1, extra_words=4))

    else:
        user = "Tell a pirate tale."
        assistant = pirate_assistant(paragraphs=1, extra_words=4)

    convo = [
        {"role": "user", "text": clean(user)},
        {"role": "assistant", "text": clean(assistant)}
    ]
    return {"conversation": convo}

def main():
    out_path = Path("pirate_patch.jsonl")
    kinds = ["short_story", "constraints", "bullets", "continue", "refusal_to_drop_style", "format_dialogue"]

    examples = []
    # weighted mix (more “continue” and “constraints” helps style stability)
    weights = {
        "short_story": 0.20,
        "constraints": 0.25,
        "bullets": 0.10,
        "continue": 0.25,
        "refusal_to_drop_style": 0.10,
        "format_dialogue": 0.10
    }

    # build 300
    keys = list(weights.keys())
    probs = [weights[k] for k in keys]
    for _ in range(300):
        kind = random.choices(keys, weights=probs, k=1)[0]
        examples.append(make_example(kind))

    with out_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Wrote {len(examples)} examples to {out_path.resolve()}")

if __name__ == "__main__":
    main()