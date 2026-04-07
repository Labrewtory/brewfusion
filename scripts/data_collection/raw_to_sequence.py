import json
from pathlib import Path
import re
import random


def clean_name(name):
    """Clean ingredient names to format them as tokens."""
    if not name:
        return "UNKNOWN"
    name = re.sub(r"[^a-zA-Z0-9_]", "_", name.replace(" ", "_"))
    return name.upper().strip("_")


def parse_recipes_to_sequences():
    data_path = (
        Path(__file__).resolve().parent.parent.parent
        / "data"
        / "raw"
        / "punkapi_beers.json"
    )
    out_dir = Path(__file__).resolve().parent.parent.parent / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(data_path, "r", encoding="utf-8") as f:
        recipes = json.load(f)

    sequences = []

    # ... Token generation logic ...
    for recipe in recipes:
        seq = []

        abv = recipe.get("abv", 0)
        ibu = recipe.get("ibu", 0)
        srm = recipe.get("srm", 0)
        seq.append(f"<TARGET_ABV> {abv}")
        seq.append(f"<TARGET_IBU> {ibu}")
        seq.append(f"<TARGET_SRM> {srm}")

        method = recipe.get("method", {})
        ingredients = recipe.get("ingredients", {})

        mash_temps = method.get("mash_temp", [])
        for mash in mash_temps:
            temp = mash.get("temp", {}).get("value", "UNKNOWN")
            duration = mash.get("duration", "UNKNOWN")
            seq.append(f"<MASH_STEP> {temp} <C> {duration} <M>")

        malts = ingredients.get("malt", [])
        for malt in malts:
            name = clean_name(malt.get("name"))
            amount = malt.get("amount", {}).get("value", 0)
            unit = malt.get("amount", {}).get("unit", "kg")
            seq.append(f"<MALT> {name} {amount} <{unit.upper()}>")

        ferm = method.get("fermentation", {})
        ferm_temp = ferm.get("temp", {}).get("value", "UNKNOWN")
        seq.append(f"<FERM> {ferm_temp} <C>")

        yeast = clean_name(ingredients.get("yeast"))
        seq.append(f"<YEAST> {yeast}")

        hops = ingredients.get("hops", [])
        hop_phases = {"start": [], "middle": [], "end": [], "dry hop": []}

        for hop in hops:
            add_phase = hop.get("add", "unknown")
            if add_phase in hop_phases:
                hop_phases[add_phase].append(hop)

        if hop_phases["start"]:
            seq.append("<BOIL_START>")
            for h in hop_phases["start"]:
                name = clean_name(h.get("name"))
                amount = h.get("amount", {}).get("value", 0)
                seq.append(f"<HOP> {name} {amount} <G>")

        if hop_phases["middle"]:
            seq.append("<BOIL_MIDDLE>")
            for h in hop_phases["middle"]:
                name = clean_name(h.get("name"))
                amount = h.get("amount", {}).get("value", 0)
                seq.append(f"<HOP> {name} {amount} <G>")

        if hop_phases["end"]:
            seq.append("<BOIL_END>")
            for h in hop_phases["end"]:
                name = clean_name(h.get("name"))
                amount = h.get("amount", {}).get("value", 0)
                seq.append(f"<HOP> {name} {amount} <G>")

        if hop_phases["dry hop"]:
            seq.append("<DRY_HOP>")
            for h in hop_phases["dry hop"]:
                name = clean_name(h.get("name"))
                amount = h.get("amount", {}).get("value", 0)
                seq.append(f"<HOP> {name} {amount} <G>")

        sequences.append(" ".join(seq))

    # Shuffle and Split (80% Train, 10% Val, 10% Test)
    random.seed(42)
    random.shuffle(sequences)

    n_total = len(sequences)
    n_train = int(n_total * 0.8)
    n_val = int(n_total * 0.1)

    splits = {
        "train": sequences[:n_train],
        "val": sequences[n_train : n_train + n_val],
        "test": sequences[n_train + n_val :],
    }

    for split_name, split_data in splits.items():
        out_file = out_dir / f"brewing_sequences_{split_name}.txt"
        with open(out_file, "w", encoding="utf-8") as f:
            for s in split_data:
                f.write(s + "\n")
        print(
            f"[{split_name.upper()}] Saved {len(split_data)} sequences to {out_file.name}"
        )

    # We still output the combined file for tokenizer training purposes
    combined_path = out_dir / "brewing_sequences.txt"
    with open(combined_path, "w", encoding="utf-8") as f:
        for s in sequences:
            f.write(s + "\n")

    if sequences:
        print("\n--- Example Sequence (Recipe 1) ---")
        print(sequences[0])


if __name__ == "__main__":
    parse_recipes_to_sequences()
