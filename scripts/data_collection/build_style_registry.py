import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent.parent.parent
    recipes_txt = root / "recipes_full.txt"
    out_dir = root / "src" / "brewfusion" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "style_registry.json"

    print(f"Loading {recipes_txt}...")
    with open(recipes_txt, "r", encoding="utf-8") as f:
        data = json.load(f)

    unique_styles = set()
    for recipe in data.values():
        style = recipe.get("style")
        if isinstance(style, str) and style.strip() not in ("", "--"):
            unique_styles.add(style.strip())

    # Sort alphabetically to get consistent indices
    sorted_styles = sorted(list(unique_styles))

    registry = {style: idx for idx, style in enumerate(sorted_styles)}

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)

    print(f"Extracted {len(registry)} unique styles.")
    print(f"Saved registry to {out_file}")


if __name__ == "__main__":
    main()
