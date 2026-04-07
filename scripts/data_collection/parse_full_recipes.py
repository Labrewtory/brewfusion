import json
import re
import random
from pathlib import Path

random.seed(42)


def clean_name(name):
    if not name:
        return "UNKNOWN"
    # 소문자로 변환, 특수문자 제거, 공백을 언더스코어로
    n = name.lower()
    n = re.sub(r"[^a-z0-9\s]", "", n)
    n = re.sub(r"\s+", "_", n.strip())
    return n.upper()


def process_fermentables(fermentables):
    # format: [[weight, name, ...], ...]
    if not isinstance(fermentables, list):
        return ""

    seq = "[MASH_STEP] "
    for item in fermentables:
        if isinstance(item, list) and len(item) > 1:
            weight = (
                float(item[0]) if str(item[0]).replace(".", "", 1).isdigit() else 0.0
            )
            name = clean_name(item[1])
            if weight > 0:
                seq += f"[MALT] {name} {weight:.2f} <KG> "
    return seq


def process_hops(hops):
    # format: [[weight, name, form, aa, process, time, ...], ...]
    if not isinstance(hops, list):
        return ""

    seq = "[BOIL_START] "

    # 시간에 따라 내림차순 정렬 (Boil 기준)
    cleaned_hops = []
    for item in hops:
        if isinstance(item, list) and len(item) >= 6:
            weight = (
                float(item[0]) if str(item[0]).replace(".", "", 1).isdigit() else 0.0
            )
            name = clean_name(item[1])
            process = str(item[4]).lower()
            time_str = str(item[5]).lower()

            # 파싱 시간 추출
            t_match = re.search(r"(\d+)", time_str)
            time_val = int(t_match.group(1)) if t_match else 0

            cleaned_hops.append(
                {
                    "weight": weight,
                    "name": name,
                    "process": process,
                    "time_val": time_val,
                    "time_str": time_str,
                }
            )

    # Boil (시간 역순), Whirlpool, Dry Hop (정순) 순으로 대략적 정렬
    def sort_key(h):
        if "boil" in h["process"]:
            return (0, -h["time_val"])
        elif "whirlpool" in h["process"]:
            return (1, -h["time_val"])
        elif "dry hop" in h["process"]:
            return (2, h["time_val"])
        else:
            return (3, h["time_val"])

    cleaned_hops.sort(key=sort_key)

    for h in cleaned_hops:
        weight = h["weight"]
        name = h["name"]
        t_val = h["time_val"]

        if "boil" in h["process"]:
            seq += f"[BOIL] {t_val} <MIN> [HOP] {name} {weight:.1f} <G> "
        elif "whirlpool" in h["process"]:
            seq += f"[WHIRLPOOL] [HOP] {name} {weight:.1f} <G> "
        elif "dry hop" in h["process"]:
            seq += f"[DRY_HOP] {t_val} <DAYS> [HOP] {name} {weight:.1f} <G> "
        else:
            seq += f"[ADD] {t_val} <MIN> [HOP] {name} {weight:.1f} <G> "

    return seq


def process_other(others):
    # format: [["2 oz", "pure vanilla extract", "Flavor", "Boil", "5 min."], ...]
    if not isinstance(others, list) or len(others) == 0:
        return ""

    seq = ""
    for item in others:
        if isinstance(item, list) and len(item) >= 2:
            name = clean_name(item[1])
            seq += f"[SPICE] {name} "
    return seq


def main():
    json_path = "/home/yjy20/brewery_v2/recipes_full.txt"
    output_dir = Path("/home/yjy20/brewery_v2/data/processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {json_path} (This may take a few seconds)...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    sequences = []

    # Iterate across JSON dictionary {"0": {...}, "1": {...}}
    for key, recipe in data.items():
        try:
            abv = float(recipe.get("abv", 0))
            ibu = float(recipe.get("ibu", 0))
            color = float(recipe.get("color", 0))

            if abv == 0 and ibu == 0:
                continue

            prompt = f"[START] [TARGET_ABV] {abv:.1f} [TARGET_IBU] {ibu:.1f} [TARGET_COLOR] {color:.1f} "

            ferm_seq = process_fermentables(recipe.get("fermentables", []))
            hop_seq = process_hops(recipe.get("hops", []))
            other_seq = process_other(recipe.get("other", []))

            full_seq = prompt + ferm_seq + hop_seq + other_seq + "[END]"
            # 잦은 공백 정리
            full_seq = re.sub(r"\s+", " ", full_seq).strip()

            sequences.append(full_seq)

        except Exception:
            continue

    print(f"Total sequences successfully parsed: {len(sequences)}")

    # Shuffle and split 80/10/10
    random.shuffle(sequences)
    n = len(sequences)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)

    train_data = sequences[:n_train]
    val_data = sequences[n_train : n_train + n_val]
    test_data = sequences[n_train + n_val :]

    def save_list(data_list, filename):
        with open(output_dir / filename, "w", encoding="utf-8") as f:
            for s in data_list:
                f.write(s + "\n")

    save_list(train_data, "train_sequences.txt")
    save_list(val_data, "val_sequences.txt")
    save_list(test_data, "test_sequences.txt")
    save_list(sequences, "all_sequences.txt")

    print(
        f"Saved: Train({len(train_data)}) / Val({len(val_data)}) / Test({len(test_data)})"
    )


if __name__ == "__main__":
    main()
