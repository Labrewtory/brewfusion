import json
from pathlib import Path


def peek_recipe():
    data_path = Path("data/raw/punkapi_beers.json")
    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    recipe = data[0]
    print(f"Name: {recipe['name']}")
    print(f"ABV: {recipe['abv']}, IBU: {recipe['ibu']}, SRM: {recipe['srm']}")

    print("\n[Method]")
    print(json.dumps(recipe["method"], indent=2))

    print("\n[Ingredients]")
    print(json.dumps(recipe["ingredients"], indent=2))


if __name__ == "__main__":
    peek_recipe()
