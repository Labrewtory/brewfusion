import json
import logging
import requests
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")


def fetch_punk_recipes():
    # Primary URL for full BrewDog JSON dataset (often used in data science)
    url = (
        "https://raw.githubusercontent.com/samjbmason/punkapi-db/master/data/beers.json"
    )

    out_dir = Path(__file__).resolve().parent.parent.parent / "data" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "setup_punkapi_beers.json"

    if out_file.exists():
        logging.info("Dataset already exists at %s", out_file)
        return

    logging.info("Downloading PunkAPI Dataset from %s...", url)

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

        logging.info("Successfully downloaded %d recipes.", len(data))

        # Print a sample's method to verify thermodynamics are present.
        if data:
            sample = data[0]
            logging.info(
                "\n--- Verification: Sample Recipe Method (%s) ---", sample.get("name")
            )
            logging.info(json.dumps(sample.get("method", {}), indent=2))

    except Exception as e:
        logging.error("Failed to fetch data: %s", e)


if __name__ == "__main__":
    fetch_punk_recipes()
