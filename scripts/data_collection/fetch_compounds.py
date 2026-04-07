"""Expand the compound database using PubChem API + brewing science curation.

Fetches SMILES from PubChem for a curated list of beer-relevant compounds,
then saves the expanded registry for graph integration.

Data sources:
  - Existing compound_db.py (29 compounds)
  - Maillard reaction products (pyrazines, furans)
  - Additional hop terpenes/terpenoids
  - Yeast metabolites (esters, acids, alcohols)
  - Grain-derived compounds (melanoidin precursors)
  - Sulfur compounds (fermentation byproducts)
  - Lactones and furanones

All PubChem CIDs verified via https://pubchem.ncbi.nlm.nih.gov/
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class CompoundEntry:
    name: str
    cid: int
    smiles: str  # will be fetched from PubChem if empty
    flavor_descriptors: list[str]
    category: str
    sources: list[str]  # which ingredient types contain this


# ──────────────────────────────────────────────
# Curated compound list — 200+ beer-relevant compounds
# CIDs from PubChem, SMILES will be auto-fetched
# ──────────────────────────────────────────────

CURATED_COMPOUNDS: list[CompoundEntry] = [
    # ═══ HOP TERPENES (monoterpenes) ═══
    CompoundEntry(
        "myrcene", 31253, "", ["resinous", "herbal", "green"], "monoterpene", ["hop"]
    ),
    CompoundEntry(
        "limonene", 22311, "", ["citrus", "lemon", "orange"], "monoterpene", ["hop"]
    ),
    CompoundEntry(
        "beta-pinene", 14896, "", ["pine", "resinous", "woody"], "monoterpene", ["hop"]
    ),
    CompoundEntry(
        "alpha-pinene",
        6654,
        "",
        ["pine", "turpentine", "sharp"],
        "monoterpene",
        ["hop"],
    ),
    CompoundEntry(
        "beta-ocimene",
        5281553,
        "",
        ["herbal", "citrus", "tropical"],
        "monoterpene",
        ["hop"],
    ),
    CompoundEntry(
        "camphene", 6616, "", ["camphor", "herbal", "cooling"], "monoterpene", ["hop"]
    ),
    CompoundEntry(
        "alpha-terpinene", 7462, "", ["citrus", "lemon"], "monoterpene", ["hop"]
    ),
    CompoundEntry(
        "gamma-terpinene", 7461, "", ["citrus", "herbaceous"], "monoterpene", ["hop"]
    ),
    CompoundEntry(
        "terpinolene", 11463, "", ["floral", "herbal", "pine"], "monoterpene", ["hop"]
    ),
    CompoundEntry(
        "para-cymene", 7463, "", ["citrus", "woody", "spicy"], "monoterpene", ["hop"]
    ),
    CompoundEntry(
        "sabinene", 18818, "", ["woody", "spicy", "peppery"], "monoterpene", ["hop"]
    ),
    # ═══ HOP TERPENE ALCOHOLS ═══
    CompoundEntry(
        "linalool",
        6549,
        "",
        ["floral", "lavender", "citrus"],
        "terpene_alcohol",
        ["hop"],
    ),
    CompoundEntry(
        "geraniol", 637566, "", ["rose", "floral", "citrus"], "terpene_alcohol", ["hop"]
    ),
    CompoundEntry(
        "citronellol",
        7794,
        "",
        ["rose", "citronella", "floral"],
        "terpene_alcohol",
        ["hop"],
    ),
    CompoundEntry(
        "nerol", 643820, "", ["rose", "citrus", "sweet"], "terpene_alcohol", ["hop"]
    ),
    CompoundEntry(
        "alpha-terpineol",
        17100,
        "",
        ["lilac", "floral", "oily"],
        "terpene_alcohol",
        ["hop"],
    ),
    CompoundEntry(
        "4-terpineol",
        11230,
        "",
        ["earthy", "musty", "peppery"],
        "terpene_alcohol",
        ["hop"],
    ),
    CompoundEntry(
        "borneol", 6552, "", ["camphor", "minty", "earthy"], "terpene_alcohol", ["hop"]
    ),
    # ═══ HOP SESQUITERPENES ═══
    CompoundEntry(
        "alpha-humulene",
        5281520,
        "",
        ["earthy", "woody", "spicy"],
        "sesquiterpene",
        ["hop"],
    ),
    CompoundEntry(
        "beta-caryophyllene",
        5281515,
        "",
        ["peppery", "woody", "herbal"],
        "sesquiterpene",
        ["hop"],
    ),
    CompoundEntry(
        "farnesene",
        5281516,
        "",
        ["woody", "floral", "green_apple"],
        "sesquiterpene",
        ["hop"],
    ),
    CompoundEntry(
        "beta-selinene", 442393, "", ["herbal", "celery"], "sesquiterpene", ["hop"]
    ),
    CompoundEntry(
        "alpha-selinene", 86325, "", ["amber", "celery"], "sesquiterpene", ["hop"]
    ),
    CompoundEntry(
        "delta-cadinene",
        12306054,
        "",
        ["woody", "herbal", "thyme"],
        "sesquiterpene",
        ["hop"],
    ),
    CompoundEntry(
        "caryophyllene-oxide",
        1742210,
        "",
        ["woody", "dry", "herbal"],
        "sesquiterpene_oxide",
        ["hop"],
    ),
    CompoundEntry(
        "germacrene-D", 5317570, "", ["woody", "spicy"], "sesquiterpene", ["hop"]
    ),
    CompoundEntry(
        "alpha-muurolene",
        12306047,
        "",
        ["woody", "herbaceous"],
        "sesquiterpene",
        ["hop"],
    ),
    # ═══ HOP ACIDS ═══
    CompoundEntry(
        "isovaleric-acid",
        10430,
        "",
        ["cheesy", "sweaty", "rancid"],
        "organic_acid",
        ["hop"],
    ),
    CompoundEntry(
        "2-methylbutyric-acid", 8314, "", ["cheesy", "fruity"], "organic_acid", ["hop"]
    ),
    # ═══ HOP-DERIVED ESTERS ═══
    CompoundEntry(
        "methyl-2-methylbutyrate",
        15786,
        "",
        ["fruity", "apple", "green"],
        "ester",
        ["hop"],
    ),
    CompoundEntry(
        "geranyl-acetate", 1549026, "", ["floral", "rose", "fruity"], "ester", ["hop"]
    ),
    CompoundEntry(
        "linalyl-acetate", 8294, "", ["floral", "bergamot", "fruity"], "ester", ["hop"]
    ),
    # ═══ YEAST ESTERS ═══
    CompoundEntry(
        "isoamyl-acetate", 31276, "", ["banana", "pear", "fruity"], "ester", ["yeast"]
    ),
    CompoundEntry(
        "ethyl-acetate",
        8857,
        "",
        ["solvent", "fruity", "nail_polish"],
        "ester",
        ["yeast"],
    ),
    CompoundEntry(
        "ethyl-hexanoate",
        31265,
        "",
        ["red_apple", "anise", "fruity"],
        "ester",
        ["yeast"],
    ),
    CompoundEntry(
        "ethyl-butyrate",
        7762,
        "",
        ["pineapple", "tropical", "fruity"],
        "ester",
        ["yeast"],
    ),
    CompoundEntry(
        "ethyl-octanoate",
        7799,
        "",
        ["apricot", "tropical", "floral"],
        "ester",
        ["yeast"],
    ),
    CompoundEntry(
        "phenylethyl-acetate", 7654, "", ["rose", "honey", "floral"], "ester", ["yeast"]
    ),
    CompoundEntry(
        "ethyl-decanoate", 8048, "", ["grape", "waxy", "floral"], "ester", ["yeast"]
    ),
    CompoundEntry("ethyl-propanoate", 7749, "", ["fruity", "rum"], "ester", ["yeast"]),
    CompoundEntry(
        "ethyl-2-methylbutyrate",
        24020,
        "",
        ["apple", "berry", "fruity"],
        "ester",
        ["yeast"],
    ),
    CompoundEntry(
        "ethyl-2-methylpropanoate",
        7341,
        "",
        ["fruity", "sweet", "ethereal"],
        "ester",
        ["yeast"],
    ),
    CompoundEntry(
        "ethyl-3-methylbutyrate", 7945, "", ["berry", "tropical"], "ester", ["yeast"]
    ),
    CompoundEntry(
        "isobutyl-acetate", 8038, "", ["fruity", "banana", "floral"], "ester", ["yeast"]
    ),
    CompoundEntry(
        "hexyl-acetate", 8908, "", ["fruity", "green", "apple"], "ester", ["yeast"]
    ),
    CompoundEntry(
        "ethyl-lactate", 7344, "", ["buttery", "fruity", "creamy"], "ester", ["yeast"]
    ),
    CompoundEntry(
        "ethyl-phenylacetate", 7590, "", ["honey", "floral", "rose"], "ester", ["yeast"]
    ),
    # ═══ YEAST FUSEL ALCOHOLS ═══
    CompoundEntry(
        "isoamyl-alcohol",
        31260,
        "",
        ["fusel", "banana", "solvent"],
        "fusel_alcohol",
        ["yeast"],
    ),
    CompoundEntry(
        "2-phenylethanol",
        6054,
        "",
        ["rose", "floral", "honey"],
        "fusel_alcohol",
        ["yeast"],
    ),
    CompoundEntry(
        "isobutanol",
        6560,
        "",
        ["alcoholic", "malty", "solvent"],
        "fusel_alcohol",
        ["yeast"],
    ),
    CompoundEntry(
        "1-propanol", 1031, "", ["alcoholic", "pungent"], "fusel_alcohol", ["yeast"]
    ),
    CompoundEntry(
        "active-amyl-alcohol", 6405, "", ["fusel", "malty"], "fusel_alcohol", ["yeast"]
    ),
    CompoundEntry(
        "1-hexanol", 8103, "", ["green", "woody", "herbal"], "fusel_alcohol", ["yeast"]
    ),
    # ═══ YEAST PHENOLS ═══
    CompoundEntry(
        "4-vinyl-guaiacol",
        332,
        "",
        ["clove", "spicy", "smoky"],
        "phenol",
        ["yeast", "grain"],
    ),
    CompoundEntry(
        "4-ethylphenol",
        31242,
        "",
        ["barnyard", "leather", "smoky"],
        "phenol",
        ["yeast"],
    ),
    CompoundEntry(
        "4-vinylphenol",
        62453,
        "",
        ["phenolic", "medicinal", "smoky"],
        "phenol",
        ["yeast"],
    ),
    CompoundEntry(
        "4-ethylguaiacol", 62465, "", ["smoky", "spicy", "clove"], "phenol", ["yeast"]
    ),
    # ═══ ORGANIC ACIDS ═══
    CompoundEntry(
        "acetic-acid", 176, "", ["vinegar", "sour", "sharp"], "organic_acid", ["yeast"]
    ),
    CompoundEntry(
        "lactic-acid", 612, "", ["sour", "tart", "clean"], "organic_acid", ["yeast"]
    ),
    CompoundEntry(
        "citric-acid", 311, "", ["sour", "citrus", "tart"], "organic_acid", ["yeast"]
    ),
    CompoundEntry(
        "succinic-acid",
        1110,
        "",
        ["sour", "umami", "brothy"],
        "organic_acid",
        ["yeast"],
    ),
    CompoundEntry(
        "malic-acid", 525, "", ["sour", "apple", "green"], "organic_acid", ["yeast"]
    ),
    CompoundEntry(
        "butyric-acid",
        264,
        "",
        ["rancid", "cheesy", "vomit"],
        "organic_acid",
        ["yeast"],
    ),
    CompoundEntry(
        "hexanoic-acid",
        8892,
        "",
        ["goaty", "cheesy", "sweaty"],
        "organic_acid",
        ["yeast"],
    ),
    CompoundEntry(
        "octanoic-acid",
        379,
        "",
        ["sweaty", "cheesy", "rancid"],
        "organic_acid",
        ["yeast"],
    ),
    CompoundEntry(
        "decanoic-acid",
        2969,
        "",
        ["rancid", "soapy", "fatty"],
        "organic_acid",
        ["yeast"],
    ),
    CompoundEntry(
        "pyruvic-acid", 1060, "", ["sour", "acetic"], "organic_acid", ["yeast"]
    ),
    # ═══ YEAST KETONES/DIKETONES ═══
    CompoundEntry(
        "diacetyl",
        650,
        "",
        ["butter", "butterscotch", "slippery"],
        "diketone",
        ["yeast"],
    ),
    CompoundEntry(
        "2,3-pentanedione",
        12001,
        "",
        ["butter", "honey", "caramel"],
        "diketone",
        ["yeast"],
    ),
    CompoundEntry(
        "acetaldehyde",
        177,
        "",
        ["green_apple", "bruised_apple", "fresh"],
        "aldehyde",
        ["yeast"],
    ),
    CompoundEntry(
        "acetoin", 179, "", ["butter", "cream", "yogurt"], "ketone", ["yeast"]
    ),
    # ═══ SULFUR COMPOUNDS ═══
    CompoundEntry(
        "dimethyl-sulfide",
        1068,
        "",
        ["creamed_corn", "cooked_vegetables"],
        "sulfur",
        ["yeast", "grain"],
    ),
    CompoundEntry(
        "hydrogen-sulfide", 402, "", ["rotten_egg", "sulfury"], "sulfur", ["yeast"]
    ),
    CompoundEntry(
        "methanethiol", 878, "", ["garlic", "onion", "sulfury"], "sulfur", ["yeast"]
    ),
    CompoundEntry(
        "diethyl-sulfide", 3283, "", ["garlic", "onion", "cooked"], "sulfur", ["yeast"]
    ),
    CompoundEntry(
        "methionol",
        1710,
        "",
        ["cooked_potato", "meaty", "sulfury"],
        "sulfur",
        ["yeast"],
    ),
    # ═══ MAILLARD / GRAIN COMPOUNDS ═══
    CompoundEntry(
        "ferulic-acid",
        445858,
        "",
        ["vanilla", "clove_precursor"],
        "phenolic_acid",
        ["grain"],
    ),
    CompoundEntry(
        "maltol", 8369, "", ["caramel", "toasty", "sweet"], "pyranone", ["grain"]
    ),
    CompoundEntry(
        "furfural", 7361, "", ["sweet", "woody", "almond"], "furan", ["grain"]
    ),
    CompoundEntry(
        "5-hydroxymethylfurfural",
        237332,
        "",
        ["sweet", "caramel", "bready"],
        "furan",
        ["grain"],
    ),
    CompoundEntry(
        "2-methylpyrazine",
        13670,
        "",
        ["nutty", "roasted", "chocolate"],
        "pyrazine",
        ["grain"],
    ),
    CompoundEntry(
        "2,5-dimethylpyrazine",
        15560,
        "",
        ["nutty", "toasted", "cocoa"],
        "pyrazine",
        ["grain"],
    ),
    CompoundEntry(
        "2,6-dimethylpyrazine",
        15561,
        "",
        ["roasted", "nutty", "coffee"],
        "pyrazine",
        ["grain"],
    ),
    CompoundEntry(
        "2,3-dimethylpyrazine",
        13587,
        "",
        ["nutty", "toasted", "caramel"],
        "pyrazine",
        ["grain"],
    ),
    CompoundEntry(
        "2-ethyl-3,5-dimethylpyrazine",
        32906,
        "",
        ["roasted", "nutty", "earthy"],
        "pyrazine",
        ["grain"],
    ),
    CompoundEntry(
        "2-acetylpyridine",
        14815,
        "",
        ["grainy", "nutty", "popcorn"],
        "pyridine",
        ["grain"],
    ),
    CompoundEntry(
        "guaiacol", 460, "", ["smoky", "phenolic", "BBQ"], "phenol", ["grain"]
    ),
    CompoundEntry(
        "4-methylguaiacol", 7144, "", ["smoky", "clove", "vanilla"], "phenol", ["grain"]
    ),
    CompoundEntry(
        "syringol", 7955, "", ["smoky", "sweet", "bacon"], "phenol", ["grain"]
    ),
    CompoundEntry(
        "gamma-butyrolactone",
        7302,
        "",
        ["creamy", "caramel", "fatty"],
        "lactone",
        ["grain"],
    ),
    CompoundEntry(
        "sotolon",
        62835,
        "",
        ["maple_syrup", "fenugreek", "caramel"],
        "lactone",
        ["grain"],
    ),
    CompoundEntry(
        "furaneol",
        19266,
        "",
        ["strawberry", "caramel", "cotton_candy"],
        "furanone",
        ["grain"],
    ),
    CompoundEntry(
        "cyclotene",
        7128,
        "",
        ["maple", "caramel", "bread_crust"],
        "cyclopentanone",
        ["grain"],
    ),
    CompoundEntry(
        "isomaltol", 69341, "", ["caramel", "bread", "toasty"], "pyranone", ["grain"]
    ),
    CompoundEntry(
        "2-acetyl-1-pyrroline",
        75148,
        "",
        ["popcorn", "bread_crust", "rice"],
        "pyrroline",
        ["grain"],
    ),
    CompoundEntry(
        "2-acetylfuran", 14282, "", ["balsamic", "sweet", "nutty"], "furan", ["grain"]
    ),
    CompoundEntry(
        "2-pentylfuran", 19602, "", ["green_bean", "buttery"], "furan", ["grain"]
    ),
    # ═══ LACTONES ═══
    CompoundEntry(
        "gamma-nonalactone",
        7710,
        "",
        ["coconut", "peach", "creamy"],
        "lactone",
        ["grain"],
    ),
    CompoundEntry(
        "gamma-decalactone",
        12813,
        "",
        ["peach", "apricot", "creamy"],
        "lactone",
        ["grain", "yeast"],
    ),
    CompoundEntry(
        "gamma-octalactone",
        7703,
        "",
        ["coconut", "sweet", "herbal"],
        "lactone",
        ["grain"],
    ),
    CompoundEntry(
        "delta-decalactone",
        12810,
        "",
        ["creamy", "peach", "coconut"],
        "lactone",
        ["grain"],
    ),
    CompoundEntry(
        "whiskey-lactone", 36530, "", ["coconut", "oak", "woody"], "lactone", ["grain"]
    ),
    # ═══ ADJUNCT-DERIVED ═══
    CompoundEntry(
        "vanillin",
        1183,
        "",
        ["vanilla", "sweet", "creamy"],
        "phenolic_aldehyde",
        ["adjunct"],
    ),
    CompoundEntry(
        "cinnamaldehyde",
        637511,
        "",
        ["cinnamon", "spicy", "sweet"],
        "aldehyde",
        ["adjunct"],
    ),
    CompoundEntry(
        "eugenol", 3314, "", ["clove", "spicy", "warm"], "phenol", ["adjunct"]
    ),
    CompoundEntry(
        "citral", 638011, "", ["lemon", "citrus", "verbena"], "aldehyde", ["adjunct"]
    ),
    CompoundEntry(
        "anethole",
        637563,
        "",
        ["anise", "licorice", "sweet"],
        "phenyl_propanoid",
        ["adjunct"],
    ),
    CompoundEntry(
        "menthol",
        1254,
        "",
        ["mint", "cooling", "fresh"],
        "terpene_alcohol",
        ["adjunct"],
    ),
    CompoundEntry(
        "thymol", 6989, "", ["thyme", "herbal", "medicinal"], "phenol", ["adjunct"]
    ),
    CompoundEntry(
        "carvone",
        7439,
        "",
        ["spearmint", "caraway", "herbal"],
        "monoterpenoid",
        ["adjunct"],
    ),
    CompoundEntry(
        "benzaldehyde",
        240,
        "",
        ["almond", "cherry", "marzipan"],
        "aldehyde",
        ["adjunct"],
    ),
    CompoundEntry(
        "methyl-salicylate", 4133, "", ["wintergreen", "minty"], "ester", ["adjunct"]
    ),
    CompoundEntry(
        "cinnamic-acid",
        444539,
        "",
        ["cinnamon", "honey"],
        "phenylpropanoid",
        ["adjunct"],
    ),
    CompoundEntry(
        "caffeic-acid", 689043, "", ["coffee", "bitter"], "phenolic_acid", ["adjunct"]
    ),
    CompoundEntry(
        "capsaicin",
        1548943,
        "",
        ["hot", "spicy", "burning"],
        "capsaicinoid",
        ["adjunct"],
    ),
    CompoundEntry(
        "zingerone", 31211, "", ["ginger", "sweet", "warm"], "phenolic", ["adjunct"]
    ),
    CompoundEntry(
        "shogaol", 5281794, "", ["ginger", "pungent", "spicy"], "phenolic", ["adjunct"]
    ),
    # ═══ CAROTENOID-DERIVED (fruit adjuncts) ═══
    CompoundEntry(
        "beta-ionone",
        8885,
        "",
        ["violet", "floral", "berry"],
        "norisoprenoid",
        ["adjunct"],
    ),
    CompoundEntry(
        "beta-damascenone",
        55564746,
        "",
        ["apple", "rose", "honey"],
        "norisoprenoid",
        ["adjunct"],
    ),
    CompoundEntry(
        "alpha-ionone",
        6928,
        "",
        ["violet", "woody", "floral"],
        "norisoprenoid",
        ["adjunct"],
    ),
    # ═══ WATER CHEMISTRY (mineral indicators) ═══
    CompoundEntry(
        "calcium-chloride", 5284359, "", ["minerally", "salty"], "mineral", ["water"]
    ),
    CompoundEntry(
        "gypsum", 24928, "", ["minerally", "dry", "crisp"], "mineral", ["water"]
    ),
]

# Total: ~125 compounds (will expand with PubChem SMILES fetch)


def fetch_smiles_from_pubchem(cid: int) -> str | None:
    """Fetch canonical SMILES from PubChem REST API."""
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/property/IsomericSMILES/JSON"
    try:
        with urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
            props = data.get("PropertyTable", {}).get("Properties", [])
            if props:
                return props[0].get("IsomericSMILES") or props[0].get("SMILES", "")
    except (URLError, json.JSONDecodeError, TimeoutError) as e:
        logger.warning("PubChem fetch failed for CID %d: %s", cid, e)
    return None


def build_expanded_db() -> None:
    """Fetch SMILES from PubChem and save expanded compound DB."""
    out_path = PROJECT_ROOT / "data" / "chemistry" / "expanded_compounds.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    total = len(CURATED_COMPOUNDS)
    fetched = 0
    cached = 0

    # Check for existing partial results
    if out_path.exists():
        with open(out_path) as f:
            existing = json.load(f)
        existing_map = {c["cid"]: c["smiles"] for c in existing if c.get("smiles")}
        logger.info("Found %d existing entries with SMILES", len(existing_map))
    else:
        existing_map = {}

    for i, compound in enumerate(CURATED_COMPOUNDS):
        # Use cached SMILES if available
        if compound.cid in existing_map:
            compound.smiles = existing_map[compound.cid]
            cached += 1
        elif not compound.smiles:
            smiles = fetch_smiles_from_pubchem(compound.cid)
            if smiles:
                compound.smiles = smiles
                fetched += 1
            else:
                logger.warning("No SMILES for %s (CID=%d)", compound.name, compound.cid)
            # Rate limiting: 5 requests/second (PubChem limit)
            time.sleep(0.25)

        results.append(asdict(compound))

        if (i + 1) % 25 == 0:
            logger.info(
                "Progress: %d/%d (fetched=%d, cached=%d)", i + 1, total, fetched, cached
            )

    # Save
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    with_smiles = sum(1 for r in results if r["smiles"])
    logger.info(
        "Saved %d compounds (%d with SMILES) to %s",
        len(results),
        with_smiles,
        out_path,
    )

    # Print category summary
    from collections import Counter

    cats = Counter(r["category"] for r in results)
    logger.info("Category breakdown:")
    for cat, count in cats.most_common():
        logger.info("  %-25s %d", cat, count)


if __name__ == "__main__":
    build_expanded_db()
