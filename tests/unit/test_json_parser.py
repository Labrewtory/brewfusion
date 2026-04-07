import json
import pytest
from brewfusion.data.json_parser import parse_json

def test_parse_json(tmp_path):
    mock_file = tmp_path / "mock_recipes.txt"
    
    # 2 mock recipes for American Stout
    recipe1 = {
        "style": "American Stout",
        "abv": 6.5,
        "ibu": 40.0,
        "color": 35.0,
        "fermentables": [
            [10.0, "Pale Malt (2 Row) US", 36.0, 80.0, 36.0],
            [1.0, "Chocolate Malt", 34.0, 8.0, 34.0]
        ],
        "hops": [
            [1.0, "Cascade", 5.5, 60, "Boil"],
            [1.0, "Centennial", 10.0, 15, "Boil"]
        ],
        "yeast": [
            "Safale US-05"
        ]
    }
    
    recipe2 = {
        "style": "American Stout",
        "abv": 7.0,
        "ibu": 50.0,
        "color": 40.0,
        "fermentables": [
            [12.0, "Pale Malt (2 Row) US", 36.0, 85.0, 36.0],
            [1.5, "Roasted Barley", 25.0, 10.0, 25.0]
        ],
        "hops": [
            [2.0, "Simcoe", 13.0, 60, "Boil"]
        ],
        "yeast": [
            "Safale US-05"
        ]
    }

    with open(mock_file, "w") as f:
        json.dump({"rec1": recipe1, "rec2": recipe2}, f)

    graph = parse_json(mock_file)

    assert "American Stout" in graph.beer_styles
    # Median of ABV 6.5 and 7.0 is 6.75
    assert graph.beer_styles["American Stout"]["abv"] == 6.75
    assert graph.beer_styles["American Stout"]["recipe_count"] == 2

    # Verify normalization
    assert "pale malt 2-row" in graph.ingredients
    assert "chocolate malt" in graph.ingredients
    assert "roasted barley" in graph.ingredients

    assert "cascade" in graph.hops
    assert "simcoe" in graph.hops
    assert "centennial" in graph.hops

    assert "us-05" in graph.yeasts

    # Check edges exist
    assert any(u == "American Stout" and v == "pale malt 2-row" for u, v, _ in graph.uses_grain)
    assert any(u == "American Stout" and v == "cascade" for u, v, _ in graph.uses_hop)
    assert any(u == "American Stout" and v == "us-05" for u, v, _ in graph.uses_yeast)
