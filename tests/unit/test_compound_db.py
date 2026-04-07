from brewfusion.data.compound_db import COMPOUNDS, classify_yeast_family, CompoundInfo


def test_compounds_loaded():
    """Test if compound JSON is loaded safely and populated in dict."""
    assert len(COMPOUNDS) > 0, "COMPOUNDS dictionary should not be empty"

    # Check structure of first item
    first_key = list(COMPOUNDS.keys())[0]
    assert isinstance(COMPOUNDS[first_key], CompoundInfo)
    assert hasattr(COMPOUNDS[first_key], "smiles")


def test_classify_yeast_family():
    """Test yeast classification heuristic."""
    res1 = classify_yeast_family("wlp001 california ale")
    res2 = classify_yeast_family("wlp400 belgian wit ale")

    # It should return a string key
    assert isinstance(res1, str)
    assert isinstance(res2, str)
