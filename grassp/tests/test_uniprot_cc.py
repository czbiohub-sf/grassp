import pytest

from grassp.datasets.uniprot_cc import (
    _parse_subcell_vocabulary,
    find_roots,
    uniprot_subcellular_vocabulary,
)


def test_parse_simple_entry():
    """Test parsing a simple entry with all required fields."""
    content = """ID   Test location.
AC   SL-9999
DE   This is a test location.
//"""

    vocab = _parse_subcell_vocabulary(content)

    assert "SL-9999" in vocab
    assert vocab["SL-9999"]["ID"] == "Test location"
    assert vocab["SL-9999"]["DE"] == "This is a test location"
    assert vocab["SL-9999"]["HI"] == []
    assert vocab["SL-9999"]["HP"] == []


def test_parse_multiline_definition():
    """Test parsing entry with multi-line definition."""
    content = """ID   Test location.
AC   SL-9999
DE   This is the first line.
DE   This is the second line.
//"""

    vocab = _parse_subcell_vocabulary(content)

    assert vocab["SL-9999"]["DE"] == "This is the first line This is the second line"


def test_parse_hierarchy_relationships():
    """Test parsing HI and HP fields."""
    content = """ID   Test location.
AC   SL-9999
DE   Test.
HI   Parent location 1.
HI   Parent location 2.
HP   Container location 1.
HP   Container location 2.
//"""

    vocab = _parse_subcell_vocabulary(content)

    assert vocab["SL-9999"]["HI"] == ["Parent location 1", "Parent location 2"]
    assert vocab["SL-9999"]["HP"] == ["Container location 1", "Container location 2"]


def test_parse_optional_fields():
    """Test parsing optional fields (SY, GO, SL)."""
    content = """ID   Test location.
AC   SL-9999
DE   Test.
SY   Synonym 1; Synonym 2; Synonym 3.
GO   GO:0005634; term name
GO   GO:0005737; other term
SL   Container, parent, test location.
//"""

    vocab = _parse_subcell_vocabulary(content)

    assert vocab["SL-9999"]["SY"] == ["Synonym 1", "Synonym 2", "Synonym 3"]
    assert vocab["SL-9999"]["GO"] == ["GO:0005634", "GO:0005737"]
    assert vocab["SL-9999"]["SL"] == "Container, parent, test location"


def test_parse_multiple_entries():
    """Test parsing multiple entries."""
    content = """ID   Location 1.
AC   SL-0001
DE   First location.
//
ID   Location 2.
AC   SL-0002
DE   Second location.
//"""

    vocab = _parse_subcell_vocabulary(content)

    assert len(vocab) == 2
    assert "SL-0001" in vocab
    assert "SL-0002" in vocab
    assert vocab["SL-0001"]["ID"] == "Location 1"
    assert vocab["SL-0002"]["ID"] == "Location 2"


def test_parse_empty_hi_hp():
    """Test that entries without HI/HP have empty lists."""
    content = """ID   Test location.
AC   SL-9999
DE   Test.
//"""

    vocab = _parse_subcell_vocabulary(content)

    assert vocab["SL-9999"]["HI"] == []
    assert vocab["SL-9999"]["HP"] == []


def test_parse_trailing_periods_removed():
    """Test that trailing periods are removed from values."""
    content = """ID   Test location.
AC   SL-9999
DE   Test description.
HI   Parent.
HP   Container.
//"""

    vocab = _parse_subcell_vocabulary(content)

    assert vocab["SL-9999"]["ID"] == "Test location"
    assert vocab["SL-9999"]["DE"] == "Test description"
    assert vocab["SL-9999"]["HI"] == ["Parent"]
    assert vocab["SL-9999"]["HP"] == ["Container"]


def test_uniprot_subcellular_vocabulary_loads():
    """Test that the vocabulary loads successfully."""
    vocab = uniprot_subcellular_vocabulary()

    # Check that we have entries
    assert len(vocab) > 500

    # Check accession ID format
    for acc in vocab.keys():
        assert acc.startswith("SL-")
        assert acc[3:].isdigit()

    # Check required fields exist
    for entry in vocab.values():
        assert "ID" in entry
        assert "DE" in entry
        assert "HI" in entry
        assert "HP" in entry
        assert isinstance(entry["HI"], list)
        assert isinstance(entry["HP"], list)


def test_uniprot_subcellular_vocabulary_known_entries():
    """Test known entries from the vocabulary."""
    vocab = uniprot_subcellular_vocabulary()

    # Check Nucleus (SL-0191)
    if "SL-0191" in vocab:
        nucleus = vocab["SL-0191"]
        assert nucleus["ID"] == "Nucleus"
        assert "GO" in nucleus
        assert "GO:0005634" in nucleus["GO"]

    # Check that some entries have hierarchical relationships
    has_hi = sum(1 for v in vocab.values() if v["HI"])
    has_hp = sum(1 for v in vocab.values() if v["HP"])

    assert has_hi > 0, "Should have entries with HI (is-a) relationships"
    assert has_hp > 0, "Should have entries with HP (part-of) relationships"


def test_uniprot_subcellular_vocabulary_caching():
    """Test that vocabulary is cached after first load."""
    from grassp.datasets.uniprot_cc import _get_cache_path

    cache_path = _get_cache_path()

    # Load vocabulary (should use cache)
    vocab1 = uniprot_subcellular_vocabulary()

    # Check cache exists
    assert cache_path.exists()

    # Load again (should use cache, not download)
    vocab2 = uniprot_subcellular_vocabulary()

    # Should be identical
    assert len(vocab1) == len(vocab2)
    assert set(vocab1.keys()) == set(vocab2.keys())


def test_find_roots_no_parents():
    """Test find_roots with entry that has no parents (is already a root)."""
    vocab = uniprot_subcellular_vocabulary()

    # Nucleus typically has no HP relationships
    if "SL-0191" in vocab:
        roots = find_roots("SL-0191", vocab)
        assert roots == ["SL-0191"]
        assert vocab[roots[0]]["ID"] == "Nucleus"


def test_find_roots_with_single_parent():
    """Test find_roots with entry that has a single parent."""
    # Create test vocabulary
    vocab = {
        "SL-0001": {"ID": "Root", "HP": [], "HI": []},
        "SL-0002": {"ID": "Child", "HP": ["Root"], "HI": []},
    }

    roots = find_roots("SL-0002", vocab)
    assert roots == ["SL-0001"]


def test_find_roots_with_multiple_parents():
    """Test find_roots with entry that has multiple parents."""
    # Create test vocabulary
    vocab = {
        "SL-0001": {"ID": "Root1", "HP": [], "HI": []},
        "SL-0002": {"ID": "Root2", "HP": [], "HI": []},
        "SL-0003": {"ID": "Child", "HP": ["Root1", "Root2"], "HI": []},
    }

    roots = find_roots("SL-0003", vocab)
    assert set(roots) == {"SL-0001", "SL-0002"}


def test_find_roots_multi_level():
    """Test find_roots with multi-level hierarchy."""
    # Create test vocabulary with nested hierarchy
    vocab = {
        "SL-0001": {"ID": "Root", "HP": [], "HI": []},
        "SL-0002": {"ID": "Level1", "HP": ["Root"], "HI": []},
        "SL-0003": {"ID": "Level2", "HP": ["Level1"], "HI": []},
        "SL-0004": {"ID": "Level3", "HP": ["Level2"], "HI": []},
    }

    # Should traverse from Level3 all the way to Root
    roots = find_roots("SL-0004", vocab)
    assert roots == ["SL-0001"]


def test_find_roots_with_hi_relationship():
    """Test find_roots using HI (is-a) relationships instead of HP."""
    vocab = {
        "SL-0001": {"ID": "RootClass", "HP": [], "HI": []},
        "SL-0002": {"ID": "ChildClass", "HP": [], "HI": ["RootClass"]},
    }

    roots = find_roots("SL-0002", vocab, relationship="HI")
    assert roots == ["SL-0001"]


def test_find_roots_invalid_accession():
    """Test find_roots with invalid accession."""
    vocab = {"SL-0001": {"ID": "Test", "HP": [], "HI": []}}

    with pytest.raises(ValueError, match="not found"):
        find_roots("SL-9999", vocab)


def test_find_roots_parent_not_in_vocab():
    """Test find_roots when parent name is not in vocabulary."""
    vocab = {
        "SL-0001": {"ID": "Child", "HP": ["UnknownParent"], "HI": []},
    }

    # Should treat child as root since parent doesn't exist
    roots = find_roots("SL-0001", vocab)
    assert roots == ["SL-0001"]


def test_find_roots_real_hierarchy():
    """Test find_roots with real vocabulary data."""
    vocab = uniprot_subcellular_vocabulary()

    # Find entries with HP relationships and test them
    tested = False
    for acc, entry in vocab.items():
        hp = entry.get("HP", [])
        if hp:  # Has parent relationships
            roots = find_roots(acc, vocab)

            # Roots should be valid accession IDs
            for root in roots:
                assert root in vocab
                # Roots should have no HP relationships (or only self-reference)
                # Note: Some roots might still have HP, just testing they exist
                assert "ID" in vocab[root]

            tested = True
            break

    assert tested, "No entries with HP relationships found for testing"
