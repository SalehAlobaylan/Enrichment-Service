"""Cross-language golden fixtures for the vector-space identity contract.

These MUST match Content-Management-System/src/spaceid/spaceid_test.go and the
Media-Service mirror. A mismatch means a vector this service writes is
uncomparable to what CMS expects.
"""
from src.common.spaceid import compute_producer_id, compute_space_id

GOLDEN_SPACE = "58ce573ed10df8af2a0197fde7f7114cd26f844de65b14f0bc95633d36d8a70f"
GOLDEN_PRODUCER = "dd5e4491b5dfb06a6696e765a2a5e813de7e61ae043ebd9bc9585e3f132f0791"


def test_golden_space_id():
    assert (
        compute_space_id("test-model", "abc123", 4, True, "mean") == GOLDEN_SPACE
    )


def test_golden_producer_id():
    assert compute_producer_id(GOLDEN_SPACE, "r:v1") == GOLDEN_PRODUCER


def test_unresolved_revision_yields_empty():
    assert compute_space_id("m", "  ", 1024, True, "p") == ""
    assert compute_producer_id("", "r:v1") == ""


def test_revision_changes_space():
    a = compute_space_id("m", "rev1", 8, True, "p")
    b = compute_space_id("m", "rev2", 8, True, "p")
    assert a and b and a != b


def test_recipe_changes_producer_not_space():
    space = compute_space_id("m", "rev", 8, False, "cls")
    assert space
    assert compute_producer_id(space, "a:v1") != compute_producer_id(space, "b:v1")
