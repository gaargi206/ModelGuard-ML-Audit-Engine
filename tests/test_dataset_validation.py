import pandas as pd

from src.dataset_validation import validate_dataset


def test_valid_dataset():
    df = pd.DataFrame({
        "age": [20, 21, 22, 23],
        "target": ["Yes", "No", "Yes", "No"]
    })

    result = validate_dataset(
        df,
        "target"
    )

    assert result["valid"] is True


def test_invalid_single_class_target():
    df = pd.DataFrame({
        "age": [20, 21, 22, 23],
        "target": ["Yes", "Yes", "Yes", "Yes"]
    })

    result = validate_dataset(
        df,
        "target"
    )

    assert result["valid"] is False


def test_missing_target_column():
    df = pd.DataFrame({
        "age": [20, 21, 22, 23],
        "income": [100, 200, 300, 400]
    })

    result = validate_dataset(
        df,
        "target"
    )

    assert result["valid"] is False


def test_duplicate_warning():
    df = pd.DataFrame({
        "age": [20, 20, 21, 22],
        "target": ["Yes", "Yes", "No", "No"]
    })

    result = validate_dataset(
        df,
        "target"
    )

    assert result["valid"] is True

    assert any(
        "duplicate" in warning.lower()
        for warning in result["warnings"]
    )
    