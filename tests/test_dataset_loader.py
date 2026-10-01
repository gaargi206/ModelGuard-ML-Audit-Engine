import io
import pandas as pd

from src.dataset_loader import (
    load_csv_file,
    clean_dataset,
    get_dataset_profile
)


def test_clean_dataset():

    df = pd.DataFrame({
        " Name ": [" Alice ", " Bob "],
        "Age": [20, 21],
        "City": [" Pune ", "?"]
    })

    cleaned = clean_dataset(df)

    assert "Name" in cleaned.columns
    assert cleaned["Name"].iloc[0] == "Alice"
    assert pd.isna(cleaned["City"].iloc[1])


def test_dataset_profile():

    df = pd.DataFrame({
        "age": [20, 21, 22],
        "city": ["Pune", "Mumbai", "Delhi"]
    })

    profile = get_dataset_profile(df)

    assert profile["Rows"] == 3
    assert profile["Columns"] == 2
    assert profile["Numeric Columns"] == 1
    assert profile["Categorical Columns"] == 1
    assert profile["Missing Cells"] == 0
    assert profile["Duplicate Rows"] == 0


def test_load_csv_file():

    csv_content = """age,target
20,Yes
21,No
22,Yes
"""

    uploaded_file = io.BytesIO(
        csv_content.encode("utf-8")
    )

    df = load_csv_file(
        uploaded_file
    )

    assert df.shape == (3, 2)

    assert list(df.columns) == [
        "age",
        "target"
    ]

    assert df["target"].tolist() == [
        "Yes",
        "No",
        "Yes"
    ]