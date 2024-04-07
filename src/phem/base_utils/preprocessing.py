from __future__ import annotations

from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


def get_default_preprocessing():
    # Preprocessing only for ensemble methods that use the original data.
    # As no of our methods uses the original data but only the predictions, this is technically not used and only
    # here for compatibility.
    return ColumnTransformer(
        transformers=[
            (
                "num",
                SimpleImputer(strategy="constant", fill_value=-1),
                make_column_selector(dtype_exclude="category"),
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        (
                            "encoder",
                            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                        ),
                        ("imputer", SimpleImputer(strategy="constant", fill_value=-1)),
                    ]
                ),
                make_column_selector(dtype_include="category"),
            ),
        ],
        sparse_threshold=0,
    )
