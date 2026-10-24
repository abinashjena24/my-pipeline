import argparse
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sklearn.pipeline import Pipeline

from src.serialization import load_pipeline


class InferenceRow(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")

    feature1: int | None = Field(default=None)
    feature2: int | None = Field(default=None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run batch inference using a trained ML pipeline."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Please provide the dataset(csv) file path.",
    )
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to the saved model pipeline artifact.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path where inference results will be saved as a CSV file.",
    )
    return parser.parse_args()


def validate_paths(input_path: Path) -> None:
    if not input_path.is_file():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")


def validate_batch(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate valid rows from invalid rows while preserving row indices."""
    valid_rows: list[dict[str, object]] = []
    invalid_rows: list[dict[str, object]] = []

    for index, row in df.iterrows():
        normalized: dict[str, object] = {}

        for column, value in row.items():
            column_name = str(column)

            if pd.isna(value):
                normalized[column_name] = None
            elif isinstance(value, str):
                stripped = value.strip()

                if stripped.lstrip("-").isdigit():
                    normalized[column_name] = int(stripped)
                else:
                    normalized[column_name] = value
            elif isinstance(value, float) and value.is_integer():
                normalized[column_name] = int(value)
            else:
                normalized[column_name] = value

        try:
            validated = InferenceRow.model_validate(normalized)
            valid_rows.append({"row_index": index, **validated.model_dump()})
        except ValidationError as exc:
            invalid_rows.append({"row_index": index, "error": str(exc)})

    return pd.DataFrame(valid_rows), pd.DataFrame(invalid_rows)


def validate_cols(df: pd.DataFrame, expected_cols: list[str]) -> None:
    actual_cols = list(df.columns)

    missing = set(expected_cols) - set(actual_cols)
    extra = set(actual_cols) - set(expected_cols)

    if missing or extra:
        raise ValueError(
            f"Invalid input columns. "
            f"Missing: {sorted(missing)}; "
            f"Unexpected: {sorted(extra)}"
        )


def load_model(model_path: Path) -> Pipeline:
    return load_pipeline(model_path)


def run_inference(
    df: pd.DataFrame,
    model: Pipeline,
) -> pd.DataFrame:
    """Generate predictions for a validated batch."""
    predictions = model.predict(df)

    results = df.copy()
    results = results.assign(prediction=list(predictions))
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(df)
        if probabilities.shape[1] == 2:
            results = results.assign(probability_class_1=list(probabilities[:, 1]))

    return results


def main() -> None:
    args = parse_args()
    validate_paths(args.input)

    model = load_model(args.model)
    df = pd.read_csv(args.input)

    expected_columns = list(model.feature_names_in_)
    validate_cols(df, expected_columns)

    print("Input columns validated successfully.")

    valid_rows, invalid_rows = validate_batch(df)

    if not valid_rows.empty:
        features = valid_rows[expected_columns]
        results = run_inference(features, model)
        results.insert(0, "row_index", valid_rows["row_index"].tolist())

        args.output.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(args.output, index=False)
        print(f"Predictions saved to: {args.output}")
        print(f"Valid rows processed: {len(results)}")
    else:
        print("No valid rows to predict.")

    if not invalid_rows.empty:
        error_path = args.output.with_name(f"{args.output.stem}_errors.csv")
        error_path.parent.mkdir(parents=True, exist_ok=True)
        invalid_rows.to_csv(error_path, index=False)
        print(f"Invalid rows report saved to: {error_path}")
        print(f"Invalid rows: {len(invalid_rows)}")


if __name__ == "__main__":
    main()
