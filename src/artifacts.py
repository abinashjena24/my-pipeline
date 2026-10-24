import json
from pathlib import Path

from sklearn.pipeline import Pipeline

from src.metadata import RunMetadata
from src.serialization import save_pipeline


def save_artifacts(pipeline: Pipeline, metadata: RunMetadata, output_dir: Path) -> Path:

    run_dir = output_dir / metadata.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    model_path = run_dir / "model.joblib"
    metadata_path = run_dir / "metadata.json"

    save_pipeline(pipeline, model_path)

    saved_metadata = metadata.model_copy(update={"artifact_path": model_path})

    metadata_path.write_text(
        json.dumps(
            saved_metadata.model_dump(mode="json"),
            indent=4,
        ),
        encoding="utf-8",
    )
    return run_dir
