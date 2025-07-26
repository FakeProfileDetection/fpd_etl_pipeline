"""
Template for pipeline stages
Copy this file to create new pipeline stages
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

# NOTE: When implementing actual stages, update imports in run_pipeline.py
from scripts.utils.enhanced_version_manager import (
    EnhancedVersionManager as VersionManager,
)

logger = logging.getLogger(__name__)


class ExtractRelavance:
    """
    Class for the relavance stage of the pipeline.

    This class encapsulates the logic for processing data in the relavance stage.
    It handles input validation, data processing, and output generation.
    """

    def __init__(
        self,
        version_id: str,
        config: Dict[str, Any],
        dry_run: bool = False,
        local_only: bool = False,
        version_manager: Optional[VersionManager] = None,
    ):
        self.version_id = version_id
        self.config = config
        self.dry_run = dry_run
        self.local_only = local_only
        self.version_manager = version_manager or VersionManager()


def run(
    version_id: str,
    config: Dict[str, Any],
    dry_run: bool = False,
    local_only: bool = False,
    version_manager: Optional[VersionManager] = None,
):
    """
    Run the relavance stage of the pipeline.

    Args:
        version_id: Unique version identifier for this pipeline run
        config: Configuration dictionary from ConfigManager
        artifact_manager: Optional CloudArtifactManager for uploading artifacts
        **kwargs: Additional stage-specific parameters

    Returns:
        Path: Output directory where results were saved

    Raises:
        ValueError: If inputs are invalid
        RuntimeError: If processing fails
    """
    self.version_id = version_id
    self.config = config

    logger.info(f"Starting relavance stage for version {version_id}")
    start_time = datetime.now()

    # Setup paths
    # TODO: Update these based on your stage
    input_dir = Path(config["PREV_STAGE_DIR"].format(version_id=version_id))
    output_dir = Path(config["RELAVANCE_DIR"].format(version_id=version_id))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize metadata collection
    metadata = {
        "stage": "relavance",
        "version_id": version_id,
        "start_time": start_time.isoformat(),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "config": config,
    }

    try:
        # Validate inputs
        if not input_dir.exists():
            raise ValueError(f"Input directory not found: {input_dir}")

        # TODO: Add specific input validation
        # Example:
        # required_files = ["data.parquet", "metadata.json"]
        # for file in required_files:
        #     if not (input_dir / file).exists():
        #         raise ValueError(f"Required input file not found: {file}")

        # Load input data
        logger.info("Loading input data...")
        # TODO: Implement data loading
        # Example:
        # input_data = pd.read_parquet(input_dir / "data.parquet")
        # metadata["input_records"] = len(input_data)

        # Process data
        logger.info("Processing data...")
        # TODO: Implement your processing logic
        # Example:
        # processed_data = process_function(input_data)
        # metadata["output_records"] = len(processed_data)

        # Collect processing statistics
        # TODO: Track what happened during processing
        # Example:
        # metadata["statistics"] = {
        #     "records_processed": len(processed_data),
        #     "records_filtered": len(input_data) - len(processed_data),
        #     "processing_time": (datetime.now() - start_time).total_seconds()
        # }

        # Save outputs
        logger.info("Saving outputs...")
        # TODO: Save your processed data
        # Example:
        # output_file = output_dir / "processed_data.parquet"
        # processed_data.to_parquet(output_file)

        # Save metadata
        metadata["end_time"] = datetime.now().isoformat()
        metadata["duration_seconds"] = (datetime.now() - start_time).total_seconds()
        metadata["status"] = "success"

        metadata_file = output_dir / "stage_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        # Upload artifacts if manager provided
        if artifact_manager and config.get("UPLOAD_ARTIFACTS"):
            logger.info("Uploading artifacts to cloud...")

            # TODO: Upload your specific artifacts
            # Example:
            # artifact_manager.upload_artifact(
            #     local_path=output_file,
            #     artifact_type="processed_data",
            #     stage="relavance",
            #     description="Processed data from relavance stage",
            #     metadata={"records": len(processed_data)}
            # )

            # Always upload metadata
            artifact_manager.upload_artifact(
                local_path=metadata_file,
                artifact_type="metadata",
                stage="relavance",
                description="Stage processing metadata",
            )

        logger.info(
            f"✅ relavance stage completed in {metadata['duration_seconds']:.1f}s"
        )
        return output_dir

    except Exception as e:
        # Log error and update metadata
        logger.error(f"❌ relavance stage failed: {e}")

        metadata["end_time"] = datetime.now().isoformat()
        metadata["duration_seconds"] = (datetime.now() - start_time).total_seconds()
        metadata["status"] = "failed"
        metadata["error"] = str(e)

        # Save error metadata
        error_metadata_file = output_dir / "stage_metadata_error.json"
        with open(error_metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        # Re-raise the exception
        raise


# Optional: Add stage-specific helper functions below
def validate_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Validate data quality and return issues found"""
    issues = []

    # TODO: Implement validation logic
    # Example:
    # if df.isnull().any().any():
    #     null_counts = df.isnull().sum()
    #     issues.append({
    #         "type": "missing_data",
    #         "severity": "warning",
    #         "details": null_counts[null_counts > 0].to_dict()
    #     })

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "checked_at": datetime.now().isoformat(),
    }


def process_function(data: pd.DataFrame) -> pd.DataFrame:
    """Main processing logic for this stage"""
    # TODO: Implement your processing logic
    # This is where the main work happens

    # Example:
    # processed = data.copy()
    # processed['new_column'] = processed['old_column'].apply(transform)
    # return processed

    return data


if __name__ == "__main__":
    # Test the stage independently
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from scripts.utils.config_manager import get_config

    # Test configuration
    test_config = get_config().get_all()
    test_version = "test_" + datetime.now().strftime("%Y%m%d_%H%M%S")

    # Run test
    try:
        output = run(
            version_id=test_version,
            config=test_config,
            artifact_manager=None,  # No cloud uploads in test
        )
        print(f"✅ Test successful. Output: {output}")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise
