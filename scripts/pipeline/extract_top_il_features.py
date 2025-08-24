#!/usr/bin/env python3
"""
Extract Top-K IL Features Stage
Extracts the top K most frequent Inter-key Latency (IL) features from statistical features

This stage:
- Loads existing statistical features (platform, session, video levels)
- Identifies the top K most frequent IL_<key1><key2> features
- Creates new datasets with only these top K features
- Preserves metadata columns (user_id, platform_id, etc.)
- Saves filtered features in statistical_IL_top_k_features/
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.utils.enhanced_version_manager import (
    EnhancedVersionManager as VersionManager,
)

logger = logging.getLogger(__name__)


class ExtractTopILFeaturesStage:
    """Extract top-K most frequent IL features from statistical features"""

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

        # Statistics tracking
        self.stats = {
            "features_processed": {},
            "top_features_selected": {},
            "processing_time": {},
            "errors": [],
        }

    def identify_top_digrams_from_keypairs(
        self, keypairs_file: Path, k: int = 10
    ) -> List[str]:
        """
        Identify the top K digrams using intersection-based approach:
        1. Find digrams that ALL users have typed (intersection)
        2. Select top k by frequency from the intersection

        Returns list of digrams like ['eKey.space', 'th', 'av', ...]
        """
        # Load keypairs data
        logger.info(f"Loading keypairs data from {keypairs_file}")

        if keypairs_file.suffix == ".parquet":
            keypairs_df = pd.read_parquet(keypairs_file)
        else:
            keypairs_df = pd.read_csv(keypairs_file)

        # Filter valid keypairs only (including outliers for digram selection)
        valid_keypairs = keypairs_df[keypairs_df["valid"]].copy()

        # Create digram column by concatenating key1 and key2
        valid_keypairs["digram"] = valid_keypairs["key1"] + valid_keypairs["key2"]

        # Get digrams for each user (across all their data)
        logger.info("Finding digrams that appear for ALL users...")
        user_digram_sets = []
        for user_id, user_data in valid_keypairs.groupby("user_id"):
            # Filter out NaN digrams
            user_digrams = set(user_data["digram"].dropna().unique())
            user_digram_sets.append(user_digrams)

        # Calculate intersection - digrams that ALL users have typed
        if user_digram_sets:
            intersection = user_digram_sets[0].copy()
            for s in user_digram_sets[1:]:
                intersection.intersection_update(s)
        else:
            intersection = set()

        logger.info(
            f"  Found {len(intersection)} digrams that ALL {len(user_digram_sets)} users have typed"
        )

        # If we have at least k digrams in intersection, select top k by frequency
        if len(intersection) >= k:
            # Count frequency of each digram in the intersection
            digram_counts = {}
            for digram in intersection:
                digram_counts[digram] = len(
                    valid_keypairs[valid_keypairs["digram"] == digram]
                )

            # Sort by frequency and take top k
            sorted_digrams = sorted(
                digram_counts.items(), key=lambda x: x[1], reverse=True
            )
            top_digrams = [digram for digram, count in sorted_digrams[:k]]

            logger.info(
                f"Selected top {k} digrams by frequency from user-level intersection:"
            )
            for i, (digram, count) in enumerate(sorted_digrams[:k]):
                logger.info(f"  {i+1}. '{digram}' (count: {count})")
        else:
            # Use all digrams in intersection if fewer than k
            top_digrams = list(intersection)
            logger.info(
                f"  Only {len(top_digrams)} digrams in intersection, using all of them"
            )

        return top_digrams

    def get_statistical_features_for_digrams(
        self, df: pd.DataFrame, digrams: List[str]
    ) -> List[str]:
        """
        Get all statistical feature columns for the given digrams

        For each digram, includes: _mean, _median, _q1, _q3, _std
        """
        stats_suffixes = ["_mean", "_median", "_q1", "_q3", "_std"]
        feature_columns = []

        for digram in digrams:
            for suffix in stats_suffixes:
                feature_name = f"IL_{digram}{suffix}"
                if feature_name in df.columns:
                    feature_columns.append(feature_name)
                else:
                    logger.warning(
                        f"Feature not found in statistical data: {feature_name}"
                    )

        logger.info(
            f"Found {len(feature_columns)} statistical features for {len(digrams)} digrams"
        )
        return feature_columns

    def extract_top_features(
        self, df: pd.DataFrame, top_features: List[str], aggregation_level: str
    ) -> pd.DataFrame:
        """Extract only the top K features plus metadata columns"""
        # Identify metadata columns based on aggregation level
        metadata_cols = ["user_id", "platform_id"]

        if aggregation_level in ["session", "video"]:
            metadata_cols.append("session_id")
        if aggregation_level == "video":
            metadata_cols.append("video_id")

        # Combine metadata and top features
        columns_to_keep = metadata_cols + top_features

        # Filter to only existing columns
        existing_columns = [col for col in columns_to_keep if col in df.columns]

        # Create filtered dataframe
        filtered_df = df[existing_columns].copy()

        logger.info(f"Filtered dataset shape: {filtered_df.shape}")
        logger.info(f"Metadata columns: {metadata_cols}")
        logger.info(
            f"IL features retained: {len([c for c in existing_columns if c.startswith('IL_')])}"
        )

        return filtered_df

    def process_feature_type(
        self,
        input_dir: Path,
        keypairs_dir: Path,
        feature_type: str,
        k: int,
        top_digrams: List[str],
    ) -> Optional[pd.DataFrame]:
        """Process a single feature type (platform, session, or video)"""
        # Load the statistical features
        feature_file = input_dir / f"statistical_{feature_type}" / "features.csv"

        if not feature_file.exists():
            # Try parquet format
            feature_file = (
                input_dir / f"statistical_{feature_type}" / "features.parquet"
            )
            if not feature_file.exists():
                logger.error(f"Feature file not found: {feature_file}")
                return None

        logger.info(f"Loading statistical features from {feature_file}")

        try:
            if feature_file.suffix == ".parquet":
                df = pd.read_parquet(feature_file)
            else:
                df = pd.read_csv(feature_file)

            logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")

            # Get the statistical feature columns for our top digrams
            feature_columns = self.get_statistical_features_for_digrams(df, top_digrams)

            if not feature_columns:
                logger.warning(f"No matching IL features found for {feature_type}")
                return None

            # Extract selected features
            filtered_df = self.extract_top_features(df, feature_columns, feature_type)

            # Track statistics
            self.stats["features_processed"][feature_type] = {
                "original_shape": df.shape,
                "filtered_shape": filtered_df.shape,
                "il_features_selected": len(feature_columns),
                "digrams_used": len(top_digrams),
            }
            self.stats["top_features_selected"][feature_type] = feature_columns

            return filtered_df

        except Exception as e:
            logger.error(f"Error processing {feature_type}: {e}")
            self.stats["errors"].append(f"{feature_type}: {e!s}")
            return None

    def run(self, k: int = 10) -> Path:
        """Execute the extract top IL features stage"""
        logger.info(
            f"Starting Extract Top IL Features stage for version {self.version_id}"
        )
        logger.info(f"Extracting top {k} digrams and their statistical features")

        # Setup directories
        artifacts_dir = (
            Path(self.config.get("ARTIFACTS_DIR", "artifacts")) / self.version_id
        )
        input_dir = artifacts_dir / "statistical_features"
        keypairs_dir = artifacts_dir / "keypairs"

        # Handle backward compatibility - check for old "features" directory
        if not input_dir.exists():
            old_input_dir = artifacts_dir / "features"
            if old_input_dir.exists():
                logger.info(f"Using legacy features directory: {old_input_dir}")
                input_dir = old_input_dir
            else:
                raise FileNotFoundError(
                    f"Statistical features directory not found: {input_dir}"
                )

        output_dir = artifacts_dir / f"statistical_IL_top_{k}_features"
        metadata_dir = (
            artifacts_dir / "etl_metadata" / f"statistical_IL_top_{k}_features"
        )

        # First, identify top k digrams from keypairs data
        keypairs_file = keypairs_dir / "keypairs.parquet"
        if not keypairs_file.exists():
            keypairs_file = keypairs_dir / "keypairs.csv"
            if not keypairs_file.exists():
                raise FileNotFoundError(f"Keypairs data not found in {keypairs_dir}")

        top_digrams = self.identify_top_digrams_from_keypairs(keypairs_file, k)

        # Store top digrams for metadata
        self.stats["top_digrams"] = top_digrams

        # Process each feature type
        feature_types = ["platform", "session", "video"]

        for feature_type in feature_types:
            logger.info(f"\nProcessing {feature_type} level features...")
            start_time = datetime.now()

            # Process the feature type with the identified top digrams
            filtered_df = self.process_feature_type(
                input_dir, keypairs_dir, feature_type, k, top_digrams
            )

            if filtered_df is not None and not self.dry_run:
                # Save filtered features
                output_subdir = output_dir / f"statistical_IL_top_{k}_{feature_type}"
                output_subdir.mkdir(parents=True, exist_ok=True)

                # Save as both CSV and parquet
                filtered_df.to_csv(output_subdir / "features.csv", index=False)
                filtered_df.to_parquet(output_subdir / "features.parquet", index=False)

                # Save feature summary
                summary = {
                    "feature_type": f"statistical_IL_top_{k}_{feature_type}",
                    "description": f"Statistical features (mean, median, q1, q3, std) for top {k} most frequent digrams",
                    "k": k,
                    "shape": filtered_df.shape,
                    "columns": filtered_df.columns.tolist(),
                    "il_features": [
                        c for c in filtered_df.columns if c.startswith("IL_")
                    ],
                    "top_digrams": top_digrams,
                    "expected_features": k * 5,  # 5 statistical measures per digram
                    "actual_features": len(
                        [c for c in filtered_df.columns if c.startswith("IL_")]
                    ),
                    "extraction_time": (datetime.now() - start_time).total_seconds(),
                }

                with open(output_subdir / "feature_summary.json", "w") as f:
                    json.dump(summary, f, indent=2)

                logger.info(f"Saved filtered features to {output_subdir}")
                logger.info(f"  Expected IL features: {k * 5} (5 stats × {k} digrams)")
                logger.info(
                    f"  Actual IL features: {len([c for c in filtered_df.columns if c.startswith('IL_')])}"
                )

            self.stats["processing_time"][feature_type] = (
                datetime.now() - start_time
            ).total_seconds()

        # Save metadata
        if not self.dry_run:
            metadata_dir.mkdir(parents=True, exist_ok=True)

            # Feature selection report
            selection_report = {
                "version_id": self.version_id,
                "timestamp": datetime.now().isoformat(),
                "k": k,
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "top_digrams": self.stats.get("top_digrams", []),
                "features_processed": self.stats["features_processed"],
                "top_features_selected": self.stats["top_features_selected"],
                "processing_time": self.stats["processing_time"],
                "total_time": sum(self.stats["processing_time"].values()),
                "errors": self.stats["errors"],
            }

            with open(metadata_dir / "selection_report.json", "w") as f:
                json.dump(selection_report, f, indent=2)

        # Log summary
        logger.info("\nTop IL feature extraction complete:")
        logger.info(
            f"  Feature types processed: {len(self.stats['features_processed'])}"
        )
        logger.info(
            f"  Total processing time: {sum(self.stats['processing_time'].values()):.2f} seconds"
        )

        if self.stats["errors"]:
            logger.warning(f"  Errors encountered: {len(self.stats['errors'])}")

        # Update version info
        if not self.dry_run:
            self.version_manager.update_stage_info(
                self.version_id,
                f"extract_top_{k}_il_features",
                {
                    "output_dir": str(output_dir),
                    "k": k,
                    "features_extracted": list(self.stats["features_processed"].keys()),
                    "completed_at": datetime.now().isoformat(),
                },
            )

        return output_dir


def run(
    version_id: str,
    config: Dict[str, Any],
    dry_run: bool = False,
    local_only: bool = False,
    k: int = 10,
) -> Path:
    """Entry point for the pipeline orchestrator"""
    stage = ExtractTopILFeaturesStage(version_id, config, dry_run, local_only)
    return stage.run(k)


if __name__ == "__main__":
    # For testing the stage independently
    import click

    from scripts.utils.config_manager import get_config

    @click.command()
    @click.option("--version-id", help="Version ID to use")
    @click.option("--k", default=10, help="Number of top features to extract")
    @click.option("--dry-run", is_flag=True, help="Preview without processing")
    def main(version_id, k, dry_run):
        """Test Extract Top IL Features stage independently"""
        logging.basicConfig(level=logging.INFO)

        config = get_config().config
        vm = VersionManager()

        if not version_id:
            # Get the latest successful version
            versions = vm.list_versions(status="successful", limit=1)
            if versions:
                version_id = versions[0]["version_id"]
                logger.info(f"Using latest successful version: {version_id}")
            else:
                logger.error("No successful versions found")
                return

        output_dir = run(version_id, config, dry_run, k=k)
        logger.info(f"Stage complete. Output: {output_dir}")

    main()
