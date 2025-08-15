#!/usr/bin/env python3
"""
Extract KVC (Key-Value Coded) Features Stage
Maps keys to unicode indices and creates train/test splits for machine learning

This stage:
- Loads keypairs data from previous stage
- Creates unicode key mapping dictionary (key -> integer index)
- Adds key1_mapped and key2_mapped columns
- Generates platform-based train/test splits
- Saves data in exact format matching the notebook for downstream ML usage
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.utils.enhanced_version_manager import (
    EnhancedVersionManager as VersionManager,
)

logger = logging.getLogger(__name__)


class ExtractKVCFeaturesStage:
    """Extract Key-Value Coded features and create train/test splits"""

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
            "total_keypairs": 0,
            "unique_keys": 0,
            "unique_users": 0,
            "train_test_splits": {},
            "processing_time": {},
            "errors": [],
        }

    def create_key_mapping(self, df: pd.DataFrame) -> Dict[str, int]:
        """
        Create unicode key mapping dictionary from keypairs data
        Exactly matches the notebook implementation
        """
        # Get all unique keys from key1 and key2 columns, filtering out NaN values
        key1_list = df["key1"].dropna().tolist()
        key2_list = df["key2"].dropna().tolist()

        # Create sorted set of all unique keys (convert all to strings for consistent sorting)
        all_keys = [str(k) for k in key1_list] + [str(k) for k in key2_list]
        unique_keys = sorted(set(all_keys))

        # Create mapping dictionary (key -> index)
        ukeys = {key: idx for idx, key in enumerate(unique_keys)}

        logger.info(f"Created key mapping with {len(ukeys)} unique keys")
        logger.debug(f"First 10 keys: {list(ukeys.keys())[:10]}")

        return ukeys

    def add_mapped_columns(
        self, df: pd.DataFrame, ukeys: Dict[str, int]
    ) -> pd.DataFrame:
        """
        Add key1_mapped and key2_mapped columns to dataframe
        Uses -1 for unmapped keys
        """
        # Add mapped columns using pandas (convert to string for lookup)
        df["key1_mapped"] = df["key1"].map(
            lambda x: ukeys.get(str(x), -1) if pd.notna(x) else -1
        )
        df["key2_mapped"] = df["key2"].map(
            lambda x: ukeys.get(str(x), -1) if pd.notna(x) else -1
        )

        # Check for unmapped keys
        unmapped_count = ((df["key1_mapped"] == -1) | (df["key2_mapped"] == -1)).sum()
        if unmapped_count > 0:
            logger.warning(f"Found {unmapped_count} rows with unmapped keys")

        return df

    def create_kvc_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create the KVC dataframe with input_id and data columns
        Exactly matches notebook format
        """
        # Create a copy and filter for valid entries only
        df_kvc = df[df["valid"] == True].copy()

        # Create input_id from platform_id, video_id, session_id
        df_kvc["input_id"] = (
            df_kvc["platform_id"].astype(str)
            + "_"
            + df_kvc["video_id"].astype(str)
            + "_"
            + df_kvc["session_id"].astype(str)
        )

        # Create data array [key1_press, key1_release, key1_mapped]
        df_kvc["data"] = df_kvc.apply(
            lambda row: [
                int(row["key1_press"]),
                int(row["key1_release"]),
                int(row["key1_mapped"]),
            ],
            axis=1,
        )

        return df_kvc

    def get_train_val_split(
        self,
        df_kvc: pd.DataFrame,
        train_platform_ids: List[int] = [1, 2],
        hold_out_platform_id: List[int] = [3],
    ) -> Tuple[Dict, Dict]:
        """
        Create train/test split based on platform IDs
        Exactly matches notebook implementation

        Returns dictionaries in format:
        {
            user_id: {
                input_id: np.ndarray of shape (n_keypairs, 3)
            }
        }
        """
        train = {}
        test = {}

        for u in df_kvc["user_id"].unique():
            train[u] = {}
            test[u] = {}

            # Get user's data
            df_user = df_kvc[df_kvc["user_id"] == u]

            if df_user.empty:
                logger.warning(f"No data found for user {u}")
                continue

            for input_id in df_user["input_id"].unique():
                df_data = df_user[df_user["input_id"] == input_id]

                platform_ids = df_data["platform_id"].unique()
                if len(platform_ids) != 1:
                    logger.warning(
                        f"Multiple platform_ids found for input_id {input_id} for user {u}: {platform_ids}"
                    )
                    continue

                # Stack data arrays into numpy array
                arr = np.stack(df_data["data"].tolist())

                if platform_ids[0] in hold_out_platform_id:
                    # If the platform_id is in hold_out_platform_id, add to test
                    test[u][input_id] = arr
                else:
                    # Otherwise, add to train
                    train[u][input_id] = arr

        return train, test

    def run(self, input_dir: Path) -> Path:
        """Run the KVC features extraction stage"""

        logger.info(f"Starting KVC features extraction for version {self.version_id}")
        start_time = datetime.now()

        # Setup paths
        artifacts_dir = Path(self.config.get("ARTIFACTS_DIR", "artifacts"))
        keypairs_dir = artifacts_dir / self.version_id / "keypairs"
        output_dir = artifacts_dir / self.version_id / "kvc_features"

        if not self.dry_run:
            output_dir.mkdir(parents=True, exist_ok=True)

        # Load keypairs data
        csv_path = keypairs_dir / "keypairs.csv"
        parquet_path = keypairs_dir / "keypairs.parquet"

        if csv_path.exists():
            logger.info(f"Loading keypairs from CSV: {csv_path}")
            df = pd.read_csv(csv_path)
        elif parquet_path.exists():
            logger.info(f"Loading keypairs from Parquet: {parquet_path}")
            df = pd.read_parquet(parquet_path)
        else:
            raise FileNotFoundError(f"No keypairs data found in {keypairs_dir}")

        self.stats["total_keypairs"] = len(df)
        self.stats["unique_users"] = df["user_id"].nunique()

        logger.info(
            f"Loaded {self.stats['total_keypairs']} keypairs from {self.stats['unique_users']} users"
        )

        # Create key mapping
        ukeys = self.create_key_mapping(df)
        self.stats["unique_keys"] = len(ukeys)

        # Save key mapping
        if not self.dry_run:
            key_mapping_file = output_dir / "key_mapping.json"
            with open(key_mapping_file, "w") as f:
                json.dump(ukeys, f, indent=2)
            logger.info(f"Saved key mapping to {key_mapping_file}")

        # Add mapped columns
        df = self.add_mapped_columns(df, ukeys)

        # Filter for error entries if needed (as in notebook)
        df_bad = df[df["error_description"] != "No error"]
        if len(df_bad) > 0:
            logger.info(
                f"Found {len(df_bad)} error entries (will be excluded from splits)"
            )

        # Create KVC dataframe
        df_kvc = self.create_kvc_dataframe(df)

        # Save enhanced keypairs with mapping
        if not self.dry_run:
            mapped_csv = output_dir / "keypairs_mapped.csv"
            df.to_csv(mapped_csv, index=False)
            logger.info(f"Saved mapped keypairs to {mapped_csv}")

        # Generate train/test splits for each platform
        for test_platform_id in [1, 2, 3]:
            hold_out_platform_id = [test_platform_id]
            train_platform_ids = list({1, 2, 3}.difference(set(hold_out_platform_id)))

            logger.info(
                f"Creating split: Train on platforms {train_platform_ids}, test on platform {hold_out_platform_id}"
            )

            train, test = self.get_train_val_split(
                df_kvc, train_platform_ids, hold_out_platform_id
            )

            # Save features
            sub_path = f"test_platform_{test_platform_id}"
            save_path = output_dir / sub_path

            if not self.dry_run:
                save_path.mkdir(parents=True, exist_ok=True)

                # Save as numpy files (exact format from notebook)
                with open(save_path / "train.npy", "wb") as f:
                    np.save(f, train)

                with open(save_path / "test.npy", "wb") as f:
                    np.save(f, test)

                logger.info(f"Saved train/test split to {save_path}")

            # Track statistics
            self.stats["train_test_splits"][f"platform_{test_platform_id}"] = {
                "train_users": len(train),
                "test_users": len(test),
                "train_samples": sum(len(user_data) for user_data in train.values()),
                "test_samples": sum(len(user_data) for user_data in test.values()),
            }

        # Save metadata
        duration = (datetime.now() - start_time).total_seconds()
        self.stats["processing_time"] = duration

        metadata = {
            "stage": "extract_kvc_features",
            "version_id": self.version_id,
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": duration,
            "stats": self.stats,
            "config": {
                "test_platforms": [1, 2, 3],
                "filter_errors": True,
                "key_mapping_size": len(ukeys),
            },
        }

        if not self.dry_run:
            metadata_file = output_dir / "metadata.json"
            with open(metadata_file, "w") as f:
                json.dump(metadata, f, indent=2)

        # Update version manager
        self.version_manager.update_stage_info(
            self.version_id,
            "extract_kvc_features",
            {
                "completed": True,
                "output_path": str(output_dir),
                "stats": self.stats,
                "duration_seconds": duration,
            },
        )

        logger.info(f"✅ KVC features extraction completed in {duration:.1f}s")
        logger.info(f"  - Created {len(ukeys)} key mappings")
        logger.info("  - Generated 3 train/test splits")
        logger.info(f"  - Output saved to {output_dir}")

        return output_dir


def run(
    version_id: str,
    config: Dict[str, Any],
    dry_run: bool = False,
    local_only: bool = False,
) -> Path:
    """
    Entry point for pipeline integration
    """
    stage = ExtractKVCFeaturesStage(
        version_id=version_id, config=config, dry_run=dry_run, local_only=local_only
    )

    # Input directory is not used directly, but kept for consistency
    artifacts_dir = Path(config.get("ARTIFACTS_DIR", "artifacts"))
    input_dir = artifacts_dir / version_id / "keypairs"

    return stage.run(input_dir)


if __name__ == "__main__":
    # Test the stage independently
    from scripts.utils.config_manager import get_config

    # Test configuration
    test_config = get_config().get_all()
    test_version = "2025-08-10_20-08-33_loris-mbp-cable-rcn-com"  # Use existing version for testing

    # Run test
    try:
        output = run(
            version_id=test_version, config=test_config, dry_run=False, local_only=True
        )
        print(f"✅ Test successful. Output: {output}")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
