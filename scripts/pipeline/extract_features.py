#!/usr/bin/env python3
"""
Extract Features Stage
Generates machine learning features from keystroke pair data

This stage:
- Loads keypair data from previous stage
- Extracts statistical features for unigrams (single keys) and digrams (key pairs)
- Supports multiple feature types and extraction strategies
- Creates features at different aggregation levels
- Handles missing data with configurable imputation
- Saves feature metadata in etl_metadata/features/
"""

import json
import logging
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.utils.enhanced_version_manager import (
    EnhancedVersionManager as VersionManager,
)

logger = logging.getLogger(__name__)


@dataclass
class DigramCoverage:
    """Track coverage statistics for a digram"""

    digram: str
    total_occurrences: int = 0
    users_with_3plus: int = 0  # Users with ≥3 occurrences at video level
    users_with_2plus: int = 0  # Users with ≥2 occurrences
    users_with_1plus: int = 0  # Users with ≥1 occurrence
    min_occurrences_per_user: int = 0  # Minimum across all users
    coverage_level: str = ""  # 'video', 'session', or 'platform'
    threshold_met: int = 0  # 3, 2, or 1


@dataclass
class ImputationRecord:
    """Track imputation details"""

    user_id: str
    feature: str
    level: str  # 'platform', 'session', or 'video'
    strategy: str  # 'user_mean', 'global_mean', or 'zero'
    original_value: float = np.nan
    imputed_value: float = 0.0
    reason: str = ""  # Why imputation was needed


@dataclass
class FeatureConfig:
    """Configuration for feature extraction"""

    name: str
    description: str
    top_n_digrams: int = 10
    use_all_unigrams: bool = True
    imputation_strategy: str = "user"  # Changed default to 'user' from 'global'
    aggregation_level: str = "user_platform"  # 'user_platform', 'session', 'video'
    keep_outliers: bool = False
    use_coverage_based_selection: bool = True  # New flag for improved selection


class BaseFeatureExtractor(ABC):
    """Base class for feature extractors"""

    @abstractmethod
    def extract(self, data: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
        """Extract features from data"""
        pass

    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Get list of feature names this extractor produces"""
        pass


class TypeNetMLFeatureExtractor(BaseFeatureExtractor):
    """
    Extracts statistical features for TypeNet ML experiments
    Based on the original TypeNet paper methodology
    """

    def __init__(self):
        self.platform_names = {1: "facebook", 2: "instagram", 3: "twitter"}
        self.feature_names = []
        self.selected_digrams: List[str] = []
        self.digram_metadata: Dict[str, DigramCoverage] = {}
        self.imputation_records: List[ImputationRecord] = []
        self.selection_fallback_used: Dict[
            str, int
        ] = {}  # Track which fallback was used

    def get_top_digrams(self, data: pd.DataFrame, n: int = 10) -> List[str]:
        """Get top N most frequent digrams across dataset"""
        # Create digram column
        data["digram"] = data["key1"] + data["key2"]

        # Get top digrams
        digram_counts = data["digram"].value_counts().head(n)
        top_digrams = digram_counts.index.tolist()

        logger.info(f"Top {n} digrams: {top_digrams}")
        return top_digrams

    def get_all_unigrams(self, data: pd.DataFrame) -> List[str]:
        """Get all unique unigrams (individual keys) in dataset"""
        # Combine key1 and key2
        all_keys = pd.concat([data["key1"], data["key2"]])
        unigrams = sorted(all_keys.unique())

        logger.info(f"Total unique unigrams: {len(unigrams)}")
        return unigrams

    def analyze_digram_coverage(
        self, data: pd.DataFrame, level: str = "video"
    ) -> Dict[str, DigramCoverage]:
        """
        Analyze digram coverage at specified level
        Returns coverage statistics for each digram
        """
        coverage_stats = {}

        # Ensure digram column exists
        if "digram" not in data.columns:
            data["digram"] = data["key1"] + data["key2"]

        # Get unique digrams
        all_digrams = data["digram"].unique()

        # Determine grouping columns based on level
        if level == "video":
            group_cols = ["user_id", "platform_id", "session_id", "video_id"]
        elif level == "session":
            group_cols = ["user_id", "platform_id", "session_id"]
        elif level == "platform":
            group_cols = ["user_id", "platform_id"]
        else:
            group_cols = ["user_id"]

        # Get all unique users
        all_users = data["user_id"].unique()
        n_users = len(all_users)

        for digram in all_digrams:
            coverage = DigramCoverage(digram=digram)
            digram_data = data[data["digram"] == digram]

            # Count occurrences per user-level combination
            user_counts = defaultdict(lambda: float("inf"))

            # For each user, find minimum occurrences across all level combinations
            for user_id in all_users:
                user_data = digram_data[digram_data["user_id"] == user_id]

                if len(user_data) == 0:
                    user_counts[user_id] = 0
                else:
                    # Get counts for each combination at this level
                    if level == "video":
                        # Count per video, take minimum
                        counts = []
                        for (p, s, v), grp in user_data.groupby(
                            ["platform_id", "session_id", "video_id"]
                        ):
                            counts.append(len(grp))
                        # For video level, we need counts for ALL possible videos
                        # If a video is missing, count is 0
                        all_combos = data[data["user_id"] == user_id][
                            ["platform_id", "session_id", "video_id"]
                        ].drop_duplicates()
                        if len(counts) < len(all_combos):
                            counts.append(0)  # Missing combo means 0 count
                        user_counts[user_id] = min(counts) if counts else 0
                    elif level == "session":
                        counts = []
                        for (p, s), grp in user_data.groupby(
                            ["platform_id", "session_id"]
                        ):
                            counts.append(len(grp))
                        all_combos = data[data["user_id"] == user_id][
                            ["platform_id", "session_id"]
                        ].drop_duplicates()
                        if len(counts) < len(all_combos):
                            counts.append(0)
                        user_counts[user_id] = min(counts) if counts else 0
                    elif level == "platform":
                        counts = []
                        for p, grp in user_data.groupby(["platform_id"]):
                            counts.append(len(grp))
                        all_platforms = data[data["user_id"] == user_id][
                            "platform_id"
                        ].unique()
                        if len(counts) < len(all_platforms):
                            counts.append(0)
                        user_counts[user_id] = min(counts) if counts else 0

            # Calculate coverage statistics
            coverage.total_occurrences = len(digram_data)
            coverage.users_with_3plus = sum(1 for c in user_counts.values() if c >= 3)
            coverage.users_with_2plus = sum(1 for c in user_counts.values() if c >= 2)
            coverage.users_with_1plus = sum(1 for c in user_counts.values() if c >= 1)
            coverage.min_occurrences_per_user = (
                min(user_counts.values()) if user_counts else 0
            )
            coverage.coverage_level = level

            # Determine threshold met (using 80% coverage as sufficient)
            min_coverage = int(n_users * 0.8)  # 80% of users must have the digram
            if coverage.users_with_3plus >= min_coverage:
                coverage.threshold_met = 3
            elif coverage.users_with_2plus >= min_coverage:
                coverage.threshold_met = 2
            elif coverage.users_with_1plus >= min_coverage:
                coverage.threshold_met = 1
            else:
                coverage.threshold_met = 0

            coverage_stats[digram] = coverage

        return coverage_stats

    def select_top_k_digrams_with_coverage(
        self, data: pd.DataFrame, k: int = 10, use_coverage: bool = True
    ) -> List[str]:
        """
        Select top-k digrams using coverage-based strategy with fallback
        """
        if not use_coverage:
            # Fall back to original frequency-based selection
            return self.get_top_digrams(data, k)

        selected = []
        used_digrams = set()

        # Ensure digram column exists
        if "digram" not in data.columns:
            data["digram"] = data["key1"] + data["key2"]

        # Start with video level (most restrictive)
        for level in ["video", "session", "platform"]:
            if len(selected) >= k:
                break

            logger.info(f"Checking {level} level for digram selection...")
            coverage_stats = self.analyze_digram_coverage(data, level)

            # Debug: show coverage distribution
            if coverage_stats:
                n_users = len(data["user_id"].unique())
                min_coverage = int(n_users * 0.8)
                stats_summary = {
                    "total_digrams": len(coverage_stats),
                    "with_80pct_at_3+": sum(
                        1
                        for c in coverage_stats.values()
                        if c.users_with_3plus >= min_coverage
                    ),
                    "with_80pct_at_2+": sum(
                        1
                        for c in coverage_stats.values()
                        if c.users_with_2plus >= min_coverage
                    ),
                    "with_80pct_at_1+": sum(
                        1
                        for c in coverage_stats.values()
                        if c.users_with_1plus >= min_coverage
                    ),
                }
                logger.info(f"  Coverage stats: {stats_summary}")

            # Remove already selected digrams
            coverage_stats = {
                d: c for d, c in coverage_stats.items() if d not in used_digrams
            }

            # Try each threshold (3, 2, 1)
            for threshold in [3, 2, 1]:
                if len(selected) >= k:
                    break

                # Find digrams meeting threshold
                eligible = [
                    (d, c)
                    for d, c in coverage_stats.items()
                    if c.threshold_met >= threshold
                ]

                if eligible:
                    # Sort by total occurrences (frequency) among eligible
                    eligible.sort(key=lambda x: x[1].total_occurrences, reverse=True)

                    # Take as many as needed
                    n_needed = k - len(selected)
                    for digram, coverage in eligible[:n_needed]:
                        selected.append(digram)
                        used_digrams.add(digram)
                        self.digram_metadata[digram] = coverage
                        self.selection_fallback_used[digram] = (level, threshold)

                    logger.info(
                        f"  Found {len(eligible[:n_needed])} digrams at {level} level "
                        f"with threshold {threshold} ({len(eligible)} eligible total)"
                    )
                else:
                    logger.info(
                        f"  No digrams found at {level} level with threshold {threshold}"
                    )

        logger.info(f"Selected {len(selected)} digrams using coverage-based strategy")
        self.selected_digrams = selected
        return selected

    def extract_statistical_features(
        self, data: pd.DataFrame, unigrams: List[str], digrams: List[str]
    ) -> Dict[str, float]:
        """
        Extract statistical features for given data subset
        Returns: Dict with features in order: median, mean, std, q1, q3
        """
        features = {}

        # Convert timing features to milliseconds if needed
        if "HL_ms" not in data.columns:
            data["HL_ms"] = data["HL"] / 1_000_000
            data["IL_ms"] = data["IL"] / 1_000_000

        # Extract unigram (HL) features
        for unigram in unigrams:
            # Filter data for this unigram
            unigram_data = data[data["key1"] == unigram]["HL_ms"]

            if len(unigram_data) > 0:
                features[f"HL_{unigram}_median"] = float(unigram_data.median())
                features[f"HL_{unigram}_mean"] = float(unigram_data.mean())
                features[f"HL_{unigram}_std"] = (
                    float(unigram_data.std()) if len(unigram_data) > 1 else 0.0
                )
                features[f"HL_{unigram}_q1"] = float(unigram_data.quantile(0.25))
                features[f"HL_{unigram}_q3"] = float(unigram_data.quantile(0.75))
            else:
                # Missing data - will be handled by imputation
                for stat in ["median", "mean", "std", "q1", "q3"]:
                    features[f"HL_{unigram}_{stat}"] = np.nan

        # Extract digram (IL) features
        for digram in digrams:
            # Filter data for this digram
            digram_data = data[data["digram"] == digram]["IL_ms"]

            if len(digram_data) > 0:
                features[f"IL_{digram}_median"] = float(digram_data.median())
                features[f"IL_{digram}_mean"] = float(digram_data.mean())
                features[f"IL_{digram}_std"] = (
                    float(digram_data.std()) if len(digram_data) > 1 else 0.0
                )
                features[f"IL_{digram}_q1"] = float(digram_data.quantile(0.25))
                features[f"IL_{digram}_q3"] = float(digram_data.quantile(0.75))
            else:
                # Missing data - will be handled by imputation
                for stat in ["median", "mean", "std", "q1", "q3"]:
                    features[f"IL_{digram}_{stat}"] = np.nan

        return features

    def apply_imputation(self, dataset: pd.DataFrame, strategy: str) -> pd.DataFrame:
        """
        Apply imputation strategy for missing values with tracking
        Priority: user-level mean > global mean > zero
        """
        feature_cols = [
            col
            for col in dataset.columns
            if col not in ["user_id", "platform_id", "session_id", "video_id"]
        ]

        # Clear previous imputation records for this run
        self.imputation_records = []

        for col in feature_cols:
            # Get initial missing mask
            missing_mask = dataset[col].isna()

            if not missing_mask.any():
                continue  # No missing values for this feature

            # Track which rows need imputation
            missing_indices = dataset[missing_mask].index

            if strategy == "user" or strategy == "user_then_global":
                # First try user-level imputation
                user_means = dataset.groupby("user_id")[col].transform("mean")

                # Apply user mean where available
                user_mean_available = missing_mask & ~user_means.isna()
                if user_mean_available.any():
                    dataset.loc[user_mean_available, col] = user_means[
                        user_mean_available
                    ]

                    # Record user-level imputations
                    for idx in dataset[user_mean_available].index:
                        row = dataset.loc[idx]
                        self.imputation_records.append(
                            ImputationRecord(
                                user_id=row["user_id"],
                                feature=col,
                                level=f"p{row.get('platform_id', '')}_s{row.get('session_id', '')}_v{row.get('video_id', '')}",
                                strategy="user_mean",
                                imputed_value=user_means[idx],
                                reason="Missing digram in this context",
                            )
                        )

                # Apply global mean for remaining NaN
                still_missing = dataset[col].isna()
                if still_missing.any():
                    global_mean = dataset[col].mean()

                    if pd.notna(global_mean):
                        dataset.loc[still_missing, col] = global_mean

                        # Record global imputations
                        for idx in dataset[still_missing].index:
                            row = dataset.loc[idx]
                            self.imputation_records.append(
                                ImputationRecord(
                                    user_id=row["user_id"],
                                    feature=col,
                                    level=f"p{row.get('platform_id', '')}_s{row.get('session_id', '')}_v{row.get('video_id', '')}",
                                    strategy="global_mean",
                                    imputed_value=global_mean,
                                    reason="No user mean available",
                                )
                            )
                    else:
                        # Last resort: zero
                        dataset.loc[still_missing, col] = 0.0
                        for idx in dataset[still_missing].index:
                            row = dataset.loc[idx]
                            self.imputation_records.append(
                                ImputationRecord(
                                    user_id=row["user_id"],
                                    feature=col,
                                    level=f"p{row.get('platform_id', '')}_s{row.get('session_id', '')}_v{row.get('video_id', '')}",
                                    strategy="zero",
                                    imputed_value=0.0,
                                    reason="No mean available",
                                )
                            )

            elif strategy == "global":
                # Use global mean only
                global_mean = dataset[col].mean()
                if pd.notna(global_mean):
                    dataset.loc[missing_mask, col] = global_mean
                    for idx in missing_indices:
                        row = dataset.loc[idx]
                        self.imputation_records.append(
                            ImputationRecord(
                                user_id=row["user_id"],
                                feature=col,
                                level=f"p{row.get('platform_id', '')}_s{row.get('session_id', '')}_v{row.get('video_id', '')}",
                                strategy="global_mean",
                                imputed_value=global_mean,
                                reason="Global strategy specified",
                            )
                        )
                else:
                    dataset.loc[missing_mask, col] = 0.0
                    for idx in missing_indices:
                        row = dataset.loc[idx]
                        self.imputation_records.append(
                            ImputationRecord(
                                user_id=row["user_id"],
                                feature=col,
                                level=f"p{row.get('platform_id', '')}_s{row.get('session_id', '')}_v{row.get('video_id', '')}",
                                strategy="zero",
                                imputed_value=0.0,
                                reason="No global mean available",
                            )
                        )

        # Log imputation summary
        if self.imputation_records:
            n_imputations = len(self.imputation_records)
            strategies_used = set(r.strategy for r in self.imputation_records)
            users_affected = set(r.user_id for r in self.imputation_records)
            logger.info(
                f"Applied {n_imputations} imputations using strategies: {strategies_used}"
            )
            logger.info(f"Users affected by imputation: {len(users_affected)}")

        return dataset

    def extract(self, data: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
        """Extract features based on configuration"""
        # Filter valid data
        data = data[data["valid"]].copy()

        # Optionally remove outliers
        if not config.keep_outliers and "outlier" in data.columns:
            data = data[~data["outlier"]]

        logger.info(f"Processing {len(data)} valid keypairs")

        # Get digrams and unigrams
        if config.use_coverage_based_selection:
            digrams = self.select_top_k_digrams_with_coverage(
                data, config.top_n_digrams
            )
        else:
            digrams = self.get_top_digrams(data, config.top_n_digrams)
        unigrams = self.get_all_unigrams(data)

        # Store feature names for later reference
        self.feature_names = []
        for unigram in unigrams:
            for stat in ["median", "mean", "std", "q1", "q3"]:
                self.feature_names.append(f"HL_{unigram}_{stat}")
        for digram in digrams:
            for stat in ["median", "mean", "std", "q1", "q3"]:
                self.feature_names.append(f"IL_{digram}_{stat}")

        # Extract features based on aggregation level
        feature_records = []

        if config.aggregation_level == "user_platform":
            # Group by user and platform
            groups = data.groupby(["user_id", "platform_id"])

            for (user_id, platform_id), group_data in groups:
                features = self.extract_statistical_features(
                    group_data, unigrams, digrams
                )
                features["user_id"] = user_id
                features["platform_id"] = platform_id
                feature_records.append(features)

        elif config.aggregation_level == "session":
            # Group by user, platform, and session
            groups = data.groupby(["user_id", "platform_id", "session_id"])

            for (user_id, platform_id, session_id), group_data in groups:
                features = self.extract_statistical_features(
                    group_data, unigrams, digrams
                )
                features["user_id"] = user_id
                features["platform_id"] = platform_id
                features["session_id"] = session_id
                feature_records.append(features)

        elif config.aggregation_level == "video":
            # Group by user, platform, session, and video
            groups = data.groupby(["user_id", "platform_id", "session_id", "video_id"])

            for (user_id, platform_id, session_id, video_id), group_data in groups:
                features = self.extract_statistical_features(
                    group_data, unigrams, digrams
                )
                features["user_id"] = user_id
                features["platform_id"] = platform_id
                features["session_id"] = session_id
                features["video_id"] = video_id
                feature_records.append(features)

        # Create DataFrame and apply imputation
        dataset = pd.DataFrame(feature_records)
        dataset = self.apply_imputation(dataset, config.imputation_strategy)

        logger.info(f"Created feature dataset with shape: {dataset.shape}")
        return dataset

    def get_feature_names(self) -> List[str]:
        """Get list of feature names"""
        return self.feature_names


class ExtractFeaturesStage:
    """Extract ML features from keypair data"""

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

        # Feature extractors registry
        self.extractors = {"statistical": TypeNetMLFeatureExtractor()}

        # Default feature configurations
        self.feature_configs = {
            "statistical_platform": FeatureConfig(
                name="statistical_platform",
                description="Statistical features aggregated by user and platform",
                top_n_digrams=10,
                use_all_unigrams=True,
                imputation_strategy="user",  # Changed from 'global' to 'user'
                aggregation_level="user_platform",
                keep_outliers=self.config.get("KEEP_OUTLIERS", False),
                use_coverage_based_selection=True,  # Enable new selection strategy
            ),
            "statistical_session": FeatureConfig(
                name="statistical_session",
                description="Statistical features aggregated by user, platform, and session",
                top_n_digrams=10,
                use_all_unigrams=True,
                imputation_strategy="user",  # Changed from 'global' to 'user'
                aggregation_level="session",
                keep_outliers=self.config.get("KEEP_OUTLIERS", False),
                use_coverage_based_selection=True,  # Enable new selection strategy
            ),
            "statistical_video": FeatureConfig(
                name="statistical_video",
                description="Statistical features aggregated by user, platform, session, and video (most granular)",
                top_n_digrams=10,
                use_all_unigrams=True,
                imputation_strategy="user",
                aggregation_level="video",
                keep_outliers=self.config.get("KEEP_OUTLIERS", False),
                use_coverage_based_selection=True,  # Enable new selection strategy
            ),
        }

        # Statistics tracking
        self.stats = {
            "features_extracted": {},
            "processing_time": {},
            "feature_counts": {},
            "errors": [],
        }

    def run(self, input_dir: Path, feature_types: Optional[List[str]] = None) -> Path:
        """Execute the extract features stage"""
        logger.info(f"Starting Extract Features stage for version {self.version_id}")
        logger.info(f"Input directory: {input_dir}")

        # Setup output directories
        artifacts_dir = (
            Path(self.config.get("ARTIFACTS_DIR", "artifacts")) / self.version_id
        )
        output_dir = artifacts_dir / "statistical_features"
        metadata_dir = artifacts_dir / "etl_metadata" / "statistical_features"

        # Load keypair data
        keypair_file = input_dir / "keypairs.parquet"
        if not keypair_file.exists():
            # Try CSV fallback
            keypair_file = input_dir / "keypairs.csv"
            if not keypair_file.exists():
                raise FileNotFoundError(f"No keypair data found in {input_dir}")

        logger.info(f"Loading keypair data from {keypair_file}")
        if keypair_file.suffix == ".parquet":
            data = pd.read_parquet(keypair_file)
        else:
            data = pd.read_csv(keypair_file)

        logger.info(f"Loaded {len(data)} keypairs")

        # Determine which features to extract
        if feature_types is None:
            feature_types = list(self.feature_configs.keys())

        # Extract each feature type
        for feature_type in feature_types:
            if feature_type not in self.feature_configs:
                logger.warning(f"Unknown feature type: {feature_type}")
                continue

            try:
                start_time = datetime.now()

                # Get configuration and extractor
                config = self.feature_configs[feature_type]
                extractor_name = feature_type.split("_")[0]  # e.g., 'statistical'
                extractor = self.extractors.get(extractor_name)

                if not extractor:
                    logger.error(f"No extractor found for {extractor_name}")
                    continue

                logger.info(f"Extracting features: {feature_type}")
                logger.info(f"  Description: {config.description}")
                logger.info(f"  Aggregation level: {config.aggregation_level}")
                logger.info(f"  Imputation: {config.imputation_strategy}")
                logger.info(f"  Keep outliers: {config.keep_outliers}")

                # Extract features
                features_df = extractor.extract(data, config)

                # Save features
                if not self.dry_run:
                    feature_subdir = output_dir / feature_type
                    feature_subdir.mkdir(parents=True, exist_ok=True)

                    # Save as parquet and CSV
                    features_df.to_parquet(
                        feature_subdir / "features.parquet", index=False
                    )
                    features_df.to_csv(feature_subdir / "features.csv", index=False)

                    # Save feature summary
                    feature_summary = {
                        "feature_type": feature_type,
                        "description": config.description,
                        "config": asdict(config),
                        "shape": features_df.shape,
                        "columns": features_df.columns.tolist(),
                        "feature_names": extractor.get_feature_names(),
                        "extraction_time": (
                            datetime.now() - start_time
                        ).total_seconds(),
                    }

                    with open(feature_subdir / "feature_summary.json", "w") as f:
                        json.dump(feature_summary, f, indent=2)

                # Update statistics
                self.stats["features_extracted"][feature_type] = {
                    "records": len(features_df),
                    "features": len(features_df.columns)
                    - len(["user_id", "platform_id", "session_id", "video_id"]),
                }
                self.stats["processing_time"][feature_type] = (
                    datetime.now() - start_time
                ).total_seconds()
                self.stats["feature_counts"][feature_type] = features_df.shape

                logger.info(
                    f"  Extracted {features_df.shape[0]} records with {features_df.shape[1]} columns"
                )

            except Exception as e:
                logger.error(f"Error extracting {feature_type}: {e}")
                self.stats["errors"].append(f"{feature_type}: {e!s}")

        # Save extraction metadata
        if not self.dry_run:
            metadata_dir.mkdir(parents=True, exist_ok=True)

            # Feature registry
            feature_registry = {
                "version_id": self.version_id,
                "timestamp": datetime.now().isoformat(),
                "features_available": list(self.stats["features_extracted"].keys()),
                "feature_configs": {
                    k: asdict(v) for k, v in self.feature_configs.items()
                },
                "extraction_summary": self.stats,
            }

            with open(metadata_dir / "feature_registry.json", "w") as f:
                json.dump(feature_registry, f, indent=2)

            # Extraction statistics
            extraction_stats = {
                "version_id": self.version_id,
                "timestamp": datetime.now().isoformat(),
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "features_extracted": self.stats["features_extracted"],
                "processing_time": self.stats["processing_time"],
                "total_time": sum(self.stats["processing_time"].values()),
                "errors": self.stats["errors"],
            }

            with open(metadata_dir / "extraction_stats.json", "w") as f:
                json.dump(extraction_stats, f, indent=2)

            # Save enhanced metadata if using coverage-based selection
            if (
                hasattr(self.extractors["statistical"], "selected_digrams")
                and self.extractors["statistical"].selected_digrams
            ):
                extractor = self.extractors["statistical"]

                # Generate digram selection metadata
                digram_metadata = {
                    "method": "coverage_based_with_fallback",
                    "selected_digrams": extractor.selected_digrams,
                    "selection_details": {},
                }

                for digram in extractor.selected_digrams:
                    if digram in extractor.selection_fallback_used:
                        level, threshold = extractor.selection_fallback_used[digram]
                        coverage = extractor.digram_metadata.get(digram)
                        digram_metadata["selection_details"][digram] = {
                            "level": level,
                            "threshold": threshold,
                            "users_with_3plus": coverage.users_with_3plus
                            if coverage
                            else 0,
                            "users_with_2plus": coverage.users_with_2plus
                            if coverage
                            else 0,
                            "users_with_1plus": coverage.users_with_1plus
                            if coverage
                            else 0,
                            "total_occurrences": coverage.total_occurrences
                            if coverage
                            else 0,
                        }

                with open(metadata_dir / "digram_selection.json", "w") as f:
                    json.dump(digram_metadata, f, indent=2)

                # Save imputation metadata
                if extractor.imputation_records:
                    from collections import defaultdict

                    imputation_metadata = {
                        "total_imputations": len(extractor.imputation_records),
                        "by_strategy": defaultdict(int),
                        "by_user": defaultdict(int),
                        "by_feature": defaultdict(int),
                        "details": [],
                    }

                    for record in extractor.imputation_records:
                        imputation_metadata["by_strategy"][record.strategy] += 1
                        imputation_metadata["by_user"][record.user_id] += 1
                        imputation_metadata["by_feature"][record.feature] += 1

                        # Add first 100 details as examples
                        if len(imputation_metadata["details"]) < 100:
                            imputation_metadata["details"].append(
                                {
                                    "user_id": record.user_id,
                                    "feature": record.feature,
                                    "level": record.level,
                                    "strategy": record.strategy,
                                    "imputed_value": record.imputed_value,
                                    "reason": record.reason,
                                }
                            )

                    # Convert defaultdicts to regular dicts for JSON serialization
                    imputation_metadata["by_strategy"] = dict(
                        imputation_metadata["by_strategy"]
                    )
                    imputation_metadata["by_user"] = dict(
                        imputation_metadata["by_user"]
                    )
                    imputation_metadata["by_feature"] = dict(
                        imputation_metadata["by_feature"]
                    )

                    # Add list of users who needed imputation
                    imputation_metadata["users_with_imputation"] = list(
                        imputation_metadata["by_user"].keys()
                    )

                    with open(metadata_dir / "imputation_tracking.json", "w") as f:
                        json.dump(imputation_metadata, f, indent=2)

                    logger.info(
                        f"Saved imputation tracking: {imputation_metadata['total_imputations']} imputations for {len(imputation_metadata['users_with_imputation'])} users"
                    )

        # Log summary
        logger.info("Feature extraction complete:")
        logger.info(f"  Features extracted: {len(self.stats['features_extracted'])}")
        logger.info(
            f"  Total processing time: {sum(self.stats['processing_time'].values()):.2f} seconds"
        )

        if self.stats["errors"]:
            logger.warning(f"  Errors encountered: {len(self.stats['errors'])}")

        # Update version info
        if not self.dry_run:
            self.version_manager.update_stage_info(
                self.version_id,
                "extract_features",
                {
                    "output_dir": str(output_dir),
                    "features_extracted": list(self.stats["features_extracted"].keys()),
                    "completed_at": datetime.now().isoformat(),
                },
            )

        return output_dir


def run(
    version_id: str,
    config: Dict[str, Any],
    dry_run: bool = False,
    local_only: bool = False,
    feature_types: Optional[List[str]] = None,
) -> Path:
    """Entry point for the pipeline orchestrator"""
    # Get input directory from previous stage
    vm = VersionManager()
    version_info = vm.get_version(version_id)

    if not version_info or "extract_keypairs" not in version_info.get("stages", {}):
        # Default input directory
        artifacts_dir = Path(config.get("ARTIFACTS_DIR", "artifacts")) / version_id
        input_dir = artifacts_dir / "keypairs"
    else:
        # Handle both output_dir and output_path for compatibility
        keypairs_info = version_info["stages"]["extract_keypairs"]
        path_key = "output_path" if "output_path" in keypairs_info else "output_dir"
        input_dir = Path(keypairs_info[path_key])

    stage = ExtractFeaturesStage(version_id, config, dry_run, local_only)
    return stage.run(input_dir, feature_types)


if __name__ == "__main__":
    # For testing the stage independently
    import click

    from scripts.utils.config_manager import get_config

    @click.command()
    @click.option("--version-id", help="Version ID to use")
    @click.option("--input-dir", help="Input directory (overrides default)")
    @click.option(
        "--feature-types", multiple=True, help="Specific feature types to extract"
    )
    @click.option("--dry-run", is_flag=True, help="Preview without processing")
    def main(version_id, input_dir, feature_types, dry_run):
        """Test Extract Features stage independently"""
        logging.basicConfig(level=logging.INFO)

        config = get_config()._config
        vm = VersionManager()

        if not version_id:
            version_id = vm.create_version_id()
            logger.info(f"Created version ID: {version_id}")

        feature_types = list(feature_types) if feature_types else None

        if input_dir:
            stage = ExtractFeaturesStage(version_id, config, dry_run)
            output_dir = stage.run(Path(input_dir), feature_types)
        else:
            output_dir = run(version_id, config, dry_run, feature_types=feature_types)

        logger.info(f"Stage complete. Output: {output_dir}")

    main()
