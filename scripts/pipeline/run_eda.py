#!/usr/bin/env python3
"""
Run EDA Stage
Generates exploratory data analysis reports and visualizations

This stage:
- Analyzes data quality and completeness
- Generates statistical summaries
- Creates visualizations for timing features
- Produces data quality reports
- Analyzes keystroke patterns and anomalies
- Generates HTML reports for easy sharing
- Saves all outputs in eda_reports/
"""

import base64
import json
import logging
import shutil
import sys
import warnings
from collections import defaultdict
from datetime import datetime
from json import JSONEncoder
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class NumpyEncoder(JSONEncoder):
    """Custom JSON encoder that handles NumPy types"""

    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        return super().default(obj)


from scripts.utils.enhanced_version_manager import (
    EnhancedVersionManager as VersionManager,
)

logger = logging.getLogger(__name__)


class DataQualityAnalyzer:
    """Analyzes data quality issues in keystroke data"""

    def __init__(self):
        self.skip_keys = {
            "Key.shift",
            "Key.ctrl",
            "Key.alt",
            "Key.cmd",
            "Key.caps_lock",
        }

    def analyze_raw_keystrokes(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze raw keystroke data for quality issues"""
        results = {
            "total_events": len(df),
            "unique_keys": df["key"].nunique(),
            "issues": [],
            "issue_counts": defaultdict(int),
            "unreleased_keys": {},
            "key_stats": defaultdict(
                lambda: {"presses": 0, "releases": 0, "issues": 0}
            ),
        }

        # Group by user and file to analyze each session
        for (user_id, filename), session_df in df.groupby(["user_id", "source_file"]):
            session_issues = self._analyze_session(session_df, user_id, filename)
            results["issues"].extend(session_issues["issues"])

            # Aggregate issue counts
            for issue_type, count in session_issues["issue_counts"].items():
                results["issue_counts"][issue_type] += count

            # Track unreleased keys
            if session_issues["unreleased_keys"]:
                results["unreleased_keys"][f"{user_id}_{filename}"] = session_issues[
                    "unreleased_keys"
                ]

            # Aggregate key stats
            for key, stats in session_issues["key_stats"].items():
                for stat_type, value in stats.items():
                    results["key_stats"][key][stat_type] += value

        return results

    def _analyze_session(
        self, df: pd.DataFrame, user_id: str, filename: str
    ) -> Dict[str, Any]:
        """Analyze a single session for issues"""
        active_keys = {}
        issues = []
        issue_counts = defaultdict(int)
        key_stats = defaultdict(lambda: {"presses": 0, "releases": 0, "issues": 0})

        # Sort by timestamp
        df = df.sort_values("timestamp")

        for idx, row in df.iterrows():
            key = row["key"]
            event_type = row["type"]
            timestamp = row["timestamp"]

            # Update statistics
            if event_type == "P":
                key_stats[key]["presses"] += 1
            else:
                key_stats[key]["releases"] += 1

            # Skip modifier keys for issue detection
            if key in self.skip_keys:
                continue

            if event_type == "P":
                if key in active_keys:
                    # Double press issue
                    issues.append(
                        {
                            "type": "double_press",
                            "user_id": user_id,
                            "file": filename,
                            "key": key,
                            "first_press": active_keys[key],
                            "second_press": timestamp,
                            "index": idx,
                        }
                    )
                    issue_counts["double_press"] += 1
                    key_stats[key]["issues"] += 1
                active_keys[key] = timestamp

            elif event_type == "R":
                if key not in active_keys:
                    # Orphan release
                    issues.append(
                        {
                            "type": "orphan_release",
                            "user_id": user_id,
                            "file": filename,
                            "key": key,
                            "time": timestamp,
                            "index": idx,
                        }
                    )
                    issue_counts["orphan_release"] += 1
                    key_stats[key]["issues"] += 1
                else:
                    # Check for negative hold time
                    hold_time = timestamp - active_keys[key]
                    if hold_time < 0:
                        issues.append(
                            {
                                "type": "negative_hold_time",
                                "user_id": user_id,
                                "file": filename,
                                "key": key,
                                "press_time": active_keys[key],
                                "release_time": timestamp,
                                "hold_time": hold_time,
                                "index": idx,
                            }
                        )
                        issue_counts["negative_hold_time"] += 1
                        key_stats[key]["issues"] += 1
                    del active_keys[key]

        # Record unreleased keys
        unreleased = {
            key: time for key, time in active_keys.items() if key not in self.skip_keys
        }

        return {
            "issues": issues,
            "issue_counts": dict(issue_counts),
            "unreleased_keys": unreleased,
            "key_stats": dict(key_stats),
        }


class FeatureAnalyzer:
    """Analyzes extracted features and timing patterns"""

    def analyze_timing_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze timing feature distributions"""
        timing_features = ["HL", "IL", "PL", "RL"]
        results = {}

        # Filter valid data
        valid_df = df[df["valid"]]

        for feature in timing_features:
            if feature in valid_df.columns:
                # Values are already in milliseconds
                values = valid_df[feature]

                # Remove nulls
                values = values.dropna()

                if len(values) > 0:
                    results[feature] = {
                        "count": len(values),
                        "mean": float(values.mean()),
                        "std": float(values.std()),
                        "min": float(values.min()),
                        "max": float(values.max()),
                        "median": float(values.median()),
                        "q25": float(values.quantile(0.25)),
                        "q75": float(values.quantile(0.75)),
                        "negative_count": int((values < 0).sum()),
                        "zero_count": int((values == 0).sum()),
                        "outlier_threshold_low": float(values.quantile(0.01)),
                        "outlier_threshold_high": float(values.quantile(0.99)),
                    }
                else:
                    results[feature] = {"count": 0}

        return results

    def load_user_metadata(self, artifacts_dir: Path) -> Dict[str, str]:
        """Load user metadata including consent timestamps"""
        metadata_path = (
            artifacts_dir / "cleaned_data" / "desktop" / "metadata" / "metadata.csv"
        )
        if metadata_path.exists():
            metadata_df = pd.read_csv(metadata_path)
            return dict(
                zip(metadata_df["user_id"], metadata_df["consent_timestamp"].fillna(""))
            )
        return {}

    def analyze_negative_values(
        self, df: pd.DataFrame, user_metadata: Dict[str, str] = None
    ) -> Dict[str, Any]:
        """Analyze negative IL and RL values with scalable summary format"""
        valid_df = df[df["valid"]]
        if user_metadata is None:
            user_metadata = {}

        # Define expected negative IL combinations (modifier keys)
        modifier_keys = {
            "Key.shift",
            "Key.shift_r",
            "Key.shift_l",
            "Key.ctrl",
            "Key.ctrl_r",
            "Key.ctrl_l",
            "Key.alt",
            "Key.alt_r",
            "Key.alt_l",
            "Key.option",
            "Key.option_r",
            "Key.option_l",
            "Key.cmd",
            "Key.cmd_r",
            "Key.cmd_l",
            "Key.win",
            "Key.win_r",
            "Key.win_l",
            "Key.fn",
            "Key.caps_lock",
            "Key.tab",
        }

        results = {
            "summary": {
                "total_users": valid_df["user_id"].nunique(),
                "users_with_negative_il": 0,
                "users_with_negative_rl": 0,
                "modifier_keys": sorted(list(modifier_keys)),
            },
            "negative_IL": {
                "total_count": 0,
                "expected_count": 0,
                "unexpected_count": 0,
                "top_patterns": [],  # List format for easier HTML rendering
            },
            "negative_RL": {
                "total_count": 0,
                "with_negative_IL": 0,
                "standalone": 0,
                "top_patterns": [],
            },
            "user_summary": [],  # Scalable summary table
        }

        # Analyze negative IL values
        negative_il_by_user = {}
        if "IL" in valid_df.columns:
            negative_il_df = valid_df[valid_df["IL"] < 0].copy()
            results["negative_IL"]["total_count"] = len(negative_il_df)

            if len(negative_il_df) > 0:
                # Add classification
                negative_il_df["is_expected"] = negative_il_df["key1"].isin(
                    modifier_keys
                )

                # Overall counts
                results["negative_IL"]["expected_count"] = int(
                    negative_il_df["is_expected"].sum()
                )
                results["negative_IL"]["unexpected_count"] = int(
                    (~negative_il_df["is_expected"]).sum()
                )

                # Top patterns (limit to 15 for scalability)
                key_pairs = (
                    negative_il_df.groupby(["key1", "key2", "is_expected"])
                    .size()
                    .reset_index(name="count")
                )
                key_pairs = key_pairs.sort_values("count", ascending=False).head(15)

                results["negative_IL"]["top_patterns"] = [
                    {
                        "pattern": f"{row['key1']} → {row['key2']}",
                        "count": int(row["count"]),
                        "type": "expected" if row["is_expected"] else "unexpected",
                    }
                    for _, row in key_pairs.iterrows()
                ]

                # Group by user for summary
                for user_id, user_df in negative_il_df.groupby("user_id"):
                    # Get top 3 patterns for this user
                    user_patterns = user_df.groupby(["key1", "key2"]).size().nlargest(3)
                    top_patterns_str = ", ".join(
                        [f"{k1}→{k2}" for (k1, k2) in user_patterns.index[:3]]
                    )

                    negative_il_by_user[user_id] = {
                        "total": len(user_df),
                        "expected": int(user_df["is_expected"].sum()),
                        "unexpected": int((~user_df["is_expected"]).sum()),
                        "top_patterns": top_patterns_str,
                    }

        # Analyze negative RL values
        negative_rl_by_user = {}
        if "RL" in valid_df.columns:
            negative_rl_df = valid_df[valid_df["RL"] < 0].copy()
            results["negative_RL"]["total_count"] = len(negative_rl_df)

            if len(negative_rl_df) > 0:
                # Check overlap with negative IL
                if "IL" in negative_rl_df.columns:
                    both_negative = negative_rl_df[negative_rl_df["IL"] < 0]
                    results["negative_RL"]["with_negative_IL"] = len(both_negative)
                    results["negative_RL"]["standalone"] = len(negative_rl_df) - len(
                        both_negative
                    )

                # Top patterns
                key_pairs = (
                    negative_rl_df.groupby(["key1", "key2"])
                    .size()
                    .reset_index(name="count")
                )
                key_pairs = key_pairs.sort_values("count", ascending=False).head(15)

                results["negative_RL"]["top_patterns"] = [
                    {
                        "pattern": f"{row['key1']} → {row['key2']}",
                        "count": int(row["count"]),
                    }
                    for _, row in key_pairs.iterrows()
                ]

                # Group by user
                for user_id, user_df in negative_rl_df.groupby("user_id"):
                    negative_rl_by_user[user_id] = {
                        "total": len(user_df),
                        "with_negative_il": len(user_df[user_df["IL"] < 0])
                        if "IL" in user_df.columns
                        else 0,
                    }

        # Create scalable user summary table
        all_users = set(negative_il_by_user.keys()) | set(negative_rl_by_user.keys())
        results["summary"]["users_with_negative_il"] = len(negative_il_by_user)
        results["summary"]["users_with_negative_rl"] = len(negative_rl_by_user)

        for user_id in sorted(all_users):
            user_row = {
                "user_id": user_id,  # Full user ID for search/copy
                "total_keypairs": int(len(valid_df[valid_df["user_id"] == user_id])),
                "negative_il_total": 0,
                "negative_il_expected": 0,
                "negative_il_unexpected": 0,
                "negative_il_percent": 0.0,
                "negative_rl_total": 0,
                "negative_rl_percent": 0.0,
                "top_il_patterns": "",
                "timestamp": user_metadata.get(user_id, "") if user_metadata else "",
            }

            if user_id in negative_il_by_user:
                il_data = negative_il_by_user[user_id]
                user_row["negative_il_total"] = il_data["total"]
                user_row["negative_il_expected"] = il_data["expected"]
                user_row["negative_il_unexpected"] = il_data["unexpected"]
                user_row["negative_il_percent"] = round(
                    il_data["total"] / user_row["total_keypairs"] * 100, 1
                )
                user_row["top_il_patterns"] = il_data["top_patterns"]

            if user_id in negative_rl_by_user:
                rl_data = negative_rl_by_user[user_id]
                user_row["negative_rl_total"] = rl_data["total"]
                user_row["negative_rl_percent"] = round(
                    rl_data["total"] / user_row["total_keypairs"] * 100, 1
                )

            results["user_summary"].append(user_row)

        return results

    def analyze_extreme_hold_latency(
        self, df: pd.DataFrame, threshold_ms: float = 5000
    ) -> Dict[str, Any]:
        """Analyze extreme Hold Latency outliers

        Args:
            df: Keypairs dataframe
            threshold_ms: Threshold for extreme HL values (default 5000ms = 5 seconds)
        """
        # Filter valid data with HL values
        valid_df = df[df["valid"] & df["HL"].notna()].copy()

        # Find extreme HL values
        extreme_hl = valid_df[valid_df["HL"] > threshold_ms].copy()

        results = {
            "threshold_ms": threshold_ms,
            "total_extreme_count": len(extreme_hl),
            "percentage_of_valid": (len(extreme_hl) / len(valid_df) * 100)
            if len(valid_df) > 0
            else 0,
            "extreme_cases": [],
        }

        if len(extreme_hl) > 0:
            # Sort by HL value descending
            extreme_hl = extreme_hl.sort_values("HL", ascending=False)

            # Analyze key distribution
            key_counts = extreme_hl["key1"].value_counts()
            results["key_distribution"] = {
                key: int(count) for key, count in key_counts.head(20).items()
            }

            # Get top extreme cases
            for _, row in extreme_hl.head(50).iterrows():
                case = {
                    "user_id": row["user_id"],
                    "key": row["key1"],
                    "hl_ms": float(row["HL"]),
                    "hl_seconds": float(row["HL"] / 1000),
                    "next_key": row["key2"] if "key2" in row else "N/A",
                    "device_type": row.get("device_type", "unknown"),
                }

                # Add session info if available
                if "session_id" in row:
                    case["session_id"] = int(row["session_id"])
                if "video_id" in row:
                    case["video_id"] = int(row["video_id"])

                results["extreme_cases"].append(case)

            # Statistical summary of extreme cases
            results["extreme_stats"] = {
                "mean_ms": float(extreme_hl["HL"].mean()),
                "median_ms": float(extreme_hl["HL"].median()),
                "max_ms": float(extreme_hl["HL"].max()),
                "max_seconds": float(extreme_hl["HL"].max() / 1000),
                "min_ms": float(extreme_hl["HL"].min()),
                "std_ms": float(extreme_hl["HL"].std()),
            }

            # User distribution
            users_with_extreme = extreme_hl["user_id"].nunique()
            total_users = valid_df["user_id"].nunique()
            results["user_stats"] = {
                "users_with_extreme_hl": users_with_extreme,
                "total_users": total_users,
                "percentage_users_affected": (users_with_extreme / total_users * 100)
                if total_users > 0
                else 0,
            }

            # Analyze by user
            user_extreme_counts = (
                extreme_hl.groupby("user_id").size().sort_values(ascending=False)
            )
            results["users_most_affected"] = [
                {"user_id": user_id, "count": int(count)}
                for user_id, count in user_extreme_counts.head(10).items()
            ]

        return results

    def analyze_user_performance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze per-user statistics"""
        user_stats = []

        for user_id, user_df in df.groupby("user_id"):
            total_count = len(user_df)
            valid_count = user_df["valid"].sum()

            stats = {
                "user_id": user_id,
                "total_keypairs": total_count,
                "valid_keypairs": int(valid_count),
                "invalid_keypairs": int(total_count - valid_count),
                "validity_rate": float(valid_count / total_count * 100)
                if total_count > 0
                else 0,
            }

            # Add outlier stats if available
            if "outlier" in user_df.columns:
                outlier_count = user_df["outlier"].sum()
                stats["outlier_count"] = int(outlier_count)
                stats["outlier_rate"] = (
                    float(outlier_count / valid_count * 100) if valid_count > 0 else 0
                )

            user_stats.append(stats)

        return pd.DataFrame(user_stats)


class ReportGenerator:
    """Generates HTML reports and visualizations"""

    def __init__(self, output_dir: Path, embed_images: bool = True):
        self.output_dir = output_dir
        self.figures_dir = output_dir / "figures"
        self.tables_dir = output_dir / "tables"
        self.embed_images = embed_images  # Flag to embed images as base64

        # Create directories
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)

        # Set plotting style
        plt.style.use("seaborn-v0_8-darkgrid")
        sns.set_palette("husl")

    def image_to_base64(self, image_path: Path) -> str:
        """Convert image to base64 data URL"""
        if not image_path.exists():
            return ""

        with open(image_path, "rb") as img_file:
            encoded = base64.b64encode(img_file.read()).decode("utf-8")

        # Determine MIME type
        suffix = image_path.suffix.lower()
        mime_type = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
        }.get(suffix, "image/png")

        return f"data:{mime_type};base64,{encoded}"

    def get_image_src(self, relative_path: str) -> str:
        """Get image source - either base64 or relative path"""
        if self.embed_images:
            full_path = self.output_dir / relative_path
            return self.image_to_base64(full_path)
        else:
            return relative_path

    def create_timing_distributions(
        self, timing_stats: Dict[str, Any], keypairs_df: pd.DataFrame = None
    ) -> str:
        """Create insightful distribution plots for timing features"""
        # Create two separate figures: one with all data, one without outliers
        filenames = []

        features = ["HL", "IL", "PL", "RL"]
        feature_names = {
            "HL": "Hold Latency",
            "IL": "Inter-key Latency",
            "PL": "Press Latency",
            "RL": "Release Latency",
        }

        # If we have the actual data, use it for better visualizations
        if keypairs_df is not None and not keypairs_df.empty:
            valid_df = keypairs_df[keypairs_df["valid"]].copy()

            # Create two versions: with all data and without outliers
            for include_outliers in [True, False]:
                fig = plt.figure(figsize=(16, 12))
                fig.suptitle(
                    "Timing Feature Distributions - All Valid Data"
                    if include_outliers
                    else "Timing Feature Distributions - Outliers Removed",
                    fontsize=16,
                )

                for idx, feature in enumerate(features):
                    ax = plt.subplot(3, 2, idx + 1)

                    if (
                        feature in valid_df.columns
                        and valid_df[feature].notna().sum() > 0
                    ):
                        data_all = valid_df[feature].dropna()

                        if include_outliers:
                            data = data_all
                        else:
                            # Remove outliers using IQR method
                            q1 = data_all.quantile(0.25)
                            q3 = data_all.quantile(0.75)
                            iqr = q3 - q1
                            lower_bound = q1 - 3 * iqr
                            upper_bound = q3 + 3 * iqr

                            # Keep negative values for IL and RL (they're meaningful)
                            if feature in ["IL", "RL"]:
                                data = data_all[
                                    (data_all >= lower_bound)
                                    & (data_all <= upper_bound)
                                ]
                            else:
                                data = data_all[
                                    (data_all >= 0) & (data_all <= upper_bound)
                                ]

                        # Create histogram with appropriate scale
                        if feature == "HL" and data.max() > 1000:
                            # For HL with wide range, use log scale
                            bins = np.logspace(
                                np.log10(max(1, data.min())), np.log10(data.max()), 50
                            )
                            ax.hist(
                                data,
                                bins=bins,
                                alpha=0.7,
                                edgecolor="black",
                                color="skyblue",
                            )
                            ax.set_xscale("log")
                        else:
                            # For others, use regular scale
                            ax.hist(
                                data,
                                bins=50,
                                alpha=0.7,
                                edgecolor="black",
                                color="skyblue",
                            )

                        ax.set_xlabel(f"{feature_names[feature]} (ms)")
                        ax.set_ylabel("Frequency")
                        ax.set_title(f"{feature_names[feature]} Distribution")

                        # Calculate statistics on the displayed data
                        mean_val = data.mean()
                        median_val = data.median()
                        std_val = data.std()

                        # Add statistics box
                        stats_text = f"Mean: {mean_val:.1f}ms\n"
                        stats_text += f"Median: {median_val:.1f}ms\n"
                        stats_text += f"Std: {std_val:.1f}ms\n"
                        if feature in ["IL", "RL"]:
                            stats_text += f"Negative: {(data < 0).sum():,}"
                        if not include_outliers:
                            outlier_count = len(data_all) - len(data)
                            stats_text += f"\nOutliers removed: {outlier_count}"

                        ax.text(
                            0.95,
                            0.95,
                            stats_text,
                            transform=ax.transAxes,
                            verticalalignment="top",
                            horizontalalignment="right",
                            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                            fontsize=9,
                        )

                        # Add statistical lines
                        # Add median line
                        ax.axvline(
                            median_val,
                            color="red",
                            linestyle="--",
                            label=f"Median: {median_val:.1f}ms",
                            alpha=0.8,
                            linewidth=2,
                        )

                        # Add mean line
                        ax.axvline(
                            mean_val,
                            color="green",
                            linestyle="-",
                            label=f"Mean: {mean_val:.1f}ms",
                            alpha=0.8,
                            linewidth=2,
                        )

                        # Add standard deviation lines only if they're within reasonable bounds
                        if mean_val - std_val > data.min() * 0.8:
                            ax.axvline(
                                mean_val - std_val,
                                color="green",
                                linestyle=":",
                                label="Mean - SD",
                                alpha=0.6,
                                linewidth=1.5,
                            )
                        if mean_val + std_val < data.max() * 1.2:
                            ax.axvline(
                                mean_val + std_val,
                                color="green",
                                linestyle=":",
                                label="Mean + SD",
                                alpha=0.6,
                                linewidth=1.5,
                            )

                        # Position legend based on data distribution
                        if feature == "HL" and mean_val > median_val * 2:
                            # Skewed distribution, put legend on left
                            ax.legend(loc="upper left", fontsize=8)
                        else:
                            # Use best location
                            ax.legend(loc="best", fontsize=8)

                        ax.grid(True, alpha=0.3)
                    else:
                        ax.text(
                            0.5,
                            0.5,
                            "No data available",
                            transform=ax.transAxes,
                            ha="center",
                            va="center",
                        )
                        ax.set_title(f"{feature_names[feature]} Distribution")

                # Add box plots for better quartile visualization (bottom row)
                ax5 = plt.subplot(3, 2, 5)
                ax6 = plt.subplot(3, 2, 6)

                # Prepare data for box plots
                box_data = []
                box_labels = []
                for feature in features:
                    if feature in valid_df.columns:
                        data_all = valid_df[feature].dropna()
                        if len(data_all) > 0:
                            if include_outliers:
                                data = data_all
                            else:
                                # Remove extreme outliers for visualization
                                q1 = data_all.quantile(0.25)
                                q3 = data_all.quantile(0.75)
                                iqr = q3 - q1
                                lower_bound = q1 - 3 * iqr
                                upper_bound = q3 + 3 * iqr

                                if feature in ["IL", "RL"]:
                                    data = data_all[
                                        (data_all >= lower_bound)
                                        & (data_all <= upper_bound)
                                    ]
                                else:
                                    data = data_all[
                                        (data_all >= 0) & (data_all <= upper_bound)
                                    ]

                            box_data.append(data)
                            box_labels.append(feature)

                if box_data:
                    # Box plot for all features
                    ax5.boxplot(
                        box_data, labels=box_labels, showfliers=not include_outliers
                    )
                    ax5.set_ylabel("Time (ms)")
                    ax5.set_title("Timing Feature Distributions (Box Plot)")
                    ax5.grid(True, alpha=0.3)

                    # Violin plot for better distribution shape visualization
                    parts = ax6.violinplot(box_data, showmeans=True, showmedians=True)
                    ax6.set_xticks(range(1, len(box_labels) + 1))
                    ax6.set_xticklabels(box_labels)
                    ax6.set_ylabel("Time (ms)")
                    ax6.set_title("Timing Feature Distributions (Violin Plot)")
                    ax6.grid(True, alpha=0.3)

                    # Color the violin plots
                    for pc in parts["bodies"]:
                        pc.set_facecolor("lightblue")
                        pc.set_alpha(0.7)

                plt.tight_layout()

                # Save the figure
                suffix = "_all_data" if include_outliers else "_no_outliers"
                filename = f"timing_distributions{suffix}.png"
                filepath = self.figures_dir / filename
                plt.savefig(filepath, dpi=300, bbox_inches="tight")
                plt.close()
                filenames.append(filename)

        else:
            # Fallback to using summary statistics only
            for idx, feature in enumerate(features):
                ax = plt.subplot(2, 2, idx + 1)

                if feature in timing_stats and timing_stats[feature]["count"] > 0:
                    stats = timing_stats[feature]

                    # Create a better visualization using the statistics
                    # Show the quartiles as a box plot style visualization
                    positions = [1]
                    box_data = [
                        [
                            stats["min"],
                            stats["q25"],
                            stats["median"],
                            stats["q75"],
                            stats["max"],
                        ]
                    ]

                    bp = ax.boxplot(
                        box_data,
                        positions=positions,
                        widths=0.6,
                        patch_artist=True,
                        showfliers=False,
                    )

                    # Color the box
                    for patch in bp["boxes"]:
                        patch.set_facecolor("lightblue")
                        patch.set_alpha(0.7)

                    # Add mean as a point
                    ax.scatter(
                        [1],
                        [stats["mean"]],
                        color="red",
                        s=100,
                        zorder=3,
                        label=f'Mean: {stats["mean"]:.1f}ms',
                    )

                    ax.set_xlim(0.5, 1.5)
                    ax.set_xticks([1])
                    ax.set_xticklabels([feature])
                    ax.set_ylabel("Time (ms)")
                    ax.set_title(f"{feature_names[feature]} Distribution")

                    # Add statistics text
                    stats_text = f"Count: {stats['count']:,}\n"
                    stats_text += f"Std: {stats['std']:.1f}ms"
                    if stats.get("negative_count", 0) > 0:
                        stats_text += f"\nNegative: {stats['negative_count']:,}"

                    ax.text(
                        0.95,
                        0.95,
                        stats_text,
                        transform=ax.transAxes,
                        verticalalignment="top",
                        horizontalalignment="right",
                        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
                    )

                    ax.legend()
                    ax.grid(True, alpha=0.3, axis="y")
                else:
                    ax.text(
                        0.5,
                        0.5,
                        "No data available",
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                    )
                    ax.set_title(f"{feature_names[feature]} Distribution")

                plt.tight_layout()
                filename = "timing_distributions.png"
                filepath = self.figures_dir / filename
                plt.savefig(filepath, dpi=300, bbox_inches="tight")
                plt.close()
                filenames.append(filename)

        # Return the list of filenames (will be multiple if we have actual data)
        return filenames if filenames else ["timing_distributions.png"]

    def create_user_quality_chart(self, user_stats: pd.DataFrame) -> str:
        """Create scalable visualization of user data quality"""
        num_users = len(user_stats)

        # Sort by validity rate
        user_stats = user_stats.sort_values("validity_rate", ascending=False)

        if num_users <= 20:
            # For small number of users, use bar charts
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Validity rates
            x = range(len(user_stats))
            ax1.bar(x, user_stats["validity_rate"])
            ax1.set_xlabel("User Index")
            ax1.set_ylabel("Validity Rate (%)")
            ax1.set_title("Data Validity Rate by User")
            ax1.grid(True, alpha=0.3)

            # Total keypairs
            ax2.bar(x, user_stats["total_keypairs"])
            ax2.set_xlabel("User Index")
            ax2.set_ylabel("Total Keypairs")
            ax2.set_title("Total Keypairs by User")
            ax2.grid(True, alpha=0.3)

        else:
            # For large number of users, use different visualizations
            fig = plt.figure(figsize=(15, 10))

            # 1. Validity rate distribution (histogram)
            ax1 = plt.subplot(2, 2, 1)
            ax1.hist(user_stats["validity_rate"], bins=20, edgecolor="black", alpha=0.7)
            ax1.set_xlabel("Validity Rate (%)")
            ax1.set_ylabel("Number of Users")
            ax1.set_title("Distribution of Validity Rates")
            ax1.grid(True, alpha=0.3)

            # Add statistics
            mean_validity = user_stats["validity_rate"].mean()
            median_validity = user_stats["validity_rate"].median()
            ax1.axvline(
                mean_validity,
                color="red",
                linestyle="--",
                label=f"Mean: {mean_validity:.1f}%",
            )
            ax1.axvline(
                median_validity,
                color="green",
                linestyle="--",
                label=f"Median: {median_validity:.1f}%",
            )
            ax1.legend()

            # 2. Keypair count distribution (log scale)
            ax2 = plt.subplot(2, 2, 2)
            ax2.hist(
                user_stats["total_keypairs"], bins=30, edgecolor="black", alpha=0.7
            )
            ax2.set_xlabel("Total Keypairs")
            ax2.set_ylabel("Number of Users")
            ax2.set_title("Distribution of Keypair Counts")
            ax2.set_yscale("log")
            ax2.grid(True, alpha=0.3)

            # 3. Scatter plot: validity rate vs keypair count
            ax3 = plt.subplot(2, 2, 3)
            ax3.scatter(
                user_stats["total_keypairs"], user_stats["validity_rate"], alpha=0.6
            )
            ax3.set_xlabel("Total Keypairs")
            ax3.set_ylabel("Validity Rate (%)")
            ax3.set_title("Validity Rate vs. Data Volume")
            ax3.grid(True, alpha=0.3)

            # Color points by validity rate
            colors = [
                "red" if v < 80 else "yellow" if v < 95 else "green"
                for v in user_stats["validity_rate"]
            ]
            ax3.scatter(
                user_stats["total_keypairs"],
                user_stats["validity_rate"],
                c=colors,
                alpha=0.6,
            )

            # 4. Top/Bottom performers
            ax4 = plt.subplot(2, 2, 4)
            top_10 = user_stats.head(10)["validity_rate"]
            bottom_10 = user_stats.tail(10)["validity_rate"]

            positions = list(range(10))
            width = 0.35

            ax4.barh(
                [p - width / 2 for p in positions],
                top_10,
                width,
                label="Top 10",
                color="green",
                alpha=0.7,
            )
            ax4.barh(
                [p + width / 2 for p in positions],
                bottom_10,
                width,
                label="Bottom 10",
                color="red",
                alpha=0.7,
            )

            ax4.set_yticks(positions)
            ax4.set_yticklabels([f"Rank {i+1}" for i in range(10)])
            ax4.set_xlabel("Validity Rate (%)")
            ax4.set_title("Top 10 vs Bottom 10 Users")
            ax4.legend()
            ax4.grid(True, alpha=0.3, axis="x")

        plt.tight_layout()
        filename = "user_data_quality.png"
        filepath = self.figures_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()

        return filename

    def generate_html_report(self, analysis_results: Dict[str, Any]) -> str:
        """Generate comprehensive HTML report"""
        template_str = """
<!DOCTYPE html>
<html>
<head>
    <title>Keystroke Data Analysis Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        h1, h2, h3 { color: #333; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #4CAF50; color: white; position: sticky; top: 0; z-index: 10; }
        tr:nth-child(even) { background-color: #f2f2f2; }
        .metric { background-color: #f0f0f0; padding: 10px; margin: 10px 0; border-radius: 5px; }
        .warning { color: #ff6b6b; }
        .good { color: #51cf66; }
        img { max-width: 100%; height: auto; margin: 20px 0; }

        /* Scrollable table container */
        .table-container {
            max-height: 500px;
            overflow-y: auto;
            border: 1px solid #ddd;
            margin: 20px 0;
        }

        /* Table of Contents */
        .toc {
            background-color: #f9f9f9;
            border: 1px solid #ddd;
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
        }
        .toc h2 {
            margin-top: 0;
        }
        .toc ul {
            list-style-type: none;
            padding-left: 0;
        }
        .toc ul ul {
            padding-left: 20px;
        }
        .toc a {
            text-decoration: none;
            color: #4CAF50;
        }
        .toc a:hover {
            text-decoration: underline;
        }

        /* Back to top link */
        .back-to-top {
            float: right;
            font-size: 0.9em;
        }

        /* Code blocks */
        pre {
            background-color: #f5f5f5;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 10px;
            overflow-x: auto;
        }

        code {
            font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <h1>Keystroke Data Analysis Report</h1>
    <p><strong>Generated:</strong> {{ timestamp }}</p>
    <p><strong>Version ID:</strong> {{ version_id }}</p>

    <!-- Data Sources -->
    <div class="metric">
        <h3>Data Sources</h3>
        <p>All data used in this report can be found at the following locations:</p>
        <ul>
            <li><strong>Keypairs Data:</strong> <code>{{ data_paths.keypairs }}</code></li>
            <li><strong>User Metadata:</strong> <code>{{ data_paths.metadata }}</code></li>
            <li><strong>Raw Keystroke Data:</strong> <code>{{ data_paths.raw_data }}</code></li>
            <li><strong>Analysis Results (JSON):</strong> <code>{{ data_paths.analysis_results }}</code></li>
        </ul>
    </div>

    <!-- Table of Contents -->
    <div class="toc">
        <h2>Table of Contents</h2>
        <ul>
            <li><a href="#feature-definitions">1. Feature Definitions</a></li>
            <li><a href="#executive-summary">2. Executive Summary</a></li>
            {% if quality_issues %}
            <li><a href="#data-quality-issues">3. Data Quality Issues</a></li>
            {% endif %}
            <li><a href="#timing-features">4. Timing Feature Analysis</a></li>
            <li><a href="#user-performance">5. User Performance</a></li>
            <li><a href="#visualizations">6. Visualizations</a></li>
            {% if extreme_hl_analysis and extreme_hl_analysis.total_extreme_count > 0 %}
            <li><a href="#extreme-hl">7. Extreme Hold Latency Analysis</a></li>
            {% endif %}
            {% if negative_analysis %}
            <li><a href="#negative-analysis">8. Negative Value Analysis</a>
                <ul>
                    <li><a href="#expected-combinations">8.1 Expected Combinations</a></li>
                    <li><a href="#il-summary">8.2 IL Negative Values</a></li>
                    <li><a href="#rl-summary">8.3 RL Negative Values</a></li>
                    <li><a href="#user-breakdown">8.4 User Summary</a></li>
                </ul>
            </li>
            {% endif %}
            <li><a href="#typing-patterns">9. Typing Pattern Analysis</a></li>
            <li><a href="#quality-assessment">10. Data Quality Assessment</a></li>
            <li><a href="#recommendations">11. Recommendations</a></li>
            <li><a href="#reproducibility">12. Reproducing This Analysis</a></li>
        </ul>
    </div>

    <h2 id="feature-definitions">Feature Definitions</h2>
    <a href="#" class="back-to-top">↑ Back to top</a>
    <img src="{{ typenet_features_src }}" alt="TypeNet Feature Definitions" style="max-width: 800px;">
    <p><em>Figure: Visual representation of the four timing features extracted from keystroke data.</em></p>
    <p><small>Source: Acien, A., Morales, A., Monaco, J. V., Vera-Rodríguez, R., & Fiérrez, J. (2021, January 14). TypeNet: Deep Learning Keystroke Biometrics. arXiv. <a href="https://arxiv.org/abs/2101.05570" target="_blank">https://arxiv.org/abs/2101.05570</a></small></p>

    <h2 id="executive-summary">Executive Summary</h2>
    <a href="#" class="back-to-top">↑ Back to top</a>
    <div class="metric">
        <h3>Dataset Overview</h3>
        <ul>
            <li><strong>Total Keypairs:</strong> {{ summary.total_keypairs|number }}</li>
            <li><strong>Valid Keypairs:</strong> {{ summary.valid_keypairs|number }} ({{ summary.validity_rate|round(1) }}%)</li>
            <li><strong>Invalid Keypairs:</strong> {{ summary.invalid_keypairs|number }}</li>
            <li><strong>Unique Users:</strong> {{ summary.unique_users }}</li>
            <li><strong>Devices:</strong> {{ summary.device_types|join(', ') }}</li>
        </ul>
    </div>

    {% if quality_issues %}
    <h2 id="data-quality-issues">Data Quality Issues</h2>
    <a href="#" class="back-to-top">↑ Back to top</a>
    <div class="metric">
        <h3>Issue Summary</h3>
        <ul>
            <li><strong>Total Issues:</strong> {{ quality_issues.total_issues }}</li>
            {% for issue_type, count in quality_issues.issue_counts.items() %}
            <li><strong>{{ issue_type|replace('_', ' ')|title }}:</strong> {{ count }}</li>
            {% endfor %}
        </ul>

        {% if quality_issues.unreleased_keys %}
        <h3 class="warning">Unreleased Keys</h3>
        <p>{{ quality_issues.unreleased_keys|length }} sessions have unreleased keys</p>
        {% endif %}
    </div>
    {% endif %}

    <h2 id="timing-features">Timing Feature Analysis</h2>
    <a href="#" class="back-to-top">↑ Back to top</a>
    <p><small>Data source: <code>{{ data_paths.keypairs }}</code></small></p>
    {% if timing_stats %}
    <table>
        <tr>
            <th>Feature</th>
            <th>Count</th>
            <th>Mean (ms)</th>
            <th>Std (ms)</th>
            <th>Min (ms)</th>
            <th>Max (ms)</th>
            <th>Median (ms)</th>
            <th>Negative Values</th>
        </tr>
        {% for feature, stats in timing_stats.items() %}
        {% if stats.count > 0 %}
        <tr>
            <td>{{ feature }}</td>
            <td>{{ stats.count|number }}</td>
            <td>{{ stats.mean|round(1) }}</td>
            <td>{{ stats.std|round(1) }}</td>
            <td>{{ stats.min|round(1) }}</td>
            <td>{{ stats.max|round(1) }}</td>
            <td>{{ stats.median|round(1) }}</td>
            <td class="{% if stats.negative_count > 0 %}warning{% else %}good{% endif %}">
                {{ stats.negative_count }}
            </td>
        </tr>
        {% endif %}
        {% endfor %}
    </table>
    {% endif %}

    <h2 id="user-performance">User Performance</h2>
    <a href="#" class="back-to-top">↑ Back to top</a>
    <p><small>Data source: <code>{{ data_paths.keypairs }}</code></small></p>
    <img src="{{ user_data_quality_src }}" alt="User Data Quality">

    <h3>All Users by Validity Rate</h3>
    <div class="table-container">
        <table>
            <tr>
                <th>User ID</th>
                <th>Total Keypairs</th>
                <th>Valid Keypairs</th>
                <th>Validity Rate (%)</th>
                {% if 'outlier_rate' in user_stats.columns %}
                <th>Outlier Rate (%)</th>
                {% endif %}
            </tr>
            {% for _, user in all_users.iterrows() %}
            <tr>
                <td>{{ user.user_id }}</td>
                <td>{{ user.total_keypairs }}</td>
                <td>{{ user.valid_keypairs }}</td>
                <td>{{ user.validity_rate|round(1) }}</td>
                {% if 'outlier_rate' in user %}
                <td>{{ user.outlier_rate|round(1) }}</td>
                {% endif %}
            </tr>
            {% endfor %}
        </table>
    </div>

    <h2 id="visualizations">Visualizations</h2>
    <a href="#" class="back-to-top">↑ Back to top</a>
    <p><small>Data source: <code>{{ data_paths.keypairs }}</code></small></p>
    <h3>Timing Feature Distributions - All Valid Data</h3>
    <img src="{{ timing_distributions_all_src }}" alt="Timing Distributions - All Data">

    <h3>Timing Feature Distributions - Outliers Removed</h3>
    <img src="{{ timing_distributions_outliers_src }}" alt="Timing Distributions - No Outliers">

    {% if extreme_hl_analysis and extreme_hl_analysis.total_extreme_count > 0 %}
    <h2 id="extreme-hl">Extreme Hold Latency Analysis</h2>
    <a href="#" class="back-to-top">↑ Back to top</a>
    <div class="metric">
        <h3>Summary</h3>
        <p>Keys held for more than {{ extreme_hl_analysis.threshold_ms|number }} ms ({{ (extreme_hl_analysis.threshold_ms / 1000)|round(1) }} seconds)</p>
        <ul>
            <li><strong>Total extreme cases:</strong> {{ extreme_hl_analysis.total_extreme_count }}</li>
            <li><strong>Percentage of valid keypairs:</strong> {{ extreme_hl_analysis.percentage_of_valid|round(2) }}%</li>
            <li><strong>Users affected:</strong> {{ extreme_hl_analysis.user_stats.users_with_extreme_hl }} out of {{ extreme_hl_analysis.user_stats.total_users }} ({{ extreme_hl_analysis.user_stats.percentage_users_affected|round(1) }}%)</li>
        </ul>

        {% if extreme_hl_analysis.extreme_stats %}
        <h3>Extreme Hold Latency Statistics</h3>
        <ul>
            <li><strong>Maximum:</strong> {{ extreme_hl_analysis.extreme_stats.max_ms|number }} ms ({{ extreme_hl_analysis.extreme_stats.max_seconds|round(1) }} seconds)</li>
            <li><strong>Mean:</strong> {{ extreme_hl_analysis.extreme_stats.mean_ms|number }} ms</li>
            <li><strong>Median:</strong> {{ extreme_hl_analysis.extreme_stats.median_ms|number }} ms</li>
            <li><strong>Minimum (of extremes):</strong> {{ extreme_hl_analysis.extreme_stats.min_ms|number }} ms</li>
        </ul>
        {% endif %}

        {% if extreme_hl_analysis.key_distribution %}
        <h3>Most Common Keys with Extreme Hold Times</h3>
        <table>
            <tr>
                <th>Key</th>
                <th>Count</th>
            </tr>
            {% for key, count in extreme_hl_analysis.key_distribution.items() %}
            <tr>
                <td>{{ key }}</td>
                <td>{{ count }}</td>
            </tr>
            {% endfor %}
        </table>
        {% endif %}

        {% if extreme_hl_analysis.extreme_cases %}
        <h3>Top Extreme Cases</h3>
        <div class="table-container">
            <table>
                <tr>
                    <th>User ID</th>
                    <th>Key</th>
                    <th>Hold Time (seconds)</th>
                    <th>Next Key</th>
                    <th>Device</th>
                </tr>
                {% for case in extreme_hl_analysis.extreme_cases[:20] %}
                <tr class="warning">
                    <td>{{ case.user_id }}</td>
                    <td>{{ case.key }}</td>
                    <td>{{ case.hl_seconds|round(1) }}</td>
                    <td>{{ case.next_key }}</td>
                    <td>{{ case.device_type }}</td>
                </tr>
                {% endfor %}
            </table>
        </div>
        {% endif %}

        {% if extreme_hl_analysis.users_most_affected %}
        <h3>Users Most Affected</h3>
        <table>
            <tr>
                <th>User ID</th>
                <th>Extreme HL Count</th>
            </tr>
            {% for user in extreme_hl_analysis.users_most_affected %}
            <tr>
                <td>{{ user.user_id }}</td>
                <td>{{ user.count }}</td>
            </tr>
            {% endfor %}
        </table>
        {% endif %}
    </div>
    {% endif %}

    {% if negative_analysis %}
    <h2 id="negative-analysis">Negative Value Analysis</h2>
    <a href="#" class="back-to-top">↑ Back to top</a>
    <p><small>Data sources: <code>{{ data_paths.keypairs }}</code>, <code>{{ data_paths.metadata }}</code></small></p>
    <div class="metric">
        <h3>Summary</h3>
        <ul>
            <li><strong>Total users analyzed:</strong> {{ negative_analysis.summary.total_users }}</li>
            <li><strong>Users with negative IL values:</strong> {{ negative_analysis.summary.users_with_negative_il }} ({{ (negative_analysis.summary.users_with_negative_il / negative_analysis.summary.total_users * 100)|round(1) }}%)</li>
            <li><strong>Users with negative RL values:</strong> {{ negative_analysis.summary.users_with_negative_rl }} ({{ (negative_analysis.summary.users_with_negative_rl / negative_analysis.summary.total_users * 100)|round(1) }}%)</li>
        </ul>

        <h3 id="expected-combinations">Expected Negative Value Combinations</h3>
        <p>Negative IL and RL values are expected when these modifier keys are held while pressing other keys:</p>
        <p><code>{{ negative_analysis.summary.modifier_keys|join(', ') }}</code></p>

        <h3 id="il-summary">Inter-key Latency (IL) Negative Values</h3>
        <ul>
            <li><strong>Total negative IL values:</strong> {{ negative_analysis.negative_IL.total_count }}</li>
            <li><strong>Expected (modifier keys):</strong> {{ negative_analysis.negative_IL.expected_count }} ({{ (negative_analysis.negative_IL.expected_count / negative_analysis.negative_IL.total_count * 100)|round(1) }}%)</li>
            <li><strong>Unexpected (fast typing):</strong> {{ negative_analysis.negative_IL.unexpected_count }} ({{ (negative_analysis.negative_IL.unexpected_count / negative_analysis.negative_IL.total_count * 100)|round(1) }}%)</li>
        </ul>

        {% if negative_analysis.negative_IL.top_patterns %}
        <h4>Top Negative IL Patterns</h4>
        <table>
            <tr>
                <th>Pattern</th>
                <th>Count</th>
                <th>Type</th>
            </tr>
            {% for pattern in negative_analysis.negative_IL.top_patterns %}
            <tr>
                <td>{{ pattern.pattern }}</td>
                <td>{{ pattern.count }}</td>
                <td class="{% if pattern.type == 'expected' %}good{% else %}{% endif %}">{{ pattern.type|title }}</td>
            </tr>
            {% endfor %}
        </table>
        {% endif %}

        <h3 id="rl-summary">Release Latency (RL) Negative Values</h3>
        <ul>
            <li><strong>Total negative RL values:</strong> {{ negative_analysis.negative_RL.total_count }}</li>
            <li><strong>With negative IL:</strong> {{ negative_analysis.negative_RL.with_negative_IL }} ({{ (negative_analysis.negative_RL.with_negative_IL / negative_analysis.negative_RL.total_count * 100)|round(1) }}%)</li>
            <li><strong>Standalone negative RL:</strong> {{ negative_analysis.negative_RL.standalone }}</li>
        </ul>

        {% if negative_analysis.negative_RL.top_patterns %}
        <h4>Top Negative RL Patterns</h4>
        <table>
            <tr>
                <th>Pattern</th>
                <th>Count</th>
            </tr>
            {% for pattern in negative_analysis.negative_RL.top_patterns %}
            <tr>
                <td>{{ pattern.pattern }}</td>
                <td>{{ pattern.count }}</td>
            </tr>
            {% endfor %}
        </table>
        {% endif %}

        <h3 id="user-breakdown">User Summary</h3>
        <p>Click on column headers to sort. Full user IDs are provided for easy searching.</p>
        <div class="table-container">
            <table id="user-negative-summary" style="cursor: pointer;">
                <thead>
                    <tr>
                        <th onclick="sortTable(0, 'user-negative-summary')">User ID</th>
                        <th onclick="sortTable(1, 'user-negative-summary', 'numeric')">Total Keypairs</th>
                        <th onclick="sortTable(2, 'user-negative-summary', 'numeric')">Neg IL Total</th>
                        <th onclick="sortTable(3, 'user-negative-summary', 'numeric')">Neg IL Expected</th>
                        <th onclick="sortTable(4, 'user-negative-summary', 'numeric')">Neg IL Unexpected</th>
                        <th onclick="sortTable(5, 'user-negative-summary', 'numeric')">Neg IL %</th>
                        <th onclick="sortTable(6, 'user-negative-summary', 'numeric')">Neg RL Total</th>
                        <th onclick="sortTable(7, 'user-negative-summary', 'numeric')">Neg RL %</th>
                        <th>Top IL Patterns</th>
                        <th>Timestamp</th>
                    </tr>
                </thead>
                <tbody>
                    {% for user in negative_analysis.user_summary %}
                    <tr>
                        <td style="font-family: monospace;">{{ user.user_id }}</td>
                        <td>{{ user.total_keypairs }}</td>
                        <td>{{ user.negative_il_total }}</td>
                        <td>{{ user.negative_il_expected }}</td>
                        <td>{{ user.negative_il_unexpected }}</td>
                        <td>{{ user.negative_il_percent }}%</td>
                        <td>{{ user.negative_rl_total }}</td>
                        <td>{{ user.negative_rl_percent }}%</td>
                        <td style="font-size: 0.9em;">{{ user.top_il_patterns }}</td>
                        <td style="font-size: 0.9em;">{{ user.timestamp }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
    </div>

    <script>
    function sortTable(columnIndex, tableId, type = 'text') {
        var table = document.getElementById(tableId);
        var tbody = table.getElementsByTagName('tbody')[0];
        var rows = Array.from(tbody.getElementsByTagName('tr'));

        var ascending = table.getAttribute('data-sort-column') == columnIndex &&
                       table.getAttribute('data-sort-order') == 'asc';

        rows.sort(function(a, b) {
            var aValue = a.cells[columnIndex].innerText.trim();
            var bValue = b.cells[columnIndex].innerText.trim();

            if (type === 'numeric') {
                aValue = parseFloat(aValue.replace('%', '')) || 0;
                bValue = parseFloat(bValue.replace('%', '')) || 0;
                return ascending ? bValue - aValue : aValue - bValue;
            } else {
                return ascending ?
                    bValue.localeCompare(aValue) :
                    aValue.localeCompare(bValue);
            }
        });

        while (tbody.firstChild) {
            tbody.removeChild(tbody.firstChild);
        }

        rows.forEach(function(row) {
            tbody.appendChild(row);
        });

        table.setAttribute('data-sort-column', columnIndex);
        table.setAttribute('data-sort-order', ascending ? 'desc' : 'asc');
    }
    </script>
    {% endif %}

    <h2 id="typing-patterns">Typing Pattern Analysis</h2>
    <a href="#" class="back-to-top">↑ Back to top</a>
    <div class="metric">
        <h3>Understanding Negative Values in Typing Data</h3>
        <p>Our analysis reveals that negative timing values are a <strong>normal and expected</strong> part of natural typing behavior, not data quality issues:</p>

        <h4>What Negative Values Mean:</h4>
        <ul>
            <li><strong>Negative IL (Inter-key Latency):</strong> The next key is pressed before the previous key is released. This is extremely common in fast typing and represents overlapping keystrokes.</li>
            <li><strong>Negative RL (Release Latency):</strong> Keys are released in a different order than they were pressed. This happens when users quickly type combinations of keys.</li>
        </ul>

        <h4>Key Findings:</h4>
        <ul>
            <li><strong>{{ (negative_analysis.negative_IL.unexpected_count / negative_analysis.negative_IL.total_count * 100)|round(1) }}%</strong> of negative IL values are from regular typing (not modifier keys)</li>
            <li><strong>{{ (negative_analysis.negative_RL.with_negative_IL / negative_analysis.negative_RL.total_count * 100)|round(1) }}%</strong> of negative RL values coincide with negative IL values, indicating they represent the same fast-typing phenomenon</li>
            <li>Common patterns include letter combinations that are typical of rapid typing sequences</li>
            <li>These patterns are consistent across users and represent individual typing styles and speeds</li>
        </ul>

        <h4>Implications for Analysis:</h4>
        <p>Negative values should <strong>not</strong> be filtered out as errors. They contain valuable information about typing dynamics and are essential for accurate user authentication and typing pattern analysis. The high percentage of "unexpected" negative values ({{ (negative_analysis.negative_IL.unexpected_count / negative_analysis.negative_IL.total_count * 100)|round(1) }}%) actually represents normal fast typing behavior rather than data quality issues.</p>
    </div>

    <h2 id="quality-assessment">Data Quality Assessment</h2>
    <a href="#" class="back-to-top">↑ Back to top</a>
    <div class="metric">
        <p><strong>Overall Quality Rating:</strong>
        <span class="{% if summary.validity_rate >= 90 %}good{% elif summary.validity_rate >= 70 %}{% else %}warning{% endif %}">
            {% if summary.validity_rate >= 95 %}Excellent{% elif summary.validity_rate >= 85 %}Good{% elif summary.validity_rate >= 70 %}Fair{% else %}Poor{% endif %}
        </span>
        ({{ summary.validity_rate|round(1) }}% valid)
        </p>
    </div>

    <h2 id="recommendations">Recommendations</h2>
    <a href="#" class="back-to-top">↑ Back to top</a>
    <ul>
        {% for rec in recommendations %}
        <li>{{ rec }}</li>
        {% endfor %}
    </ul>

    <h2 id="reproducibility">Reproducing This Analysis</h2>
    <a href="#" class="back-to-top">↑ Back to top</a>
    <div class="metric">
        <h3>Loading the Data</h3>
        <p>To reproduce any analysis or create additional visualizations, use the following Python code:</p>
        <pre><code>import pandas as pd
import json

# Load keypairs data
keypairs_df = pd.read_parquet('{{ data_paths.keypairs }}')

# Load user metadata
metadata_df = pd.read_csv('{{ data_paths.metadata }}')

# Load analysis results
with open('{{ data_paths.analysis_results }}', 'r') as f:
    analysis_results = json.load(f)

# Filter valid keypairs only
valid_df = keypairs_df[keypairs_df['valid']]

# Example: Create custom analysis
print(f"Total keypairs: {len(keypairs_df)}")
print(f"Valid keypairs: {len(valid_df)}")
print(f"Unique users: {keypairs_df['user_id'].nunique()}")
</code></pre>

        <h3>Generating Additional Plots</h3>
        <p>Example code for creating custom visualizations:</p>
        <pre><code>import matplotlib.pyplot as plt
import seaborn as sns

# Example: Distribution of Hold Latency by user
plt.figure(figsize=(12, 6))
for user_id in valid_df['user_id'].unique()[:5]:  # First 5 users
    user_data = valid_df[valid_df['user_id'] == user_id]['HL']
    plt.hist(user_data, bins=50, alpha=0.5, label=f'User {user_id}')
plt.xlabel('Hold Latency (ms)')
plt.ylabel('Frequency')
plt.title('Hold Latency Distribution by User')
plt.legend()
plt.show()
</code></pre>
    </div>
</body>
</html>
        """

        # Create a Jinja2 environment with custom filters
        from jinja2 import Environment

        env = Environment()

        # Add custom filters
        env.filters["number"] = lambda x: f"{int(x):,}" if pd.notna(x) else "N/A"
        env.filters["round"] = lambda x, n=1: round(x, n) if pd.notna(x) else "N/A"

        # Add image sources to the context
        image_sources = {
            "typenet_features_src": self.get_image_src("figures/typenet_features.png"),
            "user_data_quality_src": self.get_image_src(
                "figures/user_data_quality.png"
            ),
            "timing_distributions_all_src": self.get_image_src(
                "figures/timing_distributions_all_data.png"
            ),
            "timing_distributions_outliers_src": self.get_image_src(
                "figures/timing_distributions_no_outliers.png"
            ),
        }

        # Merge with analysis results
        context = {**analysis_results, **image_sources}

        # Create template from string
        template = env.from_string(template_str)

        return template.render(**context)


class RunEDAStage:
    """Run exploratory data analysis and generate reports"""

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

        # Initialize analyzers
        self.quality_analyzer = DataQualityAnalyzer()
        self.feature_analyzer = FeatureAnalyzer()

    def load_raw_keystrokes(self, cleaned_data_dir: Path) -> Optional[pd.DataFrame]:
        """Load raw keystroke data for quality analysis"""
        all_data = []

        # Get device types from config
        from scripts.utils.config_manager import get_config

        config_manager = get_config()
        device_types = config_manager.get_device_types()

        for device_type in device_types:
            raw_data_dir = cleaned_data_dir / device_type / "raw_data"
            if not raw_data_dir.exists():
                continue

            for user_dir in raw_data_dir.iterdir():
                if not user_dir.is_dir():
                    continue

                user_id = user_dir.name

                # Load CSV files
                for csv_file in user_dir.glob("*.csv"):
                    # Skip non-keystroke files
                    if not self._is_keystroke_file(csv_file):
                        continue

                    try:
                        # First, check if file has headers by reading first line
                        with open(csv_file) as f:
                            first_line = f.readline().strip()

                        # If first line contains non-numeric data in the timestamp column, skip header
                        if (
                            first_line
                            and not first_line.split(",")[2]
                            .replace(".", "")
                            .replace("-", "")
                            .isdigit()
                        ):
                            df = pd.read_csv(
                                csv_file,
                                skiprows=1,
                                header=None,
                                names=["type", "key", "timestamp"],
                                dtype={"timestamp": float},
                            )
                        else:
                            df = pd.read_csv(
                                csv_file,
                                header=None,
                                names=["type", "key", "timestamp"],
                                dtype={"timestamp": float},
                            )

                        df["user_id"] = user_id
                        df["device_type"] = device_type
                        df["source_file"] = csv_file.name
                        all_data.append(df)
                    except Exception as e:
                        logger.warning(f"Could not load {csv_file}: {e}")

        return pd.concat(all_data, ignore_index=True) if all_data else None

    def _is_keystroke_file(self, filepath: Path) -> bool:
        """Check if file is a keystroke data file"""
        # Pattern: platform_video_session_user.csv
        parts = filepath.stem.split("_")
        if len(parts) != 4:
            return False
        try:
            # First 3 should be numeric
            int(parts[0])
            int(parts[1])
            int(parts[2])
            return True
        except ValueError:
            return False

    def generate_recommendations(
        self, summary: Dict, quality_issues: Dict, timing_stats: Dict
    ) -> List[str]:
        """Generate analysis recommendations"""
        recommendations = []

        # Data quality recommendations
        validity_rate = summary.get("validity_rate", 0)
        if validity_rate < 90:
            recommendations.append(
                f"**Data Quality:** Validity rate is {validity_rate:.1f}%. "
                "Consider reviewing data collection procedures."
            )

        # Issue-based recommendations
        if quality_issues:
            total_issues = quality_issues.get("total_issues", 0)
            if total_issues > summary["total_keypairs"] * 0.05:
                recommendations.append(
                    f"**High Error Rate:** {total_issues} quality issues detected. "
                    "Investigate data collection synchronization."
                )

            if quality_issues.get("unreleased_keys"):
                recommendations.append(
                    "**Unreleased Keys:** Multiple sessions have unreleased keys. "
                    "May indicate incomplete data capture."
                )

        # Timing recommendations
        for feature, stats in timing_stats.items():
            if stats.get("negative_count", 0) > 0:
                recommendations.append(
                    f"**{feature} Timing:** {stats['negative_count']} negative values detected. "
                    "Check timestamp synchronization."
                )

        if not recommendations:
            recommendations.append(
                "**Overall Assessment:** Data quality appears good. "
                "No major issues detected."
            )

        return recommendations

    def run(self) -> Path:
        """Execute the EDA stage"""
        logger.info(f"Starting EDA stage for version {self.version_id}")

        # Setup directories
        artifacts_dir = (
            Path(self.config.get("ARTIFACTS_DIR", "artifacts")) / self.version_id
        )
        output_dir = artifacts_dir / "eda_reports" / "data_quality"

        # Get input directories from previous stages
        cleaned_data_dir = artifacts_dir / "cleaned_data"
        keypairs_dir = artifacts_dir / "keypairs"
        features_dir = artifacts_dir / "statistical_features"

        # Initialize results
        analysis_results = {
            "version_id": self.version_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        # Check if required input directories exist
        if not keypairs_dir.exists():
            logger.error(f"Keypairs directory not found: {keypairs_dir}")
            raise FileNotFoundError(
                "Cannot run EDA: keypairs directory missing. Did previous stages complete successfully?"
            )

        # Load and analyze keypair data
        keypair_file = keypairs_dir / "keypairs.parquet"
        if not keypair_file.exists():
            keypair_file = keypairs_dir / "keypairs.csv"

        if not keypair_file.exists():
            logger.error(f"No keypair data found in {keypairs_dir}")
            raise FileNotFoundError(
                "Cannot run EDA: No keypair data file found. Did the extract_keypairs stage complete successfully?"
            )

        logger.info(f"Loading keypair data from {keypair_file}")
        try:
            if keypair_file.suffix == ".parquet":
                keypairs_df = pd.read_parquet(keypair_file)
            else:
                keypairs_df = pd.read_csv(keypair_file)
        except Exception as e:
            logger.error(f"Failed to load keypair data: {e}")
            raise RuntimeError(f"Cannot run EDA: Failed to load keypair data - {e}")

        if keypairs_df.empty:
            logger.error("Keypair data is empty")
            raise ValueError("Cannot run EDA: Keypair data file is empty")

        # Verify required columns exist
        required_columns = ["valid", "user_id", "device_type"]
        missing_columns = [
            col for col in required_columns if col not in keypairs_df.columns
        ]
        if missing_columns:
            logger.error(f"Missing required columns in keypair data: {missing_columns}")
            raise ValueError(
                f"Cannot run EDA: Missing required columns - {missing_columns}"
            )

        if keypair_file.exists():
            logger.info("Analyzing keypair data...")

            # Basic summary
            summary = {
                "total_keypairs": len(keypairs_df),
                "valid_keypairs": keypairs_df["valid"].sum(),
                "invalid_keypairs": (~keypairs_df["valid"]).sum(),
                "validity_rate": keypairs_df["valid"].mean() * 100,
                "unique_users": keypairs_df["user_id"].nunique(),
                "device_types": keypairs_df["device_type"].unique().tolist(),
            }
            analysis_results["summary"] = summary

            # Timing analysis
            timing_stats = self.feature_analyzer.analyze_timing_features(keypairs_df)
            analysis_results["timing_stats"] = timing_stats

            # Load user metadata for negative value analysis
            user_metadata = self.feature_analyzer.load_user_metadata(artifacts_dir)

            # Negative value analysis
            negative_analysis = self.feature_analyzer.analyze_negative_values(
                keypairs_df, user_metadata
            )
            analysis_results["negative_analysis"] = negative_analysis

            # Extreme Hold Latency analysis
            extreme_hl_analysis = self.feature_analyzer.analyze_extreme_hold_latency(
                keypairs_df, threshold_ms=5000
            )
            analysis_results["extreme_hl_analysis"] = extreme_hl_analysis

            # User performance
            user_stats = self.feature_analyzer.analyze_user_performance(keypairs_df)
            analysis_results["user_stats"] = user_stats
            analysis_results["all_users"] = user_stats.sort_values(
                "validity_rate", ascending=False
            )

        # Analyze raw keystroke quality (optional)
        if cleaned_data_dir.exists():
            logger.info("Analyzing raw keystroke quality...")
            raw_df = self.load_raw_keystrokes(cleaned_data_dir)

            if raw_df is not None:
                quality_results = self.quality_analyzer.analyze_raw_keystrokes(raw_df)
                analysis_results["quality_issues"] = {
                    "total_issues": len(quality_results["issues"]),
                    "issue_counts": dict(quality_results["issue_counts"]),
                    "unreleased_keys": quality_results["unreleased_keys"],
                }
            else:
                analysis_results["quality_issues"] = None

        # Generate recommendations
        recommendations = self.generate_recommendations(
            analysis_results.get("summary", {}),
            analysis_results.get("quality_issues", {}),
            analysis_results.get("timing_stats", {}),
        )
        analysis_results["recommendations"] = recommendations

        # Add data paths for reproducibility
        analysis_results["data_paths"] = {
            "keypairs": str(keypair_file)
            if "keypair_file" in locals()
            else str(keypairs_dir / "keypairs.parquet"),
            "metadata": str(cleaned_data_dir / "desktop" / "metadata" / "metadata.csv"),
            "raw_data": str(cleaned_data_dir),
            "analysis_results": str(output_dir / "analysis_results.json"),
        }

        # Generate report
        if not self.dry_run:
            output_dir.mkdir(parents=True, exist_ok=True)

            # Initialize report generator
            # Embed images for portability (can be changed via config if needed)
            embed_images = self.config.get("EDA_EMBED_IMAGES", True)
            report_gen = ReportGenerator(output_dir, embed_images=embed_images)

            # Create visualizations
            if "timing_stats" in analysis_results:
                # Pass keypairs_df for better visualizations
                report_gen.create_timing_distributions(
                    analysis_results["timing_stats"],
                    keypairs_df if "keypairs_df" in locals() else None,
                )

            if "user_stats" in analysis_results:
                report_gen.create_user_quality_chart(analysis_results["user_stats"])

            # Copy typenet_features.png if it exists
            typenet_png_src = (
                Path(__file__).parent.parent.parent
                / "documentation"
                / "typenet_features.png"
            )
            if typenet_png_src.exists():
                typenet_png_dst = output_dir / "figures" / "typenet_features.png"
                shutil.copy2(typenet_png_src, typenet_png_dst)
                logger.info(f"Copied typenet_features.png to {typenet_png_dst}")

            # Generate HTML report
            try:
                html_content = report_gen.generate_html_report(analysis_results)

                with open(output_dir / "report.html", "w") as f:
                    f.write(html_content)
            except Exception as e:
                logger.error(f"Failed to generate HTML report: {e}")
                logger.error(f"Analysis results keys: {list(analysis_results.keys())}")
                raise

            # Save analysis results as JSON
            with open(output_dir / "analysis_results.json", "w") as f:
                # Convert DataFrame to dict for JSON serialization
                results_for_json = analysis_results.copy()
                if "user_stats" in results_for_json:
                    results_for_json["user_stats"] = results_for_json[
                        "user_stats"
                    ].to_dict("records")
                if "all_users" in results_for_json:
                    results_for_json["all_users"] = results_for_json[
                        "all_users"
                    ].to_dict("records")

                json.dump(results_for_json, f, indent=2, cls=NumpyEncoder)

            # Save summary statistics
            if "summary" in analysis_results:
                with open(output_dir / "summary_stats.json", "w") as f:
                    json.dump(
                        analysis_results["summary"], f, indent=2, cls=NumpyEncoder
                    )

            logger.info(f"Reports saved to {output_dir}")

        # Update version info
        if not self.dry_run:
            self.version_manager.update_stage_info(
                self.version_id,
                "run_eda",
                {
                    "output_dir": str(output_dir),
                    "reports_generated": ["data_quality"],
                    "completed_at": datetime.now().isoformat(),
                },
            )

        return output_dir


def run(
    version_id: str,
    config: Dict[str, Any],
    dry_run: bool = False,
    local_only: bool = False,
) -> Path:
    """Entry point for the pipeline orchestrator"""
    stage = RunEDAStage(version_id, config, dry_run, local_only)
    return stage.run()


if __name__ == "__main__":
    # For testing the stage independently
    import click

    from scripts.utils.config_manager import get_config

    @click.command()
    @click.option("--version-id", help="Version ID to use")
    @click.option("--dry-run", is_flag=True, help="Preview without generating reports")
    def main(version_id, dry_run):
        """Test EDA stage independently"""
        logging.basicConfig(level=logging.INFO)

        config = get_config().config
        vm = VersionManager()

        if not version_id:
            version_id = vm.create_version_id()
            logger.info(f"Created version ID: {version_id}")

        output_dir = run(version_id, config, dry_run)
        logger.info(f"Stage complete. Output: {output_dir}")

    main()
