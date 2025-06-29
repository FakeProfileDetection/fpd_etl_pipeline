import polars as pl
import numpy as np
import os
from pathlib import Path
from collections import deque, defaultdict
from typing import List, Tuple, Dict, Optional
import socket
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class HBOS:
    def __init__(self, n_bins=10, contamination=0.1, alpha=0.1, tol=0.5):
        """
        HBOS with smoothing (alpha) and edge-tolerance (tol).
        Parameters:
        - n_bins: Base number of bins per feature (can also be 'auto' rules).
        - contamination: Proportion of expected outliers.
        - alpha: Small constant added to every bin density.
        - tol: Fraction of one bin-width to tolerate just-outside values.
        """
        self.n_bins = n_bins
        self.contamination = contamination
        self.alpha = alpha
        self.tol = tol
        self.histograms = []  # List of 1D arrays: per-feature bin densities
        self.bin_edges = []   # List of 1D arrays: per-feature edges
        self.feature_names = []  # Keys order
        
    def fit(self, data: defaultdict):
        """Build (smoothed) histograms for each feature."""
        self.feature_names = list(data.keys())
        X = np.column_stack([data[f] for f in self.feature_names])
        self.histograms.clear()
        self.bin_edges.clear()
        
        for col in X.T:
            # 1) build raw histogram
            hist, edges = np.histogram(col, bins=self.n_bins, density=True)
            # 2) smooth: add alpha everywhere
            hist = hist + self.alpha
            self.histograms.append(hist)
            self.bin_edges.append(edges)
            
    def _compute_score(self, x: np.ndarray) -> float:
        """
        Negative-log-sum of per-feature densities with alpha & tol handling.
        Higher score = more anomalous.
        """
        score = 0.0
        for i, xi in enumerate(x):
            edges = self.bin_edges[i]
            hist = self.histograms[i]
            n_bins = hist.shape[0]
            
            # compute first/last bin widths
            width_low = edges[1] - edges[0]
            width_high = edges[-1] - edges[-2]
            
            # 1) too far below range?
            if xi < edges[0]:
                if edges[0] - xi <= self.tol * width_low:
                    # snap into first bin
                    density = hist[0]
                else:
                    # true out-of-range → worst density
                    density = self.alpha
                score += -np.log(density)
                continue
                
            # 2) too far above range?
            if xi > edges[-1]:
                if xi - edges[-1] <= self.tol * width_high:
                    # snap into last bin
                    density = hist[-1]
                else:
                    density = self.alpha
                score += -np.log(density)
                continue
                
            # 3) within [min, max] → find bin index
            bin_idx = np.searchsorted(edges, xi, side="right") - 1
            bin_idx = np.clip(bin_idx, 0, n_bins - 1)
            density = hist[bin_idx]
            score += -np.log(density)
            
        return score
        
    def decision_function(self, data: defaultdict) -> np.ndarray:
        """Return log-space HBOS scores for all points."""
        X = np.column_stack([data[f] for f in self.feature_names])
        return np.array([self._compute_score(row) for row in X])
        
    def predict_outliers(self, data: defaultdict) -> np.ndarray:
        """
        Return boolean array where True indicates an outlier.
        """
        scores = self.decision_function(data)
        threshold = np.percentile(scores, 100 * (1 - self.contamination))
        return scores > threshold

class TypeNetAnalysisReport:
    """
    Comprehensive analysis and reporting for TypeNet keystroke data.
    Integrated into the extraction pipeline.
    """
    
    def __init__(self, df: pl.DataFrame, output_dir: str):
        """
        Initialize the analysis report generator.
        
        Args:
            df: The extracted TypeNet features DataFrame
            output_dir: Directory to save analysis outputs
        """
        self.df = df
        self.output_dir = output_dir
        
        # Analysis results storage
        self.stats = {}
        self.plots_dir = Path(output_dir) / "plots"
        self.plots_dir.mkdir(exist_ok=True)
        
        # Set plotting style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def generate_basic_statistics(self) -> Dict:
        """Generate basic dataset statistics."""
        stats = {}
        
        # Overall statistics
        stats['total_records'] = len(self.df)
        stats['valid_records'] = self.df.select(pl.col('valid').sum()).item()
        stats['invalid_records'] = stats['total_records'] - stats['valid_records']
        stats['outlier_records'] = self.df.select(pl.col('outlier').sum()).item()
        
        # User statistics
        stats['unique_users'] = self.df.select(pl.col('user_id').n_unique()).item()
        stats['users_list'] = sorted(self.df.select('user_id').unique().to_series().to_list())
        
        # Platform/Video/Session statistics
        stats['unique_platforms'] = self.df.select(pl.col('platform_id').n_unique()).item()
        stats['unique_videos'] = self.df.select(pl.col('video_id').n_unique()).item()
        stats['unique_sessions'] = self.df.select(pl.col('session_id').n_unique()).item()
        
        # Records per user
        records_per_user = self.df.group_by('user_id').len().sort('user_id')
        stats['avg_records_per_user'] = records_per_user.select(pl.col('len').mean()).item()
        stats['min_records_per_user'] = records_per_user.select(pl.col('len').min()).item()
        stats['max_records_per_user'] = records_per_user.select(pl.col('len').max()).item()
        
        self.stats['basic'] = stats
        return stats
    
    def analyze_timing_features(self) -> Dict:
        """Analyze timing feature distributions and statistics."""
        timing_stats = {}
        valid_df = self.df.filter(pl.col('valid'))
        
        timing_features = ['HL', 'IL', 'PL', 'RL']
        
        for feature in timing_features:
            feature_data = valid_df.select(pl.col(feature).drop_nulls().drop_nans())
            if not feature_data.is_empty():
                values = feature_data.to_series()
                
                timing_stats[feature] = {
                    'count': len(values),
                    'mean': values.mean(),
                    'std': values.std(),
                    'min': values.min(),
                    'max': values.max(),
                    'median': values.median(),
                    'q25': values.quantile(0.25),
                    'q75': values.quantile(0.75),
                    'negative_count': (values < 0).sum(),
                    'zero_count': (values == 0).sum(),
                    'outlier_threshold_low': values.quantile(0.01),
                    'outlier_threshold_high': values.quantile(0.99)
                }
        
        self.stats['timing'] = timing_stats
        return timing_stats
    
    def analyze_error_patterns(self) -> Dict:
        """Analyze error patterns and invalid data."""
        error_stats = {}
        
        # Error type distribution
        invalid_df = self.df.filter(~pl.col('valid'))
        if not invalid_df.is_empty():
            error_counts = invalid_df.select('error_description').to_series().value_counts()
            error_stats['error_types'] = {
                row[0]: row[1] for row in error_counts.iter_rows()
            }
        else:
            error_stats['error_types'] = {}
        
        # Errors by user
        error_by_user = self.df.group_by('user_id').agg([
            pl.col('valid').sum().alias('valid_count'),
            (~pl.col('valid')).sum().alias('invalid_count')
        ]).with_columns(
            (pl.col('invalid_count') / (pl.col('valid_count') + pl.col('invalid_count')) * 100).alias('error_rate')
        ).sort('error_rate', descending=True)
        
        error_stats['users_with_highest_error_rates'] = [
            {'user_id': row[0], 'error_rate': row[3], 'invalid_count': row[2]}
            for row in error_by_user.head(10).iter_rows()
        ]
        
        self.stats['errors'] = error_stats
        return error_stats
    
    def analyze_outliers(self) -> Dict:
        """Analyze outlier patterns."""
        outlier_stats = {}
        
        valid_df = self.df.filter(pl.col('valid'))
        outlier_df = valid_df.filter(pl.col('outlier'))
        
        if not outlier_df.is_empty():
            # Outlier rate
            outlier_stats['total_outliers'] = len(outlier_df)
            outlier_stats['outlier_rate'] = len(outlier_df) / len(valid_df) * 100
            
            # Outliers by user
            outliers_by_user = valid_df.group_by('user_id').agg([
                pl.col('outlier').sum().alias('outlier_count'),
                pl.col('user_id').count().alias('total_count')
            ]).with_columns(
                (pl.col('outlier_count') / pl.col('total_count') * 100).alias('outlier_rate')
            ).sort('outlier_rate', descending=True)
            
            outlier_stats['users_with_highest_outlier_rates'] = [
                {'user_id': row[0], 'outlier_rate': row[3], 'outlier_count': row[1]}
                for row in outliers_by_user.head(10).iter_rows()
            ]
        else:
            outlier_stats['total_outliers'] = 0
            outlier_stats['outlier_rate'] = 0.0
        
        self.stats['outliers'] = outlier_stats
        return outlier_stats
    
    def create_visualizations(self):
        """Create comprehensive visualizations."""
        if self.df.filter(pl.col('valid')).is_empty():
            print("No valid data available for visualizations")
            return
            
        valid_df = self.df.filter(pl.col('valid')).to_pandas()
        
        # 1. Timing feature distributions
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('TypeNet Timing Feature Distributions', fontsize=16)
        
        timing_features = ['HL', 'IL', 'PL', 'RL']
        feature_names = {
            'HL': 'Hold Latency (ms)',
            'IL': 'Inter-key Latency (ms)', 
            'PL': 'Press Latency (ms)',
            'RL': 'Release Latency (ms)'
        }
        
        for i, feature in enumerate(timing_features):
            ax = axes[i//2, i%2]
            data = valid_df[feature].dropna()
            if len(data) > 0:
                # Remove extreme outliers for better visualization
                q99 = data.quantile(0.99)
                q01 = data.quantile(0.01)
                data_filtered = data[(data >= q01) & (data <= q99)]
                
                ax.hist(data_filtered, bins=50, alpha=0.7, edgecolor='black')
                ax.set_title(f'{feature_names[feature]}')
                ax.set_xlabel('Time (ms)')
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'timing_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. User data completeness
        user_stats = self.df.group_by('user_id').agg([
            pl.col('valid').sum().alias('valid_count'),
            pl.col('user_id').count().alias('total_count'),
            pl.col('outlier').sum().alias('outlier_count')
        ]).with_columns([
            (pl.col('valid_count') / pl.col('total_count') * 100).alias('validity_rate'),
            (pl.col('outlier_count') / pl.col('valid_count') * 100).alias('outlier_rate')
        ]).sort('user_id')
        
        user_stats_pd = user_stats.to_pandas()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Validity rates by user
        ax1.bar(range(len(user_stats_pd)), user_stats_pd['validity_rate'])
        ax1.set_title('Data Validity Rate by User')
        ax1.set_xlabel('User Index')
        ax1.set_ylabel('Validity Rate (%)')
        ax1.grid(True, alpha=0.3)
        
        # Outlier rates by user
        valid_users = user_stats_pd[user_stats_pd['valid_count'] > 0]
        if not valid_users.empty:
            ax2.bar(range(len(valid_users)), valid_users['outlier_rate'])
            ax2.set_title('Outlier Rate by User (Valid Data Only)')
            ax2.set_xlabel('User Index')
            ax2.set_ylabel('Outlier Rate (%)')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.plots_dir / 'user_data_quality.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to {self.plots_dir}")
    
    def generate_markdown_report(self) -> str:
        """Generate a comprehensive markdown report."""
        
        # Run all analyses
        self.generate_basic_statistics()
        self.analyze_timing_features()
        self.analyze_error_patterns()
        self.analyze_outliers()
        
        # Generate report
        report = []
        report.append("# TypeNet Keystroke Data Analysis Report")
        report.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**Analysis Output:** {self.output_dir}")
        
        # Executive Summary
        report.append("\n## Executive Summary")
        basic = self.stats['basic']
        report.append(f"- **Total Records:** {basic['total_records']:,}")
        report.append(f"- **Valid Records:** {basic['valid_records']:,} ({basic['valid_records']/basic['total_records']*100:.1f}%)")
        report.append(f"- **Invalid Records:** {basic['invalid_records']:,} ({basic['invalid_records']/basic['total_records']*100:.1f}%)")
        report.append(f"- **Outlier Records:** {basic['outlier_records']:,}")
        report.append(f"- **Unique Users:** {basic['unique_users']}")
        report.append(f"- **Experimental Conditions:** {basic['unique_platforms']} platforms, {basic['unique_videos']} videos, {basic['unique_sessions']} sessions")
        
        # Dataset Overview
        report.append("\n## Dataset Overview")
        report.append(f"- **Users:** {basic['users_list']}")
        report.append(f"- **Records per User:** {basic['min_records_per_user']} - {basic['max_records_per_user']} (avg: {basic['avg_records_per_user']:.0f})")
        
        # Timing Features Analysis
        report.append("\n## Timing Features Analysis")
        timing = self.stats['timing']
        
        if timing:
            report.append("### Feature Statistics")
            report.append("| Feature | Count | Mean (ms) | Std (ms) | Min (ms) | Max (ms) | Median (ms) | Negative Count |")
            report.append("|---------|-------|-----------|----------|----------|----------|-------------|----------------|")
            
            for feature, stats in timing.items():
                report.append(f"| {feature} | {stats['count']:,} | {stats['mean']:.1f} | {stats['std']:.1f} | {stats['min']:.1f} | {stats['max']:.1f} | {stats['median']:.1f} | {stats['negative_count']} |")
        else:
            report.append("No valid timing features found in the dataset.")
        
        # Error Analysis
        report.append("\n## Error Analysis")
        errors = self.stats['errors']
        
        report.append("### Error Types")
        if errors['error_types']:
            for error_type, count in errors['error_types'].items():
                report.append(f"- **{error_type}:** {count:,} occurrences")
        else:
            report.append("- No errors found in the dataset")
        
        report.append("\n### Users with Highest Error Rates")
        if errors['users_with_highest_error_rates']:
            report.append("| User ID | Error Rate (%) | Invalid Count |")
            report.append("|---------|---------------|---------------|")
            for user_error in errors['users_with_highest_error_rates']:
                report.append(f"| {user_error['user_id']} | {user_error['error_rate']:.1f}% | {user_error['invalid_count']} |")
        
        # Outlier Analysis
        report.append("\n## Outlier Analysis")
        outliers = self.stats['outliers']
        
        if outliers['total_outliers'] > 0:
            report.append(f"- **Total Outliers:** {outliers['total_outliers']:,}")
            report.append(f"- **Outlier Rate:** {outliers['outlier_rate']:.2f}%")
            
            if 'users_with_highest_outlier_rates' in outliers:
                report.append("\n### Users with Highest Outlier Rates")
                report.append("| User ID | Outlier Rate (%) | Outlier Count |")
                report.append("|---------|-----------------|---------------|")
                for user_outlier in outliers['users_with_highest_outlier_rates']:
                    report.append(f"| {user_outlier['user_id']} | {user_outlier['outlier_rate']:.1f}% | {user_outlier['outlier_count']} |")
        else:
            report.append("- No outliers detected in the dataset")
        
        # Data Quality Assessment
        report.append("\n## Data Quality Assessment")
        
        validity_rate = basic['valid_records'] / basic['total_records'] * 100
        if validity_rate >= 95:
            quality_rating = "Excellent"
        elif validity_rate >= 85:
            quality_rating = "Good"
        elif validity_rate >= 70:
            quality_rating = "Fair"
        else:
            quality_rating = "Poor"
        
        report.append(f"- **Overall Data Quality:** {quality_rating} ({validity_rate:.1f}% valid)")
        
        # Recommendations
        report.append("\n## Recommendations")
        
        recommendations = []
        
        if validity_rate < 90:
            recommendations.append("- **Data Cleaning:** Consider additional preprocessing to improve data validity rate")
        
        if outliers['total_outliers'] > basic['valid_records'] * 0.15:
            recommendations.append("- **Outlier Investigation:** High outlier rate may indicate data collection issues or need for parameter tuning")
        
        if timing and any(stats['negative_count'] > 0 for stats in timing.values()):
            recommendations.append("- **Timing Validation:** Investigate negative timing values which may indicate synchronization issues")
        
        # Check for users with very low data
        low_data_users = [user for user in errors['users_with_highest_error_rates'] if user['error_rate'] > 50]
        if low_data_users:
            recommendations.append(f"- **User Data Review:** {len(low_data_users)} users have >50% error rates and may need data recollection")
        
        if not recommendations:
            recommendations.append("- **Data Quality:** Dataset appears to be in good condition for analysis")
        
        for rec in recommendations:
            report.append(rec)
        
        # Visualizations
        report.append("\n## Visualizations")
        report.append("The following plots have been generated and saved to the `plots/` directory:")
        report.append("- `timing_distributions.png` - Distribution plots for all timing features")
        report.append("- `user_data_quality.png` - Data validity and outlier rates by user")
        
        return "\n".join(report)
    
    def save_report(self):
        """Save the comprehensive analysis report."""
        try:
            self.create_visualizations()
        except Exception as e:
            print(f"Warning: Could not create visualizations: {e}")
        
        report_content = self.generate_markdown_report()
        
        # Save markdown report
        report_file = Path(self.output_dir) / "TypeNet_Analysis_Report.md"
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        # Save detailed statistics as JSON
        import json
        stats_file = Path(self.output_dir) / "analysis_statistics.json"
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        stats_json = convert_numpy_types(self.stats)
        
        with open(stats_file, 'w') as f:
            json.dump(stats_json, f, indent=2)
        
        print(f"\n{'='*60}")
        print("ANALYSIS REPORT GENERATED")
        print(f"{'='*60}")
        print(f"Report saved to: {report_file}")
        print(f"Statistics saved to: {stats_file}")
        if self.plots_dir.exists():
            print(f"Visualizations saved to: {self.plots_dir}")
        
        return report_file

class TypeNetFeatureExtractor:
    """
    Extract TypeNet keystroke features from raw keystroke data.
    Based on: Acien et al. (2021) TypeNet: Deep Learning Keystroke Biometrics
    """
    
    def __init__(self):
        self.error_types = {
            'valid': 'No error',
            'missing_key1_release': 'Missing key1 release',
            'missing_key1_press': 'Missing key1 press (orphan release)',
            'missing_key2_release': 'Missing key2 release',
            'missing_key2_press': 'Missing key2 press (orphan release)',
            'invalid_key1': 'Invalid key1 data',
            'invalid_key2': 'Invalid key2 data',
            'negative_timing': 'Negative timing value detected'
        }
    
    def is_valid_raw_file(self, filepath: str) -> bool:
        """
        Check if file appears to be a raw keystroke data file based on filename pattern.
        Expected format: platform_video_session_user.csv (e.g., 1_1_1_1001.csv)
        """
        filename = os.path.basename(filepath)
        if not filename.endswith('.csv'):
            return False
        
        # Remove .csv extension and split by underscore
        parts = filename.replace('.csv', '').split('_')
        
        # Should have exactly 4 parts: platform, video, session, user_id
        if len(parts) != 4:
            return False
        
        # All parts should be numeric
        try:
            [int(part) for part in parts]
            return True
        except ValueError:
            return False
    
    def parse_raw_file(self, filepath: str) -> pl.DataFrame:
        """
        Parse raw keystroke file with format: press-type (P or R), key, timestamp
        """
        try:
            # First check if this looks like a raw keystroke file
            if not self.is_valid_raw_file(filepath):
                print(f"Skipping non-raw file: {os.path.basename(filepath)}")
                return pl.DataFrame()
            
            # Read CSV without header, specify schema to ensure correct types
            df = pl.read_csv(
                filepath,
                has_header=False,
                new_columns=['type', 'key', 'timestamp'],
                schema_overrides={
                    'type': pl.Utf8,
                    'key': pl.Utf8,
                    'timestamp': pl.Float64
                },
                truncate_ragged_lines=True,  # Handle files with inconsistent column counts
                ignore_errors=True           # Skip problematic rows instead of failing
            )
            
            # Basic validation - should have exactly 3 columns
            if len(df.columns) != 3:
                print(f"Invalid file format (wrong number of columns): {os.path.basename(filepath)}")
                return pl.DataFrame()
            
            # Check if type column contains expected values
            valid_types = df['type'].is_in(['P', 'R']).all()
            if not valid_types:
                print(f"Invalid file format (unexpected press types): {os.path.basename(filepath)}")
                return pl.DataFrame()
            
            return df
            
        except Exception as e:
            print(f"Error reading file {filepath}: {e}")
            return pl.DataFrame()
    
    def match_press_release_pairs(self, df: pl.DataFrame) -> List[Dict]:
        """
        Match press and release events using a parentheses matching algorithm.
        Returns list of keystroke events with validity information.
        """
        events = []
        key_stacks = {}  # Stack for each unique key
        orphan_releases = []  # Store orphan release events
        
        for idx, row in enumerate(df.iter_rows(named=True)):
            key = row['key']
            event_type = row['type']
            timestamp = row['timestamp']
            
            if key not in key_stacks:
                key_stacks[key] = deque()
            
            if event_type == 'P':
                # Push press event to stack
                key_stacks[key].append({
                    'key': key,
                    'press_time': timestamp,
                    'press_idx': idx
                })
            
            elif event_type == 'R':
                if key_stacks[key]:
                    # Match with most recent press
                    press_event = key_stacks[key].pop()
                    events.append({
                        'key': key,
                        'press_time': press_event['press_time'],
                        'release_time': timestamp,
                        'press_idx': press_event['press_idx'],
                        'release_idx': idx,
                        'valid': True,
                        'error': 'valid'
                    })
                else:
                    # Orphan release - no matching press
                    events.append({
                        'key': key,
                        'press_time': None,
                        'release_time': timestamp,
                        'press_idx': None,
                        'release_idx': idx,
                        'valid': False,
                        'error': 'missing_key1_press'
                    })
        
        # Handle unmatched press events (missing releases)
        for key, stack in key_stacks.items():
            while stack:
                press_event = stack.pop()
                events.append({
                    'key': press_event['key'],
                    'press_time': press_event['press_time'],
                    'release_time': None,
                    'press_idx': press_event['press_idx'],
                    'release_idx': None,
                    'valid': False,
                    'error': 'missing_key1_release'
                })
        
        # Sort events by press time (or release time for orphan releases)
        events.sort(key=lambda x: x['press_time'] if x['press_time'] is not None else x['release_time'])
        
        return events
    
    def calculate_features(self, key1_event: Dict, key2_event: Dict) -> Dict:
        """
        Calculate TypeNet features for a key pair.
        HL: Hold Latency (key1_release - key1_press)
        IL: Inter-key Latency (key2_press - key1_release)
        PL: Press Latency (key2_press - key1_press)
        RL: Release Latency (key2_release - key1_release)
        """
        features = {
            'key1': key1_event['key'],
            'key2': key2_event['key'],
            'key1_press': key1_event['press_time'],
            'key1_release': key1_event['release_time'],
            'key2_press': key2_event['press_time'],
            'key2_release': key2_event['release_time'],
            'HL': None,
            'IL': None,
            'PL': None,
            'RL': None,
            'valid': True,
            'error_description': 'No error'
        }
        
        # Check validity and set appropriate error messages
        if not key1_event['valid']:
            features['valid'] = False
            # Adjust error message for key1 position
            if key1_event['error'] == 'missing_key1_press':
                features['error_description'] = 'Missing key1 press (orphan release)'
            elif key1_event['error'] == 'missing_key1_release':
                features['error_description'] = 'Missing key1 release'
            else:
                features['error_description'] = self.error_types[key1_event['error']]
        
        if not key2_event['valid']:
            features['valid'] = False
            # Adjust error message for key2 position
            if key2_event['error'] == 'missing_key1_press':
                features['error_description'] = 'Missing key2 press (orphan release)'
            elif key2_event['error'] == 'missing_key1_release':
                features['error_description'] = 'Missing key2 release'
            else:
                features['error_description'] = self.error_types[key2_event['error']]
        
        # Calculate HL for key1 if it has both press and release (regardless of key2 validity)
        try:
            if key1_event['press_time'] is not None and key1_event['release_time'] is not None:
                features['HL'] = key1_event['release_time'] - key1_event['press_time']
                if features['HL'] < 0:
                    features['valid'] = False
                    features['error_description'] = 'Negative HL timing'
        except Exception as e:
            pass  # HL remains None if calculation fails
        
        # Calculate other features only if both keys are valid
        if key1_event['valid'] and key2_event['valid']:
            try:
                # IL: Inter-key Latency
                if key1_event['release_time'] is not None and key2_event['press_time'] is not None:
                    features['IL'] = key2_event['press_time'] - key1_event['release_time']
                    # IL can be negative (key overlap)
                
                # PL: Press Latency
                if key1_event['press_time'] is not None and key2_event['press_time'] is not None:
                    features['PL'] = key2_event['press_time'] - key1_event['press_time']
                    if features['PL'] < 0:
                        features['valid'] = False
                        features['error_description'] = 'Negative PL timing'
                
                # RL: Release Latency
                if key1_event['release_time'] is not None and key2_event['release_time'] is not None:
                    features['RL'] = key2_event['release_time'] - key1_event['release_time']
                    # RL can be negative (release order different from press order)
            
            except Exception as e:
                features['valid'] = False
                features['error_description'] = f'Calculation error: {str(e)}'
        
        return features
    
    def extract_features_from_file(self, filepath: str) -> pl.DataFrame:
        """
        Extract all features from a single raw keystroke file.
        """
        # Parse filename to get metadata
        filename = os.path.basename(filepath)
        parts = filename.replace('.csv', '').split('_')
        
        if len(parts) != 4:
            print(f"Invalid filename format: {filename}")
            return pl.DataFrame()
        
        platform_id, video_id, session_id, user_id = parts
        
        # Read and parse raw data
        raw_df = self.parse_raw_file(filepath)
        if raw_df.is_empty():
            return pl.DataFrame()
        
        # Match press-release pairs
        events = self.match_press_release_pairs(raw_df)
        
        # Extract features for consecutive key pairs
        features_list = []
        for i in range(len(events) - 1):
            key1_event = events[i]
            key2_event = events[i + 1]
            
            features = self.calculate_features(key1_event, key2_event)
            
            # Add metadata
            features.update({
                'user_id': int(user_id),
                'platform_id': int(platform_id),
                'video_id': int(video_id),
                'session_id': int(session_id),
                'sequence_id': i,
                'key1_timestamp': key1_event['press_time'] if key1_event['press_time'] is not None else key1_event['release_time']
            })
            
            features_list.append(features)
        
        if not features_list:
            return pl.DataFrame()
        
        return pl.DataFrame(features_list)
    
    def detect_outliers(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        Apply HBOS outlier detection to the feature data.
        Adds 'outlier' column to the dataframe.
        """
        # Only process valid records for outlier detection
        valid_df = df.filter(pl.col('valid'))
        
        if len(valid_df) < 10:  # Too few records for meaningful outlier detection
            return df.with_columns(pl.lit(False).alias('outlier'))
        
        # Prepare data for HBOS
        feature_data = defaultdict(list)
        timing_features = ['HL', 'IL', 'PL', 'RL']
        
        # Only use non-null timing features
        for feature in timing_features:
            valid_values = valid_df.select(
                pl.col(feature).drop_nulls().drop_nans()
            ).to_series().to_list()
            if len(valid_values) > 0:
                feature_data[feature] = valid_values
        
        if not feature_data:  # No valid timing data
            return df.with_columns(pl.lit(False).alias('outlier'))
        
        # Ensure all features have the same length by using complete cases
        aligned_data = defaultdict(list)
        complete_case_mask = valid_df.select([
            pl.all_horizontal([
                pl.col(f).is_not_null() & pl.col(f).is_not_nan() 
                for f in timing_features
            ])
        ]).to_series()
        
        complete_cases_df = valid_df.filter(complete_case_mask)
        
        if len(complete_cases_df) < 10:
            return df.with_columns(pl.lit(False).alias('outlier'))
        
        for feature in timing_features:
            aligned_data[feature] = complete_cases_df.select(pl.col(feature)).to_series().to_list()
        
        # Apply HBOS
        hbos = HBOS(n_bins=10, contamination=0.1, alpha=0.1, tol=0.5)
        hbos.fit(aligned_data)
        outliers = hbos.predict_outliers(aligned_data)
        
        # Create outlier mapping
        complete_case_indices = complete_cases_df.select(pl.arange(0, len(complete_cases_df)).alias('idx')).to_series().to_list()
        outlier_dict = {idx: bool(outlier) for idx, outlier in zip(complete_case_indices, outliers)}
        
        # Map outliers back to original dataframe
        outlier_column = []
        valid_idx = 0
        
        for i in range(len(df)):
            if df[i, 'valid']:  # Check if this row is valid
                if valid_idx in outlier_dict:
                    outlier_column.append(outlier_dict[valid_idx])
                else:
                    outlier_column.append(False)
                if complete_case_mask[valid_idx]:
                    valid_idx += 1
            else:
                outlier_column.append(False)
        
        # Simpler approach: start fresh
        outlier_series = pl.Series('outlier', [False] * len(df))
        
        # Mark outliers for complete cases only
        valid_row_count = 0
        for i in range(len(df)):
            if df[i, 'valid']:
                # Check if this valid row has all timing features non-null
                row_data = df.row(i, named=True)
                has_all_features = all(row_data[f] is not None for f in timing_features)
                
                if has_all_features and valid_row_count < len(outliers):
                    outlier_series[i] = bool(outliers[valid_row_count])
                    valid_row_count += 1
        
        return df.with_columns(outlier_series)
    
    def process_user_directory(self, user_dir: str) -> pl.DataFrame:
        """
        Process all files for a single user.
        """
        all_features = []
        raw_file_count = 0
        processed_file_count = 0
        
        for filepath in Path(user_dir).glob('*.csv'):
            if self.is_valid_raw_file(str(filepath)):
                raw_file_count += 1
                print(f"Processing {filepath}")
                features_df = self.extract_features_from_file(str(filepath))
                if not features_df.is_empty():
                    all_features.append(features_df)
                    processed_file_count += 1
            else:
                print(f"Skipping non-raw file: {filepath}")
        
        print(f"Found {raw_file_count} raw files, successfully processed {processed_file_count}")
        
        if all_features:
            return pl.concat(all_features)
        else:
            return pl.DataFrame()
    
    def process_dataset(self, root_dir: str, output_file: str = 'typenet_features_extracted.csv'):
        """
        Process entire dataset with one directory per user.
        """
        all_user_features = []
        
        # Directories to exclude from processing
        excluded_dirs = {'broken_data', 'typenet_features'}
        
        # Process each user directory
        for user_dir in Path(root_dir).iterdir():
            if user_dir.is_dir():
                # Skip excluded directories
                if user_dir.name in excluded_dirs:
                    print(f"Skipping excluded directory: {user_dir}")
                    continue
                    
                print(f"\nProcessing user directory: {user_dir}")
                user_features = self.process_user_directory(str(user_dir))
                if not user_features.is_empty():
                    # Apply outlier detection per user
                    user_features = self.detect_outliers(user_features)
                    all_user_features.append(user_features)
        
        # Combine all features
        if all_user_features:
            final_df = pl.concat(all_user_features)
            
            # Reorder columns to match expected format
            column_order = [
                'user_id', 'platform_id', 'video_id', 'session_id', 'sequence_id',
                'key1', 'key2', 'key1_press', 'key1_release', 'key2_press', 'key2_release',
                'HL', 'IL', 'PL', 'RL', 'key1_timestamp', 'valid', 'error_description', 'outlier'
            ]
            
            final_df = final_df.select(column_order)
            
            # Save to CSV
            final_df.write_csv(output_file)
            print(f"\nFeatures extracted and saved to {output_file}")
            print(f"Total records: {len(final_df)}")
            print(f"Valid records: {final_df.select(pl.col('valid').sum()).item()}")
            print(f"Invalid records: {final_df.select((~pl.col('valid')).sum()).item()}")
            print(f"Outliers detected: {final_df.select(pl.col('outlier').sum()).item()}")
            
            # Print error summary
            print("\nError summary:")
            error_counts = final_df.filter(~pl.col('valid')).select('error_description').to_series().value_counts(sort=True)
            for row in error_counts.iter_rows():
                error_type, count = row
                print(f"  {error_type}: {count}")
            
            # Print outlier summary
            print("\nOutlier summary:")
            outlier_valid = final_df.filter(pl.col('valid') & pl.col('outlier'))
            valid_count = final_df.select(pl.col('valid').sum()).item()
            print(f"  Outliers in valid data: {len(outlier_valid)}")
            if valid_count > 0:
                print(f"  Outlier rate in valid data: {len(outlier_valid) / valid_count * 100:.2f}%")
            
            # Generate comprehensive analysis report
            print(f"\n{'='*60}")
            print("GENERATING ANALYSIS REPORT")
            print(f"{'='*60}")
            
            # Extract directory from output_file path to use as report directory
            output_dir = os.path.dirname(output_file)
            analyzer = TypeNetAnalysisReport(final_df, output_dir)
            analyzer.save_report()
            
            return final_df
        else:
            print("No features extracted from dataset")
            return pl.DataFrame()


# Example usage
if __name__ == "__main__":
    # Create extractor instance
    extractor = TypeNetFeatureExtractor()
    
    print("=== TypeNet Feature Extraction with Outlier Detection ===\n")
    
    # Process the demo dataset
    print("Processing demo dataset...")
    raw_data_dir = 'data_dump/loadable_Combined_HU_HT'
    
    # Make saved processed data filenames consistent for automated download and extraction.
    hostname = socket.gethostname()
    now = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    processed_data_save_dir = os.path.join(base_dir, f"processed_data-{now}-{hostname}")
    os.makedirs(processed_data_save_dir, exist_ok=True)
    save_data_path = os.path.join(processed_data_save_dir, 'typenet_features.csv')

    features_df = extractor.process_dataset(raw_data_dir, save_data_path)
    
    print("\n=== Extracted Features ===")
    if not features_df.is_empty():
        # Set polars display options to show all columns
        with pl.Config(tbl_width_chars=1000, tbl_cols=-1, tbl_rows=20):
            print(features_df.head(20))
    
        print("\n=== Feature Statistics ===")
        print(f"Total keypair features: {len(features_df)}")
        print(f"Valid features: {features_df.select(pl.col('valid').sum()).item()}")
        print(f"Invalid features: {features_df.select((~pl.col('valid')).sum()).item()}")
        print(f"Outliers: {features_df.select(pl.col('outlier').sum()).item()}")
        
        print("\n=== Timing Feature Ranges (valid records only) ===")
        valid_df = features_df.filter(pl.col('valid'))
        if not valid_df.is_empty():
            for feature in ['HL', 'IL', 'PL', 'RL']:
                if feature in valid_df.columns:
                    stats = valid_df.select([
                        pl.col(feature).min().alias('min'),
                        pl.col(feature).max().alias('max'),
                        pl.col(feature).mean().alias('mean')
                    ]).filter(pl.col('min').is_not_null())
                    
                    if not stats.is_empty():
                        min_val, max_val, mean_val = stats.row(0)
                        print(f"{feature}: min={min_val:.0f}ms, "
                              f"max={max_val:.0f}ms, "
                              f"mean={mean_val:.0f}ms")
        
        print("\n=== Error Analysis ===")
        error_df = features_df.filter(~pl.col('valid'))
        if not error_df.is_empty():
            print("Errors found:")
            error_counts = error_df.select('error_description').to_series().value_counts(sort=True)
            for row in error_counts.iter_rows():
                error_type, count = row
                print(f"  - {error_type}: {count} occurrences")
        
        print("\n=== Outlier Analysis ===")
        outlier_df = features_df.filter(pl.col('outlier'))
        if not outlier_df.is_empty():
            print("Outliers by validity:")
            print(f"  - Valid outliers: {outlier_df.select(pl.col('valid').sum()).item()}")
            print(f"  - Invalid outliers: {outlier_df.select((~pl.col('valid')).sum()).item()}")
            
            print("\nOutliers by platform:")
            platforms = sorted(features_df.select('platform_id').unique().to_series().to_list())
            for platform in platforms:
                platform_outliers = outlier_df.filter(pl.col('platform_id') == platform)
                print(f"  - Platform {platform}: {len(platform_outliers)} outliers")
        
        # Show full DataFrame sample with all columns visible
        print("\n=== Full Feature Sample (All Columns) ===")
        sample_df = features_df.head(10)
        # Convert to pandas temporarily for better display of all columns
        try:
            import pandas as pd
            sample_pandas = sample_df.to_pandas()
            pd.set_option('display.max_columns', None)
            pd.set_option('display.width', None)
            pd.set_option('display.max_colwidth', None)
            print(sample_pandas.to_string(index=False))
        except ImportError:
            # Fallback to polars with wide settings
            with pl.Config(tbl_width_chars=2000, tbl_cols=-1):
                print(sample_df)
    else:
        print("No valid features were extracted from the dataset.")
        