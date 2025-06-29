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

import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
from jinja2 import Template

warnings.filterwarnings('ignore')

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.utils.version_manager import VersionManager

logger = logging.getLogger(__name__)


class DataQualityAnalyzer:
    """Analyzes data quality issues in keystroke data"""
    
    def __init__(self):
        self.skip_keys = {'Key.shift', 'Key.ctrl', 'Key.alt', 'Key.cmd', 'Key.caps_lock'}
        
    def analyze_raw_keystrokes(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze raw keystroke data for quality issues"""
        results = {
            'total_events': len(df),
            'unique_keys': df['key'].nunique(),
            'issues': [],
            'issue_counts': defaultdict(int),
            'unreleased_keys': {},
            'key_stats': defaultdict(lambda: {'presses': 0, 'releases': 0, 'issues': 0})
        }
        
        # Group by user and file to analyze each session
        for (user_id, filename), session_df in df.groupby(['user_id', 'source_file']):
            session_issues = self._analyze_session(session_df, user_id, filename)
            results['issues'].extend(session_issues['issues'])
            
            # Aggregate issue counts
            for issue_type, count in session_issues['issue_counts'].items():
                results['issue_counts'][issue_type] += count
                
            # Track unreleased keys
            if session_issues['unreleased_keys']:
                results['unreleased_keys'][f"{user_id}_{filename}"] = session_issues['unreleased_keys']
                
            # Aggregate key stats
            for key, stats in session_issues['key_stats'].items():
                for stat_type, value in stats.items():
                    results['key_stats'][key][stat_type] += value
                    
        return results
        
    def _analyze_session(self, df: pd.DataFrame, user_id: str, filename: str) -> Dict[str, Any]:
        """Analyze a single session for issues"""
        active_keys = {}
        issues = []
        issue_counts = defaultdict(int)
        key_stats = defaultdict(lambda: {'presses': 0, 'releases': 0, 'issues': 0})
        
        # Sort by timestamp
        df = df.sort_values('timestamp')
        
        for idx, row in df.iterrows():
            key = row['key']
            event_type = row['type']
            timestamp = row['timestamp']
            
            # Update statistics
            if event_type == 'P':
                key_stats[key]['presses'] += 1
            else:
                key_stats[key]['releases'] += 1
                
            # Skip modifier keys for issue detection
            if key in self.skip_keys:
                continue
                
            if event_type == 'P':
                if key in active_keys:
                    # Double press issue
                    issues.append({
                        'type': 'double_press',
                        'user_id': user_id,
                        'file': filename,
                        'key': key,
                        'first_press': active_keys[key],
                        'second_press': timestamp,
                        'index': idx
                    })
                    issue_counts['double_press'] += 1
                    key_stats[key]['issues'] += 1
                active_keys[key] = timestamp
                
            elif event_type == 'R':
                if key not in active_keys:
                    # Orphan release
                    issues.append({
                        'type': 'orphan_release',
                        'user_id': user_id,
                        'file': filename,
                        'key': key,
                        'time': timestamp,
                        'index': idx
                    })
                    issue_counts['orphan_release'] += 1
                    key_stats[key]['issues'] += 1
                else:
                    # Check for negative hold time
                    hold_time = timestamp - active_keys[key]
                    if hold_time < 0:
                        issues.append({
                            'type': 'negative_hold_time',
                            'user_id': user_id,
                            'file': filename,
                            'key': key,
                            'press_time': active_keys[key],
                            'release_time': timestamp,
                            'hold_time': hold_time,
                            'index': idx
                        })
                        issue_counts['negative_hold_time'] += 1
                        key_stats[key]['issues'] += 1
                    del active_keys[key]
                    
        # Record unreleased keys
        unreleased = {key: time for key, time in active_keys.items() if key not in self.skip_keys}
        
        return {
            'issues': issues,
            'issue_counts': dict(issue_counts),
            'unreleased_keys': unreleased,
            'key_stats': dict(key_stats)
        }


class FeatureAnalyzer:
    """Analyzes extracted features and timing patterns"""
    
    def analyze_timing_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze timing feature distributions"""
        timing_features = ['HL', 'IL', 'PL', 'RL']
        results = {}
        
        # Filter valid data
        valid_df = df[df['valid']]
        
        for feature in timing_features:
            if feature in valid_df.columns:
                # Convert to milliseconds if needed
                if f'{feature}_ms' in valid_df.columns:
                    values = valid_df[f'{feature}_ms']
                else:
                    values = valid_df[feature] / 1_000_000  # nanoseconds to ms
                    
                # Remove nulls
                values = values.dropna()
                
                if len(values) > 0:
                    results[feature] = {
                        'count': len(values),
                        'mean': float(values.mean()),
                        'std': float(values.std()),
                        'min': float(values.min()),
                        'max': float(values.max()),
                        'median': float(values.median()),
                        'q25': float(values.quantile(0.25)),
                        'q75': float(values.quantile(0.75)),
                        'negative_count': int((values < 0).sum()),
                        'zero_count': int((values == 0).sum()),
                        'outlier_threshold_low': float(values.quantile(0.01)),
                        'outlier_threshold_high': float(values.quantile(0.99))
                    }
                else:
                    results[feature] = {'count': 0}
                    
        return results
        
    def analyze_user_performance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze per-user statistics"""
        user_stats = []
        
        for user_id, user_df in df.groupby('user_id'):
            total_count = len(user_df)
            valid_count = user_df['valid'].sum()
            
            stats = {
                'user_id': user_id,
                'total_keypairs': total_count,
                'valid_keypairs': int(valid_count),
                'invalid_keypairs': int(total_count - valid_count),
                'validity_rate': float(valid_count / total_count * 100) if total_count > 0 else 0
            }
            
            # Add outlier stats if available
            if 'outlier' in user_df.columns:
                outlier_count = user_df['outlier'].sum()
                stats['outlier_count'] = int(outlier_count)
                stats['outlier_rate'] = float(outlier_count / valid_count * 100) if valid_count > 0 else 0
                
            user_stats.append(stats)
            
        return pd.DataFrame(user_stats)


class ReportGenerator:
    """Generates HTML reports and visualizations"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.figures_dir = output_dir / "figures"
        self.tables_dir = output_dir / "tables"
        
        # Create directories
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        
        # Set plotting style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
    def create_timing_distributions(self, timing_stats: Dict[str, Any]) -> str:
        """Create distribution plots for timing features"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        features = ['HL', 'IL', 'PL', 'RL']
        feature_names = {
            'HL': 'Hold Latency',
            'IL': 'Inter-key Latency',
            'PL': 'Press Latency',
            'RL': 'Release Latency'
        }
        
        for idx, feature in enumerate(features):
            ax = axes[idx]
            
            if feature in timing_stats and timing_stats[feature]['count'] > 0:
                stats = timing_stats[feature]
                
                # Create text for plot
                text = f"Mean: {stats['mean']:.1f}ms\n"
                text += f"Median: {stats['median']:.1f}ms\n"
                text += f"Std: {stats['std']:.1f}ms\n"
                text += f"Count: {stats['count']:,}"
                
                # Add box with stats
                ax.text(0.95, 0.95, text, transform=ax.transAxes,
                       verticalalignment='top', horizontalalignment='right',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
                
                # Create dummy histogram representation
                ax.bar(['Min', '25%', 'Median', '75%', 'Max'],
                      [stats['min'], stats['q25'], stats['median'], stats['q75'], stats['max']])
                ax.set_ylabel('Time (ms)')
                ax.set_title(f'{feature_names[feature]} Distribution')
            else:
                ax.text(0.5, 0.5, 'No data available', 
                       transform=ax.transAxes, ha='center', va='center')
                ax.set_title(f'{feature_names[feature]} Distribution')
                
        plt.tight_layout()
        filename = 'timing_distributions.png'
        filepath = self.figures_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename
        
    def create_user_quality_chart(self, user_stats: pd.DataFrame) -> str:
        """Create bar chart of user data quality"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Sort by validity rate
        user_stats = user_stats.sort_values('validity_rate', ascending=False)
        
        # Validity rates
        x = range(len(user_stats))
        ax1.bar(x, user_stats['validity_rate'])
        ax1.set_xlabel('User Index')
        ax1.set_ylabel('Validity Rate (%)')
        ax1.set_title('Data Validity Rate by User')
        ax1.grid(True, alpha=0.3)
        
        # Total keypairs
        ax2.bar(x, user_stats['total_keypairs'])
        ax2.set_xlabel('User Index')
        ax2.set_ylabel('Total Keypairs')
        ax2.set_title('Total Keypairs by User')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        filename = 'user_data_quality.png'
        filepath = self.figures_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename
        
    def generate_html_report(self, analysis_results: Dict[str, Any]) -> str:
        """Generate comprehensive HTML report"""
        template_str = '''
<!DOCTYPE html>
<html>
<head>
    <title>Keystroke Data Analysis Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; }
        h1, h2, h3 { color: #333; }
        table { border-collapse: collapse; width: 100%; margin: 20px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #4CAF50; color: white; }
        tr:nth-child(even) { background-color: #f2f2f2; }
        .metric { background-color: #f0f0f0; padding: 10px; margin: 10px 0; border-radius: 5px; }
        .warning { color: #ff6b6b; }
        .good { color: #51cf66; }
        img { max-width: 100%; height: auto; margin: 20px 0; }
    </style>
</head>
<body>
    <h1>Keystroke Data Analysis Report</h1>
    <p><strong>Generated:</strong> {{ timestamp }}</p>
    <p><strong>Version ID:</strong> {{ version_id }}</p>
    
    <h2>Executive Summary</h2>
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
    <h2>Data Quality Issues</h2>
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
    
    <h2>Timing Feature Analysis</h2>
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
    
    <h2>User Performance</h2>
    <img src="figures/user_data_quality.png" alt="User Data Quality">
    
    <h3>Top Users by Validity Rate</h3>
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
        {% for _, user in top_users.iterrows() %}
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
    
    <h2>Visualizations</h2>
    <h3>Timing Feature Distributions</h3>
    <img src="figures/timing_distributions.png" alt="Timing Distributions">
    
    <h2>Data Quality Assessment</h2>
    <div class="metric">
        <p><strong>Overall Quality Rating:</strong> 
        <span class="{% if summary.validity_rate >= 90 %}good{% elif summary.validity_rate >= 70 %}{% else %}warning{% endif %}">
            {% if summary.validity_rate >= 95 %}Excellent{% elif summary.validity_rate >= 85 %}Good{% elif summary.validity_rate >= 70 %}Fair{% else %}Poor{% endif %}
        </span>
        ({{ summary.validity_rate|round(1) }}% valid)
        </p>
    </div>
    
    <h2>Recommendations</h2>
    <ul>
        {% for rec in recommendations %}
        <li>{{ rec }}</li>
        {% endfor %}
    </ul>
</body>
</html>
        '''
        
        template = Template(template_str)
        
        # Custom filters
        template.globals['number'] = lambda x: f"{int(x):,}" if pd.notna(x) else "N/A"
        template.globals['round'] = lambda x, n=1: round(x, n) if pd.notna(x) else "N/A"
        
        return template.render(**analysis_results)


class RunEDAStage:
    """Run exploratory data analysis and generate reports"""
    
    def __init__(self, version_id: str, config: Dict[str, Any], 
                 dry_run: bool = False, local_only: bool = False):
        self.version_id = version_id
        self.config = config
        self.dry_run = dry_run
        self.local_only = local_only
        self.version_manager = VersionManager()
        
        # Initialize analyzers
        self.quality_analyzer = DataQualityAnalyzer()
        self.feature_analyzer = FeatureAnalyzer()
        
    def load_raw_keystrokes(self, cleaned_data_dir: Path) -> Optional[pd.DataFrame]:
        """Load raw keystroke data for quality analysis"""
        all_data = []
        
        for device_type in ['desktop', 'mobile']:
            raw_data_dir = cleaned_data_dir / device_type / 'raw_data'
            if not raw_data_dir.exists():
                continue
                
            for user_dir in raw_data_dir.iterdir():
                if not user_dir.is_dir():
                    continue
                    
                user_id = user_dir.name
                
                # Load CSV files
                for csv_file in user_dir.glob('*.csv'):
                    # Skip non-keystroke files
                    if not self._is_keystroke_file(csv_file):
                        continue
                        
                    try:
                        df = pd.read_csv(csv_file, header=None, names=['type', 'key', 'timestamp'])
                        df['user_id'] = user_id
                        df['device_type'] = device_type
                        df['source_file'] = csv_file.name
                        all_data.append(df)
                    except Exception as e:
                        logger.warning(f"Could not load {csv_file}: {e}")
                        
        return pd.concat(all_data, ignore_index=True) if all_data else None
        
    def _is_keystroke_file(self, filepath: Path) -> bool:
        """Check if file is a keystroke data file"""
        # Pattern: platform_video_session_user.csv
        parts = filepath.stem.split('_')
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
            
    def generate_recommendations(self, summary: Dict, quality_issues: Dict, 
                               timing_stats: Dict) -> List[str]:
        """Generate analysis recommendations"""
        recommendations = []
        
        # Data quality recommendations
        validity_rate = summary.get('validity_rate', 0)
        if validity_rate < 90:
            recommendations.append(
                f"**Data Quality:** Validity rate is {validity_rate:.1f}%. "
                "Consider reviewing data collection procedures."
            )
            
        # Issue-based recommendations
        if quality_issues:
            total_issues = quality_issues.get('total_issues', 0)
            if total_issues > summary['total_keypairs'] * 0.05:
                recommendations.append(
                    f"**High Error Rate:** {total_issues} quality issues detected. "
                    "Investigate data collection synchronization."
                )
                
            if quality_issues.get('unreleased_keys'):
                recommendations.append(
                    "**Unreleased Keys:** Multiple sessions have unreleased keys. "
                    "May indicate incomplete data capture."
                )
                
        # Timing recommendations
        for feature, stats in timing_stats.items():
            if stats.get('negative_count', 0) > 0:
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
        artifacts_dir = Path(self.config.get("ARTIFACTS_DIR", "artifacts")) / self.version_id
        output_dir = artifacts_dir / "eda_reports" / "data_quality"
        
        # Get input directories from previous stages
        cleaned_data_dir = artifacts_dir / "cleaned_data"
        keypairs_dir = artifacts_dir / "keypairs"
        features_dir = artifacts_dir / "features"
        
        # Initialize results
        analysis_results = {
            'version_id': self.version_id,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Load and analyze keypair data
        keypair_file = keypairs_dir / "keypairs.parquet"
        if not keypair_file.exists():
            keypair_file = keypairs_dir / "keypairs.csv"
            
        if keypair_file.exists():
            logger.info("Analyzing keypair data...")
            if keypair_file.suffix == '.parquet':
                keypairs_df = pd.read_parquet(keypair_file)
            else:
                keypairs_df = pd.read_csv(keypair_file)
                
            # Basic summary
            summary = {
                'total_keypairs': len(keypairs_df),
                'valid_keypairs': keypairs_df['valid'].sum(),
                'invalid_keypairs': (~keypairs_df['valid']).sum(),
                'validity_rate': keypairs_df['valid'].mean() * 100,
                'unique_users': keypairs_df['user_id'].nunique(),
                'device_types': keypairs_df['device_type'].unique().tolist()
            }
            analysis_results['summary'] = summary
            
            # Timing analysis
            timing_stats = self.feature_analyzer.analyze_timing_features(keypairs_df)
            analysis_results['timing_stats'] = timing_stats
            
            # User performance
            user_stats = self.feature_analyzer.analyze_user_performance(keypairs_df)
            analysis_results['user_stats'] = user_stats
            analysis_results['top_users'] = user_stats.nlargest(10, 'validity_rate')
            
        # Analyze raw keystroke quality (optional)
        if cleaned_data_dir.exists():
            logger.info("Analyzing raw keystroke quality...")
            raw_df = self.load_raw_keystrokes(cleaned_data_dir)
            
            if raw_df is not None:
                quality_results = self.quality_analyzer.analyze_raw_keystrokes(raw_df)
                analysis_results['quality_issues'] = {
                    'total_issues': len(quality_results['issues']),
                    'issue_counts': dict(quality_results['issue_counts']),
                    'unreleased_keys': quality_results['unreleased_keys']
                }
            else:
                analysis_results['quality_issues'] = None
                
        # Generate recommendations
        recommendations = self.generate_recommendations(
            analysis_results.get('summary', {}),
            analysis_results.get('quality_issues', {}),
            analysis_results.get('timing_stats', {})
        )
        analysis_results['recommendations'] = recommendations
        
        # Generate report
        if not self.dry_run:
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize report generator
            report_gen = ReportGenerator(output_dir)
            
            # Create visualizations
            if 'timing_stats' in analysis_results:
                report_gen.create_timing_distributions(analysis_results['timing_stats'])
                
            if 'user_stats' in analysis_results:
                report_gen.create_user_quality_chart(analysis_results['user_stats'])
                
            # Generate HTML report
            html_content = report_gen.generate_html_report(analysis_results)
            
            with open(output_dir / 'report.html', 'w') as f:
                f.write(html_content)
                
            # Save analysis results as JSON
            with open(output_dir / 'analysis_results.json', 'w') as f:
                # Convert DataFrame to dict for JSON serialization
                results_for_json = analysis_results.copy()
                if 'user_stats' in results_for_json:
                    results_for_json['user_stats'] = results_for_json['user_stats'].to_dict('records')
                if 'top_users' in results_for_json:
                    results_for_json['top_users'] = results_for_json['top_users'].to_dict('records')
                    
                json.dump(results_for_json, f, indent=2, default=str)
                
            # Save summary statistics
            if 'summary' in analysis_results:
                with open(output_dir / 'summary_stats.json', 'w') as f:
                    json.dump(analysis_results['summary'], f, indent=2)
                    
            logger.info(f"Reports saved to {output_dir}")
            
        # Update version info
        if not self.dry_run:
            self.version_manager.update_stage_info(
                self.version_id,
                "run_eda",
                {
                    "output_dir": str(output_dir),
                    "reports_generated": ['data_quality'],
                    "completed_at": datetime.now().isoformat()
                }
            )
            
        return output_dir


def run(version_id: str, config: Dict[str, Any], 
        dry_run: bool = False, local_only: bool = False) -> Path:
    """Entry point for the pipeline orchestrator"""
    stage = RunEDAStage(version_id, config, dry_run, local_only)
    return stage.run()


if __name__ == "__main__":
    # For testing the stage independently
    import click
    from scripts.utils.config_manager import get_config
    
    @click.command()
    @click.option('--version-id', help='Version ID to use')
    @click.option('--dry-run', is_flag=True, help='Preview without generating reports')
    def main(version_id, dry_run):
        """Test EDA stage independently"""
        logging.basicConfig(level=logging.INFO)
        
        config = get_config()._config
        vm = VersionManager()
        
        if not version_id:
            version_id = vm.create_version_id()
            logger.info(f"Created version ID: {version_id}")
            
        output_dir = run(version_id, config, dry_run)
        logger.info(f"Stage complete. Output: {output_dir}")
        
    main()