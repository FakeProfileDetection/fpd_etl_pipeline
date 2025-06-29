import polars as pl
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
from itertools import combinations
import warnings
import datetime
from datetime import datetime

warnings.filterwarnings('ignore')
from typenet_feature_tools import (
    filter_IL_top_k
    )

class TypeNetMLFeatureExtractor:
    """
    Feature extraction system for TypeNet ML experiments
    Handles multiple experiment types and feature extraction strategies
    Polars version for improved performance
    """
    
    def __init__(self, data_path: str = 'typenet_features_extracted.csv', keep_outliers: bool = False):
        self.keep_outliers = keep_outliers
        """Initialize with extracted TypeNet features"""
        print(f"Loading data from {data_path}...")
        self.df = pl.read_csv(data_path)
        
        # Filter only valid data for ML
        self.df = self.df.filter(pl.col('valid'))
        if  self.keep_outliers:
            print("Keeping outliers in the dataset")
        else:
            print("Removing outliers from the dataset")
            self.df = self.df.filter(~pl.col('outlier'))
        
        # Convert to milliseconds
        for col in ['HL', 'IL', 'PL', 'RL']:
            self.df = self.df.with_columns(
                (pl.col(col) / 1_000_000).alias(f'{col}_ms')
            )
        
        self.platform_names = {1: 'facebook', 2: 'instagram', 3: 'twitter'}
        
        print(f"Loaded {len(self.df):,} valid keystroke pairs")
        print(f"Users: {self.df['user_id'].n_unique()}")
        print(f"Platforms: {sorted(self.df['platform_id'].unique().to_list())}")
        
        # Get sessions per platform using Polars
        sessions_per_platform = (
            self.df.group_by('platform_id')
            .agg(pl.col('session_id').n_unique())
            .sort('platform_id')
        )
        print(f"Sessions per platform: {dict(zip(sessions_per_platform['platform_id'].to_list(), sessions_per_platform['session_id'].to_list()))}")
        
    def get_top_digrams(self, n: int = 10) -> List[str]:
        """Get top N most frequent digrams across entire dataset"""
        # Create digram column
        self.df = self.df.with_columns(
            (pl.col('key1') + pl.col('key2')).alias('digram')
        )
        
        # Get top digrams
        digram_counts = (
            self.df.group_by('digram')
            .agg(pl.count().alias('count'))
            .sort('count', descending=True)
            .head(n)
        )
        
        top_digrams = digram_counts['digram'].to_list()
        print(f"\nTop {n} digrams: {top_digrams}")
        return top_digrams
    
    def get_all_unigrams(self) -> List[str]:
        """Get all unique unigrams (individual keys) in dataset"""
        # Combine key1 and key2 columns
        all_keys = pl.concat([
            self.df.select('key1'),
            self.df.select(pl.col('key2').alias('key1'))
        ])
        
        # Get unique keys and sort
        unigrams = sorted(all_keys['key1'].unique().to_list())
        print(f"\nTotal unique unigrams: {len(unigrams)}")
        return unigrams
    
    def extract_features(self, data: pl.DataFrame, unigrams: List[str], 
                        digrams: List[str]) -> Dict[str, float]:
        """
        Extract statistical features for given data subset
        Returns: Dict with features in order: median, mean, std, q1, q3
        """
        features = {}
        
        # Extract unigram (HL) features
        for unigram in unigrams:
            # Filter data for this unigram
            unigram_data = data.filter(pl.col('key1') == unigram).select('HL_ms')
            
            if len(unigram_data) > 0:
                hl_values = unigram_data['HL_ms']
                features[f'HL_{unigram}_median'] = float(hl_values.median())
                features[f'HL_{unigram}_mean'] = float(hl_values.mean())
                features[f'HL_{unigram}_std'] = float(hl_values.std()) if len(unigram_data) > 1 else 0.0
                features[f'HL_{unigram}_q1'] = float(hl_values.quantile(0.25))
                features[f'HL_{unigram}_q3'] = float(hl_values.quantile(0.75))
            else:
                # Missing data - will be handled by imputation strategy
                for stat in ['median', 'mean', 'std', 'q1', 'q3']:
                    features[f'HL_{unigram}_{stat}'] = np.nan
        
        # Extract digram (IL) features
        for digram in digrams:
            # Filter data for this digram
            digram_data = data.filter(pl.col('digram') == digram).select('IL_ms')
            
            if len(digram_data) > 0:
                il_values = digram_data['IL_ms']
                features[f'IL_{digram}_median'] = float(il_values.median())
                features[f'IL_{digram}_mean'] = float(il_values.mean())
                features[f'IL_{digram}_std'] = float(il_values.std()) if len(digram_data) > 1 else 0.0
                features[f'IL_{digram}_q1'] = float(il_values.quantile(0.25))
                features[f'IL_{digram}_q3'] = float(il_values.quantile(0.75))
            else:
                # Missing data - will be handled by imputation strategy
                for stat in ['median', 'mean', 'std', 'q1', 'q3']:
                    features[f'IL_{digram}_{stat}'] = np.nan
        
        return features
    
    def create_dataset_1(self, unigrams: List[str], digrams: List[str], 
                        imputation: str = 'global') -> pl.DataFrame:
        """
        Dataset 1: One set of features per user/platform
        Aggregates all sessions and videos for each user-platform combination
        """
        print("\nCreating Dataset 1: User-Platform level features...")
        
        feature_records = []
        
        # Group by user and platform
        grouped = self.df.group_by(['user_id', 'platform_id']).agg(pl.all())
        
        for row in grouped.iter_rows(named=True):
            # Reconstruct DataFrame for this group
            group_data = pl.DataFrame({
                'key1': row['key1'],
                'key2': row['key2'],
                'HL_ms': row['HL_ms'],
                'IL_ms': row['IL_ms'],
                'digram': row['digram']
            })
            
            features = self.extract_features(group_data, unigrams, digrams)
            features['user_id'] = row['user_id']
            features['platform_id'] = row['platform_id']
            feature_records.append(features)
        
        # Create DataFrame
        dataset = pl.DataFrame(feature_records)
        
        # Apply imputation strategy
        dataset = self.apply_imputation(dataset, imputation, level='platform')
        
        # Reorder columns
        id_cols = ['user_id', 'platform_id']
        feature_cols = sorted([col for col in dataset.columns if col not in id_cols])
        dataset = dataset.select(id_cols + feature_cols)
        
        print(f"Dataset 1 shape: {dataset.shape}")
        return dataset
    
    def create_dataset_2(self, unigrams: List[str], digrams: List[str], 
                        imputation: str = 'global') -> pl.DataFrame:
        """
        Dataset 2: Two sets of features per user/platform/session
        Aggregates all videos for each user-platform-session combination
        """
        print("\nCreating Dataset 2: User-Platform-Session level features...")
        
        feature_records = []
        
        # Group by user, platform, and session
        grouped = self.df.group_by(['user_id', 'platform_id', 'session_id']).agg(pl.all())
        
        for row in grouped.iter_rows(named=True):
            # Reconstruct DataFrame for this group
            group_data = pl.DataFrame({
                'key1': row['key1'],
                'key2': row['key2'],
                'HL_ms': row['HL_ms'],
                'IL_ms': row['IL_ms'],
                'digram': row['digram']
            })
            
            features = self.extract_features(group_data, unigrams, digrams)
            features['user_id'] = row['user_id']
            features['platform_id'] = row['platform_id']
            features['session_id'] = row['session_id']
            feature_records.append(features)
        
        # Create DataFrame
        dataset = pl.DataFrame(feature_records)
        
        # Apply imputation strategy
        dataset = self.apply_imputation(dataset, imputation, level='session')
        
        # Reorder columns
        id_cols = ['user_id', 'platform_id', 'session_id']
        feature_cols = sorted([col for col in dataset.columns if col not in id_cols])
        dataset = dataset.select(id_cols + feature_cols)
        
        print(f"Dataset 2 shape: {dataset.shape}")
        return dataset
    
    def create_dataset_3(self, unigrams: List[str], digrams: List[str], 
                        imputation: str = 'global') -> pl.DataFrame:
        """
        Dataset 3: Six sets of features per user/platform/session/video
        Most granular level - no aggregation
        """
        print("\nCreating Dataset 3: User-Platform-Session-Video level features...")
        
        feature_records = []
        
        # Group by user, platform, session, and video
        grouped = self.df.group_by(['user_id', 'platform_id', 'session_id', 'video_id']).agg(pl.all())
        
        for row in grouped.iter_rows(named=True):
            # Reconstruct DataFrame for this group
            group_data = pl.DataFrame({
                'key1': row['key1'],
                'key2': row['key2'],
                'HL_ms': row['HL_ms'],
                'IL_ms': row['IL_ms'],
                'digram': row['digram']
            })
            
            features = self.extract_features(group_data, unigrams, digrams)
            features['user_id'] = row['user_id']
            features['platform_id'] = row['platform_id']
            features['session_id'] = row['session_id']
            features['video_id'] = row['video_id']
            feature_records.append(features)
        
        # Create DataFrame
        dataset = pl.DataFrame(feature_records)
        
        # Apply imputation strategy
        dataset = self.apply_imputation(dataset, imputation, level='video')
        
        # Reorder columns
        id_cols = ['user_id', 'platform_id', 'session_id', 'video_id']
        feature_cols = sorted([col for col in dataset.columns if col not in id_cols])
        dataset = dataset.select(id_cols + feature_cols)
        
        print(f"Dataset 3 shape: {dataset.shape}")
        return dataset
    
    def apply_imputation(self, dataset: pl.DataFrame, strategy: str, level: str) -> pl.DataFrame:
        """
        Apply imputation strategy for missing values
        strategy: 'global' (average over all users) or 'user' (average over user's data)
        level: 'platform', 'session', or 'video'
        """
        feature_cols = [col for col in dataset.columns if col not in ['user_id', 'platform_id', 'session_id', 'video_id']]
        
        if strategy == 'global':
            # Replace NaN with global mean
            for col in feature_cols:
                global_mean = dataset[col].mean()
                dataset = dataset.with_columns(
                    pl.col(col).fill_null(global_mean).fill_nan(global_mean)
                )
                
        elif strategy == 'user':
            # Replace NaN with user-specific mean
            for col in feature_cols:
                # First try user-level mean
                user_means = dataset.group_by('user_id').agg(
                    pl.col(col).mean().alias(f'{col}_user_mean')
                )
                
                # Join back to get user means
                dataset = dataset.join(user_means, on='user_id', how='left')
                
                # Fill nulls and nans with user mean
                dataset = dataset.with_columns(
                    pl.when(pl.col(col).is_null() | pl.col(col).is_nan())
                    .then(pl.col(f'{col}_user_mean'))
                    .otherwise(pl.col(col))
                    .alias(col)
                )
                
                # Drop the temporary user mean column
                dataset = dataset.drop(f'{col}_user_mean')
                
                # If still NaN (user has no data for this feature), use global mean
                global_mean = dataset[col].mean()
                dataset = dataset.with_columns(
                    pl.col(col).fill_null(global_mean).fill_nan(global_mean)
                )
        
        # Final check - if still any NaN (e.g., all values were NaN), fill with 0
        for col in feature_cols:
            dataset = dataset.with_columns(
                pl.col(col).fill_null(0.0).fill_nan(0.0)
            )
        
        return dataset
    
    def create_experiment_splits(self, dataset: pl.DataFrame, experiment_type: str, 
                               experiment_config: Dict) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """
        Create train/test splits based on experiment configuration
        """
        if experiment_type == 'session':
            # Session 1 vs Session 2
            train_data = dataset.filter(pl.col('session_id') == 1)
            test_data = dataset.filter(pl.col('session_id') == 2)
            
        elif experiment_type == 'platform_3c2':
            # 3-choose-2 platform experiments
            train_platforms = experiment_config['train_platforms']
            test_platform = experiment_config['test_platform']
            
            train_data = dataset.filter(pl.col('platform_id').is_in(train_platforms))
            test_data = dataset.filter(pl.col('platform_id') == test_platform)
            
        elif experiment_type == 'platform_3c1':
            # 3-choose-1 platform experiments
            train_platform = experiment_config['train_platform']
            test_platform = experiment_config['test_platform']
            
            train_data = dataset.filter(pl.col('platform_id') == train_platform)
            test_data = dataset.filter(pl.col('platform_id') == test_platform)
        
        else:
            raise ValueError(f"Unknown experiment type: {experiment_type}")
        
        return train_data, test_data
    
    def generate_all_experiments(self, output_dir: str = 'ml_experiments'):
        """
        Generate all datasets for all experiment configurations
        """
        # Create output directory structure
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Get features
        unigrams = self.get_all_unigrams()
        digrams = self.get_top_digrams(n=10)
        
        # Save feature lists for reference
        feature_info = {
            'unigrams': unigrams,
            'digrams': digrams,
            'feature_order': ['median', 'mean', 'std', 'q1', 'q3']
        }
        with open(output_path / 'feature_info.json', 'w') as f:
            json.dump(feature_info, f, indent=2)
        
        # Define all experiments
        experiments = []
        
        # # Session experiments
        # experiments.append({
        #     'name': 'session_1vs2',
        #     'type': 'session',
        #     'description': 'Session 1 (train) vs Session 2 (test)',
        #     'config': {}
        # })
        
        # # 3-choose-2 platform experiments
        # platforms = [1, 2, 3]
        # for test_platform in platforms:
        #     train_platforms = [p for p in platforms if p != test_platform]
        #     experiments.append({
        #         'name': f'platform_3c2_train{train_platforms[0]}{train_platforms[1]}_test{test_platform}',
        #         'type': 'platform_3c2',
        #         'description': f'Train on platforms {train_platforms}, test on platform {test_platform}',
        #         'config': {
        #             'train_platforms': train_platforms,
        #             'test_platform': test_platform
        #         }
        #     })
        
        # # 3-choose-1 platform experiments
        # for train_platform, test_platform in combinations(platforms, 2):
        #     experiments.append({
        #         'name': f'platform_3c1_train{train_platform}_test{test_platform}',
        #         'type': 'platform_3c1',
        #         'description': f'Train on platform {train_platform}, test on platform {test_platform}',
        #         'config': {
        #             'train_platform': train_platform,
        #             'test_platform': test_platform
        #         }
        #     })
            
            # # Also the reverse
            # experiments.append({
            #     'name': f'platform_3c1_train{test_platform}_test{train_platform}',
            #     'type': 'platform_3c1',
            #     'description': f'Train on platform {test_platform}, test on platform {train_platform}',
            #     'config': {
            #         'train_platform': test_platform,
            #         'test_platform': train_platform
            #     }
            # })
        
        # Generate datasets with both imputation strategies
        for imputation in ['global', 'user']:
            imp_dir = output_path / f'imputation_{imputation}'
            imp_dir.mkdir(exist_ok=True)
            
            print(f"\n{'='*60}")
            print(f"Generating datasets with {imputation} imputation")
            print(f"{'='*60}")
            
            # Create all three dataset types
            datasets = {
                'dataset_1': self.create_dataset_1(unigrams, digrams, imputation),
                'dataset_2': self.create_dataset_2(unigrams, digrams, imputation),
                'dataset_3': self.create_dataset_3(unigrams, digrams, imputation)
            }
            
            # Save full datasets
            for (dataset_name, dataset), levels in zip(datasets.items(), ['platform_id', 'session_id', 'video_id']):
                dataset_il_filtered = filter_IL_top_k(
                    dataset, level=levels, k=10
                )
                if self.keep_outliers:
                    dataset.write_csv(str(imp_dir / f'{dataset_name}_full_with_outliers.csv'), include_header=True)
                    dataset_il_filtered.write_csv(str(imp_dir / f'{dataset_name}_full_with_outliers_IL_filtered.csv'), include_header=True)
                    
                else:
                    dataset.write_csv(str(imp_dir / f'{dataset_name}_full_without_outliers.csv'), include_header=True)
                    dataset_il_filtered.write_csv(str(imp_dir / f'{dataset_name}_full_without_outliers_IL_filtered.csv'), include_header=True)
            
            # Generate experiment splits
            for experiment in experiments:
                print(f"\nProcessing experiment: {experiment['name']}")
                exp_dir = imp_dir / experiment['name']
                exp_dir.mkdir(exist_ok=True)
                
                # Save experiment info
                with open(exp_dir / 'experiment_info.json', 'w') as f:
                    json.dump(experiment, f, indent=2)
                
                # Generate splits for each dataset type
                for dataset_name, dataset in datasets.items():
                    # Skip dataset 1 for session experiments (no session_id column)
                    if experiment['type'] == 'session' and dataset_name == 'dataset_1':
                        print(f"  Skipping {dataset_name} - no session information at platform level")
                        continue
                    
                    try:
                        train_data, test_data = self.create_experiment_splits(
                            dataset, experiment['type'], experiment['config']
                        )
                        
                        # Save train/test splits
                        train_data.write_csv(str(exp_dir / f'{dataset_name}_train.csv'))
                        test_data.write_csv(str(exp_dir / f'{dataset_name}_test.csv'))
                        
                        print(f"  {dataset_name}: Train shape {train_data.shape}, Test shape {test_data.shape}")
                        
                    except Exception as e:
                        print(f"  Error processing {dataset_name}: {e}")
        
        # Generate summary report
        self.generate_summary_report(output_path, experiments)
        
        print(f"\n✅ All experiments generated in '{output_dir}' directory")
        print(f"Total experiments: {len(experiments)}")
        print(f"Dataset types: 3 (user-platform, user-platform-session, user-platform-session-video)")
        print(f"Imputation strategies: 2 (global, user)")
    
    def generate_summary_report(self, output_path: Path, experiments: List[Dict]):
        """Generate a summary report of all experiments"""
        
        report = f"""# TypeNet ML Experiments Summary

## Directory Structure
```
{output_path.name}/
├── feature_info.json          # List of unigrams, digrams, and feature order
├── imputation_global/         # Global mean imputation
│   ├── dataset_1_full.csv     # User-Platform level features
│   ├── dataset_2_full.csv     # User-Platform-Session level features
│   ├── dataset_3_full.csv     # User-Platform-Session-Video level features
└── imputation_user/           # User-specific mean imputation
    └── ... (same structure)
```

## Dataset Descriptions

### Dataset 1: User-Platform Level
- One feature vector per user per platform
- Aggregates all sessions and videos
- Suitable for experiments where you want platform-level representation

### Dataset 2: User-Platform-Session Level  
- Two feature vectors per user per platform (one per session)
- Aggregates all videos within each session
- Suitable for session-based experiments

### Dataset 3: User-Platform-Session-Video Level
- Six feature vectors per user per platform (3 videos × 2 sessions)
- Most granular level - no aggregation
- Suitable for fine-grained analysis

## Feature Structure
Each feature vector contains:
- Unigram features (HL - Hold Latency): 5 statistics × N unigrams
- Digram features (IL - Inter-key Latency): 5 statistics × 10 top digrams
- Statistics order: median, mean, std, q1, q3

## Experiments

"""
        
        # Group experiments by type
        session_exps = [e for e in experiments if e['type'] == 'session']
        platform_3c2_exps = [e for e in experiments if e['type'] == 'platform_3c2']
        platform_3c1_exps = [e for e in experiments if e['type'] == 'platform_3c1']
        
        report += f"### Session Experiments ({len(session_exps)})\n"
        for exp in session_exps:
            report += f"- **{exp['name']}**: {exp['description']}\n"
        
        report += f"\n### Platform 3-choose-2 Experiments ({len(platform_3c2_exps)})\n"
        for exp in platform_3c2_exps:
            report += f"- **{exp['name']}**: {exp['description']}\n"
        
        report += f"\n### Platform 3-choose-1 Experiments ({len(platform_3c1_exps)})\n"
        for exp in platform_3c1_exps:
            report += f"- **{exp['name']}**: {exp['description']}\n"
        
        report += """

## Platform Mapping
- Platform 1: Facebook
- Platform 2: Instagram  
- Platform 3: Twitter

## Usage Example
```python
import polars as pl

# Load a specific experiment
train_data = pl.read_csv('ml_experiments/imputation_global/session_1vs2/dataset_1_train.csv')
test_data = pl.read_csv('ml_experiments/imputation_global/session_1vs2/dataset_1_test.csv')

# Features start from column index 2 (after user_id and platform_id)
X_train = train_data.select(pl.all().exclude(['user_id', 'platform_id'])).to_numpy()
y_train = train_data['user_id'].to_numpy()

X_test = test_data.select(pl.all().exclude(['user_id', 'platform_id'])).to_numpy()
y_test = test_data['user_id'].to_numpy()
```

## Performance Note
This version uses Polars instead of Pandas for significantly faster processing, 
especially beneficial for large datasets with many users and features.
"""
        
        with open(output_path / 'README.md', 'w') as f:
            f.write(report)

def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset_path",
        type=str,
        default="processed_data/processed_data-2025-05-31_140105-UbuntuSungoddess/typenet_features.csv",
        help="Path to the TypeNet features CSV file",
    )
    parser.add_argument(
        "-k",
        "--keep_outliers",
        action="store_true",
        help="Keep outlier records in the dataset",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="ml_experiments",
        help="Directory to save the generated ML experiments",
    )
    return parser.parse_args()


# Main execution
if __name__ == "__main__":
    args = parse_args()
    # Initialize feature extractor
    extractor = TypeNetMLFeatureExtractor(data_path=args.dataset_path, keep_outliers=args.keep_outliers)

    # Generate all experiments
    extractor.generate_all_experiments(output_dir=args.output_dir+datetime.now().strftime("%Y-%m-%d_%H%M%S"))

    print("\n📊 Next steps:")
    print("1. Review ml_experiments/README.md for experiment details")
    print("2. Check ml_experiments/feature_info.json for feature specifications")
    print("3. Load train/test datasets for your ML experiments")
    print("4. Implement similarity search and top-k accuracy evaluation")
