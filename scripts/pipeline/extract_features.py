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

import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime
import json
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.utils.version_manager import VersionManager

logger = logging.getLogger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature extraction"""
    name: str
    description: str
    top_n_digrams: int = 10
    use_all_unigrams: bool = True
    imputation_strategy: str = 'global'  # 'global' or 'user'
    aggregation_level: str = 'user_platform'  # 'user_platform', 'session', 'video'
    keep_outliers: bool = False


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
        self.platform_names = {1: 'facebook', 2: 'instagram', 3: 'twitter'}
        self.feature_names = []
        
    def get_top_digrams(self, data: pd.DataFrame, n: int = 10) -> List[str]:
        """Get top N most frequent digrams across dataset"""
        # Create digram column
        data['digram'] = data['key1'] + data['key2']
        
        # Get top digrams
        digram_counts = data['digram'].value_counts().head(n)
        top_digrams = digram_counts.index.tolist()
        
        logger.info(f"Top {n} digrams: {top_digrams}")
        return top_digrams
        
    def get_all_unigrams(self, data: pd.DataFrame) -> List[str]:
        """Get all unique unigrams (individual keys) in dataset"""
        # Combine key1 and key2
        all_keys = pd.concat([data['key1'], data['key2']])
        unigrams = sorted(all_keys.unique())
        
        logger.info(f"Total unique unigrams: {len(unigrams)}")
        return unigrams
        
    def extract_statistical_features(self, data: pd.DataFrame, unigrams: List[str], 
                                   digrams: List[str]) -> Dict[str, float]:
        """
        Extract statistical features for given data subset
        Returns: Dict with features in order: median, mean, std, q1, q3
        """
        features = {}
        
        # Convert timing features to milliseconds if needed
        if 'HL_ms' not in data.columns:
            data['HL_ms'] = data['HL'] / 1_000_000
            data['IL_ms'] = data['IL'] / 1_000_000
            
        # Extract unigram (HL) features
        for unigram in unigrams:
            # Filter data for this unigram
            unigram_data = data[data['key1'] == unigram]['HL_ms']
            
            if len(unigram_data) > 0:
                features[f'HL_{unigram}_median'] = float(unigram_data.median())
                features[f'HL_{unigram}_mean'] = float(unigram_data.mean())
                features[f'HL_{unigram}_std'] = float(unigram_data.std()) if len(unigram_data) > 1 else 0.0
                features[f'HL_{unigram}_q1'] = float(unigram_data.quantile(0.25))
                features[f'HL_{unigram}_q3'] = float(unigram_data.quantile(0.75))
            else:
                # Missing data - will be handled by imputation
                for stat in ['median', 'mean', 'std', 'q1', 'q3']:
                    features[f'HL_{unigram}_{stat}'] = np.nan
                    
        # Extract digram (IL) features
        for digram in digrams:
            # Filter data for this digram
            digram_data = data[data['digram'] == digram]['IL_ms']
            
            if len(digram_data) > 0:
                features[f'IL_{digram}_median'] = float(digram_data.median())
                features[f'IL_{digram}_mean'] = float(digram_data.mean())
                features[f'IL_{digram}_std'] = float(digram_data.std()) if len(digram_data) > 1 else 0.0
                features[f'IL_{digram}_q1'] = float(digram_data.quantile(0.25))
                features[f'IL_{digram}_q3'] = float(digram_data.quantile(0.75))
            else:
                # Missing data - will be handled by imputation
                for stat in ['median', 'mean', 'std', 'q1', 'q3']:
                    features[f'IL_{digram}_{stat}'] = np.nan
                    
        return features
        
    def apply_imputation(self, dataset: pd.DataFrame, strategy: str) -> pd.DataFrame:
        """Apply imputation strategy for missing values"""
        feature_cols = [col for col in dataset.columns 
                       if col not in ['user_id', 'platform_id', 'session_id', 'video_id', 'device_type']]
        
        if strategy == 'global':
            # Replace NaN with global mean
            for col in feature_cols:
                global_mean = dataset[col].mean()
                dataset[col] = dataset[col].fillna(global_mean)
                
        elif strategy == 'user':
            # Replace NaN with user-specific mean
            for col in feature_cols:
                # Calculate user means
                user_means = dataset.groupby('user_id')[col].transform('mean')
                
                # Fill with user mean
                mask = dataset[col].isna()
                dataset.loc[mask, col] = user_means[mask]
                
                # If still NaN (user has no data for this feature), use global mean
                global_mean = dataset[col].mean()
                dataset[col] = dataset[col].fillna(global_mean)
                
        # Final check - if still any NaN, fill with 0
        dataset[feature_cols] = dataset[feature_cols].fillna(0.0)
        
        return dataset
        
    def extract(self, data: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
        """Extract features based on configuration"""
        # Filter valid data
        data = data[data['valid']].copy()
        
        # Optionally remove outliers
        if not config.keep_outliers and 'outlier' in data.columns:
            data = data[~data['outlier']]
            
        logger.info(f"Processing {len(data)} valid keypairs")
        
        # Get digrams and unigrams
        digrams = self.get_top_digrams(data, config.top_n_digrams)
        unigrams = self.get_all_unigrams(data)
        
        # Store feature names for later reference
        self.feature_names = []
        for unigram in unigrams:
            for stat in ['median', 'mean', 'std', 'q1', 'q3']:
                self.feature_names.append(f'HL_{unigram}_{stat}')
        for digram in digrams:
            for stat in ['median', 'mean', 'std', 'q1', 'q3']:
                self.feature_names.append(f'IL_{digram}_{stat}')
                
        # Extract features based on aggregation level
        feature_records = []
        
        if config.aggregation_level == 'user_platform':
            # Group by user and platform
            groups = data.groupby(['user_id', 'platform_id', 'device_type'])
            
            for (user_id, platform_id, device_type), group_data in groups:
                features = self.extract_statistical_features(group_data, unigrams, digrams)
                features['user_id'] = user_id
                features['platform_id'] = platform_id
                features['device_type'] = device_type
                feature_records.append(features)
                
        elif config.aggregation_level == 'session':
            # Group by user, platform, and session
            groups = data.groupby(['user_id', 'platform_id', 'session_id', 'device_type'])
            
            for (user_id, platform_id, session_id, device_type), group_data in groups:
                features = self.extract_statistical_features(group_data, unigrams, digrams)
                features['user_id'] = user_id
                features['platform_id'] = platform_id
                features['session_id'] = session_id
                features['device_type'] = device_type
                feature_records.append(features)
                
        elif config.aggregation_level == 'video':
            # Group by user, platform, session, and video
            groups = data.groupby(['user_id', 'platform_id', 'session_id', 'video_id', 'device_type'])
            
            for (user_id, platform_id, session_id, video_id, device_type), group_data in groups:
                features = self.extract_statistical_features(group_data, unigrams, digrams)
                features['user_id'] = user_id
                features['platform_id'] = platform_id
                features['session_id'] = session_id
                features['video_id'] = video_id
                features['device_type'] = device_type
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
    
    def __init__(self, version_id: str, config: Dict[str, Any], 
                 dry_run: bool = False, local_only: bool = False,
                 version_manager: Optional[VersionManager] = None):
        self.version_id = version_id
        self.config = config
        self.dry_run = dry_run
        self.local_only = local_only
        self.version_manager = version_manager or VersionManager()
        
        # Feature extractors registry
        self.extractors = {
            'typenet_ml': TypeNetMLFeatureExtractor()
        }
        
        # Default feature configurations
        self.feature_configs = {
            'typenet_ml_user_platform': FeatureConfig(
                name='typenet_ml_user_platform',
                description='TypeNet ML features aggregated by user and platform',
                top_n_digrams=10,
                use_all_unigrams=True,
                imputation_strategy='global',
                aggregation_level='user_platform',
                keep_outliers=self.config.get('KEEP_OUTLIERS', False)
            ),
            'typenet_ml_session': FeatureConfig(
                name='typenet_ml_session',
                description='TypeNet ML features aggregated by session',
                top_n_digrams=10,
                use_all_unigrams=True,
                imputation_strategy='global',
                aggregation_level='session',
                keep_outliers=self.config.get('KEEP_OUTLIERS', False)
            ),
            'typenet_ml_video': FeatureConfig(
                name='typenet_ml_video',
                description='TypeNet ML features at video level (most granular)',
                top_n_digrams=10,
                use_all_unigrams=True,
                imputation_strategy='user',
                aggregation_level='video',
                keep_outliers=self.config.get('KEEP_OUTLIERS', False)
            )
        }
        
        # Statistics tracking
        self.stats = {
            "features_extracted": {},
            "processing_time": {},
            "feature_counts": {},
            "errors": []
        }
        
    def run(self, input_dir: Path, feature_types: Optional[List[str]] = None) -> Path:
        """Execute the extract features stage"""
        logger.info(f"Starting Extract Features stage for version {self.version_id}")
        logger.info(f"Input directory: {input_dir}")
        
        # Setup output directories
        artifacts_dir = Path(self.config.get("ARTIFACTS_DIR", "artifacts")) / self.version_id
        output_dir = artifacts_dir / "features"
        metadata_dir = artifacts_dir / "etl_metadata" / "features"
        
        # Load keypair data
        keypair_file = input_dir / "keypairs.parquet"
        if not keypair_file.exists():
            # Try CSV fallback
            keypair_file = input_dir / "keypairs.csv"
            if not keypair_file.exists():
                raise FileNotFoundError(f"No keypair data found in {input_dir}")
                
        logger.info(f"Loading keypair data from {keypair_file}")
        if keypair_file.suffix == '.parquet':
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
                extractor_name = feature_type.split('_')[0] + '_' + feature_type.split('_')[1]  # e.g., 'typenet_ml'
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
                    features_df.to_parquet(feature_subdir / "features.parquet", index=False)
                    features_df.to_csv(feature_subdir / "features.csv", index=False)
                    
                    # Save feature summary
                    feature_summary = {
                        "feature_type": feature_type,
                        "description": config.description,
                        "config": asdict(config),
                        "shape": features_df.shape,
                        "columns": features_df.columns.tolist(),
                        "feature_names": extractor.get_feature_names(),
                        "extraction_time": (datetime.now() - start_time).total_seconds()
                    }
                    
                    with open(feature_subdir / "feature_summary.json", 'w') as f:
                        json.dump(feature_summary, f, indent=2)
                        
                # Update statistics
                self.stats["features_extracted"][feature_type] = {
                    "records": len(features_df),
                    "features": len(features_df.columns) - len(['user_id', 'platform_id', 'session_id', 'video_id', 'device_type'])
                }
                self.stats["processing_time"][feature_type] = (datetime.now() - start_time).total_seconds()
                self.stats["feature_counts"][feature_type] = features_df.shape
                
                logger.info(f"  Extracted {features_df.shape[0]} records with {features_df.shape[1]} columns")
                
            except Exception as e:
                logger.error(f"Error extracting {feature_type}: {e}")
                self.stats["errors"].append(f"{feature_type}: {str(e)}")
                
        # Save extraction metadata
        if not self.dry_run:
            metadata_dir.mkdir(parents=True, exist_ok=True)
            
            # Feature registry
            feature_registry = {
                "version_id": self.version_id,
                "timestamp": datetime.now().isoformat(),
                "features_available": list(self.stats["features_extracted"].keys()),
                "feature_configs": {k: asdict(v) for k, v in self.feature_configs.items()},
                "extraction_summary": self.stats
            }
            
            with open(metadata_dir / "feature_registry.json", 'w') as f:
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
                "errors": self.stats["errors"]
            }
            
            with open(metadata_dir / "extraction_stats.json", 'w') as f:
                json.dump(extraction_stats, f, indent=2)
                
        # Log summary
        logger.info(f"Feature extraction complete:")
        logger.info(f"  Features extracted: {len(self.stats['features_extracted'])}")
        logger.info(f"  Total processing time: {sum(self.stats['processing_time'].values()):.2f} seconds")
        
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
                    "completed_at": datetime.now().isoformat()
                }
            )
            
        return output_dir


def run(version_id: str, config: Dict[str, Any], 
        dry_run: bool = False, local_only: bool = False,
        feature_types: Optional[List[str]] = None) -> Path:
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
    @click.option('--version-id', help='Version ID to use')
    @click.option('--input-dir', help='Input directory (overrides default)')
    @click.option('--feature-types', multiple=True, help='Specific feature types to extract')
    @click.option('--dry-run', is_flag=True, help='Preview without processing')
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