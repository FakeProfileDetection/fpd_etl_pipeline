# Pipeline Stages API Reference

## Stage 1: DownloadDataStage

Downloads raw data from Google Cloud Storage.

```python
from scripts.pipeline.download_data import DownloadDataStage

stage = DownloadDataStage(
    version_id="2024-12-15_10-30-00",
    config=config_dict,
    dry_run=False,
    local_only=False
)
output_dir = stage.run(source_path="uploads/")
```

### Parameters
- `version_id`: Unique identifier for this pipeline run
- `config`: Configuration dictionary
- `dry_run`: If True, shows what would be done without executing
- `local_only`: If True, skips cloud operations

### Returns
- `Path`: Output directory containing downloaded data

## Stage 2: CleanDataStage

Validates and organizes user data by completeness and device type.

```python
from scripts.pipeline.clean_data import CleanDataStage

stage = CleanDataStage(version_id, config)
output_dir = stage.run(input_dir)
```

### Key Methods
- `validate_user_files()`: Checks if user has all 18 required files
- `extract_user_id()`: Extracts 32-char hex ID from filename
- `convert_csv_filename()`: Converts web app format to TypeNet format

## Stage 3: ExtractKeypairsStage

Extracts keystroke timing features from raw data.

```python
from scripts.pipeline.extract_keypairs import ExtractKeypairsStage

stage = ExtractKeypairsStage(version_id, config)
output_dir = stage.run(cleaned_data_dir)
```

### Features Calculated
- **HL** (Hold Latency): Time key is held down
- **IL** (Inter-key Latency): Time between key1 release and key2 press
- **PL** (Press Latency): Time between consecutive presses
- **RL** (Release Latency): Time between consecutive releases

### Output Format
- `keypairs.parquet`: All keystroke pairs with timing features
- `keypairs.csv`: Same data in CSV format

## Stage 4: ExtractFeaturesStage

Generates ML-ready statistical features at multiple aggregation levels.

```python
from scripts.pipeline.extract_features import ExtractFeaturesStage

stage = ExtractFeaturesStage(version_id, config)
output_dir = stage.run(
    keypairs_dir,
    feature_types=['typenet_ml_user_platform', 'typenet_ml_session', 'typenet_ml_video']
)
```

### Feature Types
- `typenet_ml_user_platform`: Aggregated per user and platform
- `typenet_ml_session`: Aggregated per session
- `typenet_ml_video`: Aggregated per video

### Statistical Features
For each key combination and timing metric:
- mean, median, std
- min, max, range
- skew, kurtosis
- percentiles (25th, 75th)

## Stage 5: RunEDAStage

Generates exploratory data analysis reports and visualizations.

```python
from scripts.pipeline.run_eda import RunEDAStage

stage = RunEDAStage(version_id, config)
stage.run(artifacts_dir)
```

### Generated Reports
- `eda_summary.html`: Main analysis report
- `data_quality_report.json`: Quality metrics
- `timing_distributions/`: Histogram plots
- `feature_correlations.png`: Feature correlation matrix

## Utility Classes

### VersionManager
Manages pipeline versions and tracks stage completion.

```python
from scripts.utils.version_manager import VersionManager

vm = VersionManager()
version_id = vm.create_version_id()
vm.register_version(version_id, metadata)
vm.update_stage_info(version_id, "clean_data", stage_info)
```

### ConfigManager
Handles configuration from environment files.

```python
from scripts.utils.config_manager import get_config

config = get_config()
device_types = config.get_device_types()  # ['desktop'] by default
should_upload = config.get("UPLOAD_ARTIFACTS", False)
```

### DataValidator
Validates pipeline outputs at each stage.

```python
from scripts.utils.data_validator import validate_pipeline_output

validation = validate_pipeline_output(version_id, artifacts_dir)
if validation['valid']:
    print("All stages completed successfully")
```