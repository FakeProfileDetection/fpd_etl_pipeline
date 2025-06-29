# FPD ETL Pipeline

A robust ETL (Extract, Transform, Load) pipeline for processing keystroke dynamics data from the Fake Profile Detection (FPD) project.

## 🎯 Quick Start for Different Teams

### Data Science Team (Full Analysis)
```bash
# Process all data including mobile devices and generate comprehensive reports
python scripts/pipeline/run_pipeline.py --mode full --device-types desktop,mobile --generate-reports
```

### Research Team (Review for Paper)
```bash
# Download and process desktop data only (default)
python scripts/pipeline/run_pipeline.py --mode full
```

### Development Team
```bash
# Run specific stages with dry-run to test changes
python scripts/pipeline/run_pipeline.py --dry-run -s clean -s keypairs

# Run tests
python run_tests.py
```

## 📋 Overview

This pipeline processes keystroke dynamics data collected from a web application where users complete typing tasks across three platforms (Facebook, Instagram, Twitter). The pipeline:

1. **Downloads** raw data from Google Cloud Storage
2. **Cleans** and validates data (requires all 18 tasks per user)
3. **Extracts** keystroke timing features (Hold, Inter-key, Press, Release latencies)
4. **Generates** ML-ready feature sets at multiple aggregation levels
5. **Creates** comprehensive EDA reports and visualizations

## 🚀 Installation

### Prerequisites
- Python 3.8+
- Google Cloud SDK (for data download)
- Git

### Setup
```bash
# Clone the repository
git clone https://github.com/FakeProfileDetection/fpd_etl_pipeline.git
cd fpd_etl_pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy configuration template
cp config/.env.base config/.env.local
# Edit config/.env.local with your settings
```

## 🔧 Configuration

### Environment Variables
Create `config/.env.local` based on `.env.base`:

```bash
# Essential for cloud data access
PROJECT_ID=your-gcp-project
BUCKET_NAME=your-gcs-bucket

# Processing options
DEVICE_TYPES=desktop          # Options: desktop, mobile, or desktop,mobile
UPLOAD_ARTIFACTS=false        # Set to true for production
INCLUDE_PII=false            # Set to true to include demographics
GENERATE_REPORTS=true        # Generate EDA reports
```

## 📊 Pipeline Stages

### 1. Download Data (`download`)
Downloads raw keystroke data from Google Cloud Storage.
```bash
python scripts/pipeline/run_pipeline.py -s download
```

### 2. Clean Data (`clean`)
- Validates user data completeness (requires all 18 files)
- Separates complete vs. incomplete users
- Organizes by device type (desktop/mobile)
```bash
python scripts/pipeline/run_pipeline.py -s clean
```

### 3. Extract Keypairs (`keypairs`)
- Extracts keystroke pairs and timing features
- Calculates latencies: HL, IL, PL, RL
- Identifies data quality issues
```bash
python scripts/pipeline/run_pipeline.py -s keypairs
```

### 4. Extract Features (`features`)
- Generates statistical features per key combination
- Creates features at multiple levels: user-platform, session, video
- Handles missing data with imputation
```bash
python scripts/pipeline/run_pipeline.py -s features
```

### 5. Run EDA (`eda`)
- Generates comprehensive analysis reports
- Creates visualizations for timing distributions
- Produces data quality summaries
```bash
python scripts/pipeline/run_pipeline.py -s eda
```

## 🔄 Pipeline Modes

The pipeline supports three execution modes:

### Full Mode (`--mode full`)
- Creates a **new version ID** for tracking
- Runs all requested stages from scratch
- Best for: Initial processing or complete reprocessing
- Default stages: download, clean, keypairs, features (excludes EDA)

```bash
# Process new data completely
python scripts/pipeline/run_pipeline.py --mode full

# Include EDA
python scripts/pipeline/run_pipeline.py --mode full -s eda
```

### Incremental Mode (`--mode incr`) - Default
- Uses the **current version ID** from versions.json
- Only runs stages that haven't been completed yet
- Checks completed stages and skips them automatically
- Best for: Continuing interrupted processing or adding new stages

```bash
# Continue processing where left off
python scripts/pipeline/run_pipeline.py --mode incr

# Example: If download and clean are done, will only run keypairs and features
python scripts/pipeline/run_pipeline.py  # incr is default
```

### Force Mode (`--mode force`)
- Creates a **new version ID** with suffix `_force`
- Links to parent version for traceability
- Re-runs ALL stages regardless of completion status
- Best for: Re-processing with updated code or parameters

```bash
# Force re-run all stages (creates new version)
python scripts/pipeline/run_pipeline.py --mode force

# Force re-run specific stages only
python scripts/pipeline/run_pipeline.py --mode force -s clean -s keypairs

# Force re-run from specific parent version
python scripts/pipeline/run_pipeline.py --mode force --version-id 2025-06-29_16-14-37
```

### Mode Examples

```bash
# Scenario 1: Fresh start with new data
python scripts/pipeline/run_pipeline.py --mode full --upload-artifacts

# Scenario 2: Pipeline failed at features stage, continue from there
python scripts/pipeline/run_pipeline.py --mode incr

# Scenario 3: Updated feature extraction code, need to re-run
python scripts/pipeline/run_pipeline.py --mode force -s features

# Scenario 4: Add EDA to already processed data
python scripts/pipeline/run_pipeline.py -s eda
```

## 📁 Output Structure

**Local Artifacts:**
```
artifacts/
└── {version_id}/
    ├── raw_data/           # Downloaded data
    │   └── web_app_data/   # Raw CSV and JSON files
    ├── cleaned_data/       # Organized by user
    │   ├── desktop/
    │   │   ├── raw_data/   # Complete users
    │   │   └── broken_data/# Incomplete users
    │   └── mobile/
    ├── keypairs/           # Extracted timing features
    │   ├── keypairs.parquet
    │   └── keypairs.csv
    ├── features/           # ML-ready features
    │   ├── typenet_ml_user_platform/
    │   ├── typenet_ml_session/
    │   └── typenet_ml_video/
    ├── reports/            # EDA reports and plots
    └── etl_metadata/       # Processing logs and stats
```

**Cloud Storage Structure:**
```
gs://{bucket}/
├── uploads/                # Raw web app data (source)
├── artifacts/              # Processed pipeline outputs
│   └── {version_id}/       # Mirrors local structure
│       ├── artifact_manifest.json
│       ├── raw_data/
│       ├── keypairs/
│       └── features/
└── raw_data_from_web_app/  # Legacy tar.gz files (old pipeline)
```

## 🎯 Common Use Cases

### Running with Real GCS Data

**First-time setup:**
```bash
# Authenticate with Google Cloud
gcloud auth login
gcloud config set project fake-profile-detection-460117

# Test access
python scripts/standalone/test_gcs_access.py
```

**Download and process real data:**
```bash
# Full pipeline with GCS download
python scripts/pipeline/run_pipeline.py --mode full --upload-artifacts
```

### Team Collaboration Workflow

**Important:** The `--upload-artifacts` flag enables cloud operations (like downloading from GCS) but does NOT automatically upload results. Use the separate upload script after reviewing results.

**Team Member 1 - Process and Share Data:**
```bash
# 1. Process data with cloud download enabled
python scripts/pipeline/run_pipeline.py --mode full --upload-artifacts

# 2. Review results locally
ls artifacts/2025-06-29_16-14-37_loris-mbpcablercncom/

# 3. Upload artifacts for team access (excludes PII by default)
python scripts/standalone/upload_artifacts.py --version-id 2025-06-29_16-14-37_loris-mbpcablercncom

# 4. Commit and push versions.json to share version info
git add versions.json
git commit -m "Add processed version 2025-06-29_16-14-37"
git push
```

**Team Member 2 - Use Shared Data:**
```bash
# 1. Pull latest version info
git pull

# 2. Download all artifacts from a version
python scripts/standalone/download_artifacts.py --version-id 2025-06-29_16-14-37_loris-mbpcablercncom

# 3. Or download specific stages only
python scripts/standalone/download_artifacts.py \
    --version-id 2025-06-29_16-14-37_loris-mbpcablercncom \
    --stages keypairs features
```

**Upload Options:**
```bash
# Include PII in upload (requires explicit flag)
python scripts/standalone/upload_artifacts.py \
    --version-id {version_id} \
    --include-pii

# Upload specific stages only
python scripts/standalone/upload_artifacts.py \
    --version-id {version_id} \
    --stages features reports

# Force re-upload if artifacts already exist
python scripts/standalone/upload_artifacts.py \
    --version-id {version_id} \
    --force
```

### Running with Sample Data (No GCS Required)

**Create sample data for testing:**
```bash
# Generate 10 sample users
python scripts/standalone/create_sample_data.py --num-users 10
```

### 👥 Team-Specific Workflows

#### 🔬 Research Team

**Primary Goal**: Analyze processed data and EDA artifacts for publications

**Finding and Downloading Latest Data:**
```bash
# 1. Find the latest complete version with artifacts
python scripts/standalone/list_versions.py --uploaded-only

# 2. Download all artifacts from the latest version
python scripts/standalone/download_artifacts.py --version-id {latest_version_id}

# 3. Access comprehensive reports
open artifacts/{version_id}/reports/index.html
```

**Quick Access to Latest:**
```bash
# Create an alias in your ~/.bashrc or ~/.zshrc
alias fpd-latest='python scripts/standalone/list_versions.py --limit 1 --uploaded-only'
alias fpd-download-latest='python scripts/standalone/download_artifacts.py --version-id $(python scripts/standalone/list_versions.py --uploaded-only --json | jq -r ".[0].version_id")'
```

**What You Get:**
- Comprehensive EDA reports in `reports/` directory
- Processed features in multiple formats (CSV, Parquet)
- Data quality summaries and statistics
- Visualizations and plots

#### 📊 Data Science Team

**Primary Goal**: Perform custom analysis and develop ML models

**Getting Started with Data:**
```bash
# 1. Download latest processed data
python scripts/standalone/list_versions.py --uploaded-only
python scripts/standalone/download_artifacts.py --version-id {latest_version_id}

# 2. Load data in your notebooks
# Example: Load keypairs data
import pandas as pd
keypairs = pd.read_parquet('artifacts/{version_id}/keypairs/keypairs.parquet')
features = pd.read_parquet('artifacts/{version_id}/features/typenet_ml_user_platform/features.parquet')
```

**Working with Different Feature Sets:**
```bash
# Download only specific stages you need
python scripts/standalone/download_artifacts.py \
    --version-id {version_id} \
    --stages keypairs features
```

**Requesting Pipeline Changes:**
1. Document your requirements in a GitHub issue
2. Include sample code showing desired output format
3. Tag the development team for implementation

**Custom Analysis Setup:**
```python
# Create your analysis scripts in notebooks/
# Example: notebooks/custom_feature_analysis.ipynb

# Standard imports for FPD analysis
import pandas as pd
import numpy as np
from pathlib import Path

# Load version info
import json
with open('versions.json') as f:
    versions = json.load(f)
    latest = versions['current']

# Load artifacts
base_path = Path(f'artifacts/{latest}')
keypairs = pd.read_parquet(base_path / 'keypairs/keypairs.parquet')
```

#### 💻 Development Team

**Primary Goal**: Extend pipeline functionality and maintain codebase

**Adding New Feature Extraction Scripts:**

1. **Create Feature Definition** in `config/feature_definitions.yaml`:
```yaml
# Example: CNN-ready image features
cnn_image_features:
  description: "Image-like representation of keystroke patterns for CNN"
  module: "scripts.pipeline.features.cnn_features"
  aggregation_level: "user_platform"
  output_format: "numpy"
  parameters:
    image_width: 224
    image_height: 224
    channels: 3
```

2. **Implement Feature Extractor**:
```python
# scripts/pipeline/features/cnn_features.py
from scripts.pipeline.features.base_feature_extractor import BaseFeatureExtractor

class CNNFeatureExtractor(BaseFeatureExtractor):
    def extract(self, keypairs_df: pd.DataFrame) -> np.ndarray:
        """Convert keystroke patterns to image format"""
        # Implementation here
        pass
```

3. **Register in Pipeline**:
```python
# scripts/pipeline/extract_features.py
FEATURE_EXTRACTORS = {
    'typenet_ml': TypeNetMLExtractor,
    'cnn_image': CNNFeatureExtractor,  # Add your extractor
    'lstm_sequence': LSTMFeatureExtractor,
    'transformer_tokens': TransformerFeatureExtractor,
}
```

**Adding New EDA Tools:**

1. **Create EDA Module** in `scripts/eda/`:
```python
# scripts/eda/timing_distribution_analysis.py
def generate_timing_plots(keypairs_df: pd.DataFrame, output_dir: Path):
    """Generate timing distribution visualizations"""
    # Implementation
```

2. **Register in EDA Pipeline**:
```python
# scripts/pipeline/run_eda.py
from scripts.eda.timing_distribution_analysis import generate_timing_plots

EDA_MODULES = [
    generate_summary_stats,
    generate_timing_plots,  # Add your module
    generate_user_comparison,
]
```

**Development Best Practices:**
```bash
# 1. Create feature branch
git checkout -b feature/lstm-sequences

# 2. Use test-driven development
python -m pytest tests/pipeline/features/test_lstm_features.py -v

# 3. Test with sample data
./scripts/dev_workflow.sh test

# 4. Run full pipeline locally
./scripts/dev_workflow.sh run

# 5. Submit PR with tests and documentation
```

**Debugging Pipeline Issues:**
```bash
# Run specific stage with debug logging
python scripts/pipeline/run_pipeline.py \
    -s features \
    --log-level DEBUG \
    --version-id test_debug_$(date +%s)

# Check intermediate outputs
ls -la artifacts/test_debug_*/etl_metadata/

# Run with Python debugger
python -m pdb scripts/pipeline/run_pipeline.py -s clean
```

**Processing Text Files (.txt):**
```python
# Example: Add text feature extraction
# scripts/pipeline/extract_text_features.py

def process_user_text_files(user_dir: Path) -> pd.DataFrame:
    """Extract features from user .txt files"""
    text_files = list(user_dir.glob("*.txt"))
    
    features = []
    for txt_file in text_files:
        content = txt_file.read_text()
        
        # Extract features
        feature_dict = {
            'user_id': user_dir.name,
            'file_name': txt_file.name,
            'char_count': len(content),
            'word_count': len(content.split()),
            'avg_word_length': np.mean([len(w) for w in content.split()]),
            # Add more features
        }
        features.append(feature_dict)
    
    return pd.DataFrame(features)
```

### For Developers

**Development Workflow Helper Script:**

The `dev_workflow.sh` script provides convenient commands for common development tasks:

```bash
# Make it executable (first time only)
chmod +x scripts/dev_workflow.sh

# Run full pipeline locally (no uploads)
./scripts/dev_workflow.sh run

# Review results, then upload artifacts
./scripts/dev_workflow.sh upload

# Quick test with temporary version
./scripts/dev_workflow.sh test

# Download team member's artifacts
./scripts/dev_workflow.sh download                    # Latest version
./scripts/dev_workflow.sh download 2025-06-29_16-14  # Specific version

# Check pipeline status
./scripts/dev_workflow.sh status

# Run specific stage
./scripts/dev_workflow.sh stage clean

# Clean up local artifacts
./scripts/dev_workflow.sh clean
```

**What dev_workflow.sh does:**
- `run`: Executes full pipeline locally without uploading, shows artifact location
- `upload`: Shows what will be uploaded (dry run), then uploads after confirmation
- `test`: Creates temporary version for testing without contaminating real artifacts
- `download`: Downloads artifacts from GCS (latest or specific version)
- `status`: Shows current version, local artifacts, cloud config, and environment
- `stage`: Runs only specified stage (clean, keypairs, features, or eda)
- `clean`: Removes all local artifacts and cache (with confirmation)

**Manual testing:**
```bash
# Dry run to see what would happen
python scripts/pipeline/run_pipeline.py --dry-run --mode full

# Run specific tests
python -m pytest tests/pipeline/test_clean_data.py -v

# Run all tests
python run_tests.py
```

**Debug specific stage:**
```bash
# Run with debug logging
python scripts/pipeline/run_pipeline.py \
    -s clean \
    --log-level DEBUG
```

## 🛡️ Data Privacy

By default, the pipeline:
- **Excludes PII** (demographics, email, consent forms)
- **Processes desktop data only** (mobile users didn't follow instructions)
- **Keeps data local** (no cloud uploads)

To include PII for analysis:
```bash
python scripts/pipeline/run_pipeline.py --include-pii
```

## 📈 Data Requirements

For a user's data to be considered complete:
- Must have all **18 keystroke files** (3 platforms × 3 videos × 2 sessions)
- Must have required metadata files (consent, demographics, start_time)
- Files must follow the naming convention: `{platform}_{user_id}_{sequence}.csv`

## 🐛 Troubleshooting

### Common Issues

1. **"No complete platform data"**
   - User is missing some of the 18 required files
   - Check `cleaned_data/{device}/broken_data/` for incomplete users

2. **"Could not extract user ID"**
   - File naming doesn't match expected pattern
   - User IDs must be 32-character hex strings

3. **"No keypair data found"**
   - Previous stages haven't run successfully
   - Check that cleaned_data contains users in raw_data folders

### Logs and Debugging

- Logs are written to console and can be redirected: `python run_pipeline.py > pipeline.log 2>&1`
- Use `--log-level DEBUG` for detailed output
- Check `artifacts/{version}/etl_metadata/` for stage-specific metadata

## 🤝 Contributing

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Run tests: `python run_tests.py`
3. Ensure code quality: `flake8 scripts/ tests/`
4. Submit a pull request

## 🗂️ Version Management

### Managing Versions

The pipeline tracks all processing runs with version IDs. Use the `manage_versions.py` script to:

```bash
# Show version statistics
python scripts/standalone/manage_versions.py stats

# Clean up old versions (keeps 10 most recent by default)
python scripts/standalone/manage_versions.py cleanup

# Dry run to see what would be deleted
python scripts/standalone/manage_versions.py cleanup --dry-run

# Keep more versions
python scripts/standalone/manage_versions.py cleanup --keep-count 20

# Show details for a specific version
python scripts/standalone/manage_versions.py show 2025-06-29_16-14-37_loris-mbpcablercncom
```

### Version Storage (Planned Enhancement)

Currently using a single `versions.json` file. Future migration to directory structure:
```
versions/
├── index.json              # Current version and quick lookups
├── 2025-06-29_16-14-37.json  # Individual version files
├── 2025-06-29_17-10-25.json
└── ...
```

Benefits:
- Better performance with many versions
- Easier to manage individual versions
- Supports parallel access
- Simpler cleanup operations

## 🚀 Quick Command Reference

### For All Teams
```bash
# Find latest version with artifacts
python scripts/standalone/list_versions.py --uploaded-only

# Download latest artifacts
python scripts/standalone/download_artifacts.py --version-id {version_id}

# Check what's in a version
ls -la artifacts/{version_id}/
```

### For Research Team
```bash
# Get latest and open reports
VERSION=$(python scripts/standalone/list_versions.py --uploaded-only --json | jq -r ".[0].version_id")
python scripts/standalone/download_artifacts.py --version-id $VERSION
open artifacts/$VERSION/reports/index.html
```

### For Data Science Team
```bash
# Download specific stages for analysis
python scripts/standalone/download_artifacts.py \
    --version-id {version_id} \
    --stages keypairs features

# Quick data exploration
python -c "import pandas as pd; df=pd.read_parquet('artifacts/{version_id}/keypairs/keypairs.parquet'); print(df.info())"
```

### For Development Team
```bash
# Full development cycle
./scripts/dev_workflow.sh run       # Process locally
./scripts/dev_workflow.sh upload    # Share with team
./scripts/dev_workflow.sh status    # Check status

# Test new feature
python scripts/pipeline/run_pipeline.py \
    -s features \
    --feature-types your_new_feature \
    --version-id test_$(date +%s)
```

## 📚 Additional Documentation

- [Planning and Design](documentation/planning.md) - Original design decisions
- [Feature Architecture](docs/feature_architecture.md) - Guide for adding new features
- [Troubleshooting Guide](docs/troubleshooting.md) - Common issues and solutions
- [GCS Setup Guide](docs/gcs_setup.md) - Cloud storage configuration
- [Data Schema](documentation/schema.md) - File formats and structures

## 📄 License

This project is part of the Fake Profile Detection research. See LICENSE for details.

## 👥 Contact

For questions about:
- **Pipeline usage**: Contact the development team
- **Data analysis**: Contact the data science team  
- **Research**: Contact the principal investigators