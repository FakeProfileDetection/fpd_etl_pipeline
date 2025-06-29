# ETL/EDA Pipeline Implementation Plan
## Fake Profile Detection Research Project

**Version**: 1.0  
**Date**: January 2025  
**Team**: Data Science Research Team

---

## Executive Summary

This document outlines the implementation plan for a modular, versioned ETL/EDA pipeline for processing keystroke dynamics data collected via a web application. The pipeline is designed for academic research with requirements for reproducibility, team collaboration, and eventual public data release.

### Key Features
- **Modular pipeline** with independent stages that can be run separately or together
- **Version control** for all data processing runs with unique identifiers
- **Cloud-based artifact storage** (Google Cloud Storage) with local caching
- **Safe defaults**: No cloud uploads or PII processing without explicit flags
- **Academic research focus**: Reproducibility, documentation, and publication readiness
- **Team collaboration**: Shared artifacts, GitHub Pages reports, and version tracking

---

## Project Context

### Current Situation
- Transitioning from old dataset to new web app collected data
- Existing EDA scripts need refactoring for new pipeline
- Team of 4 members working on different aspects
- 5-hour prototype timeline for initial validation

### Technical Environment
- **Cloud**: Google Cloud Storage (GCS) with IAM-controlled access
- **Development**: Mac/Linux environments
- **Version Control**: Git/GitHub (open source)
- **Languages**: Python (primary), Bash (scripts)
- **Data Privacy**: PII contained only in demographics files

### Requirements
1. **Reproducibility**: Complete environment and parameter tracking
2. **Modularity**: Run individual stages or full pipeline
3. **Safety**: Prevent accidental PII exposure or cloud clutter
4. **Collaboration**: Easy sharing of processed data and results
5. **Development**: Local-only mode for pipeline development

---

## Architecture Overview

### Directory Structure
```
project/
├── config/
│   ├── .env.base              # Base configuration (in git)
│   ├── .env.local             # Local overrides (gitignored)
│   └── versions.json          # Version registry (in git)
├── scripts/
│   ├── pipeline/
│   │   ├── 01_download_data.py
│   │   ├── 02_clean_data.py
│   │   ├── 03_extract_keypairs.py
│   │   ├── 04_extract_features.py
│   │   ├── 05_run_eda.py
│   │   └── run_pipeline.py
│   ├── standalone/
│   │   ├── download_artifacts.py
│   │   ├── upload_artifacts.py
│   │   └── publish_reports.py
│   ├── utils/
│   │   ├── artifact_manager.py
│   │   ├── cloud_artifact_manager.py
│   │   ├── version_manager.py
│   │   ├── config_manager.py
│   │   └── safety_checks.py
│   └── dev_workflow.sh
├── features/
│   ├── registry.py
│   └── extractors/
│       ├── base.py
│       ├── typenet_ml.py
│       └── (future extractors)
├── eda/
│   ├── registry.py
│   └── reports/
│       ├── base.py
│       ├── data_quality.py
│       └── (custom reports)
├── artifacts/              # Local artifacts (gitignored)
├── docs/
│   └── reports/           # GitHub Pages reports
└── templates/
    └── reports/           # Report templates
```

### Data Flow
1. **Web App Data** (GCS) → Download → **Local Raw Data**
2. **Raw Data** → Clean → **Cleaned Data** + Metadata
3. **Cleaned Data** → Extract Keypairs → **Keypair Data**
4. **Cleaned/Keypair Data** → Extract Features → **Feature Sets**
5. **All Stages** → Generate Reports → **HTML/PDF Reports**

### Artifact Storage
- **Local**: `artifacts/{version_id}/{stage}/`
- **Cloud**: `gs://bucket/artifacts/{version_id}/{stage}/`
- **Metadata**: `versions.json` (in git) tracks all versions

---

## Implementation Strategy

### Phase 1: Core Infrastructure (Hour 1)
1. **Version Manager**
   ```python
   # utils/version_manager.py
   - create_version_id() → "{timestamp}_{hostname}"
   - get_current_version()
   - register_stage()
   - update_versions_json()
   ```

2. **Configuration System**
   ```python
   # utils/config_manager.py
   - Load .env.base and .env.local
   - Cloud configuration (bucket, paths)
   - Pipeline parameters
   ```

3. **Directory Setup Script**
   ```bash
   # scripts/setup_project.sh
   - Create directory structure
   - Install git hooks
   - Initialize configs
   ```

### Phase 2: Pipeline Stages (Hours 2-3)
1. **Download Stage** (`01_download_data.py`)
   - Download from GCS web app data
   - Create local raw data directory
   - Generate download manifest

2. **Clean Stage** (`02_clean_data.py`)
   - Implement existing cleaning logic
   - Collect cleaning metadata
   - Save outliers and validation errors

3. **Keypair Extraction** (`03_extract_keypairs.py`)
   - Adapt existing extraction code
   - Generate extraction statistics

### Phase 3: Feature System (Hour 4)
1. **Feature Registry**
   ```python
   # features/registry.py
   - Register feature extractors
   - Track which features processed per version
   - Handle dependencies
   ```

2. **Base Feature Extractor**
   ```python
   # features/extractors/base.py
   - Standard interface for all extractors
   - Input/output validation
   - Metadata collection
   ```

3. **Initial Feature Extractor**
   ```python
   # features/extractors/typenet_ml.py
   - Port existing non-DL feature extraction
   ```

### Phase 4: Integration & Testing (Hour 5)
1. **Pipeline Orchestrator**
   ```python
   # scripts/pipeline/run_pipeline.py
   - Command-line interface
   - Stage coordination
   - Safe defaults (no upload, no PII)
   ```

2. **Artifact Management**
   ```python
   # utils/cloud_artifact_manager.py
   - Upload/download artifacts
   - Update manifests
   - PII filtering
   ```

3. **Initial Testing**
   - Run on sample data
   - Verify outputs
   - Check safety features

---

## Key Design Decisions

### 1. Versioning Strategy
- **Version ID Format**: `YYYY-MM-DD_HH-MM-SS_hostname`
- **Purpose**: Unique identification, chronological ordering, source tracking
- **Storage**: Metadata in git, artifacts in cloud/local

### 2. Safe Defaults
- **No Upload**: `--upload-artifacts` flag required
- **No PII**: `--include-pii` flag required
- **Rationale**: Prevent accidental data exposure or cloud clutter

### 3. Modular Feature Extraction
- **Registry Pattern**: New features don't require pipeline changes
- **Independent Execution**: Can run only new features on old data
- **Dependency Management**: Features can depend on stages or other features

### 4. Artifact Separation
- **ETL Metadata**: Processing statistics, errors, validation results
- **EDA Reports**: Visualizations, insights, analysis
- **Storage**: Cloud for sharing, local for development

### 5. PII Handling
- **Isolation**: PII only in `*demographics*` files
- **Filtering**: Automatic exclusion in downloads/uploads
- **Future**: Clean PII before public data release

---

## Usage Examples

### Development Workflow
```bash
# 1. Run pipeline locally (no upload)
python scripts/pipeline/run_pipeline.py --mode full

# 2. Review results
ls artifacts/2025-01-15_10-30-00_macbook-dev/

# 3. Upload if satisfied
python scripts/standalone/upload_artifacts.py

# 4. Share version with team
git add versions.json
git commit -m "Add pipeline run 2025-01-15_10-30-00"
git push
```

### Team Collaboration
```bash
# Download teammate's artifacts
python scripts/standalone/download_artifacts.py \
    --version 2025-01-15_10-30-00_macbook-alice

# Run only new feature extraction
python scripts/pipeline/04_extract_features.py \
    --version-id current \
    --feature-types new_behavioral_features
```

### Production Run
```bash
# Full pipeline with uploads and reports
python scripts/pipeline/run_pipeline.py \
    --mode full \
    --upload-artifacts \
    --generate-reports \
    --publish-to-github
```

---

## Future Enhancements

### Near Term (After Prototype)
1. **Parallel Processing**: Run independent stages/features in parallel
2. **Incremental Mode**: Only process changed data
3. **Data Validation**: Schema enforcement between stages
4. **Enhanced Reports**: Interactive dashboards, statistical summaries

### Long Term
1. **Public Data Release**: Automated PII removal and packaging
2. **DOI Integration**: Mint DOIs for dataset versions
3. **Containerization**: Docker images for exact reproduction
4. **CI/CD Pipeline**: Automated testing and quality checks

---

## Safety Considerations

### Git Safety
- **.gitignore**: Excludes all data files (`*.csv`, `*.parquet`, etc.)
- **Pre-commit Hook**: Prevents accidental data commits
- **Only Metadata**: Only `versions.json` and manifests in git

### PII Protection
- **Default Exclusion**: `--include-pii` flag required
- **Pattern Matching**: `*demographics*`, `*consent*`, `*email*`
- **Cloud Isolation**: PII files require special IAM permissions

### Upload Safety
- **Confirmation Prompts**: For PII or large uploads
- **Dry Run Mode**: Preview before upload
- **Size Limits**: Configurable max file size

---

## Getting Started

### Initial Setup
```bash
# 1. Clone repository
git clone https://github.com/your-org/fake-profile-detection
cd fake-profile-detection

# 2. Set up environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Configure
cp config/.env.base config/.env.local
# Edit .env.local with your settings

# 4. Initialize
python scripts/setup_project.py

# 5. Run test
python scripts/pipeline/run_pipeline.py --mode full --dry-run
```

### Environment Variables
```bash
# Required in .env.local
PROJECT_ID="your-gcp-project"
BUCKET_DIR="your-bucket-name"
GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
```

---

## Appendix: Code Templates

### Basic Stage Template
```python
# scripts/pipeline/0X_stage_name.py
from pathlib import Path
from typing import Dict, Any
import pandas as pd
from utils.artifact_manager import ArtifactManager

def run(version_id: str, config: Dict[str, Any], 
        artifact_manager: ArtifactManager, **kwargs) -> Path:
    """Stage description"""
    
    # Setup paths
    input_dir = Path(config['PREV_STAGE_DIR']) / version_id
    output_dir = Path(config['THIS_STAGE_DIR']) / version_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process data
    # ... your logic here ...
    
    # Save artifacts
    artifact_manager.save_dataframe(
        df=result_df,
        stage='stage_name',
        name='output_data',
        description='Processed data from stage'
    )
    
    return output_dir
```

### Feature Extractor Template
```python
# features/extractors/new_feature.py
from features.extractors.base import BaseFeatureExtractor

class NewFeatureExtractor(BaseFeatureExtractor):
    """Extract new feature type"""
    
    def validate_inputs(self, input_dir: Path) -> bool:
        required_file = input_dir / "cleaned_data.parquet"
        return required_file.exists()
    
    def extract(self, input_dir: Path) -> pd.DataFrame:
        # Load input data
        data = pd.read_parquet(input_dir / "cleaned_data.parquet")
        
        # Extract features
        features = self._compute_features(data)
        
        return features
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            "user_id": "string",
            "feature_1": "float",
            # ... define expected schema
        }
```

---

## Contact for Questions

This implementation plan should provide sufficient detail to continue development. Key files to implement first:

1. `utils/version_manager.py` - Core versioning logic
2. `scripts/pipeline/run_pipeline.py` - Main orchestrator
3. `scripts/pipeline/01_download_data.py` - First stage
4. `utils/cloud_artifact_manager.py` - Cloud integration

The modular design allows implementing and testing each component independently before full integration.