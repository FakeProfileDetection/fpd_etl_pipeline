# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Fake Profile Detection (FPD) ETL/EDA Pipeline for academic research on keystroke dynamics data. The project processes web app data containing keystroke timings, with infrastructure partially implemented and data extraction pipeline fully functional.

## Quick Start

```bash
# Initial setup
python setup_project.py

# Configure environment (edit config/.env.local)
# Add: PROJECT_ID, BUCKET_NAME, GOOGLE_APPLICATION_CREDENTIALS

# Run extraction pipeline on web app data
cd temp_old_scripts
./run_extraction_pipeline.sh ../example_web_app_data/web_app_single_user_data

# Run new pipeline (when implemented)
python scripts/pipeline/run_pipeline.py --mode full
```

## Data Flow

### Input Data Structure
Web app data arrives as individual files per user:
- `{user_id}_consent.json` - Consent information
- `{user_id}_demographics.json` - User demographics (PII)
- `{user_id}_start_time.json` - Session start time
- `{user_id}_completion.json` - Session completion status
- `f_{user_id}_{idx}.csv` - Free text typing data
- `i_{user_id}_{idx}.csv` - Image description typing data
- `t_{user_id}_{idx}.csv` - Transcription typing data
- `*_metadata.json` - Metadata for each typing file
- `*_raw.txt` - Raw text that was typed

### CSV Format
```csv
Press or Release,Key,Time
P,h,1735660002620
R,h,1735660002672
```

### Current Processing Pipeline (run_extraction_pipeline.sh)
1. **Download from GCS**: `gs://fake-profile-detection-eda-bucket/uploads/`
2. **Map to User Directories**: Organizes files into user folders, separates broken/complete data
3. **Extract TypeNet Features**: Processes keystroke data into features
4. **Extract ML Features**: Generates machine learning features
5. **Upload Results**: Archives and uploads to GCS

### Output Structure (after extraction)
```
raw_data-{timestamp}-{hostname}/
├── desktop/
│   ├── raw_data/       # Complete user data
│   ├── broken_data/    # Incomplete user data
│   ├── text/           # Text files
│   └── metadata/       # User lists and metadata
└── mobile/             # Same structure
```

## Key Components

### Existing Scripts (temp_old_scripts/)

**run_extraction_pipeline.sh**
- Main orchestrator for data processing
- Creates timestamped directories for versioning
- Handles GCS authentication and transfers
- Runs Python extraction scripts

**map_new_data_to_user_dirs.py**
- Maps flat web app files to user directories
- Identifies complete vs broken data
- Generates metadata about processing

**typenet_extraction_polars.py**
- Extracts keystroke pair features using Polars
- Handles hash-based user IDs
- Generates TypeNet feature set

**typenet_ml_features_polars.py**
- Creates ML features from keystroke data
- Includes outlier handling options
- Outputs feature CSV for modeling

**analyze_keystrokes.py**
- Compares JavaScript vs WASM implementations
- Detects timing issues and anomalies
- Now using WASM only, but analysis useful for data quality

### New Pipeline Structure (scripts/)

**Pipeline Stages** (to be implemented):
1. `01_download_data.py` - Download from GCS web app bucket
2. `02_clean_data.py` - Map files and validate (replaces map_new_data_to_user_dirs.py)
3. `03_extract_keypairs.py` - Extract keystroke pairs (replaces typenet_extraction_polars.py)
4. `04_extract_features.py` - Generate ML features (replaces typenet_ml_features_polars.py)
5. `05_run_eda.py` - Generate analysis reports (incorporate analyze_keystrokes.py logic)

## Development Commands

### Current Workflow
```bash
# Process web app data
cd temp_old_scripts
./run_extraction_pipeline.sh [web_app_data_dir]

# Analyze keystroke quality
python analyze_keystrokes.py file1.csv file2.csv

# Download existing data from GCS
./download_data.sh
```

### New Pipeline (when implemented)
```bash
# Full pipeline
python scripts/pipeline/run_pipeline.py --mode full

# Single stage
python scripts/pipeline/run_pipeline.py --stage clean

# Development workflow
./scripts/dev_workflow.sh run
./scripts/dev_workflow.sh upload
```

## Configuration

**Environment Variables** (config/.env.local):
```bash
# Cloud settings
PROJECT_ID=your-gcp-project
BUCKET_NAME=fake-profile-detection-eda-bucket
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json

# Safety settings
UPLOAD_ARTIFACTS=false
INCLUDE_PII=false

# PII patterns (demographics files)
PII_EXCLUDE_PATTERNS=*demographics*,*consent*,*email*
```

## Data Quality Checks

The analyze_keystrokes.py script revealed important findings:
- JavaScript implementation had timing issues (orphan releases, unreleased keys)
- WASM implementation is cleaner with no timing issues
- We're now using WASM exclusively

Include these checks in the EDA stage:
- Orphan releases (R without P)
- Double presses (P before R)
- Negative hold times
- Unreleased keys
- Consecutive key timing patterns

## Migration Guide

To port the existing pipeline to the new structure:

1. **Stage 2 (Clean)**: Use logic from `map_new_data_to_user_dirs.py`
   - Map files to user directories
   - Identify complete vs broken data
   - Generate metadata

2. **Stage 3 (Extract Keypairs)**: Use logic from `typenet_extraction_polars.py`
   - Process raw keystroke files
   - Extract keypair features
   - Handle hash-based user IDs

3. **Stage 4 (Features)**: Use logic from `typenet_ml_features_polars.py`
   - Generate ML features
   - Keep outliers by default
   - Output versioned feature files

4. **Stage 5 (EDA)**: Incorporate `analyze_keystrokes.py`
   - Data quality analysis
   - Timing pattern detection
   - Generate HTML reports

## Important Notes

1. **User IDs are MD5 hashes** (32-character hex strings)
2. **Keep outliers** in feature extraction (--keep-outliers flag)
3. **WASM data only** - JavaScript comparison no longer needed
4. **PII in demographics files** - Handle with care
5. **Versioning pattern**: `{type}-{timestamp}-{hostname}`

## Testing

```bash
# Test with example data
./run_extraction_pipeline.sh ../example_web_app_data/web_app_single_user_data

# Verify extraction results
ls raw_data-*/desktop/raw_data/
ls processed_data-*/

# Check for data quality issues
python analyze_keystrokes.py [csv_file]
```

## Related Repositories

- **FakeProfileDetection/eda**: Analysis scripts and notebooks
- **FakeProfileDetection/keystroke-scripts**: Core algorithms
- **FakeProfileDetection/web-data-collection**: Data collection web app