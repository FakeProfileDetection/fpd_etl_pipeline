# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Fake Profile Detection (FPD) ETL/EDA Pipeline for academic research on keystroke dynamics data. The project processes web app data containing keystroke timings, with infrastructure partially implemented and data extraction pipeline fully functional.

## Quick Start

```bash
# Initial setup
python setup_project.py

# Configure environment (edit .env)
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

**Pipeline Stages** (implemented):
1. `download_data.py` - Download from GCS web app bucket
2. `clean_data.py` - Map files and validate (replaces map_new_data_to_user_dirs.py)
3. `extract_keypairs.py` - Extract keystroke pairs (replaces typenet_extraction_polars.py)
4. `extract_features.py` - Generate ML features (replaces typenet_ml_features_polars.py)
5. `run_eda.py` - Generate analysis reports (incorporates analyze_keystrokes.py logic)

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

### New Pipeline
```bash
# Full pipeline with LLM validation
python scripts/pipeline/run_pipeline.py --mode full --with-llm-check

# Full pipeline without LLM check (no API key needed)
python scripts/pipeline/run_pipeline.py --mode full

# Run specific stages
python scripts/pipeline/run_pipeline.py -s clean -s features

# Run only LLM check on existing data
python scripts/pipeline/run_pipeline.py -s llm_check --local-only --with-llm-check

# Dry run to preview
python scripts/pipeline/run_pipeline.py --mode full --dry-run

# Development workflow
./scripts/dev_workflow.sh run
./scripts/dev_workflow.sh upload
```

## Configuration

**Environment Variables** (.env):
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

# LLM Check settings (optional)
OPENAI_API_KEY=sk-your-api-key  # Required for LLM validation
LLM_CHECK_MODEL=gpt-4o-mini     # Cost-efficient model
LLM_CHECK_THRESHOLD=40           # Pass threshold (0-100)
LLM_CHECK_MAX_CONCURRENT=5       # Parallel API calls (reduce if rate limited)
LLM_CHECK_MAX_TOKENS=500         # Max tokens for response (use 2000+ for local models)
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

## Pipeline Stages

### Core Stages
1. **Download**: Download from GCS web app bucket
2. **Clean**: Map files and validate (replaces map_new_data_to_user_dirs.py)
3. **LLM Check (Optional)**: Validate user text responses with OpenAI API
4. **Extract Keypairs**: Extract keystroke pairs (replaces typenet_extraction_polars.py)
5. **KVC Features**: Generate unicode key mappings for ML
6. **Features**: Generate ML features (replaces typenet_ml_features_polars.py)
7. **Top IL**: Extract top inter-key latency features
8. **EDA**: Generate analysis reports

### LLM Check Stage (Optional)
The LLM check validates user responses to ensure they watched and engaged with videos.
This stage is **OPTIONAL** and requires an OpenAI API key.

**Running with LLM Check:**
```bash
# First time - will prompt for API key if needed
python scripts/pipeline/run_pipeline.py --mode full --with-llm-check

# Subsequent runs (API key saved in .env)
python scripts/pipeline/run_pipeline.py --mode full --with-llm-check

# CI/CD environments (non-interactive)
export OPENAI_API_KEY=sk-...
python scripts/pipeline/run_pipeline.py --mode full --with-llm-check --non-interactive

# Run only LLM check stage
python scripts/pipeline/run_pipeline.py --stages llm_check --local-only
```

**Running without LLM Check (default):**
```bash
python scripts/pipeline/run_pipeline.py --mode full
```

**Setting up API Key:**
1. Get key from https://platform.openai.com/api-keys
2. Add to .env file: `OPENAI_API_KEY=sk-...`
3. Or let the pipeline prompt you interactively

**LLM Check Outputs:**
- `scores.csv` - All text scores with pass/fail status
- `scores.json` - Detailed results with full text
- `summary_report.html` - Interactive searchable report
- `flagged_users.csv` - Users failing validation threshold

The stage validates against 3 video categories (Coach Carter, Oscars Slap, Trump-Ukraine)
and determines if users genuinely engaged with the content for MTurk payment validation.

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
