# FPD ETL Pipeline - Quick Reference

## 🚀 Quick Start

```bash
# 1. Setup (first time only)
./setup.sh              # Installs uv, Python 3.12.10, and all dependencies
source activate.sh      # Activate environment

# 2. Configure
cp .env.example .env
# Edit .env with your GCS settings

# 3. Run pipeline
./scripts/dev_workflow.sh run

# 4. Upload (after review)
./scripts/dev_workflow.sh upload
```

## 📁 Project Structure

```
fpd_etl_pipeline/
├── config/              # Configuration & version tracking
│   ├── versions_successful.json
│   ├── versions_failed.json
│   └── versions/       # Individual version files
├── scripts/
│   ├── pipeline/        # Pipeline stages
│   ├── standalone/      # Utility scripts
│   └── utils/          # Core utilities
├── features/           # Feature extractors
├── eda/               # EDA reports
├── artifacts/         # Local outputs (gitignored)
├── .venv/             # Virtual environment (gitignored)
└── docs/              # Documentation
```

## 🔧 Common Commands

### Pipeline Operations
```bash
# Run full pipeline (local only)
python scripts/pipeline/run_pipeline.py --mode full

# Run specific stages
python scripts/pipeline/run_pipeline.py -s clean -s features

# Run with top-k IL features
python scripts/pipeline/run_pipeline.py -s top_il_features --top-k 20

# Run with uploads (production)
python scripts/pipeline/run_pipeline.py --mode full --upload-artifacts

# Dry run (see what would happen)
python scripts/pipeline/run_pipeline.py --mode full --dry-run
```

### Development Workflow
```bash
# Run pipeline locally
./scripts/dev_workflow.sh run

# Upload after review
./scripts/dev_workflow.sh upload

# Quick test
./scripts/dev_workflow.sh test

# Check status
./scripts/dev_workflow.sh status

# Run single stage
./scripts/dev_workflow.sh stage clean
```

### Team Collaboration
```bash
# Download latest artifacts
python scripts/standalone/download_artifacts.py

# Download specific version
python scripts/standalone/download_artifacts.py --version 2025-01-15_10-00-00_hostname

# List available artifacts
python scripts/standalone/download_artifacts.py --list-only

# Upload existing artifacts
python scripts/standalone/upload_artifacts.py
```

### Version Management
```bash
# List versions (various formats)
python scripts/standalone/version_tools.py list              # Table
python scripts/standalone/version_tools.py list --format ids # Just IDs
python scripts/standalone/version_tools.py list --format json # JSON

# Show version details
python scripts/standalone/version_tools.py show VERSION_ID

# Clean up old versions
python scripts/standalone/version_tools.py cleanup --days 7

# Development cleanup (BE CAREFUL!)
python scripts/standalone/cleanup_dev_versions.py --interactive
python scripts/standalone/purge_development_versions.py --dry-run
```

## ⚙️ Configuration

### Key Settings in `.env`
```bash
# Cloud settings
GCS_PROJECT_ID="your-gcp-project"
GCS_BUCKET_NAME="your-bucket"

# Pipeline settings
ARTIFACTS_DIR=artifacts
LOG_LEVEL=INFO
DEFAULT_TOP_K_FEATURES=10

# Safe defaults
UPLOAD_ARTIFACTS=false    # Don't upload by default
INCLUDE_PII=false        # Exclude PII by default
```

### PII Patterns (excluded by default)
- `*demographics*`
- `*consent*`
- `*email*`
- `*user_id*`

## 📊 Pipeline Stages

1. **Download** (`01_download_data.py`)
   - Downloads web app data from GCS
   - Creates raw data directory

2. **Clean** (`02_clean_data.py`)
   - Cleans and validates data
   - Removes outliers
   - Generates cleaning metadata

3. **Extract Keypairs** (`03_extract_keypairs.py`)
   - Extracts keystroke pair features
   - Creates keypair datasets

4. **Extract Features** (`04_extract_features.py`)
   - Runs registered feature extractors
   - Supports multiple feature types
   - Can run only new extractors
   - Outputs to `statistical_features/`

4a. **Extract Top-k IL Features** (`extract_top_il_features.py`)
   - Finds top k most frequent digrams
   - Extracts 5 statistical measures per digram
   - Creates k×5 feature sets

5. **Run EDA** (`05_run_eda.py`)
   - Generates analysis reports
   - Creates visualizations
   - Outputs HTML/Jupyter reports

## 🔒 Safety Features

- **No accidental uploads**: Use `--upload-artifacts` flag
- **No PII by default**: Use `--include-pii` flag
- **Git safety**: Pre-commit hooks prevent data commits
- **Confirmation prompts**: For risky operations
- **Dry run mode**: Test without side effects

## 📝 Version Management

Versions follow format: `YYYY-MM-DD_HH-MM-SS_hostname`

The enhanced version system uses separate files:
- `config/versions_successful.json` - Successful runs
- `config/versions_failed.json` - Failed runs
- `config/versions/` - Individual version details
- `config/current_version.txt` - Current version pointer

```bash
# Check current version
cat config/current_version.txt

# List all versions
python scripts/standalone/version_tools.py list

# Get version IDs for scripting
python scripts/standalone/version_tools.py list --format ids
```

## 🐛 Troubleshooting

### Common Issues

1. **"No bucket configured"**
   - Set `BUCKET_NAME` in `config/.env.local`

2. **"Cloud storage not available"**
   - Check `GOOGLE_APPLICATION_CREDENTIALS`
   - Ensure `google-cloud-storage` is installed

3. **"Version not found"**
   - Run with `--mode full` to create new version
   - Check `versions.json` for available versions

4. **Git won't let me commit**
   - Check for data files: `git status`
   - Remove with: `git reset HEAD <file>`

### Debug Mode
```bash
# Run with debug logging
python scripts/pipeline/run_pipeline.py --log-level DEBUG

# Test single stage
python scripts/pipeline/02_clean_data.py
```

## 📚 Next Steps for Implementation

1. **Implement Pipeline Stages**
   - Copy `stage_template.py` for each stage
   - Update imports in `run_pipeline.py`
   - Add your processing logic

2. **Add Feature Extractors**
   - Create in `features/extractors/`
   - Inherit from `BaseFeatureExtractor`
   - Register in feature registry

3. **Create EDA Reports**
   - Create in `eda/reports/`
   - Inherit from `BaseEDAReport`
   - Register in EDA registry

## 🔗 Useful Links

- [Implementation Plan](docs/IMPLEMENTATION_PLAN.md)
- [Google Cloud Storage Docs](https://cloud.google.com/storage/docs)
- [Pandas Documentation](https://pandas.pydata.org/docs/)

## 💡 Tips

- Always work in a virtual environment
- Run `./scripts/dev_workflow.sh status` to check setup
- Use `--dry-run` to test commands safely
- Keep `versions.json` updated in git
- Review artifacts before uploading
- Document your feature extractors

---

For questions or issues, check the implementation plan or create an issue on GitHub.
