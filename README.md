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

## 📁 Output Structure

```
artifacts/
└── {version_id}/
    ├── raw_data/           # Downloaded data
    ├── cleaned_data/       # Organized by user
    │   ├── desktop/
    │   │   ├── raw_data/   # Complete users
    │   │   └── broken_data/# Incomplete users
    │   └── mobile/
    ├── keypairs/           # Extracted timing features
    ├── features/           # ML-ready features
    └── reports/            # EDA reports and plots
```

## 🎯 Common Use Cases

### For Data Scientists

**Full pipeline with all data:**
```bash
# Process everything including mobile data
python scripts/pipeline/run_pipeline.py \
    --mode full \
    --device-types desktop,mobile \
    --generate-reports
```

**Feature extraction for specific analysis:**
```bash
# Extract features at session level only
python scripts/pipeline/extract_features.py \
    --version-id {your_version} \
    --feature-types typenet_ml_session
```

### For Researchers

**Process clean desktop data only:**
```bash
# Default behavior - desktop only, no PII
python scripts/pipeline/run_pipeline.py --mode full
```

**Download specific version for review:**
```bash
# Download previously processed data
python scripts/pipeline/download_data.py \
    --version-id 2024-12-15_10-30-00_hostname
```

### For Developers

**Test pipeline changes:**
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

## 📚 Additional Documentation

- [Planning and Design](documentation/planning.md) - Original design decisions
- [Quick Reference](QUICK_REFERENCE.md) - Command cheatsheet
- [API Documentation](documentation/api/) - Detailed function docs
- [Data Schema](documentation/schema.md) - File formats and structures

## 📄 License

This project is part of the Fake Profile Detection research. See LICENSE for details.

## 👥 Contact

For questions about:
- **Pipeline usage**: Contact the development team
- **Data analysis**: Contact the data science team  
- **Research**: Contact the principal investigators