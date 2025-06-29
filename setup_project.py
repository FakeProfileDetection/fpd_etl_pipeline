#!/usr/bin/env python3
"""
Setup script for FPD ETL Pipeline
Creates directory structure, installs git hooks, and initializes configuration
"""

import os
import sys
from pathlib import Path
import json
import shutil


def create_directory_structure():
    """Create the project directory structure"""
    directories = [
        "config",
        "scripts/pipeline",
        "scripts/standalone", 
        "scripts/utils",
        "features/extractors",
        "eda/reports",
        "artifacts",
        "docs/reports",
        "templates/reports",
        ".artifact_cache"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created {dir_path}")
    
    # Create __init__.py files for Python packages
    python_packages = [
        "scripts",
        "scripts/pipeline",
        "scripts/standalone",
        "scripts/utils",
        "features",
        "features/extractors",
        "eda",
        "eda/reports"
    ]
    
    for package in python_packages:
        init_file = Path(package) / "__init__.py"
        init_file.touch()
        print(f"✓ Created {init_file}")


def create_gitignore():
    """Create comprehensive .gitignore file"""
    gitignore_content = """# Data files - NEVER commit these
*.csv
*.parquet
*.json
*.xlsx
*.h5
*.pkl
*.pickle

# Except configuration files
!config/*.json
!versions.json
!.env.base
!requirements.txt

# Local data directories
data/
artifacts/
web-app/data/
eda_reports/
.artifact_cache/

# But DO track metadata and manifests
!artifacts/**/artifact_manifest.json
!artifacts/**/metadata.json

# Temporary files
*.tmp
.DS_Store
__pycache__/
*.pyc
.pytest_cache/
.coverage
htmlcov/

# Local environment
.env.local
.env.current
venv/
env/
.venv/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
Thumbs.db
.Spotlight-V100
.Trashes

# Jupyter
.ipynb_checkpoints/
*.ipynb
!examples/*.ipynb
"""
    
    with open(".gitignore", "w") as f:
        f.write(gitignore_content)
    print("✓ Created .gitignore")


def create_git_hooks():
    """Install git pre-commit hook to prevent data commits"""
    git_dir = Path(".git")
    if not git_dir.exists():
        print("⚠️  Not a git repository. Skipping git hooks.")
        return
    
    hooks_dir = git_dir / "hooks"
    hooks_dir.mkdir(exist_ok=True)
    
    pre_commit_content = """#!/bin/bash
# Pre-commit hook to prevent data files from being committed

echo "🔍 Checking for data files..."

# Check for data file extensions
data_extensions="csv parquet json xlsx h5 pkl pickle"
for ext in $data_extensions; do
    files=$(git diff --cached --name-only | grep "\\.$ext$" | grep -v "^config/" | grep -v "versions.json" | grep -v "requirements.txt")
    if [ -n "$files" ]; then
        echo "❌ ERROR: Attempting to commit data files:"
        echo "$files"
        echo ""
        echo "Please remove these files from staging:"
        echo "  git reset HEAD <file>..."
        exit 1
    fi
done

# Check for demographics files specifically (PII risk)
demographics_files=$(git diff --cached --name-only | grep -i "demographics" | grep -v ".py$")
if [ -n "$demographics_files" ]; then
    echo "❌ ERROR: Attempting to commit demographics files (PII risk):"
    echo "$demographics_files"
    echo ""
    echo "Demographics files should never be committed to git."
    exit 1
fi

# Check for large files
large_files=$(git diff --cached --name-only | xargs -I {} find {} -size +10M 2>/dev/null)
if [ -n "$large_files" ]; then
    echo "⚠️  WARNING: Large files detected (>10MB):"
    echo "$large_files"
    echo ""
    read -p "Are you sure you want to commit these large files? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "✅ Pre-commit checks passed"
exit 0
"""
    
    pre_commit_path = hooks_dir / "pre-commit"
    with open(pre_commit_path, "w") as f:
        f.write(pre_commit_content)
    
    # Make executable
    os.chmod(pre_commit_path, 0o755)
    print("✓ Installed git pre-commit hook")


def create_env_files():
    """Create environment configuration files"""
    
    # .env.base - committed to git
    env_base_content = """# Base configuration - committed to git
# Copy to .env.local and modify for your environment

# Safe defaults for development
UPLOAD_ARTIFACTS=false
INCLUDE_PII=false
GENERATE_REPORTS=true
PUBLISH_REPORTS=false

# Cloud settings (when uploads are enabled)
PROJECT_ID="fake-profile-detection-460117"
BUCKET_NAME="fake-profile-detection-eda-bucket"

# Directory settings
WEB_APP_DATA_SOURCE="uploads"
RAW_DATA_DIR="./artifacts/{version_id}/raw_data"
CLEANED_DATA_DIR="./artifacts/{version_id}/cleaned_data"
KEYPAIRS_DIR="./artifacts/{version_id}/keypairs"
FEATURES_DIR="./artifacts/{version_id}/features"

# Artifacts configuration
ARTIFACT_RETENTION_DAYS=90
MAX_ARTIFACT_SIZE_MB=100

# PII safety patterns (comma-separated)
PII_EXCLUDE_PATTERNS="*demographics*,*consent*,*email*,*user_id*"

# Processing parameters
PARALLEL_WORKERS=4
SAMPLE_SIZE_DEV=1000
CHUNK_SIZE=10000

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
"""
    
    with open("config/.env.base", "w") as f:
        f.write(env_base_content)
    print("✓ Created config/.env.base")
    
    # Create .env.local if it doesn't exist
    local_env_path = Path("config/.env.local")
    if not local_env_path.exists():
        shutil.copy("config/.env.base", local_env_path)
        print("✓ Created config/.env.local (copy of .env.base)")
    else:
        print("✓ config/.env.local already exists, skipping")


def create_versions_json():
    """Create initial versions.json file"""
    versions_path = Path("versions.json")
    if versions_path.exists():
        print("✓ versions.json already exists, skipping")
        return
    
    initial_versions = {
        "versions": [],
        "current": None,
        "schema_version": "1.0"
    }
    
    with open(versions_path, "w") as f:
        json.dump(initial_versions, f, indent=2)
    print("✓ Created versions.json")


def create_requirements_txt():
    """Create requirements.txt with necessary dependencies"""
    requirements = """# Core dependencies
pandas>=1.5.0
numpy>=1.20.0
click>=8.0.0
python-dotenv>=0.19.0
google-cloud-storage>=2.10.0

# Data processing
pyarrow>=10.0.0  # For parquet files
openpyxl>=3.0.0  # For Excel files
h5py>=3.0.0      # For HDF5 files

# Analysis and visualization
matplotlib>=3.5.0
seaborn>=0.12.0
plotly>=5.0.0
scipy>=1.9.0
scikit-learn>=1.0.0

# EDA and reporting
jinja2>=3.0.0
jupyter>=1.0.0
nbconvert>=6.0.0

# Development and testing
pytest>=7.0.0
pytest-cov>=3.0.0
black>=22.0.0
flake8>=4.0.0
mypy>=0.990
pre-commit>=2.0.0

# Optional: PII detection
presidio-analyzer>=2.2.0
presidio-anonymizer>=2.2.0

# Documentation
markdown>=3.3.0
"""
    
    with open("requirements.txt", "w") as f:
        f.write(requirements)
    print("✓ Created requirements.txt")


def create_readme():
    """Create initial README.md"""
    readme_content = """# FPD ETL Pipeline

ETL/EDA Pipeline for Fake Profile Detection Research

## Quick Start

1. **Setup Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\\Scripts\\activate
   pip install -r requirements.txt
   ```

2. **Configure**
   ```bash
   cp config/.env.base config/.env.local
   # Edit config/.env.local with your settings
   ```

3. **Run Pipeline**
   ```bash
   # Development mode (no uploads)
   python scripts/pipeline/run_pipeline.py --mode full
   
   # With uploads (after review)
   python scripts/standalone/upload_artifacts.py
   ```

## Documentation

See `docs/IMPLEMENTATION_PLAN.md` for detailed documentation.

## Safety Notes

- Data files are NEVER committed to git
- PII is excluded by default
- Uploads require explicit flags
"""
    
    with open("README.md", "w") as f:
        f.write(readme_content)
    print("✓ Created README.md")


def main():
    """Run all setup tasks"""
    print("🚀 Setting up FPD ETL Pipeline project...")
    print("")
    
    # Check if we're in the right directory
    if not Path(".git").exists():
        print("⚠️  Warning: Not in a git repository root")
        response = input("Continue anyway? [y/N] ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Run setup tasks
    create_directory_structure()
    print("")
    
    create_gitignore()
    create_git_hooks()
    print("")
    
    create_env_files()
    create_versions_json()
    print("")
    
    create_requirements_txt()
    create_readme()
    print("")
    
    print("✅ Project setup complete!")
    print("")
    print("Next steps:")
    print("1. Create virtual environment: python -m venv venv")
    print("2. Activate it: source venv/bin/activate")
    print("3. Install dependencies: pip install -r requirements.txt")
    print("4. Configure: edit config/.env.local")
    print("5. Run pipeline: python scripts/pipeline/run_pipeline.py --help")


if __name__ == "__main__":
    main()

