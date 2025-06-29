#!/bin/bash
# Development workflow helper script
# Provides convenient commands for common development tasks

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Change to project root
cd "$PROJECT_ROOT"

# Function to print colored output
print_info() {
    echo -e "${GREEN}$1${NC}"
}

print_warning() {
    echo -e "${YELLOW}$1${NC}"
}

print_error() {
    echo -e "${RED}$1${NC}"
}

# Function to get latest version
get_latest_version() {
    python -c "
from scripts.utils.version_manager import VersionManager
vm = VersionManager()
version_id = vm.get_current_version_id()
if version_id:
    print(version_id)
" 2>/dev/null || echo ""
}

# Function to run pipeline locally and review
dev_run() {
    print_info "🔧 Running pipeline in development mode (no uploads)..."
    
    # Check if virtual environment is activated
    if [ -z "$VIRTUAL_ENV" ]; then
        print_warning "⚠️  Virtual environment not activated. Activate it with:"
        print_warning "    source venv/bin/activate"
        return 1
    fi
    
    # Run pipeline
    python scripts/pipeline/run_pipeline.py \
        --mode full \
        --generate-reports \
        --log-level INFO
    
    # Get the version that was created
    VERSION=$(get_latest_version)
    
    if [ -n "$VERSION" ]; then
        print_info "✅ Pipeline complete. Version: $VERSION"
        print_info "📊 Review artifacts in: ./artifacts/$VERSION/"
        
        # Check if reports were generated
        if [ -d "./artifacts/$VERSION/unified_reports" ]; then
            print_info "📈 View reports at: file://$PROJECT_ROOT/artifacts/$VERSION/unified_reports/index.html"
        fi
    else
        print_error "❌ Failed to get version ID"
    fi
}

# Function to upload after review
dev_upload() {
    print_info "📤 Uploading artifacts after review..."
    
    VERSION=$(get_latest_version)
    if [ -z "$VERSION" ]; then
        print_error "❌ No version found. Run 'dev_run' first."
        return 1
    fi
    
    print_info "Version to upload: $VERSION"
    
    # Show what will be uploaded (dry run)
    python scripts/standalone/upload_artifacts.py --dry-run
    
    echo ""
    read -p "Proceed with upload? [y/N] " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python scripts/standalone/upload_artifacts.py
        
        # Optionally publish reports
        echo ""
        read -p "Also publish reports to GitHub Pages? [y/N] " -n 1 -r
        echo ""
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            # TODO: Implement publish_reports.py
            print_warning "Report publishing not yet implemented"
        fi
        
        # Remind to commit
        print_info "📝 Don't forget to commit and push versions.json:"
        print_info "    git add versions.json"
        print_info "    git commit -m 'Upload artifacts for $VERSION'"
        print_info "    git push"
    fi
}

# Function to run quick test without contaminating artifacts
dev_test() {
    print_info "🧪 Running quick test (temp version)..."
    
    # Create temporary version ID
    TEMP_VERSION="test_$(date +%Y%m%d_%H%M%S)_$(hostname | tr ' ' '-' | tr '[:upper:]' '[:lower:]')"
    
    print_info "Test version: $TEMP_VERSION"
    
    python scripts/pipeline/run_pipeline.py \
        --version-id "$TEMP_VERSION" \
        --stages clean \
        --local-only \
        --log-level DEBUG
    
    print_info "Test artifacts in: ./artifacts/$TEMP_VERSION"
    print_warning "Remember to clean up test artifacts when done:"
    print_warning "    rm -rf ./artifacts/$TEMP_VERSION"
}

# Function to download team member's artifacts
dev_download() {
    if [ -z "$1" ]; then
        # Download latest
        print_info "📥 Downloading latest artifacts..."
        python scripts/standalone/download_artifacts.py
    else
        # Download specific version
        print_info "📥 Downloading artifacts for version: $1"
        python scripts/standalone/download_artifacts.py --version "$1"
    fi
}

# Function to clean up local artifacts
dev_clean() {
    print_warning "⚠️  This will remove all local artifacts!"
    echo "Directories to remove:"
    echo "  - ./artifacts/"
    echo "  - ./.artifact_cache/"
    
    read -p "Are you sure? [y/N] " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf ./artifacts/
        rm -rf ./.artifact_cache/
        print_info "✅ Local artifacts cleaned"
    fi
}

# Function to show status
dev_status() {
    print_info "📊 Pipeline Status"
    echo ""
    
    # Current version
    VERSION=$(get_latest_version)
    if [ -n "$VERSION" ]; then
        echo "Current version: $VERSION"
    else
        echo "Current version: None"
    fi
    
    # Local artifacts
    if [ -d "./artifacts" ]; then
        ARTIFACT_COUNT=$(find ./artifacts -type f -name "*.parquet" -o -name "*.csv" -o -name "*.json" | wc -l)
        ARTIFACT_SIZE=$(du -sh ./artifacts 2>/dev/null | cut -f1)
        echo "Local artifacts: $ARTIFACT_COUNT files ($ARTIFACT_SIZE)"
    else
        echo "Local artifacts: None"
    fi
    
    # Check cloud config
    if grep -q "BUCKET_NAME=" config/.env.local 2>/dev/null; then
        BUCKET=$(grep "BUCKET_NAME=" config/.env.local | cut -d'=' -f2 | tr -d '"')
        echo "Cloud bucket: $BUCKET"
    else
        echo "Cloud bucket: Not configured"
    fi
    
    # Python environment
    if [ -n "$VIRTUAL_ENV" ]; then
        echo "Virtual env: $(basename $VIRTUAL_ENV) ✓"
    else
        echo "Virtual env: Not activated ✗"
    fi
}

# Function to run specific stage
dev_stage() {
    STAGE=$1
    if [ -z "$STAGE" ]; then
        print_error "❌ Please specify a stage: clean, keypairs, features, or eda"
        return 1
    fi
    
    print_info "🔧 Running stage: $STAGE"
    
    python scripts/pipeline/run_pipeline.py \
        --mode incr \
        --stages "$STAGE" \
        --log-level INFO
}

# Main menu
case "$1" in
    run)
        dev_run
        ;;
    upload)
        dev_upload
        ;;
    test)
        dev_test
        ;;
    download)
        dev_download "$2"
        ;;
    clean)
        dev_clean
        ;;
    status)
        dev_status
        ;;
    stage)
        dev_stage "$2"
        ;;
    *)
        echo "FPD ETL Pipeline - Development Workflow"
        echo ""
        echo "Usage: $0 {run|upload|test|download|clean|status|stage}"
        echo ""
        echo "Commands:"
        echo "  run       - Run full pipeline locally (no uploads)"
        echo "  upload    - Upload artifacts after review"
        echo "  test      - Quick test run with temp version"
        echo "  download  - Download artifacts (latest or specific version)"
        echo "  clean     - Remove all local artifacts"
        echo "  status    - Show current status"
        echo "  stage     - Run specific stage (clean, keypairs, features, eda)"
        echo ""
        echo "Examples:"
        echo "  $0 run                    # Run pipeline locally"
        echo "  $0 upload                 # Upload after review"
        echo "  $0 download               # Download latest"
        echo "  $0 download 2025-01-15... # Download specific version"
        echo "  $0 stage clean           # Run only cleaning stage"
        ;;
esac

