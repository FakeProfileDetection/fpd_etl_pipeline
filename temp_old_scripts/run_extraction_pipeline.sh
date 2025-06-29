#!/usr/bin/env bash
# run_extraction_pipeline.sh
# Complete pipeline to extract raw data from web app, process it, and upload to GCS
# Usage: ./run_extraction_pipeline.sh [web_app_data_dir]
#
# Arguments:
#   web_app_data_dir - Directory containing web app data (default: ./uploads)

set -euo pipefail

# Source utility functions if available
if [[ -f "./utils.sh" ]]; then
    source "./utils.sh"
else
    # Define minimal versions if utils.sh is not available
    print_step() { echo -e "\n=== $1 ===\n"; }
    print_info() { echo "$1"; }
    print_error() { echo "ERROR: $1" >&2; }
    print_warning() { echo "WARNING: $1"; }
fi

# ============================================================================
# CONFIGURATION SECTION
# ============================================================================

# GCS Configuration
BUCKET="fake-profile-detection-eda-bucket"

# Define timestamp and hostname ONCE for the entire pipeline run
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
HOSTNAME=$(hostname)

# Command line arguments
WEB_APP_DATA_SOURCE="${1:-./uploads}"  # Default to ./uploads if not specified

# Always keep outliers (as per requirement)
KEEP_OUTLIERS="--keep-outliers"

# Directory names (all using the same timestamp)
WEB_APP_DATA_DIR="${WEB_APP_DATA_SOURCE}"
RAW_DATA_DIR="raw_data-${TIMESTAMP}-${HOSTNAME}"
PROCESSED_DATA_DIR="processed_data-${TIMESTAMP}-${HOSTNAME}"
ML_EXPERIMENTS_DIR="ml_experiments-${TIMESTAMP}-${HOSTNAME}"

# File names (all using the same timestamp)
TYPENET_FEATURES_FILE="${PROCESSED_DATA_DIR}/typenet-team-features-${TIMESTAMP}-${HOSTNAME}.csv"
ML_FEATURES_FILE="typenet-ml-features-${TIMESTAMP}-${HOSTNAME}.csv"

# Display configuration
print_info "Pipeline execution started"
print_info "Timestamp for this run: ${TIMESTAMP}"
print_info "Hostname: ${HOSTNAME}"
print_info "Web app data source: ${WEB_APP_DATA_SOURCE}"
print_info "All output files will use timestamp: ${TIMESTAMP}"
print_info "Outliers will be kept in ML features"

# ============================================================================
# MODULAR FUNCTIONS - Comment out calls in main() to skip steps
# ============================================================================

# Step 1: Authenticate with gcloud
step1_authenticate() {
    print_step "Step 1: Checking Google Cloud authentication"
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
        print_warning "Not authenticated with gcloud. Please run: gcloud auth application-default login"
        exit 1
    else
        print_info "Already authenticated with gcloud"
    fi
}

# Step 2: Download web app data from GCS
step2_download_webapp_data() {
    print_step "Step 2: Downloading unprocessed web app data from GCS"
    
    # Create directory if it doesn't exist
    mkdir -p "${WEB_APP_DATA_DIR}"
    
    print_info "Downloading from: gs://${BUCKET}/uploads/*.*"
    print_info "Destination: ${WEB_APP_DATA_DIR}/"
    
    gsutil -m cp "gs://${BUCKET}/uploads/*.*" "${WEB_APP_DATA_DIR}/" || {
        print_error "Failed to download data from GCS"
        exit 1
    }
    
    file_count=$(find "${WEB_APP_DATA_DIR}" -type f | wc -l)
    print_info "Downloaded/Found $file_count files in ${WEB_APP_DATA_DIR}"
}

# Step 3: Map web app data to user directories
step3_map_webapp_to_users() {
    print_step "Step 3: Mapping web app data to user directories"
    
    if [[ ! -d "${WEB_APP_DATA_DIR}" ]]; then
        print_error "Web app data directory not found: ${WEB_APP_DATA_DIR}"
        exit 1
    fi
    
    file_count=$(find "${WEB_APP_DATA_DIR}" -type f | wc -l)
    print_info "Processing $file_count files from ${WEB_APP_DATA_DIR}"
    
    python map_new_data_to_user_dirs.py "${WEB_APP_DATA_DIR}" "./${RAW_DATA_DIR}" || {
        print_error "Failed to map web app data"
        exit 1
    }
    
    print_info "Created raw data directory: ${RAW_DATA_DIR}"
}

# Step 4: Upload raw data to GCS
step4_upload_raw_data() {
    print_step "Step 4: Uploading raw data to GCS"
    
    if [[ ! -d "${RAW_DATA_DIR}" ]]; then
        print_error "Raw data directory not found: ${RAW_DATA_DIR}"
        exit 1
    fi
    
    tar czf "${RAW_DATA_DIR}.tar.gz" "${RAW_DATA_DIR}"
    gsutil cp "${RAW_DATA_DIR}.tar.gz" "gs://${BUCKET}/raw_data_from_web_app/"
    rm "${RAW_DATA_DIR}.tar.gz"
    
    print_info "Uploaded: gs://${BUCKET}/raw_data_from_web_app/${RAW_DATA_DIR}.tar.gz"
}

# Step 5: Extract TypeNet features
step5_extract_typenet_features() {
    print_step "Step 5: Extracting TypeNet features"
    
    mkdir -p "./${PROCESSED_DATA_DIR}"
    
    # Check both desktop/raw_data and desktop/text directories
    DESKTOP_RAW_DIR="${RAW_DATA_DIR}/desktop/raw_data"
    DESKTOP_TEXT_DIR="${RAW_DATA_DIR}/desktop/text"
    
    # Determine which directory to use
    if [[ -d "$DESKTOP_TEXT_DIR" ]] && [[ $(find "$DESKTOP_TEXT_DIR" -name "*.txt" -type f | wc -l) -gt 0 ]]; then
        # If we have text files, we need to process them first
        print_info "Found text files in ${DESKTOP_TEXT_DIR}"
        DATA_DIR="$DESKTOP_TEXT_DIR"
    elif [[ -d "$DESKTOP_RAW_DIR" ]]; then
        print_info "Using raw data directory: ${DESKTOP_RAW_DIR}"
        DATA_DIR="$DESKTOP_RAW_DIR"
    else
        print_error "No data directory found"
        exit 1
    fi
    
    # Create a temporary Python script that handles the hash-based user IDs
    cat > temp_extract.py << 'EOF'
import sys
import os
sys.path.append('.')
from typenet_extraction_polars import TypeNetFeatureExtractor
import polars as pl
from pathlib import Path

# Monkey patch the is_valid_raw_file method to handle hash-based user IDs
def is_valid_raw_file_with_hash(self, filepath: str) -> bool:
    """Check if file appears to be a raw keystroke data file based on filename pattern."""
    filename = os.path.basename(filepath)
    if not filename.endswith('.csv'):
        return False
    
    # Remove .csv extension and split by underscore
    parts = filename.replace('.csv', '').split('_')
    
    # Should have exactly 4 parts: platform, video, session, user_id
    if len(parts) != 4:
        return False
    
    # First 3 parts should be numeric, last part can be a hash
    try:
        [int(part) for part in parts[:3]]  # Check first 3 parts are numeric
        # 4th part should be a 32-character hex string (MD5 hash)
        if len(parts[3]) == 32 and all(c in '0123456789abcdef' for c in parts[3]):
            return True
    except ValueError:
        pass
    
    return False

# Apply the monkey patch
TypeNetFeatureExtractor.is_valid_raw_file = is_valid_raw_file_with_hash

# Now run the extraction
extractor = TypeNetFeatureExtractor()
extractor.process_dataset('${DATA_DIR}', '${TYPENET_FEATURES_FILE}')
EOF
        
    python temp_extract.py || {
        print_error "Failed to extract TypeNet features"
        rm -f temp_extract.py
        
        # Check if the file was created but empty
        if [[ -f "${TYPENET_FEATURES_FILE}" ]] && [[ ! -s "${TYPENET_FEATURES_FILE}" ]]; then
            print_error "TypeNet features file was created but is empty"
            rm -f "${TYPENET_FEATURES_FILE}"
        fi
        
        exit 1
    }
    rm -f temp_extract.py
    
    # Verify the file was created and has content
    if [[ -f "${TYPENET_FEATURES_FILE}" ]] && [[ -s "${TYPENET_FEATURES_FILE}" ]]; then
        line_count=$(wc -l < "${TYPENET_FEATURES_FILE}")
        print_info "Created TypeNet features file with $line_count lines: ${TYPENET_FEATURES_FILE}"
    else
        print_error "TypeNet features file was not created or is empty"
        exit 1
    fi
}

# Step 6: Upload TypeNet features to GCS
step6_upload_typenet_features() {
    print_step "Step 6: Uploading TypeNet features to GCS"
    
    if [[ ! -f "${TYPENET_FEATURES_FILE}" ]]; then
        print_error "TypeNet features file not found: ${TYPENET_FEATURES_FILE}"
        exit 1
    fi
    
    gsutil cp "${TYPENET_FEATURES_FILE}" "gs://${BUCKET}/typenet_features/"
    print_info "Uploaded: gs://${BUCKET}/typenet_features/$(basename ${TYPENET_FEATURES_FILE})"
}

# Step 7: Extract ML features
step7_extract_ml_features() {
    print_step "Step 7: Extracting ML features"
    
    if [[ ! -f "${TYPENET_FEATURES_FILE}" ]]; then
        print_error "TypeNet features file not found: ${TYPENET_FEATURES_FILE}"
        exit 1
    fi
    
    python typenet_ml_features_polars.py \
        --dataset_path "${TYPENET_FEATURES_FILE}" \
        --output_dir "${ML_EXPERIMENTS_DIR}" \
        ${KEEP_OUTLIERS} || {
        print_error "Failed to extract ML features"
        exit 1
    }
    
    print_info "Created ML experiments directory: ${ML_EXPERIMENTS_DIR}"
}

# Step 8: Upload ML features to GCS
step8_upload_ml_features() {
    print_step "Step 8: Uploading ML features to GCS"
    
    if [[ ! -d "${ML_EXPERIMENTS_DIR}" ]]; then
        print_error "ML experiments directory not found: ${ML_EXPERIMENTS_DIR}"
        exit 1
    fi
    
    # Archive the entire ml_experiments directory
    tar czf "${ML_EXPERIMENTS_DIR}.tar.gz" "${ML_EXPERIMENTS_DIR}"
    gsutil cp "${ML_EXPERIMENTS_DIR}.tar.gz" "gs://${BUCKET}/typenet_ml_features/"
    rm "${ML_EXPERIMENTS_DIR}.tar.gz"
    
    print_info "Uploaded: gs://${BUCKET}/typenet_ml_features/${ML_EXPERIMENTS_DIR}.tar.gz"
    
    # Also upload individual ML feature CSV files
    if [[ -d "${ML_EXPERIMENTS_DIR}/imputation_global" ]]; then
        # Copy and rename the main dataset files
        for dataset in dataset_1 dataset_2 dataset_3; do
            for variant in "full_without_outliers" "full_with_outliers" "full_without_outliers_IL_filtered" "full_with_outliers_IL_filtered"; do
                src_file="${ML_EXPERIMENTS_DIR}/imputation_global/${dataset}_${variant}.csv"
                if [[ -f "$src_file" ]]; then
                    dest_file="${dataset}_${variant}-${TIMESTAMP}-${HOSTNAME}.csv"
                    cp "$src_file" "${PROCESSED_DATA_DIR}/${dest_file}"
                    gsutil cp "${PROCESSED_DATA_DIR}/${dest_file}" "gs://${BUCKET}/typenet_ml_features/"
                fi
            done
        done
    fi
}

# Step 9: Generate summary report
step9_generate_summary() {
    print_step "Generating Pipeline Summary"
    
    cat > pipeline_summary_${TIMESTAMP}.txt << EOF
TypeNet Data Extraction Pipeline Summary
========================================
Timestamp: ${TIMESTAMP}
Hostname: ${HOSTNAME}
Web App Data Source: ${WEB_APP_DATA_SOURCE}
Keep Outliers: Yes (always enabled)

Local Directories Created:
- Web App Data: ${WEB_APP_DATA_DIR}/
- Raw Data: ./${RAW_DATA_DIR}/
- Processed Data: ./${PROCESSED_DATA_DIR}/
- ML Experiments: ./${ML_EXPERIMENTS_DIR}/

Files Uploaded to GCS:
- Raw Data Archive: gs://${BUCKET}/raw_data_from_web_app/${RAW_DATA_DIR}.tar.gz
- TypeNet Features: gs://${BUCKET}/typenet_features/typenet-team-features-${TIMESTAMP}-${HOSTNAME}.csv
- ML Features Archive: gs://${BUCKET}/typenet_ml_features/${ML_EXPERIMENTS_DIR}.tar.gz
- Individual ML CSVs: gs://${BUCKET}/typenet_ml_features/dataset_*-${TIMESTAMP}-${HOSTNAME}.csv

Generated: $(date)
EOF

    cat pipeline_summary_${TIMESTAMP}.txt
    cp pipeline_summary_${TIMESTAMP}.txt "${PROCESSED_DATA_DIR}/"
    
    print_info ""
    print_info "Pipeline execution completed successfully!"
    print_info "Summary saved to: pipeline_summary_${TIMESTAMP}.txt"
}

# ============================================================================
# MAIN EXECUTION - Controlled by pipeline_control.conf or manual commenting
# ============================================================================

# Load configuration if available
if [[ -f "./pipeline_control.conf" ]]; then
    print_info "Loading pipeline control configuration from pipeline_control.conf"
    source ./pipeline_control.conf
else
    # Default: run all steps
    RUN_STEP1_AUTH=true
    RUN_STEP2_DOWNLOAD=true
    RUN_STEP3_MAP_DATA=true
    RUN_STEP4_UPLOAD_RAW=true
    RUN_STEP5_EXTRACT_FEATURES=true
    RUN_STEP6_UPLOAD_FEATURES=true
    RUN_STEP7_EXTRACT_ML=true
    RUN_STEP8_UPLOAD_ML=true
    RUN_STEP9_SUMMARY=true
fi

main() {
    # Step 1: Check authentication
    [[ "$RUN_STEP1_AUTH" == "true" ]] && step1_authenticate
    
    # Step 2: Download web app data (skip if already downloaded)
    [[ "$RUN_STEP2_DOWNLOAD" == "true" ]] && step2_download_webapp_data
    
    # Step 3: Map web app data to user directories
    [[ "$RUN_STEP3_MAP_DATA" == "true" ]] && step3_map_webapp_to_users
    
    # Step 4: Upload raw data to GCS
    [[ "$RUN_STEP4_UPLOAD_RAW" == "true" ]] && step4_upload_raw_data
    
    # Step 5: Extract TypeNet features
    [[ "$RUN_STEP5_EXTRACT_FEATURES" == "true" ]] && step5_extract_typenet_features
    
    # Step 6: Upload TypeNet features to GCS
    [[ "$RUN_STEP6_UPLOAD_FEATURES" == "true" ]] && step6_upload_typenet_features
    
    # Step 7: Extract ML features
    [[ "$RUN_STEP7_EXTRACT_ML" == "true" ]] && step7_extract_ml_features
    
    # Step 8: Upload ML features to GCS
    [[ "$RUN_STEP8_UPLOAD_ML" == "true" ]] && step8_upload_ml_features
    
    # Step 9: Generate summary
    [[ "$RUN_STEP9_SUMMARY" == "true" ]] && step9_generate_summary
}

# Run the main pipeline
main