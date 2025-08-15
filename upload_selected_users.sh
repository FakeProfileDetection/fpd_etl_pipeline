#!/bin/bash

# Upload files for selected users to GCS
# Usage: ./upload_selected_users.sh

BUCKET_NAME="fake-profile-detection-eda-bucket"
SOURCE_DIR="artifacts/2025-08-10_19-25-21_loris-mbp-cable-rcn-com/raw_data/web_app_data"
TARGET_PREFIX="uploads"
USER_LIST_FILE="temp_good_users.txt/upload_to_uploads.txt"

# Check if source directory exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Source directory $SOURCE_DIR does not exist"
    exit 1
fi

# Check if user list file exists
if [ ! -f "$USER_LIST_FILE" ]; then
    echo "Error: User list file $USER_LIST_FILE does not exist"
    exit 1
fi

echo "Starting upload of selected user files to gs://$BUCKET_NAME/$TARGET_PREFIX/"
echo "Source directory: $SOURCE_DIR"
echo ""

# Read user IDs and process each one
while IFS= read -r user_id || [ -n "$user_id" ]; do
    # Skip empty lines
    if [ -z "$user_id" ]; then
        continue
    fi

    # Trim whitespace
    user_id=$(echo "$user_id" | tr -d '[:space:]')

    # Skip if still empty after trimming
    if [ -z "$user_id" ]; then
        continue
    fi

    echo "Processing user: $user_id"

    # Find all files for this user
    file_count=0
    for file in "$SOURCE_DIR"/*"${user_id}"*; do
        if [ -f "$file" ]; then
            filename=$(basename "$file")
            echo "  Uploading: $filename"

            # Upload file to GCS
            gcloud storage cp "$file" "gs://$BUCKET_NAME/$TARGET_PREFIX/$filename"

            if [ $? -eq 0 ]; then
                ((file_count++))
            else
                echo "  ERROR: Failed to upload $filename"
            fi
        fi
    done

    if [ $file_count -eq 0 ]; then
        echo "  WARNING: No files found for user $user_id"
    else
        echo "  Uploaded $file_count files for user $user_id"
    fi
    echo ""

done < "$USER_LIST_FILE"

echo "Upload complete!"
echo ""
echo "To verify uploads, run:"
echo "gcloud storage ls gs://$BUCKET_NAME/$TARGET_PREFIX/ | head -20"
