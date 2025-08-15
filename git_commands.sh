#!/bin/bash

# Git commands to commit the pipeline improvements

echo "Adding pipeline changes..."

# Add the modified pipeline files
git add scripts/pipeline/download_data.py
git add scripts/pipeline/clean_data.py
git add scripts/pipeline/run_pipeline.py

# Add the new KVC features extraction stage
git add scripts/pipeline/extract_kvc_features.py

# Add utility scripts
git add scripts/fix_gstmp_files.py
git add upload_selected_users.sh

# Add documentation
git add docs/kvc_features_integration.md

# Add verification script
git add verify_kvc_output.py

echo "Files staged for commit. Creating commit..."

# Create the commit with the detailed message
git commit -F git_commit_message.txt

echo "Commit created. To push to remote repository, run:"
echo "  git push origin main"

# Show what was committed
echo ""
echo "Committed changes:"
git show --stat
