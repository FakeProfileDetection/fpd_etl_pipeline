# Google Cloud Storage Setup Guide

## Overview
The FPD ETL Pipeline downloads data from Google Cloud Storage (GCS). This guide helps you set up authentication to access the project buckets.

## Prerequisites
- Google account with access to the FPD project (via Google Workspace)
- `gcloud` CLI installed (comes with Google Cloud SDK)

## Authentication Steps

### 1. Set up Application Default Credentials
```bash
# This will open a browser for authentication
gcloud auth application-default login

# Select your Google Workspace account when prompted
```

### 2. (Optional) Log in to gcloud CLI
```bash
# Only needed if you want to use gcloud commands directly
gcloud auth login
```

### 3. Set the Project (Optional)
```bash
# Optional: Set the FPD project as default for gcloud CLI
gcloud config set project fake-profile-detection-460117
```

### 4. Verify Access
```bash
# Test your access to the project bucket
python scripts/standalone/test_gcs_access.py
```

If successful, you should see:
```
✅ Successfully accessed bucket!
✅ All tests passed! You can run the pipeline.
```

## Running the Pipeline with GCS Data

Once authenticated, you can download and process real data:

```bash
# Download only (to inspect data first)
python scripts/pipeline/run_pipeline.py --stages download --upload-artifacts

# Full pipeline (download and process all stages)
python scripts/pipeline/run_pipeline.py --mode full --upload-artifacts

# Process specific device types
python scripts/pipeline/run_pipeline.py --mode full --upload-artifacts --device-types desktop,mobile
```

## Troubleshooting

### "Access Denied" Error
- Ensure your Google account has been granted access to the project
- Contact your project administrator to verify IAM permissions

### "Reauthentication Failed" Error
- Your credentials have expired
- Run `gcloud auth login` again

### Wrong Project Error
- Check current project: `gcloud config get-value project`
- Switch to correct project: `gcloud config set project fake-profile-detection-460117`

### Using Service Account (Alternative)
If you have a service account key file:
```bash
# Add to .env.local
echo "GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json" >> config/.env.local
```

## Data Location
- **GCS Bucket**: `fake-profile-detection-eda-bucket`
- **Upload Path**: `uploads/` (raw web app data)
- **Local Download**: `artifacts/{version_id}/raw_data/web_app_data/`

## Security Notes
- Never commit service account keys to the repository
- The `config/.env.shared` file contains project/bucket names (not secrets - these are public to team members)
- Authentication is handled via Google Cloud's Application Default Credentials
- Access is controlled by IAM permissions in Google Cloud Console
- No manual .env configuration is needed - defaults are in `config/.env.shared`