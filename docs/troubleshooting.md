# Troubleshooting Guide

## Common Issues and Solutions

### CSV Header Errors in processing_log.json

**Symptom**: Errors like `"could not convert string to float: 'Time'"` in the processing log.

**Cause**: The pipeline expects CSV files with a specific format. Raw data from the web app includes headers, while test data might not.

**Solution**: The pipeline now automatically detects headers by checking if the first line contains "Press" or "Time". If you still see these errors:

1. Check the timestamp of the error - it might be from an earlier run
2. Verify the CSV format:
   ```bash
   head -3 your_file.csv
   ```
   
**Expected formats**:
- With header: `Press or Release,Key,Time`
- Without header: `P,h,1735660002620`

### Authentication Issues with GCS

**Symptom**: "Reauthentication failed" or "Access denied" errors.

**Solution**:
1. Run `gcloud auth login`
2. Select your Google Workspace account
3. Run `python scripts/standalone/test_gcs_access.py` to verify

### Python Version Compatibility

**Symptom**: `gsutil requires Python version 3.8-3.12`

**Solution**: The pipeline now uses `gcloud storage` commands instead of `gsutil` for Python 3.13+ compatibility.

### Missing Dependencies

**Symptom**: Pipeline stages fail with "Dependencies not met"

**Cause**: The pipeline tracks which stages have completed for each version.

**Solution**: Run the missing stages or run the full pipeline:
```bash
python scripts/pipeline/run_pipeline.py --mode full
```

### No Features Extracted

**Symptom**: "Processing 0 valid keypairs" in feature extraction

**Possible causes**:
1. All keypairs marked as outliers (check outlier thresholds)
2. No valid keypairs extracted (check CSV format)
3. Wrong device type filter (check DEVICE_TYPES in config)

**Solution**: Check the keypairs data:
```python
import pandas as pd
df = pd.read_parquet('artifacts/{version_id}/keypairs/keypairs.parquet')
print(f"Valid: {df['valid'].sum()}")
print(f"Outliers: {df['outlier'].sum()}")
print(f"Device types: {df['device_type'].unique()}")
```