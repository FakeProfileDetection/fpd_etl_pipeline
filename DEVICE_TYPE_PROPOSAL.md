# Device Type Processing Proposal

## Current Behavior
- clean_data: Sorts users into desktop/mobile based on demographics
- extract_keypairs: Processes BOTH desktop and mobile raw_data
- extract_features: Works on combined keypair data
- run_eda: Generates reports for both desktop and mobile

## Proposed Changes

### 1. Add device_types parameter to pipeline stages
Add an optional `device_types` parameter to control which device types to process:
- Default: `["desktop"]` (desktop only)
- Options: `["desktop"]`, `["mobile"]`, or `["desktop", "mobile"]`

### 2. Update affected stages:
- **extract_keypairs**: Filter device types in the loop
- **run_eda**: Filter device types for report generation

### 3. Add configuration option
Add to config.yaml:
```yaml
# Device types to process
DEVICE_TYPES: ["desktop"]  # Options: ["desktop"], ["mobile"], ["desktop", "mobile"]
```

### 4. Add CLI option to run_pipeline.py
```bash
python run_pipeline.py --device-types desktop
python run_pipeline.py --device-types mobile  
python run_pipeline.py --device-types desktop,mobile
```

## Benefits
- Default behavior focuses on desktop data (the valid data)
- Flexibility to process mobile data if needed in the future
- Backward compatible (defaults to desktop only)
- Clear configuration and documentation

## Implementation Notes
- clean_data stage still sorts ALL data (no change needed)
- Only downstream processing stages need the filter
- Config option provides easy control without code changes