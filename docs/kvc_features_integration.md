# KVC Features Integration Documentation

## Overview
The KVC (Key-Value Coded) features stage has been successfully integrated into the FPD ETL pipeline. This stage transforms keystroke data into unicode-mapped features suitable for machine learning models, exactly matching the format from the `map_unicode_new_users_10August2025.ipynb` notebook.

## What It Does
1. **Unicode Key Mapping**: Creates a dictionary mapping each unique key to an integer index (0-103)
2. **Key Encoding**: Adds `key1_mapped` and `key2_mapped` columns with integer representations
3. **Train/Test Splits**: Generates platform-based cross-validation splits for ML training
4. **Data Formatting**: Outputs numpy arrays in the exact format required by downstream ML scripts

## Output Structure
```
artifacts/{version_id}/kvc_features/
├── key_mapping.json          # Unicode to integer mapping dictionary
├── keypairs_mapped.csv       # Enhanced keypairs with mapping columns
├── metadata.json             # Stage metadata and statistics
├── test_platform_1/
│   ├── train.npy            # Training data (platforms 2,3)
│   └── test.npy             # Test data (platform 1)
├── test_platform_2/
│   ├── train.npy            # Training data (platforms 1,3)
│   └── test.npy             # Test data (platform 2)
└── test_platform_3/
    ├── train.npy            # Training data (platforms 1,2)
    └── test.npy             # Test data (platform 3)
```

## Data Format
Each `.npy` file contains a dictionary structure:
```python
{
    user_id: {
        input_id: np.ndarray  # Shape: (n_keypairs, 3)
                              # Columns: [key1_press, key1_release, key1_mapped]
    }
}
```

## Pipeline Integration

### Stage Position
- **Location**: Between `keypairs` and `features` stages
- **Dependencies**: Requires `extract_keypairs` stage output
- **CLI name**: `kvc`

### Usage Examples
```bash
# Run full pipeline including KVC
python scripts/pipeline/run_pipeline.py --mode full

# Run only KVC stage
python scripts/pipeline/run_pipeline.py --stages kvc --local-only

# Run keypairs and KVC together
python scripts/pipeline/run_pipeline.py --stages keypairs kvc

# Skip KVC for traditional workflow
python scripts/pipeline/run_pipeline.py --stages keypairs features top_il
```

## Key Features
- **Exact Format Match**: Output matches the notebook implementation precisely
- **Platform-based Splits**: Automatic generation of 3 cross-validation splits
- **Error Handling**: Filters out entries with error descriptions
- **Valid Data Only**: Processes only entries where `valid == True`
- **Metadata Tracking**: Full statistics on processing and splits

## Statistics Tracked
- Total keypairs processed
- Unique users and keys
- Train/test split sizes
- Processing time
- Unmapped keys (if any)

## Implementation Details
- **File**: `scripts/pipeline/extract_kvc_features.py`
- **Dependencies**: numpy, pandas (no additional requirements)
- **Input**: `keypairs.csv` from extract_keypairs stage
- **Processing Time**: ~0.8 seconds for 33k keypairs

## Verification
The output format has been verified to exactly match the notebook implementation:
- Numpy array structure: ✅
- Dictionary hierarchy: ✅
- Data types (int64): ✅
- Platform splits: ✅

## Notes
- Keys are converted to strings for consistent sorting
- Missing/NaN values are mapped to -1
- Only valid entries (valid=True) are included in splits
- Error entries are automatically excluded
