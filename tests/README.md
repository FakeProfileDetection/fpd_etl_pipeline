# Pipeline Tests

This directory contains tests for the FPD ETL pipeline stages.

## Test Structure

```
tests/
├── pipeline/          # Unit tests for each pipeline stage
│   ├── test_clean_data.py
│   ├── test_extract_keypairs.py
│   └── test_extract_features.py
├── test_integration.py    # Integration tests
├── test_utils.py          # Test utilities and data generators
└── README.md
```

## Running Tests

### Run all tests
```bash
python run_tests.py
```

### Run specific stage tests
```bash
python run_tests.py clean_data
python run_tests.py extract_keypairs
python run_tests.py extract_features
```

### Run tests with unittest directly
```bash
python -m unittest discover tests
python -m unittest tests.pipeline.test_clean_data
```

## Test Coverage

### Clean Data Stage (`test_clean_data.py`)
- ✅ Complete user data processing
- ✅ Incomplete user data handling
- ✅ Mixed data scenarios
- ✅ Dry run mode
- ✅ Empty input handling
- ✅ Processing summary generation

### Extract Keypairs Stage (`test_extract_keypairs.py`)
- ✅ Basic keypair extraction
- ✅ Data quality issue handling
- ✅ Timing calculation accuracy
- ✅ Outlier detection
- ✅ Dry run mode
- ✅ Empty input handling
- ✅ Metadata tracking

### Extract Features Stage (`test_extract_features.py`)
- ✅ Basic feature extraction
- ✅ Missing value imputation
- ✅ Aggregation levels
- ✅ Outlier handling
- ✅ Feature registry creation
- ✅ Specific feature type selection
- ✅ Statistical feature calculations

### Integration Tests (`test_integration.py`)
- ✅ Complete pipeline flow
- ✅ Broken data handling
- ✅ Error recovery

## Data Validation

The `scripts/utils/data_validator.py` module provides validation utilities:

### KeystrokeDataValidator
- CSV format validation
- Timing consistency checks
- Data quality issue detection

### UserDataValidator
- User data completeness checks
- Required file validation

### FeatureDataValidator
- Feature dataframe validation
- NaN and infinite value detection
- Value range checking

### Pipeline Output Validation
```python
from scripts.utils.data_validator import validate_pipeline_output

results = validate_pipeline_output(version_id, artifacts_dir)
```

## Test Utilities

The `test_utils.py` module provides:

### TestDataGenerator
- Creates realistic test keystroke data
- Generates user metadata files
- Supports data quality issue injection

### TestValidator
- Validates cleaned data structure
- Checks keypair extraction output
- Verifies feature extraction results

### Example Usage
```python
from tests.test_utils import TestDataGenerator

gen = TestDataGenerator()
gen.create_complete_user_data(output_dir, user_id)
gen.create_keystroke_csv(csv_path, include_errors=True)
```

## Adding New Tests

1. Create test file in appropriate directory
2. Import test utilities and validators
3. Subclass `unittest.TestCase`
4. Use `setUp()` and `tearDown()` for test isolation
5. Follow naming convention: `test_<functionality>`

Example:
```python
class TestNewStage(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.config = create_test_config()
        
    def tearDown(self):
        shutil.rmtree(self.test_dir)
        
    def test_basic_functionality(self):
        # Test implementation
        pass
```