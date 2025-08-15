#!/usr/bin/env python3
"""Verify KVC output format matches notebook"""

from pathlib import Path

import numpy as np

# Load the generated train/test files
version_id = "2025-08-10_20-08-33_loris-mbp-cable-rcn-com"
test_platform = 1

train_path = Path(
    f"artifacts/{version_id}/kvc_features/test_platform_{test_platform}/train.npy"
)
test_path = Path(
    f"artifacts/{version_id}/kvc_features/test_platform_{test_platform}/test.npy"
)

# Load the numpy files
train = np.load(train_path, allow_pickle=True).item()
test = np.load(test_path, allow_pickle=True).item()

print("Train data structure:")
print(f"  Number of users: {len(train)}")
print(f"  First user ID: {list(train.keys())[0]}")

# Check structure for first user
first_user = list(train.keys())[0]
print(f"\n  User '{first_user}' data:")
print(f"    Number of input_ids: {len(train[first_user])}")
print(f"    Input IDs: {list(train[first_user].keys())[:3]}...")

# Check data format for first input_id
first_input = list(train[first_user].keys())[0]
data_array = train[first_user][first_input]
print(f"\n  Data for input_id '{first_input}':")
print(f"    Shape: {data_array.shape}")
print(f"    Dtype: {data_array.dtype}")
print("    First 3 rows:")
for i in range(min(3, len(data_array))):
    print(f"      {data_array[i]}")

print("\nTest data structure:")
print(f"  Number of users: {len(test)}")
print(f"  Total test samples: {sum(len(user_data) for user_data in test.values())}")

# Verify the structure matches expected format
expected_format_correct = True
for user_id, user_data in train.items():
    for input_id, arr in user_data.items():
        # Check that it's a numpy array with 3 columns
        if not isinstance(arr, np.ndarray) or arr.shape[1] != 3:
            expected_format_correct = False
            print(
                f"ERROR: Unexpected format for user {user_id}, input {input_id}: {arr.shape}"
            )
            break
    if not expected_format_correct:
        break

if expected_format_correct:
    print("\n✅ Output format matches notebook implementation!")
else:
    print("\n❌ Output format does not match expected structure")
