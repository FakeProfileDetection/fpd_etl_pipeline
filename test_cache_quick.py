#!/usr/bin/env python3
"""Quick test to verify cache integration works"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Set environment for testing
os.environ["LLM_CHECK_USE_LOCAL"] = "true"
os.environ["LLM_CHECK_USE_CACHE"] = "true"
os.environ["LLM_CHECK_MODEL"] = "gpt-oss-20b"

from scripts.pipeline.llm_cache import LLMCheckCache

# Test basic cache operations
print("Testing LLM Cache Integration...")
print("=" * 50)

cache = LLMCheckCache()

# Show current cache stats
stats = cache.get_cache_stats()
print("Current cache stats:")
print(f"  Total users: {stats['total_users']}")
print(f"  Total entries: {stats['total_entries']}")
print(f"  Cache size: {stats['cache_size_bytes'] / 1024:.1f} KB")

# Test data
test_user = "integration_test_user"
test_files = [
    ("file1.txt", "Test content about Coach Carter"),
    ("file2.txt", "More test content"),
]

# Check if cached
print(f"\nChecking cache for user '{test_user}'...")
cached = cache.get_cached_results(test_user, "desktop", test_files)

if cached:
    print(f"✓ Found cached results: {cached}")
else:
    print("✗ Not cached, storing test results...")
    test_results = [
        {"Coach Carter": 85, "Oscars Slap": 0, "Trump-Ukraine Meeting": 0},
        {"Coach Carter": 0, "Oscars Slap": 0, "Trump-Ukraine Meeting": 0},
    ]
    cache.store_results(test_user, "desktop", test_files, test_results, "test")

    # Verify it was stored
    cached = cache.get_cached_results(test_user, "desktop", test_files)
    if cached:
        print(f"✓ Successfully cached and retrieved: {cached}")
    else:
        print("✗ Cache storage failed!")

# Show updated stats
stats = cache.get_cache_stats()
print("\nUpdated cache stats:")
print(f"  Total users: {stats['total_users']}")
print(f"  Total entries: {stats['total_entries']}")

print("\n✓ Cache integration test complete!")
cache.close()
