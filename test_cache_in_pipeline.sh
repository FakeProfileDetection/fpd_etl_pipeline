#!/bin/bash
# Test script to verify caching works in the pipeline

echo "============================================"
echo "Testing LLM Cache Integration in Pipeline"
echo "============================================"

# Set up test environment
export LLM_CHECK_USE_LOCAL=true
export LLM_CHECK_USE_CACHE=true

# Use a small subset of data for testing
TEST_VERSION="test_cache_$(date +%Y%m%d_%H%M%S)"

echo ""
echo "First run - should have all cache misses"
echo "----------------------------------------"
.venv/bin/python scripts/pipeline/run_pipeline.py \
    --stages llm_check \
    --local-only \
    --with-llm-check \
    --dry-run 2>&1 | grep -E "(Cache|API calls|processed)"

echo ""
echo "Second run - should have cache hits"
echo "------------------------------------"
.venv/bin/python scripts/pipeline/run_pipeline.py \
    --stages llm_check \
    --local-only \
    --with-llm-check \
    --dry-run 2>&1 | grep -E "(Cache|API calls|processed)"

echo ""
echo "============================================"
echo "Cache test complete!"
echo "============================================"
