#!/usr/bin/env python3
"""
Test script showing how LLM cache integration works
"""

import asyncio
from typing import Dict, List, Tuple

# Simulate the cache
from scripts.pipeline.llm_cache import LLMCheckCache


async def process_users_with_cache(user_data: Dict[str, List[Tuple[str, str]]]):
    """
    Demonstrate how caching would work in the pipeline

    Args:
        user_data: Dict of user_id -> list of (filename, content) tuples
    """
    cache = LLMCheckCache()
    device_type = "desktop"

    # Track statistics
    stats = {
        "total_users": len(user_data),
        "cache_hits": 0,
        "cache_misses": 0,
        "api_calls": 0,
    }

    # Results to return
    all_results = {}

    # Users that need processing
    users_to_process = []

    print(f"\n=== Processing {len(user_data)} users ===\n")

    # Check cache for each user
    for user_id, text_files in user_data.items():
        cached_results = cache.get_cached_results(user_id, device_type, text_files)

        if cached_results:
            print(f"✓ Cache HIT for user {user_id}: {len(cached_results)} scores")
            all_results[user_id] = cached_results
            stats["cache_hits"] += 1
        else:
            print(f"✗ Cache MISS for user {user_id}: need to process")
            users_to_process.append(user_id)
            stats["cache_misses"] += 1

    # Process uncached users (simulate API calls)
    if users_to_process:
        print(f"\n=== Processing {len(users_to_process)} uncached users ===\n")

        for user_id in users_to_process:
            text_files = user_data[user_id]

            # Simulate API processing
            print(f"  Processing user {user_id}...")
            results = []
            for filename, content in text_files:
                # Simulate score generation
                if "Coach Carter" in content:
                    score = {
                        "Coach Carter": 90,
                        "Oscars Slap": 0,
                        "Trump-Ukraine Meeting": 0,
                    }
                elif "Oscars" in content:
                    score = {
                        "Coach Carter": 0,
                        "Oscars Slap": 85,
                        "Trump-Ukraine Meeting": 0,
                    }
                else:
                    score = {
                        "Coach Carter": 0,
                        "Oscars Slap": 0,
                        "Trump-Ukraine Meeting": 0,
                    }

                results.append(score)
                stats["api_calls"] += 1

            # Store in cache
            cache.store_results(user_id, device_type, text_files, results, "test-model")
            all_results[user_id] = results
            print(f"  ✓ Cached results for user {user_id}")

    # Print statistics
    print("\n=== Statistics ===")
    print(f"Total users: {stats['total_users']}")
    print(
        f"Cache hits: {stats['cache_hits']} ({stats['cache_hits']/stats['total_users']*100:.1f}%)"
    )
    print(
        f"Cache misses: {stats['cache_misses']} ({stats['cache_misses']/stats['total_users']*100:.1f}%)"
    )
    print(f"API calls made: {stats['api_calls']}")
    print(f"API calls saved: {stats['cache_hits'] * 3}")  # Assuming 3 files per user

    # Show cache stats
    cache_stats = cache.get_cache_stats()
    print("\n=== Cache Database Stats ===")
    print(f"Total cached users: {cache_stats['total_users']}")
    print(f"Total cached entries: {cache_stats['total_entries']}")
    print(f"Cache size: {cache_stats['cache_size_bytes'] / 1024:.1f} KB")

    return all_results


async def main():
    """Test the caching system"""

    # Simulate user data (user_id -> list of text files)
    user_data = {
        "user_001": [
            ("f_user_001_1.txt", "I watched Coach Carter and it was inspiring"),
            ("f_user_001_2.txt", "The movie taught me about dedication"),
            ("f_user_001_3.txt", "Great film overall"),
        ],
        "user_002": [
            ("f_user_002_1.txt", "The Oscars slap was shocking"),
            ("f_user_002_2.txt", "Will Smith's actions were unexpected"),
            ("f_user_002_3.txt", "Chris Rock handled it well"),
        ],
        "user_003": [
            ("f_user_003_1.txt", "Random text about nothing specific"),
            ("f_user_003_2.txt", "Just some generic content"),
            ("f_user_003_3.txt", "No particular topic"),
        ],
    }

    print("=" * 60)
    print("FIRST RUN - All users should be cache misses")
    print("=" * 60)

    results1 = await process_users_with_cache(user_data)

    print("\n" + "=" * 60)
    print("SECOND RUN - All users should be cache hits")
    print("=" * 60)

    results2 = await process_users_with_cache(user_data)

    # Modify one user's data
    user_data["user_001"][0] = (
        "f_user_001_1.txt",
        "CHANGED: New content about Coach Carter",
    )

    print("\n" + "=" * 60)
    print("THIRD RUN - user_001 changed, should be cache miss")
    print("=" * 60)

    results3 = await process_users_with_cache(user_data)

    # Add a new user
    user_data["user_004"] = [
        ("f_user_004_1.txt", "New user talking about Coach Carter"),
        ("f_user_004_2.txt", "More content"),
    ]

    print("\n" + "=" * 60)
    print("FOURTH RUN - Added user_004, should be cache miss")
    print("=" * 60)

    results4 = await process_users_with_cache(user_data)


if __name__ == "__main__":
    asyncio.run(main())
