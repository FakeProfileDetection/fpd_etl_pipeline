#!/usr/bin/env python3
"""
LLM Check Results Cache
Stores and retrieves LLM check results to avoid re-processing users
"""

import hashlib
import json
import logging
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class LLMCheckCache:
    """SQLite-based cache for LLM check results"""

    def __init__(self, db_path: str = "data/llm_check_cache.db"):
        """Initialize cache with database connection"""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row  # Enable column access by name
        self.init_database()

    def init_database(self):
        """Create tables if they don't exist"""
        cursor = self.conn.cursor()

        # Main cache table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS llm_check_results (
                user_id TEXT PRIMARY KEY,
                device_type TEXT NOT NULL,
                results_json TEXT NOT NULL,
                text_hash TEXT NOT NULL,
                model_used TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        # Index for faster lookups
        cursor.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_user_device
            ON llm_check_results(user_id, device_type)
        """
        )

        # Metadata table for tracking cache statistics
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS cache_metadata (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )

        self.conn.commit()

    def get_text_hash(self, text_files: List[Tuple[str, str]]) -> str:
        """Generate hash of all text content for a user"""
        # Sort files by filename for consistent hashing
        sorted_files = sorted(text_files, key=lambda x: x[0])
        combined = "".join(f"{fname}:{content}" for fname, content in sorted_files)
        return hashlib.sha256(combined.encode()).hexdigest()

    def get_cached_results(
        self, user_id: str, device_type: str, text_files: List[Tuple[str, str]]
    ) -> Optional[List[Dict]]:
        """
        Retrieve cached results for a user if text hasn't changed

        Args:
            user_id: User identifier
            device_type: desktop or mobile
            text_files: List of (filename, content) tuples

        Returns:
            List of result dicts if cached and unchanged, None otherwise
        """
        cursor = self.conn.cursor()

        # Calculate hash of current text
        current_hash = self.get_text_hash(text_files)

        # Query cache
        cursor.execute(
            """
            SELECT results_json, text_hash
            FROM llm_check_results
            WHERE user_id = ? AND device_type = ?
        """,
            (user_id, device_type),
        )

        row = cursor.fetchone()

        if row:
            cached_hash = row["text_hash"]

            # Check if text content has changed
            if cached_hash == current_hash:
                logger.info(f"Cache hit for user {user_id} ({device_type})")
                return json.loads(row["results_json"])
            else:
                logger.info(
                    f"Cache invalidated for user {user_id} ({device_type}) - text changed"
                )
                return None

        logger.info(f"Cache miss for user {user_id} ({device_type})")
        return None

    def store_results(
        self,
        user_id: str,
        device_type: str,
        text_files: List[Tuple[str, str]],
        results: List[Dict],
        model: str = None,
    ):
        """
        Store LLM check results in cache

        Args:
            user_id: User identifier
            device_type: desktop or mobile
            text_files: List of (filename, content) tuples
            results: List of result dictionaries
            model: Model name used for processing
        """
        cursor = self.conn.cursor()

        text_hash = self.get_text_hash(text_files)
        results_json = json.dumps(results)

        # Insert or replace existing entry
        cursor.execute(
            """
            INSERT OR REPLACE INTO llm_check_results
            (user_id, device_type, results_json, text_hash, model_used, updated_at)
            VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """,
            (user_id, device_type, results_json, text_hash, model),
        )

        self.conn.commit()
        logger.info(f"Cached results for user {user_id} ({device_type})")

    def get_cache_stats(self) -> Dict:
        """Get statistics about the cache"""
        cursor = self.conn.cursor()

        stats = {}

        # Total cached users
        cursor.execute("SELECT COUNT(DISTINCT user_id) as count FROM llm_check_results")
        stats["total_users"] = cursor.fetchone()["count"]

        # Total cached entries
        cursor.execute("SELECT COUNT(*) as count FROM llm_check_results")
        stats["total_entries"] = cursor.fetchone()["count"]

        # Entries by device type
        cursor.execute(
            """
            SELECT device_type, COUNT(*) as count
            FROM llm_check_results
            GROUP BY device_type
        """
        )
        stats["by_device"] = {
            row["device_type"]: row["count"] for row in cursor.fetchall()
        }

        # Cache size
        cursor.execute(
            "SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()"
        )
        stats["cache_size_bytes"] = cursor.fetchone()["size"]

        return stats

    def clear_cache(self, older_than_days: Optional[int] = None):
        """
        Clear cache entries

        Args:
            older_than_days: Only clear entries older than this many days
        """
        cursor = self.conn.cursor()

        if older_than_days:
            cursor.execute(
                """
                DELETE FROM llm_check_results
                WHERE updated_at < datetime('now', '-' || ? || ' days')
            """,
                (older_than_days,),
            )
            logger.info(f"Cleared cache entries older than {older_than_days} days")
        else:
            cursor.execute("DELETE FROM llm_check_results")
            logger.info("Cleared all cache entries")

        self.conn.commit()

        # Vacuum to reclaim space
        self.conn.execute("VACUUM")

    def export_cache(self, output_path: str):
        """Export cache to JSON for backup"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM llm_check_results")

        rows = []
        for row in cursor.fetchall():
            rows.append(
                {
                    "user_id": row["user_id"],
                    "device_type": row["device_type"],
                    "results": json.loads(row["results_json"]),
                    "text_hash": row["text_hash"],
                    "model_used": row["model_used"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }
            )

        with open(output_path, "w") as f:
            json.dump(rows, f, indent=2)

        logger.info(f"Exported {len(rows)} cache entries to {output_path}")

    def close(self):
        """Close database connection"""
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# Example usage
if __name__ == "__main__":
    # Test the cache
    cache = LLMCheckCache()

    # Example text files for a user
    test_files = [
        ("file1.txt", "I watched Coach Carter"),
        ("file2.txt", "The movie was inspiring"),
    ]

    # Check if cached
    results = cache.get_cached_results("test_user", "desktop", test_files)
    print(f"Cached results: {results}")

    if not results:
        # Simulate processing
        fake_results = [
            {"Coach Carter": 90, "Oscars Slap": 0, "Trump-Ukraine Meeting": 0},
            {"Coach Carter": 85, "Oscars Slap": 0, "Trump-Ukraine Meeting": 0},
        ]

        # Store in cache
        cache.store_results(
            "test_user", "desktop", test_files, fake_results, "gpt-oss-20b"
        )

        # Verify it's cached
        results = cache.get_cached_results("test_user", "desktop", test_files)
        print(f"After caching: {results}")

    # Show stats
    print(f"\nCache stats: {cache.get_cache_stats()}")

    cache.close()
