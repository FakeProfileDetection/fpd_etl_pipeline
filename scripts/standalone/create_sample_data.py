#!/usr/bin/env python3
"""
Create Sample Data for Testing
Generates realistic sample data matching the expected web app format
"""

import json
import random
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.utils.enhanced_version_manager import (
    EnhancedVersionManager as VersionManager,
)
from scripts.utils.test_data_generator import FakeDataGenerator


def create_sample_users(num_users: int = 5) -> None:
    """Create sample user data for testing"""

    # Create a test version
    version_manager = VersionManager()
    version_id = f"sample_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    version_manager.register_version(version_id)

    # Setup output directory
    output_dir = Path("artifacts") / version_id / "raw_data" / "web_app_data"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📁 Creating sample data in: {output_dir}")

    # Initialize generator
    generator = FakeDataGenerator()

    # Generate users
    created_files = 0
    for i in range(num_users):
        # Mix of complete and incomplete users
        is_complete = i < (num_users * 0.7)  # 70% complete users

        if is_complete:
            # Complete user with all 18 keystroke files
            user_id = f"complete_user_{i:03d}" + "a" * 20  # 32 chars total
            files = generator.generate_complete_user(output_dir, user_id)
            created_files += len(files)
            print(f"✅ Created complete user: {user_id[:20]}... ({len(files)} files)")
        else:
            # Incomplete user (missing some files)
            user_id = f"incomplete_user_{i:03d}" + "b" * 16  # 32 chars total
            # Generate subset of files
            num_files = random.randint(5, 15)
            files = generator.generate_user_files(
                output_dir, user_id, num_files=num_files
            )
            created_files += len(files)
            print(f"⚠️  Created incomplete user: {user_id[:20]}... ({len(files)} files)")

    # Add some users with only metadata (no keystroke files)
    for i in range(2):
        user_id = f"metadata_only_{i:03d}" + "c" * 18  # 32 chars total
        metadata_file = output_dir / f"{user_id}.json"

        metadata = {
            "user_id": user_id,
            "collection_timestamp": datetime.now().isoformat(),
            "consent_provided": True,
            "age_group": random.choice(["18-25", "26-35", "36-45", "46+"]),
            "gender": random.choice(["male", "female", "other", "prefer_not_to_say"]),
            "education_level": random.choice(
                ["high_school", "bachelors", "masters", "phd"]
            ),
            "typing_experience": random.choice(
                ["beginner", "intermediate", "advanced", "expert"]
            ),
            "primary_language": "english",
            "device_type": random.choice(["desktop", "mobile"]),
            "keyboard_layout": "qwerty",
        }

        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        created_files += 1
        print(f"📄 Created metadata-only user: {user_id[:20]}...")

    # Create manifest
    manifest = {
        "version_id": version_id,
        "created_at": datetime.now().isoformat(),
        "total_users": num_users + 2,
        "complete_users": int(num_users * 0.7),
        "incomplete_users": num_users - int(num_users * 0.7),
        "metadata_only_users": 2,
        "total_files": created_files,
        "data_type": "sample_data",
        "generator": "create_sample_data.py",
    }

    with open(output_dir / "sample_data_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n✅ Sample data created successfully!")
    print("📊 Summary:")
    print(f"   - Total files: {created_files}")
    print(f"   - Complete users: {manifest['complete_users']}")
    print(f"   - Incomplete users: {manifest['incomplete_users']}")
    print(f"   - Metadata-only users: {manifest['metadata_only_users']}")
    print("\n🚀 To process this data, run:")
    print(
        f"   python scripts/pipeline/run_pipeline.py --version-id {version_id} --stages clean,keypairs,features,eda"
    )


def main():
    """Create sample data"""
    import argparse

    parser = argparse.ArgumentParser(description="Create sample data for testing")
    parser.add_argument(
        "--num-users",
        type=int,
        default=10,
        help="Number of users to generate (default: 10)",
    )

    args = parser.parse_args()

    print("🎲 Creating sample data for testing...")
    print("=" * 60)

    create_sample_users(args.num_users)


if __name__ == "__main__":
    main()
