#!/usr/bin/env python3
"""
Test Google Cloud Storage Access
Tests authentication and access to the FPD project buckets
"""

import subprocess
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.utils.config_manager import get_config


def test_auth():
    """Test GCP authentication status"""
    print("🔍 Checking GCP authentication...\n")
    
    # Check current account
    try:
        result = subprocess.run(
            ["gcloud", "auth", "list"],
            capture_output=True,
            text=True,
            check=True
        )
        print("Current authenticated accounts:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to check auth: {e}")
        return False
        
    # Check current project
    try:
        result = subprocess.run(
            ["gcloud", "config", "get-value", "project"],
            capture_output=True,
            text=True,
            check=True
        )
        current_project = result.stdout.strip()
        print(f"Current project: {current_project}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to check project: {e}")
        return False
        
    return True


def test_bucket_access():
    """Test access to configured bucket"""
    config = get_config()
    project_id = config.get("PROJECT_ID")
    bucket_name = config.get("BUCKET_NAME")
    
    print(f"\n🪣 Testing access to bucket: {bucket_name}")
    print(f"📁 In project: {project_id}\n")
    
    # Try to list bucket contents
    try:
        cmd = ["gcloud", "storage", "ls", f"gs://{bucket_name}/"]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Successfully accessed bucket!")
            print("\nSample contents:")
            print(result.stdout)
            return True
        else:
            print(f"❌ Failed to access bucket: {result.stderr}")
            
            # Check if it's an auth issue
            if "Reauthentication failed" in result.stderr:
                print("\n⚠️  Authentication required!")
                print("\nTo authenticate, run:")
                print(f"  gcloud auth login")
                print(f"  gcloud config set project {project_id}")
            elif "403" in result.stderr or "does not have" in result.stderr:
                print("\n⚠️  Access denied!")
                print("\nMake sure your Google account has access to the project.")
                print("Contact your project administrator for access.")
                
            return False
            
    except Exception as e:
        print(f"❌ Error testing bucket: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("Google Cloud Storage Access Test")
    print("=" * 60)
    
    # Test authentication
    if not test_auth():
        print("\n❌ Authentication test failed")
        sys.exit(1)
        
    # Test bucket access
    if not test_bucket_access():
        print("\n❌ Bucket access test failed")
        sys.exit(1)
        
    print("\n✅ All tests passed! You can run the pipeline.")
    print("\nTo download data from GCS, run:")
    print("  python scripts/pipeline/run_pipeline.py --stages download --upload-artifacts")


if __name__ == "__main__":
    main()