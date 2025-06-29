"""
Enhanced Version Management System
Uses directory-based storage for better scalability
"""

import json
import os
import socket
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
import shutil

logger = logging.getLogger(__name__)


class VersionManagerV2:
    """Enhanced version manager using directory-based storage"""
    
    def __init__(self, versions_dir: Path = Path("versions")):
        self.versions_dir = versions_dir
        self.versions_dir.mkdir(exist_ok=True)
        
        # Main index file for current version and quick lookups
        self.index_file = self.versions_dir / "index.json"
        
        # Initialize index if it doesn't exist
        if not self.index_file.exists():
            self._init_index()
    
    def _init_index(self):
        """Initialize the version index"""
        index = {
            "current": None,
            "latest_complete": None,
            "schema_version": "2.0",
            "created_at": datetime.now().isoformat(),
            "versions_count": 0
        }
        self._save_index(index)
    
    def _load_index(self) -> Dict[str, Any]:
        """Load the version index"""
        with open(self.index_file, 'r') as f:
            return json.load(f)
    
    def _save_index(self, index: Dict[str, Any]):
        """Save the version index"""
        with open(self.index_file, 'w') as f:
            json.dump(index, f, indent=2)
    
    def _get_version_file(self, version_id: str) -> Path:
        """Get the file path for a specific version"""
        return self.versions_dir / f"{version_id}.json"
    
    def create_version_id(self) -> str:
        """Create a new version ID"""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        hostname = socket.gethostname().replace(' ', '-').replace('.', '-').lower()
        return f"{timestamp}_{hostname}"
    
    def register_version(self, version_id: str, metadata: Dict[str, Any]) -> bool:
        """Register a new version"""
        version_file = self._get_version_file(version_id)
        
        if version_file.exists():
            logger.warning(f"Version {version_id} already exists")
            return False
        
        # Create version data
        version_data = {
            "version_id": version_id,
            "created_at": datetime.now().isoformat(),
            "created_by": socket.gethostname(),
            "user": os.getenv("USER", "unknown"),
            "status": "in_progress",
            "stages": {},
            "features": {},
            "metadata": metadata or {},
            "parent_version": metadata.get("parent_version"),  # For force mode
            "is_force_rerun": metadata.get("is_force_rerun", False)
        }
        
        # Save version file
        with open(version_file, 'w') as f:
            json.dump(version_data, f, indent=2)
        
        # Update index
        index = self._load_index()
        index["current"] = version_id
        index["versions_count"] += 1
        self._save_index(index)
        
        logger.info(f"Registered version: {version_id}")
        return True
    
    def get_version(self, version_id: str) -> Optional[Dict[str, Any]]:
        """Get version information"""
        version_file = self._get_version_file(version_id)
        
        if not version_file.exists():
            return None
        
        with open(version_file, 'r') as f:
            return json.load(f)
    
    def update_version_info(self, version_id: str, updates: Dict[str, Any]):
        """Update version information"""
        version_data = self.get_version(version_id)
        if not version_data:
            logger.error(f"Version {version_id} not found")
            return False
        
        # Update the data
        for key, value in updates.items():
            if key != "stages":  # Don't overwrite stages
                version_data[key] = value
        
        # Save updated version
        version_file = self._get_version_file(version_id)
        with open(version_file, 'w') as f:
            json.dump(version_data, f, indent=2)
        
        return True
    
    def update_stage_info(self, version_id: str, stage_name: str, stage_info: Dict[str, Any]):
        """Update stage information for a version"""
        version_data = self.get_version(version_id)
        if not version_data:
            logger.error(f"Version {version_id} not found")
            return False
        
        # Update stage info
        if "stages" not in version_data:
            version_data["stages"] = {}
        
        version_data["stages"][stage_name] = {
            **stage_info,
            "completed_at": datetime.now().isoformat()
        }
        
        # Save updated version
        version_file = self._get_version_file(version_id)
        with open(version_file, 'w') as f:
            json.dump(version_data, f, indent=2)
        
        logger.info(f"Updated stage '{stage_name}' for version {version_id}")
        return True
    
    def mark_version_complete(self, version_id: str, summary: Dict[str, Any] = None):
        """Mark a version as complete"""
        version_data = self.get_version(version_id)
        if not version_data:
            return False
        
        version_data["status"] = "complete"
        version_data["completed_at"] = datetime.now().isoformat()
        if summary:
            version_data["summary"] = summary
        
        # Save updated version
        version_file = self._get_version_file(version_id)
        with open(version_file, 'w') as f:
            json.dump(version_data, f, indent=2)
        
        # Update index if this is a complete version with artifacts
        if summary and summary.get("artifacts_uploaded"):
            index = self._load_index()
            index["latest_complete"] = version_id
            self._save_index(index)
        
        logger.info(f"Marked version {version_id} as complete")
        return True
    
    def get_current_version_id(self) -> Optional[str]:
        """Get the current version ID"""
        index = self._load_index()
        return index.get("current")
    
    def list_versions(self, limit: int = 10, complete_only: bool = False) -> List[Dict[str, Any]]:
        """List versions with filtering options"""
        version_files = sorted(self.versions_dir.glob("*.json"), reverse=True)
        versions = []
        
        for version_file in version_files:
            if version_file.name == "index.json":
                continue
                
            try:
                with open(version_file, 'r') as f:
                    version_data = json.load(f)
                
                if complete_only and version_data.get("status") != "complete":
                    continue
                
                versions.append(version_data)
                
                if len(versions) >= limit:
                    break
                    
            except Exception as e:
                logger.warning(f"Error reading version file {version_file}: {e}")
        
        return versions
    
    def delete_version(self, version_id: str, force: bool = False) -> bool:
        """Delete a version and its artifacts"""
        version_data = self.get_version(version_id)
        if not version_data:
            logger.error(f"Version {version_id} not found")
            return False
        
        # Check if it's safe to delete
        if not force:
            if version_data.get("status") == "complete" and version_data.get("summary", {}).get("artifacts_uploaded"):
                logger.error(f"Cannot delete uploaded version {version_id} without --force")
                return False
        
        # Delete version file
        version_file = self._get_version_file(version_id)
        version_file.unlink()
        
        # Delete artifacts directory if it exists
        artifacts_dir = Path("artifacts") / version_id
        if artifacts_dir.exists():
            shutil.rmtree(artifacts_dir)
            logger.info(f"Deleted artifacts directory: {artifacts_dir}")
        
        # Update index
        index = self._load_index()
        if index.get("current") == version_id:
            # Find a new current version
            remaining_versions = self.list_versions(limit=1)
            index["current"] = remaining_versions[0]["version_id"] if remaining_versions else None
        
        if index.get("latest_complete") == version_id:
            # Find new latest complete
            complete_versions = self.list_versions(limit=1, complete_only=True)
            index["latest_complete"] = complete_versions[0]["version_id"] if complete_versions else None
        
        index["versions_count"] -= 1
        self._save_index(index)
        
        logger.info(f"Deleted version: {version_id}")
        return True
    
    def cleanup_old_versions(self, keep_count: int = 10, dry_run: bool = False) -> List[str]:
        """Clean up old test versions"""
        all_versions = self.list_versions(limit=1000)
        
        # Separate by type
        test_versions = []
        complete_uploaded = []
        complete_local = []
        in_progress = []
        
        for v in all_versions:
            if "test" in v.get("version_id", "") or v.get("metadata", {}).get("test"):
                test_versions.append(v)
            elif v.get("status") == "complete":
                if v.get("summary", {}).get("artifacts_uploaded"):
                    complete_uploaded.append(v)
                else:
                    complete_local.append(v)
            else:
                in_progress.append(v)
        
        deleted = []
        
        # Always delete test versions older than 1 day
        for v in test_versions:
            created = datetime.fromisoformat(v["created_at"].replace('Z', '+00:00'))
            if (datetime.now() - created).days > 1:
                if not dry_run:
                    self.delete_version(v["version_id"], force=True)
                deleted.append(v["version_id"])
        
        # Keep only recent local complete versions
        if len(complete_local) > keep_count:
            for v in complete_local[keep_count:]:
                if not dry_run:
                    self.delete_version(v["version_id"], force=True)
                deleted.append(v["version_id"])
        
        # Clean up old in-progress versions (older than 7 days)
        for v in in_progress:
            created = datetime.fromisoformat(v["created_at"].replace('Z', '+00:00'))
            if (datetime.now() - created).days > 7:
                if not dry_run:
                    self.delete_version(v["version_id"], force=True)
                deleted.append(v["version_id"])
        
        return deleted
    
    def create_force_version(self, parent_version_id: str) -> str:
        """Create a new version for force mode that tracks its parent"""
        parent_data = self.get_version(parent_version_id)
        if not parent_data:
            raise ValueError(f"Parent version {parent_version_id} not found")
        
        # Create new version with force suffix
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        hostname = socket.gethostname().replace(' ', '-').replace('.', '-').lower()
        new_version_id = f"{timestamp}_{hostname}_force"
        
        # Register with parent tracking
        metadata = {
            "mode": "force",
            "parent_version": parent_version_id,
            "is_force_rerun": True,
            "parent_summary": parent_data.get("summary", {})
        }
        
        self.register_version(new_version_id, metadata)
        return new_version_id


# Migration function
def migrate_from_v1():
    """Migrate from single versions.json to directory structure"""
    old_file = Path("versions.json")
    if not old_file.exists():
        logger.info("No versions.json found, starting fresh")
        return
    
    logger.info("Migrating from versions.json to directory structure...")
    
    # Load old data
    with open(old_file, 'r') as f:
        old_data = json.load(f)
    
    # Create new manager
    vm = VersionManagerV2()
    
    # Migrate each version
    for version_data in old_data.get("versions", []):
        version_id = version_data["version_id"]
        
        # Save as individual file
        version_file = vm._get_version_file(version_id)
        with open(version_file, 'w') as f:
            json.dump(version_data, f, indent=2)
        
        logger.info(f"Migrated version: {version_id}")
    
    # Update index
    index = vm._load_index()
    index["current"] = old_data.get("current")
    index["versions_count"] = len(old_data.get("versions", []))
    
    # Find latest complete
    for v in reversed(old_data.get("versions", [])):
        if v.get("status") == "complete" and v.get("summary", {}).get("artifacts_uploaded"):
            index["latest_complete"] = v["version_id"]
            break
    
    vm._save_index(index)
    
    # Rename old file
    backup_name = f"versions.json.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    old_file.rename(backup_name)
    logger.info(f"Backed up old versions.json to {backup_name}")
    
    logger.info("Migration complete!")


if __name__ == "__main__":
    # Test or migrate
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "migrate":
        migrate_from_v1()
    else:
        # Test the new system
        vm = VersionManagerV2()
        print(f"Current version: {vm.get_current_version_id()}")
        print(f"Total versions: {vm._load_index()['versions_count']}")