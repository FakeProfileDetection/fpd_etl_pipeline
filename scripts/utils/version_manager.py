"""
Version management for pipeline runs
Handles version creation, tracking, and metadata
"""

import json
import os
import socket
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class VersionManager:
    """Manage pipeline versions and their metadata"""
    
    def __init__(self, versions_file: Path = Path("versions.json")):
        self.versions_file = versions_file
        self.versions_data = self._load_versions()
    
    def _load_versions(self) -> Dict[str, Any]:
        """Load versions from JSON file"""
        if self.versions_file.exists():
            try:
                with open(self.versions_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError as e:
                logger.error(f"Error loading versions.json: {e}")
                return {"versions": [], "current": None, "schema_version": "1.0"}
        else:
            return {"versions": [], "current": None, "schema_version": "1.0"}
    
    def _save_versions(self):
        """Save versions to JSON file"""
        with open(self.versions_file, 'w') as f:
            json.dump(self.versions_data, f, indent=2)
        logger.info(f"Updated {self.versions_file}")
    
    def create_version_id(self) -> str:
        """Create a unique version ID with timestamp and hostname"""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        hostname = socket.gethostname().replace(" ", "-").lower()
        # Sanitize hostname for filesystem
        hostname = "".join(c for c in hostname if c.isalnum() or c in "-_")
        version_id = f"{timestamp}_{hostname}"
        logger.info(f"Created version ID: {version_id}")
        return version_id
    
    def register_version(self, version_id: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Register a new version"""
        version_entry = {
            "version_id": version_id,
            "created_at": datetime.now().isoformat(),
            "created_by": socket.gethostname(),
            "user": os.getenv('USER', 'unknown'),
            "stages": {},
            "features": {},
            "metadata": metadata or {},
            "status": "in_progress"
        }
        
        # Add to versions list
        self.versions_data["versions"].insert(0, version_entry)  # Most recent first
        self.versions_data["current"] = version_id
        self._save_versions()
        
        logger.info(f"Registered version: {version_id}")
        return version_entry
    
    def get_version(self, version_id: str) -> Optional[Dict[str, Any]]:
        """Get version information"""
        if version_id == "current" or version_id == "latest":
            version_id = self.get_current_version_id()
        
        for version in self.versions_data["versions"]:
            if version["version_id"] == version_id:
                return version
        return None
    
    def get_current_version_id(self) -> Optional[str]:
        """Get the current/latest version ID"""
        if self.versions_data["current"]:
            return self.versions_data["current"]
        elif self.versions_data["versions"]:
            return self.versions_data["versions"][0]["version_id"]
        return None
    
    def update_stage_info(self, version_id: str, stage: str, info: Dict[str, Any]):
        """Update information for a pipeline stage"""
        version = self.get_version(version_id)
        if not version:
            raise ValueError(f"Version {version_id} not found")
        
        # Update in the versions list
        for v in self.versions_data["versions"]:
            if v["version_id"] == version_id:
                v["stages"][stage] = {
                    **info,
                    "completed_at": datetime.now().isoformat()
                }
                break
        
        self._save_versions()
        logger.info(f"Updated stage '{stage}' for version {version_id}")
    
    def update_feature_info(self, version_id: str, feature_type: str, info: Dict[str, Any]):
        """Update information for a feature extraction"""
        version = self.get_version(version_id)
        if not version:
            raise ValueError(f"Version {version_id} not found")
        
        # Update in the versions list
        for v in self.versions_data["versions"]:
            if v["version_id"] == version_id:
                if "features" not in v:
                    v["features"] = {}
                v["features"][feature_type] = {
                    **info,
                    "extracted_at": datetime.now().isoformat()
                }
                break
        
        self._save_versions()
        logger.info(f"Updated feature '{feature_type}' for version {version_id}")
    
    def mark_version_complete(self, version_id: str, summary: Optional[Dict[str, Any]] = None):
        """Mark a version as complete"""
        for v in self.versions_data["versions"]:
            if v["version_id"] == version_id:
                v["status"] = "complete"
                v["completed_at"] = datetime.now().isoformat()
                if summary:
                    v["summary"] = summary
                break
        
        self._save_versions()
        logger.info(f"Marked version {version_id} as complete")
    
    def list_versions(self, limit: int = 10, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List versions with optional filtering"""
        versions = self.versions_data["versions"]
        
        if status:
            versions = [v for v in versions if v.get("status") == status]
        
        return versions[:limit]
    
    def get_latest_complete_version(self) -> Optional[Dict[str, Any]]:
        """Get the most recent complete version"""
        for version in self.versions_data["versions"]:
            if version.get("status") == "complete":
                return version
        return None
    
    def create_derived_version(self, parent_version_id: str, starting_stage: str) -> str:
        """Create a new version derived from an existing one"""
        parent = self.get_version(parent_version_id)
        if not parent:
            raise ValueError(f"Parent version {parent_version_id} not found")
        
        # Create new version ID
        new_version_id = self.create_version_id()
        
        # Create version entry with parent info
        version_entry = {
            "version_id": new_version_id,
            "parent_version": parent_version_id,
            "created_at": datetime.now().isoformat(),
            "created_by": socket.gethostname(),
            "user": os.getenv('USER', 'unknown'),
            "stages": {},
            "features": {},
            "metadata": {
                "derived_from": parent_version_id,
                "starting_stage": starting_stage
            },
            "status": "in_progress"
        }
        
        # Copy stages that come before the starting stage
        # TODO: Implement stage ordering logic
        # This would copy relevant stages from parent
        
        self.versions_data["versions"].insert(0, version_entry)
        self.versions_data["current"] = new_version_id
        self._save_versions()
        
        logger.info(f"Created derived version {new_version_id} from {parent_version_id}")
        return new_version_id


# Convenience functions for command-line usage
def get_current_version() -> Optional[str]:
    """Get current version ID"""
    vm = VersionManager()
    return vm.get_current_version_id()


def create_new_version() -> str:
    """Create and register a new version"""
    vm = VersionManager()
    version_id = vm.create_version_id()
    vm.register_version(version_id)
    return version_id


if __name__ == "__main__":
    # Test version manager
    import os
    vm = VersionManager()
    
    # Create a test version
    test_id = vm.create_version_id()
    print(f"Created version: {test_id}")
    
    # Register it
    vm.register_version(test_id, {"test": True})
    
    # Get current
    current = vm.get_current_version_id()
    print(f"Current version: {current}")
    
    # List versions
    versions = vm.list_versions(limit=5)
    print(f"Found {len(versions)} versions")

