#!/usr/bin/env python3
"""
Enhanced Version Manager with separate tracking for successful/failed runs
and better support for team collaboration
"""

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import shutil
import os
try:
    import fcntl
except ImportError:
    # fcntl not available on Windows
    fcntl = None

logger = logging.getLogger(__name__)


class EnhancedVersionManager:
    """
    Enhanced version manager that:
    - Separates successful and failed versions
    - Supports archiving old versions
    - Reduces git conflicts with separate files
    - Provides cleanup utilities
    """
    
    def __init__(self, base_dir: str = "config"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Separate files for different version types
        self.successful_versions_file = self.base_dir / "versions_successful.json"
        self.failed_versions_file = self.base_dir / "versions_failed.json"
        self.archived_versions_dir = self.base_dir / "versions_archived"
        self.archived_versions_dir.mkdir(exist_ok=True)
        
        # Individual version files directory (to reduce conflicts)
        self.versions_dir = self.base_dir / "versions"
        self.versions_dir.mkdir(exist_ok=True)
        
        # Current version pointer
        self.current_version_file = self.base_dir / "current_version.txt"
        
        # Initialize files if they don't exist
        self._initialize_files()
        
        # Keep only recent versions in memory
        self.max_recent_versions = 50
        
    def _initialize_files(self):
        """Initialize version files if they don't exist"""
        for file_path in [self.successful_versions_file, self.failed_versions_file]:
            if not file_path.exists():
                with open(file_path, 'w') as f:
                    json.dump({
                        "versions": {},
                        "metadata": {
                            "created_at": datetime.now().isoformat(),
                            "last_updated": datetime.now().isoformat()
                        }
                    }, f, indent=2)
    
    def _file_lock(self, file_path: Path, mode: str = 'r'):
        """Context manager for file locking"""
        class FileLock:
            def __init__(self, path, mode):
                self.path = path
                self.mode = mode
                self.file = None
                
            def __enter__(self):
                self.file = open(self.path, self.mode)
                # Use flock on Unix-like systems
                if fcntl:
                    try:
                        fcntl.flock(self.file.fileno(), fcntl.LOCK_EX)
                    except:
                        pass
                return self.file
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                if fcntl:
                    try:
                        fcntl.flock(self.file.fileno(), fcntl.LOCK_UN)
                    except:
                        pass
                self.file.close()
                
        return FileLock(file_path, mode)
    
    def create_version_id(self, suffix: str = "") -> str:
        """Create a new version ID"""
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        hostname = os.uname().nodename.replace(' ', '-').replace('.', '-').lower()
        version_id = f"{timestamp}_{hostname}"
        if suffix:
            version_id += f"_{suffix}"
        return version_id
    
    def register_version(self, version_id: str, metadata: Dict[str, Any] = None) -> None:
        """Register a new version (initially goes to failed until marked complete)"""
        version_info = {
            "id": version_id,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "status": "in_progress",
            "metadata": metadata or {},
            "stages": {}
        }
        
        # Save to individual file
        version_file = self.versions_dir / f"{version_id}.json"
        with open(version_file, 'w') as f:
            json.dump(version_info, f, indent=2)
        
        # Add to failed versions (will move to successful when complete)
        with self._file_lock(self.failed_versions_file, 'r+') as f:
            data = json.load(f)
            data["versions"][version_id] = {
                "id": version_id,
                "created_at": version_info["created_at"],
                "status": "in_progress"
            }
            data["metadata"]["last_updated"] = datetime.now().isoformat()
            f.seek(0)
            json.dump(data, f, indent=2)
            f.truncate()
            
        logger.info(f"Registered new version: {version_id}")
    
    def mark_version_complete(self, version_id: str, summary: Dict[str, Any] = None) -> None:
        """Mark a version as successfully completed and move it to successful versions"""
        # Load from individual file
        version_file = self.versions_dir / f"{version_id}.json"
        if not version_file.exists():
            logger.error(f"Version file not found: {version_id}")
            return
            
        with open(version_file, 'r') as f:
            version_info = json.load(f)
        
        # Update version info
        version_info["status"] = "completed"
        version_info["completed_at"] = datetime.now().isoformat()
        version_info["updated_at"] = datetime.now().isoformat()
        if summary:
            version_info["summary"] = summary
            
        # Save updated version
        with open(version_file, 'w') as f:
            json.dump(version_info, f, indent=2)
        
        # Remove from failed versions
        with self._file_lock(self.failed_versions_file, 'r+') as f:
            failed_data = json.load(f)
            if version_id in failed_data["versions"]:
                del failed_data["versions"][version_id]
                failed_data["metadata"]["last_updated"] = datetime.now().isoformat()
                f.seek(0)
                json.dump(failed_data, f, indent=2)
                f.truncate()
        
        # Add to successful versions
        with self._file_lock(self.successful_versions_file, 'r+') as f:
            successful_data = json.load(f)
            successful_data["versions"][version_id] = {
                "id": version_id,
                "created_at": version_info["created_at"],
                "completed_at": version_info["completed_at"],
                "status": "completed"
            }
            successful_data["metadata"]["last_updated"] = datetime.now().isoformat()
            
            # Auto-archive if too many versions
            if len(successful_data["versions"]) > self.max_recent_versions:
                self._auto_archive_old_versions(successful_data)
                
            f.seek(0)
            json.dump(successful_data, f, indent=2)
            f.truncate()
        
        # Update current version pointer
        with open(self.current_version_file, 'w') as f:
            f.write(version_id)
            
        logger.info(f"Marked version as complete: {version_id}")
    
    def update_stage_info(self, version_id: str, stage_name: str, stage_info: Dict[str, Any]) -> None:
        """Update information for a specific stage"""
        version_file = self.versions_dir / f"{version_id}.json"
        if not version_file.exists():
            logger.error(f"Version file not found: {version_id}")
            return
            
        with self._file_lock(version_file, 'r+') as f:
            version_info = json.load(f)
            version_info["stages"][stage_name] = stage_info
            version_info["updated_at"] = datetime.now().isoformat()
            f.seek(0)
            json.dump(version_info, f, indent=2)
            f.truncate()
    
    def update_version(self, version_id: str, updates: Dict[str, Any]) -> None:
        """Update version metadata"""
        version_file = self.versions_dir / f"{version_id}.json"
        if not version_file.exists():
            logger.error(f"Version file not found: {version_id}")
            return
            
        with self._file_lock(version_file, 'r+') as f:
            version_info = json.load(f)
            version_info.update(updates)
            version_info["updated_at"] = datetime.now().isoformat()
            f.seek(0)
            json.dump(version_info, f, indent=2)
            f.truncate()
    
    def get_version(self, version_id: str) -> Optional[Dict[str, Any]]:
        """Get version info from individual file"""
        version_file = self.versions_dir / f"{version_id}.json"
        if version_file.exists():
            with open(version_file, 'r') as f:
                return json.load(f)
        return None
    
    def get_current_version_id(self) -> Optional[str]:
        """Get the current version ID"""
        if self.current_version_file.exists():
            with open(self.current_version_file, 'r') as f:
                return f.read().strip()
        return None
    
    def list_versions(self, status: str = "all", limit: int = 20) -> List[Dict[str, Any]]:
        """
        List versions by status
        
        Args:
            status: 'successful', 'failed', 'archived', or 'all'
            limit: Maximum number of versions to return
        """
        versions = []
        
        if status in ["successful", "all"]:
            with open(self.successful_versions_file, 'r') as f:
                data = json.load(f)
                for version_id, summary in data["versions"].items():
                    # Load full details if needed
                    full_version = self.get_version(version_id)
                    if full_version:
                        versions.append(full_version)
                    else:
                        versions.append(summary)
                
        if status in ["failed", "all"]:
            with open(self.failed_versions_file, 'r') as f:
                data = json.load(f)
                for version_id, summary in data["versions"].items():
                    # Load full details if needed
                    full_version = self.get_version(version_id)
                    if full_version:
                        versions.append(full_version)
                    else:
                        versions.append(summary)
        
        # Sort by creation date (newest first)
        versions.sort(key=lambda v: v.get("created_at", ""), reverse=True)
        
        return versions[:limit]
    
    def delete_version(self, version_id: str, delete_artifacts: bool = False) -> bool:
        """
        Delete a version from tracking
        
        Args:
            version_id: Version to delete
            delete_artifacts: Also delete artifacts directory
        """
        deleted = False
        
        # Delete from individual file
        version_file = self.versions_dir / f"{version_id}.json"
        if version_file.exists():
            version_file.unlink()
            deleted = True
        
        # Delete from successful versions
        with self._file_lock(self.successful_versions_file, 'r+') as f:
            data = json.load(f)
            if version_id in data["versions"]:
                del data["versions"][version_id]
                data["metadata"]["last_updated"] = datetime.now().isoformat()
                f.seek(0)
                json.dump(data, f, indent=2)
                f.truncate()
                deleted = True
        
        # Delete from failed versions
        with self._file_lock(self.failed_versions_file, 'r+') as f:
            data = json.load(f)
            if version_id in data["versions"]:
                del data["versions"][version_id]
                data["metadata"]["last_updated"] = datetime.now().isoformat()
                f.seek(0)
                json.dump(data, f, indent=2)
                f.truncate()
                deleted = True
        
        # Delete artifacts if requested
        if delete_artifacts:
            artifacts_dir = Path("artifacts") / version_id
            if artifacts_dir.exists():
                shutil.rmtree(artifacts_dir)
                logger.info(f"Deleted artifacts for version: {version_id}")
        
        if deleted:
            logger.info(f"Deleted version: {version_id}")
        else:
            logger.warning(f"Version not found: {version_id}")
            
        return deleted
    
    def archive_version(self, version_id: str) -> bool:
        """Move a version to archived status"""
        version_info = self.get_version(version_id)
        if not version_info:
            logger.error(f"Version not found: {version_id}")
            return False
        
        # Save to archive directory with year-month subdirectory
        created_date = datetime.fromisoformat(version_info["created_at"])
        archive_subdir = self.archived_versions_dir / created_date.strftime("%Y-%m")
        archive_subdir.mkdir(exist_ok=True)
        
        archive_file = archive_subdir / f"{version_id}.json"
        version_info["archived_at"] = datetime.now().isoformat()
        
        with open(archive_file, 'w') as f:
            json.dump(version_info, f, indent=2)
        
        # Remove from active tracking
        self.delete_version(version_id, delete_artifacts=False)
        
        logger.info(f"Archived version: {version_id}")
        return True
    
    def cleanup_failed_versions(self, days: int = 7) -> int:
        """Clean up old failed versions"""
        cutoff_date = datetime.now().timestamp() - (days * 24 * 60 * 60)
        deleted_count = 0
        
        with open(self.failed_versions_file, 'r') as f:
            data = json.load(f)
            
        to_delete = []
        for version_id, version_summary in data["versions"].items():
            created_at = datetime.fromisoformat(version_summary["created_at"]).timestamp()
            if created_at < cutoff_date:
                to_delete.append(version_id)
        
        for version_id in to_delete:
            if self.delete_version(version_id, delete_artifacts=True):
                deleted_count += 1
        
        logger.info(f"Cleaned up {deleted_count} old failed versions")
        return deleted_count
    
    def _auto_archive_old_versions(self, data: Dict[str, Any]) -> None:
        """Automatically archive oldest versions when limit exceeded"""
        versions = list(data["versions"].items())
        versions.sort(key=lambda x: x[1].get("created_at", ""))
        
        # Archive oldest versions
        to_archive = versions[:-self.max_recent_versions]
        for version_id, _ in to_archive:
            self.archive_version(version_id)
    
    def export_version_info(self, version_id: str, output_file: Path) -> bool:
        """Export version info for sharing or git commit"""
        version_info = self.get_version(version_id)
        if not version_info:
            logger.error(f"Version not found: {version_id}")
            return False
        
        with open(output_file, 'w') as f:
            json.dump(version_info, f, indent=2)
        
        logger.info(f"Exported version info to: {output_file}")
        return True
    
    def search_versions(self, **criteria) -> List[Dict[str, Any]]:
        """
        Search versions by various criteria
        
        Example:
            search_versions(status="failed", stage_failed="download_data")
        """
        all_versions = self.list_versions(status="all", limit=1000)
        results = []
        
        for version in all_versions:
            match = True
            
            # Check status
            if "status" in criteria and version.get("status") != criteria["status"]:
                match = False
                
            # Check failed stage
            if "stage_failed" in criteria:
                stages = version.get("stages", {})
                stage_name = criteria["stage_failed"]
                if stage_name not in stages or stages[stage_name].get("completed", True):
                    match = False
                    
            # Check date range
            if "created_after" in criteria:
                created_at = datetime.fromisoformat(version["created_at"])
                if created_at < criteria["created_after"]:
                    match = False
                    
            if match:
                results.append(version)
                
        return results