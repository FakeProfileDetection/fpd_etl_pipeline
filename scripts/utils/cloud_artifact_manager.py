"""
Cloud artifact management for Google Cloud Storage
Handles upload/download of artifacts with metadata tracking
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from google.cloud import storage

    HAS_GCS = True
except ImportError:
    HAS_GCS = False
    logging.warning("google-cloud-storage not installed. Cloud features disabled.")

from .enhanced_version_manager import EnhancedVersionManager as VersionManager

logger = logging.getLogger(__name__)


class CloudArtifactManager:
    """Manage artifacts in Google Cloud Storage"""

    def __init__(
        self,
        version_id: str,
        bucket_name: str,
        local_cache_dir: Path = Path(".artifact_cache"),
    ):
        self.version_id = version_id
        self.bucket_name = bucket_name
        self.local_cache = local_cache_dir / version_id
        self.local_cache.mkdir(parents=True, exist_ok=True)

        # Initialize GCS client if available
        if HAS_GCS and bucket_name:
            try:
                self.client = storage.Client()
                self.bucket = self.client.bucket(bucket_name)
                self.cloud_enabled = True
            except Exception as e:
                logger.warning(f"Failed to initialize GCS client: {e}")
                self.cloud_enabled = False
        else:
            self.cloud_enabled = False
            if not HAS_GCS:
                logger.warning(
                    "Cloud features disabled - google-cloud-storage not installed"
                )

        # Cloud paths
        self.artifact_prefix = f"artifacts/{version_id}"
        self.manifest_path = f"{self.artifact_prefix}/artifact_manifest.json"

        # Local manifest tracking
        self.local_manifest_path = self.local_cache / "artifact_manifest.json"
        self.manifest = self._load_or_create_manifest()

    def _load_or_create_manifest(self) -> Dict[str, Any]:
        """Load existing manifest or create new one"""
        # Try loading from local cache first
        if self.local_manifest_path.exists():
            try:
                with open(self.local_manifest_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load local manifest: {e}")

        # Try loading from cloud
        if self.cloud_enabled:
            try:
                blob = self.bucket.blob(self.manifest_path)
                if blob.exists():
                    content = blob.download_as_text()
                    manifest = json.loads(content)
                    # Save to local cache
                    with open(self.local_manifest_path, "w") as f:
                        json.dump(manifest, f, indent=2)
                    return manifest
            except Exception as e:
                logger.warning(f"Failed to load cloud manifest: {e}")

        # Create new manifest
        return {
            "version_id": self.version_id,
            "created_at": datetime.now().isoformat(),
            "artifacts": {},
            "summary": {"total_artifacts": 0, "total_size_bytes": 0},
        }

    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate MD5 checksum of a file"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def upload_artifact(
        self,
        local_path: Path,
        artifact_type: str,
        stage: str,
        description: str = "",
        metadata: Optional[Dict] = None,
        force: bool = False,
    ) -> str:
        """Upload artifact to cloud and update manifest"""

        if not local_path.exists():
            raise FileNotFoundError(f"Local file not found: {local_path}")

        # Generate cloud path
        artifact_name = local_path.name
        cloud_path = f"{self.artifact_prefix}/{stage}/{artifact_type}/{artifact_name}"

        # Check if already uploaded (unless force)
        artifact_id = f"{stage}_{artifact_type}_{artifact_name}"
        if not force and artifact_id in self.manifest["artifacts"]:
            logger.info(f"Artifact {artifact_id} already uploaded, skipping")
            return cloud_path

        # Calculate checksum
        checksum = self._calculate_checksum(local_path)
        size_bytes = local_path.stat().st_size

        # Upload to GCS if enabled
        if self.cloud_enabled:
            try:
                blob = self.bucket.blob(cloud_path)
                blob.upload_from_filename(str(local_path))
                logger.info(
                    f"Uploaded {local_path.name} to gs://{self.bucket_name}/{cloud_path}"
                )
            except Exception as e:
                logger.error(f"Failed to upload {local_path}: {e}")
                raise
        else:
            logger.info(
                f"Cloud disabled - would upload {local_path.name} to {cloud_path}"
            )

        # Update manifest
        self.manifest["artifacts"][artifact_id] = {
            "cloud_path": cloud_path,
            "local_name": artifact_name,
            "artifact_type": artifact_type,
            "stage": stage,
            "description": description,
            "checksum": checksum,
            "size_bytes": size_bytes,
            "uploaded_at": datetime.now().isoformat(),
            "metadata": metadata or {},
        }

        # Update summary
        self.manifest["summary"]["total_artifacts"] = len(self.manifest["artifacts"])
        self.manifest["summary"]["total_size_bytes"] = sum(
            a["size_bytes"] for a in self.manifest["artifacts"].values()
        )

        # Save manifest
        self._save_manifest()

        return cloud_path

    def download_artifact(self, artifact_id: str, force: bool = False) -> Path:
        """Download artifact from cloud with caching"""

        artifact_info = self.manifest["artifacts"].get(artifact_id)
        if not artifact_info:
            raise ValueError(f"Artifact {artifact_id} not found in manifest")

        local_path = self.local_cache / artifact_info["local_name"]

        # Check cache
        if local_path.exists() and not force:
            # Verify checksum
            if self._calculate_checksum(local_path) == artifact_info["checksum"]:
                logger.info(f"Using cached artifact: {artifact_id}")
                return local_path
            else:
                logger.warning(f"Cached file checksum mismatch for {artifact_id}")

        # Download from cloud
        if self.cloud_enabled:
            try:
                blob = self.bucket.blob(artifact_info["cloud_path"])
                blob.download_to_filename(str(local_path))
                logger.info(f"Downloaded {artifact_id} from cloud")

                # Verify checksum
                if self._calculate_checksum(local_path) != artifact_info["checksum"]:
                    logger.error(f"Downloaded file checksum mismatch for {artifact_id}")
                    local_path.unlink()
                    raise ValueError("Checksum verification failed")

            except Exception as e:
                logger.error(f"Failed to download {artifact_id}: {e}")
                raise
        else:
            logger.error(f"Cloud disabled - cannot download {artifact_id}")
            raise RuntimeError("Cloud storage not available")

        return local_path

    def download_stage_artifacts(
        self,
        stage: str,
        artifact_types: Optional[List[str]] = None,
        pii_filter: Optional[Callable[[str], bool]] = None,
    ) -> Dict[str, Path]:
        """Download all artifacts for a stage"""

        downloads: Dict[str, List[Path]] = {}

        # Filter artifacts
        stage_artifacts = {
            aid: info
            for aid, info in self.manifest["artifacts"].items()
            if info["stage"] == stage
            and (not artifact_types or info["artifact_type"] in artifact_types)
            and (not pii_filter or pii_filter(info["local_name"]))
        }

        if not stage_artifacts:
            logger.warning(f"No artifacts found for stage '{stage}'")
            return downloads

        # Parallel download
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_artifact = {
                executor.submit(self.download_artifact, aid): aid
                for aid in stage_artifacts
            }

            for future in as_completed(future_to_artifact):
                artifact_id = future_to_artifact[future]
                try:
                    local_path = future.result()
                    downloads[artifact_id] = local_path
                except Exception as e:
                    logger.error(f"Failed to download {artifact_id}: {e}")

        return downloads

    def list_artifacts(
        self, stage: Optional[str] = None, artifact_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List artifacts with optional filtering"""
        artifacts = []

        for aid, info in self.manifest["artifacts"].items():
            if stage and info["stage"] != stage:
                continue
            if artifact_type and info["artifact_type"] != artifact_type:
                continue

            artifacts.append({"id": aid, **info})

        return artifacts

    def _save_manifest(self):
        """Save manifest to local cache and cloud"""
        # Save locally
        with open(self.local_manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=2)

        # Save to cloud
        if self.cloud_enabled:
            try:
                manifest_blob = self.bucket.blob(self.manifest_path)
                manifest_blob.upload_from_string(
                    json.dumps(self.manifest, indent=2), content_type="application/json"
                )
                logger.info(f"Updated cloud manifest: {self.manifest_path}")
            except Exception as e:
                logger.error(f"Failed to save manifest to cloud: {e}")

        # Update versions.json
        self._update_versions_json()

    def _update_versions_json(self):
        """Update versions.json with artifact information"""
        vm = VersionManager()

        # Create artifact summary
        stages = list(set(a["stage"] for a in self.manifest["artifacts"].values()))
        artifact_types = list(
            set(a["artifact_type"] for a in self.manifest["artifacts"].values())
        )

        artifact_info = {
            "manifest_path": f"gs://{self.bucket_name}/{self.manifest_path}"
            if self.cloud_enabled
            else "local",
            "total_artifacts": self.manifest["summary"]["total_artifacts"],
            "total_size_mb": self.manifest["summary"]["total_size_bytes"] / 1024 / 1024,
            "stages": stages,
            "artifact_types": artifact_types,
        }

        vm.update_stage_info(self.version_id, "artifacts", artifact_info)

    def get_manifest_summary(self) -> Dict[str, Any]:
        """Get summary of artifacts"""
        summary = self.manifest["summary"].copy()

        # Add breakdown by stage
        summary["by_stage"] = {}
        for artifact in self.manifest["artifacts"].values():
            stage = artifact["stage"]
            if stage not in summary["by_stage"]:
                summary["by_stage"][stage] = {
                    "count": 0,
                    "size_bytes": 0,
                    "types": set(),
                }
            summary["by_stage"][stage]["count"] += 1
            summary["by_stage"][stage]["size_bytes"] += artifact["size_bytes"]
            summary["by_stage"][stage]["types"].add(artifact["artifact_type"])

        # Convert sets to lists for JSON serialization
        for stage_info in summary["by_stage"].values():
            stage_info["types"] = list(stage_info["types"])

        return summary


def create_pii_filter(patterns: List[str]) -> Callable[[str], bool]:
    """Create a PII filter function from patterns"""

    def is_not_pii(filename: str) -> bool:
        filename_lower = filename.lower()
        for pattern in patterns:
            # Simple pattern matching (could be enhanced with fnmatch)
            pattern_clean = pattern.strip("*").lower()
            if pattern_clean in filename_lower:
                return False
        return True

    return is_not_pii


if __name__ == "__main__":
    # Test artifact manager
    import tempfile

    # Create test artifact
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"test": "data"}, f)
        test_file = Path(f.name)

    # Test manager (will work in local mode if GCS not configured)
    manager = CloudArtifactManager(
        version_id="test_2025-01-15_10-00-00",
        bucket_name="",  # Empty for local testing
    )

    # Test upload
    try:
        cloud_path = manager.upload_artifact(
            local_path=test_file,
            artifact_type="test",
            stage="testing",
            description="Test artifact",
        )
        print(f"Uploaded to: {cloud_path}")
    except Exception as e:
        print(f"Upload failed (expected if no GCS): {e}")

    # List artifacts
    artifacts = manager.list_artifacts()
    print(f"Total artifacts: {len(artifacts)}")

    # Cleanup
    test_file.unlink()
