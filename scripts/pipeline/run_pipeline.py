#!/usr/bin/env python3
"""
Main pipeline orchestrator
Coordinates execution of all pipeline stages with proper dependency management
"""

import click
import logging
import sys
import socket
import tarfile
import subprocess
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.utils.version_manager import VersionManager
from scripts.utils.config_manager import get_config
from scripts.utils.cloud_artifact_manager import CloudArtifactManager
from scripts.utils.logger_config import setup_pipeline_logging, get_pipeline_logger

# Import pipeline stages
from scripts.pipeline import download_data, clean_data, extract_keypairs, extract_features, run_eda

logger = logging.getLogger(__name__)


class Pipeline:
    """Main pipeline executor"""
    
    def __init__(self, version_id: str, config: Dict[str, Any], 
                 dry_run: bool = False, local_only: bool = False,
                 force_mode: bool = False,
                 version_manager: Optional[VersionManager] = None):
        self.version_id = version_id
        self.config = config
        self.dry_run = dry_run
        self.local_only = local_only
        self.force_mode = force_mode
        
        # Initialize managers
        self.version_manager = version_manager or VersionManager()
        
        # Cloud artifact manager (only if uploads are enabled)
        if config.get("UPLOAD_ARTIFACTS"):
            self.artifact_manager = CloudArtifactManager(
                version_id=version_id,
                bucket_name=config.get("BUCKET_NAME")
            )
        else:
            self.artifact_manager = None
        
        # Track completed stages - check what's already done
        self.completed_stages = set()
        version_info = self.version_manager.get_version(version_id)
        if version_info and "stages" in version_info:
            for stage_name, stage_info in version_info["stages"].items():
                if stage_info.get("completed", False):
                    self.completed_stages.add(stage_name)
                    
        # Track pipeline statistics
        self.pipeline_stats = {
            "start_time": datetime.now(),
            "stage_times": {},
            "stage_errors": {},
            "files_processed": 0,
            "users_processed": 0,
            "data_downloaded_mb": 0
        }
    
    def run_stage(self, stage_name: str, stage_func: callable, **kwargs) -> bool:
        """Run a single pipeline stage"""
        logger.info(f"{'[DRY RUN] ' if self.dry_run else ''}Running stage: {stage_name}")
        
        if self.dry_run:
            logger.info(f"Would run {stage_name} with version_id: {self.version_id}")
            self.completed_stages.add(stage_name)
            return True
        
        try:
            start_time = datetime.now()
            
            # Run the stage - all stages have the same signature
            output_path = stage_func(
                version_id=self.version_id,
                config=self.config,
                dry_run=self.dry_run,
                local_only=self.local_only
            )
            
            # Calculate duration
            duration = (datetime.now() - start_time).total_seconds()
            self.pipeline_stats["stage_times"][stage_name] = duration
            
            # Update version info
            stage_info = {
                "output_path": str(output_path),
                "duration_seconds": duration,
                "completed": True
            }
            self.version_manager.update_stage_info(self.version_id, stage_name, stage_info)
            
            # Extract statistics from stages if available
            version_info = self.version_manager.get_version(self.version_id)
            if version_info and "stages" in version_info:
                # Get stats from clean_data stage
                if stage_name == "clean_data" and "clean_data" in version_info["stages"]:
                    stats = version_info["stages"]["clean_data"].get("stats", {})
                    self.pipeline_stats["users_processed"] = stats.get("total_users", 0)
                    self.pipeline_stats["files_processed"] = stats.get("total_files", 0)
                # Get download size from download_data stage  
                elif stage_name == "download_data" and "download_data" in version_info["stages"]:
                    # Read from download metadata if available
                    artifacts_dir = Path(self.config.get("ARTIFACTS_DIR", "artifacts")) / self.version_id
                    download_log = artifacts_dir / "etl_metadata" / "download" / "download_log.json"
                    if download_log.exists():
                        try:
                            import json
                            with open(download_log, 'r') as f:
                                log_data = json.load(f)
                                size_bytes = log_data.get("total_size", 0)
                                self.pipeline_stats["data_downloaded_mb"] = size_bytes / (1024 * 1024)
                        except:
                            pass
            
            self.completed_stages.add(stage_name)
            logger.info(f"✅ Completed {stage_name} in {duration:.1f}s")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed stage {stage_name}: {e}")
            self.pipeline_stats["stage_errors"][stage_name] = str(e)
            
            # Log detailed error information
            logger_config = get_pipeline_logger()
            logger_config.log_error_details(stage_name, e)
            
            # Update version info with failure
            self.version_manager.update_stage_info(
                self.version_id, 
                stage_name, 
                {"completed": False, "error": str(e)}
            )
            return False
    
    def run_all_stages(self, stages: List[str]) -> bool:
        """Run all specified stages in order"""
        # Define stage dependencies
        stage_order = ["download", "clean", "keypairs", "features", "eda"]
        
        # Filter and order stages
        stages_to_run = [s for s in stage_order if s in stages]
        
        logger.info(f"Running pipeline stages: {stages_to_run}")
        
        for stage in stages_to_run:
            # Check dependencies
            if not self._check_dependencies(stage):
                logger.error(f"Dependencies not met for stage: {stage}")
                return False
            
            # Run stage
            success = self._run_stage_by_name(stage)
            if not success:
                logger.error(f"Stage '{stage}' failed. Stopping pipeline.")
                return False
        
        return True
    
    def _check_dependencies(self, stage: str) -> bool:
        """Check if stage dependencies are met"""
        dependencies = {
            "download": [],
            "clean": ["download_data"],  # Use actual stage names stored in version info
            "keypairs": ["clean_data"],
            "features": ["extract_keypairs"],  # Features depend on keypairs
            "eda": ["extract_keypairs"]  # EDA can run after keypairs are extracted
        }
        
        required = dependencies.get(stage, [])
        missing = [dep for dep in required if dep not in self.completed_stages]
        
        if missing:
            logger.warning(f"Missing dependencies for {stage}: {missing}")
            return False
        
        return True
    
    def _run_stage_by_name(self, stage_name: str) -> bool:
        """Run a stage by name"""
        # Map CLI stage names to actual function names and module stage names
        stage_mappings = {
            "download": ("download_data", download_data.run),
            "clean": ("clean_data", clean_data.run),
            "keypairs": ("extract_keypairs", extract_keypairs.run),
            "features": ("extract_features", extract_features.run),
            "eda": ("run_eda", run_eda.run)
        }
        
        if stage_name not in stage_mappings:
            logger.error(f"Unknown stage: {stage_name}")
            return False
        
        actual_stage_name, stage_func = stage_mappings[stage_name]
        return self.run_stage(actual_stage_name, stage_func)
    
    def create_and_upload_tarball(self) -> bool:
        """Create tar.gz of version artifacts and upload to cloud"""
        artifacts_dir = Path(self.config.get("ARTIFACTS_DIR", "artifacts"))
        version_dir = artifacts_dir / self.version_id
        
        if not version_dir.exists():
            logger.error(f"Version directory not found: {version_dir}")
            return False
        
        # Create tar.gz file
        tarball_name = f"{self.version_id}.tar.gz"
        tarball_path = artifacts_dir / tarball_name
        
        logger.info(f"Creating archive: {tarball_name}")
        
        try:
            # Create tar.gz excluding PII if needed
            exclude_patterns = []
            if not self.config.get("INCLUDE_PII"):
                exclude_patterns = [
                    "*metadata/metadata.csv",
                    "*demographics.json", 
                    "*consent.json",
                    "*email*",
                    "*start_time.json"
                ]
            
            def should_exclude(tarinfo):
                """Filter function for tarfile to exclude PII files"""
                if not exclude_patterns:
                    return False
                    
                for pattern in exclude_patterns:
                    if pattern.replace("*", "") in tarinfo.name:
                        logger.debug(f"Excluding PII file: {tarinfo.name}")
                        return True
                return False
            
            with tarfile.open(tarball_path, "w:gz") as tar:
                tar.add(version_dir, arcname=self.version_id, filter=lambda x: None if should_exclude(x) else x)
            
            logger.info(f"Archive created: {tarball_path.stat().st_size / 1024 / 1024:.2f} MB")
            
            # Upload to cloud
            bucket_name = self.config.get("BUCKET_NAME")
            if not bucket_name:
                logger.error("BUCKET_NAME not configured")
                return False
            
            # Check if version already exists in cloud
            check_cmd = [
                "gcloud", "storage", "ls",
                f"gs://{bucket_name}/artifacts/{tarball_name}"
            ]
            
            result = subprocess.run(check_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.warning(f"Version {self.version_id} already exists in cloud")
                logger.info("Per policy: Only additions are allowed to existing versions")
                logger.info("If you need to modify existing artifacts, create a new version")
                return False
            
            # Upload the tarball
            upload_cmd = [
                "gcloud", "storage", "cp",
                str(tarball_path),
                f"gs://{bucket_name}/artifacts/{tarball_name}"
            ]
            
            logger.info(f"Uploading to gs://{bucket_name}/artifacts/{tarball_name}")
            result = subprocess.run(upload_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Upload failed: {result.stderr}")
                return False
            
            logger.info("✅ Upload completed successfully")
            
            # Clean up local tarball
            tarball_path.unlink()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to create/upload archive: {e}")
            if tarball_path.exists():
                tarball_path.unlink()
            return False
    
    def get_summary(self) -> Dict[str, Any]:
        """Generate pipeline execution summary"""
        end_time = datetime.now()
        total_duration = (end_time - self.pipeline_stats["start_time"]).total_seconds()
        
        summary = {
            "version_id": self.version_id,
            "total_duration_seconds": round(total_duration, 2),
            "stages_completed": list(self.completed_stages),
            "stages_failed": list(self.pipeline_stats["stage_errors"].keys()),
            "files_processed": self.pipeline_stats["files_processed"],
            "users_processed": self.pipeline_stats["users_processed"],
            "data_downloaded_mb": round(self.pipeline_stats["data_downloaded_mb"], 2),
            "stage_times": {k: round(v, 2) for k, v in self.pipeline_stats["stage_times"].items()}
        }
        
        return summary
    
    def print_summary(self):
        """Print a formatted summary to console"""
        summary = self.get_summary()
        
        print("\n" + "="*60)
        print("PIPELINE EXECUTION SUMMARY")
        print("="*60)
        print(f"Version: {summary['version_id']}")
        print(f"Total Duration: {summary['total_duration_seconds']} seconds")
        print(f"\nStages Completed ({len(summary['stages_completed'])}):")
        for stage in summary['stages_completed']:
            time_taken = summary['stage_times'].get(stage, 0)
            print(f"  ✓ {stage}: {time_taken}s")
        
        if summary['stages_failed']:
            print(f"\nStages Failed ({len(summary['stages_failed'])}):")
            for stage in summary['stages_failed']:
                print(f"  ✗ {stage}")
                # Show error message if available
                error_msg = self.pipeline_stats["stage_errors"].get(stage)
                if error_msg:
                    print(f"     Error: {error_msg}")
        
        if summary['files_processed'] > 0:
            print(f"\nData Processed:")
            print(f"  Files: {summary['files_processed']}")
            print(f"  Users: {summary['users_processed']}")
            
        if summary['data_downloaded_mb'] > 0:
            print(f"  Downloaded: {summary['data_downloaded_mb']} MB")
            
        print("="*60 + "\n")


@click.command()
@click.option('--mode', type=click.Choice(['full', 'incr', 'force']), 
              default='incr', help='Pipeline run mode')
@click.option('--stages', '-s', multiple=True,
              help='Specific stages to run (default: all)')
@click.option('--version-id', help='Version to work with (default: create new)')
@click.option('--local-only', is_flag=True, default=False,
              help='Development mode - use only local data, no cloud downloads')
@click.option('--upload-artifacts', is_flag=True, default=False,
              help='Upload processed artifacts to cloud storage after completion')
@click.option('--include-pii', is_flag=True, default=False,
              help='Include PII data in operations (default: False)')
@click.option('--no-eda', is_flag=True, default=False,
              help='Skip EDA stage (EDA runs by default)')
@click.option('--dry-run', is_flag=True,
              help='Show what would be done without executing')
@click.option('--log-level', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']),
              default='INFO', help='Logging level')
@click.option('--device-types', type=str, default='desktop',
              help='Device types to process (comma-separated: desktop,mobile)')
def main(mode: str, stages: List[str], version_id: Optional[str],
         upload_artifacts: bool, include_pii: bool, no_eda: bool,
         dry_run: bool, local_only: bool, log_level: str, device_types: str):
    """
    Run the keystroke data processing pipeline.
    
    DEFAULT BEHAVIOR:
    - Downloads latest data from cloud storage
    - Processes through all pipeline stages
    - Saves results locally (does NOT upload unless --upload-artifacts is used)
    - Excludes PII data (use --include-pii to include demographics)
    
    MODES:
    - full: Create new version with fresh download from cloud
    - incr: Continue processing incomplete stages from last run
    - force: Force re-run all stages using existing downloaded data
    
    FLAGS:
    - --local-only: Skip cloud download, use existing local data (for development)
    - --upload-artifacts: Upload results to cloud after processing
    - --include-pii: Include personally identifiable information in processing
    - --no-eda: Skip exploratory data analysis stage
    
    Examples:
        # Standard run - download latest data and process
        python run_pipeline.py --mode full
        
        # Development - use local data only
        python run_pipeline.py --mode full --local-only
        
        # Process and upload results to cloud
        python run_pipeline.py --mode full --upload-artifacts
        
        # Continue incomplete processing
        python run_pipeline.py --mode incr
    """
    
    # Initialize version first to get version_id
    version_mgr = VersionManager()
    
    if not version_id:
        if mode == 'full':
            # Create new version for full runs
            version_id = version_mgr.create_version_id()
            version_mgr.register_version(version_id, {"mode": mode, "stages": list(stages)})
            click.echo(f"Created new version: {version_id}")
        else:
            # Use current version for incremental/force modes
            version_id = version_mgr.get_current_version_id()
            if not version_id:
                click.echo("❌ No current version found. Run with --mode full first.")
                return
    
    # Setup centralized logging
    command_args = {
        'mode': mode,
        'stages': list(stages) if stages else 'all',
        'local_only': local_only,
        'upload_artifacts': upload_artifacts,
        'include_pii': include_pii,
        'no_eda': no_eda,
        'dry_run': dry_run,
        'log_level': log_level,
        'device_types': device_types
    }
    
    log_filename = setup_pipeline_logging(
        version_id=version_id,
        script_name='pipeline',
        command_args=command_args,
        log_level=log_level
    )
    
    logger.info(f"📝 Logging to: {log_filename}")
    
    # Load configuration
    config_mgr = get_config()
    config = config_mgr.get_all()
    
    # Override config with command line flags
    config["UPLOAD_ARTIFACTS"] = upload_artifacts
    config["INCLUDE_PII"] = include_pii
    config["LOCAL_ONLY"] = local_only
    
    # Parse and set device types
    if device_types:
        device_list = [d.strip().lower() for d in device_types.split(",") if d.strip()]
        config["DEVICE_TYPES"] = device_list
        # Also set in environment for config manager
        import os
        os.environ["DEVICE_TYPES"] = device_types
    
    # Show operation mode
    if local_only:
        click.echo("🏠 Local-only mode: Using existing local data (no cloud download)")
    else:
        click.echo("☁️  Cloud mode: Will download latest data from cloud storage")
    
    if upload_artifacts:
        click.echo("📤 Upload enabled: Results will be uploaded to cloud after processing")
        if include_pii:
            if not click.confirm("⚠️  WARNING: Upload will include PII data. Continue?"):
                return
    
    
    # Show configuration
    click.echo(f"""
Pipeline Configuration:
- Version: {version_id}
- Mode: {mode}
- Local only: {'Yes (no cloud download)' if local_only else 'No (will download from cloud)'}
- Upload results: {'Yes' if upload_artifacts else 'No'}
- Include PII: {'Yes' if include_pii else 'No (excluded)'}
- Run EDA: {'No' if no_eda else 'Yes'}
- Device types: {', '.join(config.get('DEVICE_TYPES', ['desktop']))}
- Stages: {list(stages) if stages else 'all'}
""")
    
    if dry_run:
        click.echo("🔍 DRY RUN - No actual processing will occur")
    
    if not config_mgr.validate_cloud_config() and config.get('UPLOAD_ARTIFACTS'):
        click.echo("❌ Invalid cloud configuration. Please check config/.env.local")
        return
    
    # Determine stages to run
    all_stages = ["download", "clean", "keypairs", "features", "eda"]
    if not stages:
        if mode == 'full':
            # Include all stages including EDA by default, unless --no-eda is specified
            stages = all_stages if not no_eda else all_stages[:-1]
        elif mode == 'incr':
            # Incremental mode: Only run stages not yet completed
            vm = VersionManager()
            version_info = vm.get_version(version_id)
            completed_stages = set()
            
            if version_info and "stages" in version_info:
                for stage_name, stage_info in version_info["stages"].items():
                    if stage_info.get("completed", False):
                        completed_stages.add(stage_name)
            
            # Map stage names to CLI names
            stage_name_mapping = {
                'download_data': 'download',
                'clean_data': 'clean', 
                'extract_keypairs': 'keypairs',
                'extract_features': 'features',
                'run_eda': 'eda'
            }
            
            # Determine which stages still need to run
            stages_to_run = []
            stages_to_check = all_stages if not no_eda else all_stages[:-1]
            for stage in stages_to_check:
                mapped_name = {v: k for k, v in stage_name_mapping.items()}.get(stage, stage)
                if mapped_name not in completed_stages:
                    stages_to_run.append(stage)
            
            if not stages_to_run:
                click.echo("ℹ️  All stages already completed. Use --mode force to re-run.")
                stages = []
            else:
                stages = stages_to_run
                click.echo(f"📋 Incremental mode - running incomplete stages: {', '.join(stages)}")
        elif mode == 'force':
            # Force mode: Create new version linked to parent
            if not version_id:
                version_id = version_mgr.get_current_version_id()
                if not version_id:
                    click.echo("❌ No current version found. Use --mode full for new data.")
                    return
            
            # Create new force version
            parent_version_id = version_id
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            hostname = socket.gethostname().replace(' ', '-').replace('.', '-').lower()
            version_id = f"{timestamp}_{hostname}_force"
            
            # Register with parent tracking
            metadata = {
                "mode": "force",
                "parent_version": parent_version_id,
                "is_force_rerun": True
            }
            version_mgr.register_version(version_id, metadata)
            
            click.echo(f"⚡ Force mode - creating new version: {version_id}")
            click.echo(f"   Parent version: {parent_version_id}")
            
            stages = all_stages if not no_eda else all_stages[:-1]
    
    # Initialize pipeline
    pipeline = Pipeline(
        version_id=version_id,
        config=config,
        dry_run=dry_run,
        local_only=local_only,
        force_mode=(mode == 'force')
    )
    
    # Run pipeline
    success = pipeline.run_all_stages(stages)
    
    # Always print summary
    pipeline.print_summary()
    
    # Show log file location
    logger.info(f"\n📄 Full execution log: {log_filename}")
    
    if success:
        logger.info("\n✅ Pipeline completed successfully!")
        
        # Mark version as complete
        if not dry_run:
            summary = {
                "stages_run": list(pipeline.completed_stages),
                "mode": mode,
                "artifacts_uploaded": False
            }
            version_mgr.mark_version_complete(version_id, summary)
        
        # Handle artifact upload if requested
        if upload_artifacts and not dry_run:
            logger.info("\n📤 Uploading artifacts to cloud...")
            if pipeline.create_and_upload_tarball():
                logger.info("✅ Artifacts uploaded successfully")
                # Update summary to reflect upload
                summary["artifacts_uploaded"] = True
                version_mgr.update_version(version_id, summary)
            else:
                logger.error("❌ Failed to upload artifacts")
                logger.info("You can retry with: python scripts/standalone/upload_artifacts.py --version-id " + version_id)
        elif not upload_artifacts:
            logger.info("\n💡 Tip: To upload artifacts after review, run:")
            logger.info(f"    python scripts/standalone/upload_artifacts.py --version-id {version_id}")
    else:
        logger.error("\n❌ Pipeline failed!")
        
        # Show which stage failed and why
        failed_stages = pipeline.pipeline_stats.get("stage_errors", {})
        if failed_stages:
            for stage, error in failed_stages.items():
                logger.error(f"\n🚫 Stage '{stage}' failed with error:")
                logger.error(f"   {error}")
        
        logger.error(f"\n📝 See full error details in: {log_filename}")
        sys.exit(1)


if __name__ == "__main__":
    main()

