#!/usr/bin/env python3
"""
Main pipeline orchestrator
Coordinates execution of all pipeline stages with proper dependency management
"""

import click
import logging
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.utils.version_manager import VersionManager
from scripts.utils.config_manager import get_config
from scripts.utils.cloud_artifact_manager import CloudArtifactManager

# Import pipeline stages
from scripts.pipeline import download_data, clean_data, extract_keypairs, extract_features, run_eda

logger = logging.getLogger(__name__)


class Pipeline:
    """Main pipeline executor"""
    
    def __init__(self, version_id: str, config: Dict[str, Any], 
                 dry_run: bool = False, local_only: bool = False,
                 version_manager: Optional[VersionManager] = None):
        self.version_id = version_id
        self.config = config
        self.dry_run = dry_run
        self.local_only = local_only
        
        # Initialize managers
        self.version_manager = version_manager or VersionManager()
        
        # Cloud artifact manager (if not local only)
        if not local_only and config.get("UPLOAD_ARTIFACTS"):
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
            
            # Update version info
            stage_info = {
                "output_path": str(output_path),
                "duration_seconds": duration,
                "completed": True
            }
            self.version_manager.update_stage_info(self.version_id, stage_name, stage_info)
            
            self.completed_stages.add(stage_name)
            logger.info(f"✅ Completed {stage_name} in {duration:.1f}s")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed stage {stage_name}: {e}")
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
            if not success and stage != "eda":  # EDA failure shouldn't stop pipeline
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


@click.command()
@click.option('--mode', type=click.Choice(['full', 'incr', 'force']), 
              default='incr', help='Pipeline run mode')
@click.option('--stages', '-s', multiple=True,
              help='Specific stages to run (default: all)')
@click.option('--version-id', help='Version to work with (default: create new)')
@click.option('--upload-artifacts', is_flag=True, default=False,
              help='Upload artifacts to cloud (default: False)')
@click.option('--include-pii', is_flag=True, default=False,
              help='Include PII data in operations (default: False)')
@click.option('--generate-reports', is_flag=True, default=None,
              help='Generate reports after pipeline')
@click.option('--dry-run', is_flag=True,
              help='Show what would be done without executing')
@click.option('--local-only', is_flag=True,
              help='Development mode - no cloud operations')
@click.option('--log-level', type=click.Choice(['DEBUG', 'INFO', 'WARNING', 'ERROR']),
              default='INFO', help='Logging level')
@click.option('--device-types', type=str, default='desktop',
              help='Device types to process (comma-separated: desktop,mobile)')
def main(mode: str, stages: List[str], version_id: Optional[str],
         upload_artifacts: bool, include_pii: bool, generate_reports: Optional[bool],
         dry_run: bool, local_only: bool, log_level: str, device_types: str):
    """
    Run data processing pipeline with safe defaults.
    
    By default:
    - Artifacts are NOT uploaded to cloud (use --upload-artifacts to enable)
    - PII data is EXCLUDED (use --include-pii to process demographics)
    
    Examples:
        # Local development (default)
        python run_pipeline.py
        
        # Full pipeline with uploads
        python run_pipeline.py --mode full --upload-artifacts
        
        # Run specific stages
        python run_pipeline.py -s clean -s features
        
        # Dry run to see what would happen
        python run_pipeline.py --mode full --dry-run
    """
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load configuration
    config_mgr = get_config()
    config = config_mgr.get_all()
    
    # Override config with command line flags
    if upload_artifacts:
        config["UPLOAD_ARTIFACTS"] = True
    if include_pii:
        config["INCLUDE_PII"] = True
    if generate_reports is not None:
        config["GENERATE_REPORTS"] = generate_reports
    
    # Parse and set device types
    if device_types:
        device_list = [d.strip().lower() for d in device_types.split(",") if d.strip()]
        config["DEVICE_TYPES"] = device_list
        # Also set in environment for config manager
        import os
        os.environ["DEVICE_TYPES"] = device_types
    
    # Show warnings for risky operations
    if upload_artifacts and not include_pii:
        click.echo("📌 Uploading artifacts with PII excluded (default)")
    elif upload_artifacts and include_pii:
        if not click.confirm("⚠️  WARNING: Upload will include PII data. Continue?"):
            return
    
    # Initialize version
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
    
    # Show configuration
    click.echo(f"""
Pipeline Configuration:
- Version: {version_id}
- Mode: {mode}
- Upload to cloud: {'Yes' if config.get('UPLOAD_ARTIFACTS') else 'No (local only)'}
- Include PII: {'Yes' if config.get('INCLUDE_PII') else 'No (excluded)'}
- Generate reports: {'Yes' if config.get('GENERATE_REPORTS') else 'No'}
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
            stages = all_stages[:-1]  # Exclude EDA by default
        else:
            # TODO: For incremental mode, determine what needs to run
            stages = all_stages[:-1]
    
    # Initialize pipeline
    pipeline = Pipeline(
        version_id=version_id,
        config=config,
        dry_run=dry_run,
        local_only=local_only or not config.get('UPLOAD_ARTIFACTS')
    )
    
    # Run pipeline
    success = pipeline.run_all_stages(stages)
    
    if success:
        click.echo(f"\n✅ Pipeline completed successfully!")
        
        # Mark version as complete
        if not dry_run:
            summary = {
                "stages_run": list(pipeline.completed_stages),
                "mode": mode,
                "artifacts_uploaded": config.get('UPLOAD_ARTIFACTS', False)
            }
            version_mgr.mark_version_complete(version_id, summary)
        
        # Generate reports if requested
        if config.get('GENERATE_REPORTS') and not dry_run:
            click.echo("\n📊 Generating reports...")
            # TODO: Call report generation
            click.echo("TODO: Implement report generation")
        
        # Show next steps
        if not config.get('UPLOAD_ARTIFACTS'):
            click.echo("\n💡 Tip: To upload artifacts after review, run:")
            click.echo(f"    python scripts/standalone/upload_artifacts.py --version-id {version_id}")
    else:
        click.echo(f"\n❌ Pipeline failed. Check logs for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()

