#!/usr/bin/env python3
"""
Centralized logging configuration for the FPD ETL Pipeline
Provides consistent logging setup across all modules
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
import json


class PipelineLogger:
    """Centralized logger configuration for the pipeline"""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.log_file_path = None
            self.command_info = {}
            PipelineLogger._initialized = True
    
    def setup_logging(self, 
                     version_id: str,
                     script_name: str,
                     command_args: Dict[str, Any] = None,
                     log_level: str = "INFO",
                     artifacts_dir: str = "artifacts") -> Path:
        """
        Set up dual logging (file + console) with consistent formatting
        
        Args:
            version_id: Pipeline version ID
            script_name: Name of the script being run (e.g., 'pipeline', 'upload_artifacts')
            command_args: Command line arguments as dictionary
            log_level: Console logging level (file always gets DEBUG)
            artifacts_dir: Base artifacts directory
            
        Returns:
            Path to the log file
        """
        # Create log directory
        log_dir = Path(artifacts_dir) / version_id / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log filename with script name and timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_filename = log_dir / f"{script_name}_{timestamp}.log"
        self.log_file_path = log_filename
        
        # Store command info
        self.command_info = {
            "script": script_name,
            "version_id": version_id,
            "timestamp": datetime.now().isoformat(),
            "command_args": command_args or {},
            "python_version": sys.version.split()[0],
            "platform": sys.platform
        }
        
        # Configure root logger
        root_logger = logging.getLogger()
        
        # Remove any existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
            
        root_logger.setLevel(logging.DEBUG)
        
        # File handler - detailed logs with full formatting
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        
        # Console handler - summary only with simple formatting
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level))
        
        # Custom console formatter that adds spacing
        class SpacedConsoleFormatter(logging.Formatter):
            def format(self, record):
                # Add newline after stage completion messages
                if "✅ Completed" in record.getMessage() or "❌ Failed" in record.getMessage():
                    return f"{record.getMessage()}\n"
                return record.getMessage()
        
        console_formatter = SpacedConsoleFormatter()
        console_handler.setFormatter(console_formatter)
        
        # Add handlers
        root_logger.addHandler(file_handler)
        root_logger.addHandler(console_handler)
        
        # Write command info to log file header
        self._write_log_header()
        
        return log_filename
    
    def _write_log_header(self):
        """Write command information header to log file"""
        logger = logging.getLogger('LOGGER_CONFIG')
        
        logger.info("="*80)
        logger.info("PIPELINE EXECUTION LOG")
        logger.info("="*80)
        logger.info(f"Script: {self.command_info['script']}")
        logger.info(f"Version ID: {self.command_info['version_id']}")
        logger.info(f"Start Time: {self.command_info['timestamp']}")
        logger.info(f"Python Version: {self.command_info['python_version']}")
        logger.info(f"Platform: {self.command_info['platform']}")
        
        if self.command_info['command_args']:
            logger.info("\nCommand Arguments:")
            for key, value in self.command_info['command_args'].items():
                logger.info(f"  --{key}: {value}")
        
        # Also write as JSON for programmatic access
        logger.debug(f"\nCommand Info (JSON):\n{json.dumps(self.command_info, indent=2)}")
        logger.info("="*80 + "\n")
    
    def get_log_file_path(self) -> Optional[Path]:
        """Get the current log file path"""
        return self.log_file_path
    
    def log_error_details(self, stage: str, error: Exception):
        """Log detailed error information for failed stages"""
        logger = logging.getLogger('ERROR_DETAILS')
        
        # Log to file with full details
        logger.error(f"\nDETAILED ERROR INFORMATION FOR STAGE: {stage}")
        logger.error("="*60)
        logger.error(f"Error Type: {type(error).__name__}")
        logger.error(f"Error Message: {str(error)}")
        
        # Include traceback if available
        import traceback
        tb = traceback.format_exc()
        if tb and tb != "NoneType: None\n":
            logger.error(f"Traceback:\n{tb}")
        
        logger.error("="*60 + "\n")
        
        # Show summary on console
        console_logger = logging.getLogger('CONSOLE')
        console_logger.error(f"\n🚨 STAGE FAILED: {stage}")
        console_logger.error(f"   Error: {type(error).__name__}: {str(error)}")
        console_logger.error(f"   See log for full details: {self.log_file_path}\n")


def setup_pipeline_logging(version_id: str, 
                          script_name: str = "pipeline",
                          command_args: Dict[str, Any] = None,
                          log_level: str = "INFO") -> Path:
    """
    Convenience function to set up logging for pipeline scripts
    
    Returns:
        Path to the log file
    """
    logger_config = PipelineLogger()
    return logger_config.setup_logging(
        version_id=version_id,
        script_name=script_name,
        command_args=command_args,
        log_level=log_level
    )


def get_pipeline_logger() -> PipelineLogger:
    """Get the singleton PipelineLogger instance"""
    return PipelineLogger()