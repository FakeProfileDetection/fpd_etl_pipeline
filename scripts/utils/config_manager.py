"""
Configuration management for the pipeline
Loads settings from environment files and provides a unified interface
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class ConfigManager:
    """Manage configuration from environment files"""

    def __init__(self, config_dir: Path = Path("config")):
        self.config_dir = config_dir
        self.config: Dict[str, Any] = {}
        self._load_config()

    def _load_config(self):
        """Load configuration from .env files"""
        # Load base configuration
        base_env = self.config_dir / ".env.base"
        if base_env.exists():
            load_dotenv(base_env)
            logger.info(f"Loaded base config from {base_env}")

        # Load shared team configuration
        shared_env = self.config_dir / ".env.shared"
        if shared_env.exists():
            load_dotenv(shared_env, override=True)
            logger.info(f"Loaded shared config from {shared_env}")

        # Load local overrides
        local_env = self.config_dir / ".env.local"
        if local_env.exists():
            load_dotenv(local_env, override=True)
            logger.info(f"Loaded local config from {local_env}")

        # Parse environment variables
        self._parse_env_vars()

    def _parse_env_vars(self):
        """Parse environment variables into config dict"""
        # Boolean configs
        self.config["UPLOAD_ARTIFACTS"] = self._parse_bool(
            os.getenv("UPLOAD_ARTIFACTS", "false")
        )
        self.config["INCLUDE_PII"] = self._parse_bool(os.getenv("INCLUDE_PII", "false"))
        self.config["GENERATE_REPORTS"] = self._parse_bool(
            os.getenv("GENERATE_REPORTS", "true")
        )
        self.config["PUBLISH_REPORTS"] = self._parse_bool(
            os.getenv("PUBLISH_REPORTS", "false")
        )

        # Cloud settings
        self.config["PROJECT_ID"] = os.getenv("PROJECT_ID", "")
        self.config["BUCKET_NAME"] = os.getenv("BUCKET_NAME", "")

        # Directory settings (with version placeholder)
        self.config["WEB_APP_DATA_SOURCE"] = os.getenv("WEB_APP_DATA_SOURCE", "uploads")
        self.config["RAW_DATA_DIR"] = os.getenv(
            "RAW_DATA_DIR", "./artifacts/{version_id}/raw_data"
        )
        self.config["CLEANED_DATA_DIR"] = os.getenv(
            "CLEANED_DATA_DIR", "./artifacts/{version_id}/cleaned_data"
        )
        self.config["KEYPAIRS_DIR"] = os.getenv(
            "KEYPAIRS_DIR", "./artifacts/{version_id}/keypairs"
        )
        self.config["FEATURES_DIR"] = os.getenv(
            "FEATURES_DIR", "./artifacts/{version_id}/statistical_features"
        )
        self.config["RELAVANCE_DIR"] = os.getenv(
            "RELAVANCE_DIR", "./artifacts/{version_id}/relavance"
        )

        # Artifact settings
        self.config["ARTIFACT_RETENTION_DAYS"] = int(
            os.getenv("ARTIFACT_RETENTION_DAYS", "90")
        )
        self.config["MAX_ARTIFACT_SIZE_MB"] = int(
            os.getenv("MAX_ARTIFACT_SIZE_MB", "100")
        )

        # PII patterns (comma-separated)
        pii_patterns = os.getenv(
            "PII_EXCLUDE_PATTERNS", "*demographics*,*consent*,*email*"
        )
        self.config["PII_EXCLUDE_PATTERNS"] = [
            p.strip() for p in pii_patterns.split(",") if p.strip()
        ]

        # Processing parameters
        self.config["PARALLEL_WORKERS"] = int(os.getenv("PARALLEL_WORKERS", "4"))
        self.config["SAMPLE_SIZE_DEV"] = int(os.getenv("SAMPLE_SIZE_DEV", "1000"))
        self.config["CHUNK_SIZE"] = int(os.getenv("CHUNK_SIZE", "10000"))

        # Device types to process (comma-separated)
        device_types = os.getenv("DEVICE_TYPES", "desktop")
        self.config["DEVICE_TYPES"] = [
            d.strip().lower() for d in device_types.split(",") if d.strip()
        ]

        # Logging
        self.config["LOG_LEVEL"] = os.getenv("LOG_LEVEL", "INFO")
        self.config["LOG_FORMAT"] = os.getenv("LOG_FORMAT", "json")

        # Google Cloud credentials
        self.config["GOOGLE_APPLICATION_CREDENTIALS"] = os.getenv(
            "GOOGLE_APPLICATION_CREDENTIALS", ""
        )

    def _parse_bool(self, value: str) -> bool:
        """Parse string to boolean"""
        return value.lower() in ["true", "yes", "1", "on"]

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self.config.get(key, default)

    def get_dir(self, key: str, version_id: str) -> Path:
        """Get directory path with version substitution"""
        template = self.config.get(key, "")
        if not template:
            raise ValueError(f"Directory config '{key}' not found")

        # Replace {version_id} placeholder
        path_str = template.format(version_id=version_id)
        return Path(path_str)

    def get_all(self) -> Dict[str, Any]:
        """Get all configuration values"""
        return self.config.copy()

    def validate_cloud_config(self) -> bool:
        """Validate cloud configuration is complete"""
        required = ["PROJECT_ID", "BUCKET_NAME"]

        if self.config.get("UPLOAD_ARTIFACTS"):
            # Only validate if uploads are enabled
            missing = [k for k in required if not self.config.get(k)]
            if missing:
                logger.error(f"Missing required cloud config: {missing}")
                return False

        return True

    def get_safe_config_for_logging(self) -> Dict[str, Any]:
        """Get config with sensitive values redacted"""
        safe_config = self.config.copy()

        # Redact only truly sensitive values (credentials paths)
        sensitive_keys = ["GOOGLE_APPLICATION_CREDENTIALS"]

        for key in sensitive_keys:
            if key in safe_config and safe_config[key]:
                safe_config[key] = "***REDACTED***"

        return safe_config

    def is_development_mode(self) -> bool:
        """Check if running in development mode"""
        return not self.config.get("UPLOAD_ARTIFACTS", False)

    def is_pii_excluded(self) -> bool:
        """Check if PII should be excluded"""
        return not self.config.get("INCLUDE_PII", False)

    def should_generate_reports(self) -> bool:
        """Check if reports should be generated"""
        return self.config.get("GENERATE_REPORTS", True)

    def get_pii_patterns(self) -> List[str]:
        """Get PII exclusion patterns"""
        return self.config.get("PII_EXCLUDE_PATTERNS", [])

    def get_device_types(self) -> List[str]:
        """Get device types to process"""
        return self.config.get("DEVICE_TYPES", ["desktop"])

    def save_run_config(self, version_id: str, output_dir: Path):
        """Save configuration used for this run"""
        run_config = {
            "version_id": version_id,
            "timestamp": datetime.now().isoformat(),
            "config": self.get_safe_config_for_logging(),
        }

        config_file = output_dir / "run_config.json"
        with open(config_file, "w") as f:
            json.dump(run_config, f, indent=2)

        logger.info(f"Saved run configuration to {config_file}")


# Singleton instance
_config_manager: Optional[ConfigManager] = None


def get_config() -> ConfigManager:
    """Get or create config manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


if __name__ == "__main__":
    # Test configuration loading
    from datetime import datetime

    config = get_config()

    print("Configuration loaded:")
    print(json.dumps(config.get_safe_config_for_logging(), indent=2))

    print(f"\nDevelopment mode: {config.is_development_mode()}")
    print(f"PII excluded: {config.is_pii_excluded()}")
    print(f"Generate reports: {config.should_generate_reports()}")

    # Test directory substitution
    test_version = "2025-01-15_10-00-00_test"
    print(f"\nRaw data dir: {config.get_dir('RAW_DATA_DIR', test_version)}")
