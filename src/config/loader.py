"""
Configuration loader for RAG pipeline models and settings.

Supports YAML configuration files with environment variable overrides.
Provides sensible defaults for local and cloud deployments.
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Optional, Any
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Model configuration."""
    model: str
    url: str
    temperature: float = 0.1
    max_tokens: int = 2048
    timeout: int = 120


class ConfigLoader:
    """Load and manage RAG pipeline configuration."""

    # Default configurations.
    # NOTE: both modes talk to Ollama's native /api/generate and /api/tags
    # endpoints (see src/generation/generator.py) — "cloud" just means an
    # Ollama server running on a free-tier VM instead of localhost. Hosted
    # inference APIs (Together AI, Replicate, etc.) use a different request
    # schema and are NOT drop-in compatible with this generator without an
    # adapter, so they are intentionally not used as defaults here.
    DEFAULTS = {
        "local": {
            "model": "mistral:7b",
            "url": "http://localhost:11434",
            "temperature": 0.1,
            "max_tokens": 2048,
            # Generation with a full prospectus context (multiple retrieved
            # chunks + parent expansion) routinely takes >60s on consumer
            # hardware (CPU/MPS, no dedicated inference GPU). Measured in
            # testing: 60s timed out outright; even 120s was cut close by
            # a cold-start Mistral 7B run (warm runs landed ~90s). 180s
            # gives real headroom instead of matching Ollama's bare default.
            "timeout": 180,
        },
        "cloud": {
            "model": "mistral:7b",
            "url": "http://YOUR_CLOUD_VM_IP:11434",  # Ollama on a free-tier VM
            "temperature": 0.1,
            "max_tokens": 2048,
            "timeout": 180,  # add headroom for network latency on top of generation time
        },
    }

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize config loader.

        Args:
            config_path: Path to YAML config file. If None, looks for
                        config/models.yaml relative to project root.
        """
        self.config_path = config_path or self._find_default_config()
        self.config: Dict[str, Any] = {}
        self._load_config()

    def _find_default_config(self) -> Optional[str]:
        """Find default config file location."""
        # Try relative to this file first
        default_paths = [
            Path(__file__).parent.parent.parent / "config" / "models.yaml",
            Path.cwd() / "config" / "models.yaml",
            Path.cwd() / "models.yaml",
        ]

        for path in default_paths:
            if path.exists():
                logger.info(f"Found config at {path}")
                return str(path)

        return None

    def _load_config(self):
        """Load configuration from file or use defaults."""
        if self.config_path and Path(self.config_path).exists():
            try:
                with open(self.config_path, "r") as f:
                    self.config = yaml.safe_load(f) or {}
                logger.info(f"Loaded config from {self.config_path}")
            except Exception as e:
                logger.warning(f"Failed to load config from {self.config_path}: {e}")
                self.config = self.DEFAULTS.copy()
        else:
            logger.info("Using default configuration")
            self.config = self.DEFAULTS.copy()

    def get_model_config(self, mode: str = "local") -> ModelConfig:
        """
        Get model configuration for the specified mode.

        Args:
            mode: Configuration mode ("local", "cloud", "replicate")
                 or custom key from config file.

        Returns:
            ModelConfig with settings from file or defaults,
            with environment variable overrides applied.
        """
        # Get base config
        if mode in self.config:
            cfg = self.config[mode].copy()
        elif mode in self.DEFAULTS:
            cfg = self.DEFAULTS[mode].copy()
        else:
            logger.warning(f"Unknown mode '{mode}', using 'local' defaults")
            cfg = self.DEFAULTS["local"].copy()

        # Apply environment variable overrides
        cfg["model"] = os.getenv("OLLAMA_MODEL", cfg.get("model"))
        cfg["url"] = os.getenv("OLLAMA_URL", cfg.get("url"))
        cfg["temperature"] = float(os.getenv("OLLAMA_TEMPERATURE", cfg.get("temperature", 0.1)))
        cfg["max_tokens"] = int(os.getenv("OLLAMA_MAX_TOKENS", cfg.get("max_tokens", 2048)))
        cfg["timeout"] = int(os.getenv("OLLAMA_TIMEOUT", cfg.get("timeout", 120)))

        return ModelConfig(**cfg)

    def get_evaluator_config(self) -> Dict[str, Any]:
        """
        Get evaluator configuration.

        Returns:
            Dict with evaluator settings (use_llm_judge, etc.)
        """
        eval_config = self.config.get("evaluator", {})

        # Apply environment overrides
        return {
            "use_llm_judge": os.getenv("USE_LLM_JUDGE", "false").lower() == "true",
            "quick_mode": os.getenv("QUICK_EVAL", "false").lower() == "true",
            **eval_config
        }

    def __repr__(self) -> str:
        """String representation."""
        return f"ConfigLoader(config_path={self.config_path})"


def load_config(mode: str = "local", config_path: Optional[str] = None) -> ModelConfig:
    """
    Convenience function to load config in one call.

    Args:
        mode: Configuration mode ("local", "cloud", etc.)
        config_path: Optional path to config file.

    Returns:
        ModelConfig with settings.
    """
    loader = ConfigLoader(config_path)
    return loader.get_model_config(mode)
