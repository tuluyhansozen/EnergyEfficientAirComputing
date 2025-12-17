"""Configuration loading utilities.

This module provides functions to load simulation configuration
from YAML and JSON files.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Union

logger = logging.getLogger(__name__)


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        path: Path to YAML file.

    Returns:
        Configuration dictionary.

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If file is not valid YAML.
    """
    try:
        import yaml  # type: ignore
    except ImportError:
        raise ImportError(
            "PyYAML is required for YAML config loading. " "Install with: pip install PyYAML"
        ) from ImportError

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    try:
        with path.open() as f:
            data = yaml.safe_load(f) or {}
            logger.info(f"Loaded configuration from {path}")
            return dict(data)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML file: {e}") from e


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from JSON file.

    Args:
        path: Path to JSON file.

    Returns:
        Configuration dictionary.

    Raises:
        FileNotFoundError: If file doesn't exist.
        ValueError: If file is not valid JSON.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    try:
        with path.open() as f:
            data = json.load(f)
            logger.info(f"Loaded configuration from {path}")
            return dict(data)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON file: {e}") from e


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load configuration from file (auto-detect format).

    Supports YAML (.yaml, .yml) and JSON (.json) files.

    Args:
        path: Path to configuration file.

    Returns:
        Configuration dictionary.

    Raises:
        ValueError: If file format is not supported.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in [".yaml", ".yml"]:
        return load_yaml(path)
    elif suffix == ".json":
        return load_json(path)
    else:
        raise ValueError(
            f"Unsupported configuration format: {suffix}. " "Use .yaml, .yml, or .json"
        )


def save_yaml(data: Dict[str, Any], path: Union[str, Path]) -> None:
    """Save configuration to YAML file.

    Args:
        data: Configuration dictionary.
        path: Output path.
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required. Install with: pip install PyYAML") from ImportError

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Saved configuration to {path}")


def save_json(data: Dict[str, Any], path: Union[str, Path]) -> None:
    """Save configuration to JSON file.

    Args:
        data: Configuration dictionary.
        path: Output path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Saved configuration to {path}")


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two configuration dictionaries.

    Args:
        base: Base configuration.
        override: Override values.

    Returns:
        Merged configuration.
    """
    result = base.copy()

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value

    return result


def get_default_config_path() -> Path:
    """Get path to default configuration file."""
    # Check common locations
    locations = [
        Path("config/simulation.yaml"),
        Path("config/default.yaml"),
        Path("simulation.yaml"),
    ]

    for loc in locations:
        if loc.exists():
            return loc

    # Return first location (will be created if needed)
    return locations[0]
