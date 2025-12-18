"""Unit tests for config loader module."""

import json
from pathlib import Path

import pytest

from aircompsim.config.loader import (
    get_default_config_path,
    load_config,
    load_json,
    load_yaml,
    merge_configs,
    save_json,
    save_yaml,
)


class TestLoadYAML:
    """Tests for load_yaml function."""

    def test_load_yaml_success(self, tmp_path):
        """Test successful YAML loading."""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(
            """
simulation:
  time_limit: 1000
  user_count: 20
"""
        )

        config = load_yaml(yaml_file)

        assert config["simulation"]["time_limit"] == 1000
        assert config["simulation"]["user_count"] == 20

    def test_load_yaml_file_not_found(self, tmp_path):
        """Test loading nonexistent YAML file."""
        with pytest.raises(FileNotFoundError):
            load_yaml(tmp_path / "nonexistent.yaml")

    def test_load_yaml_empty_file(self, tmp_path):
        """Test loading empty YAML file."""
        yaml_file = tmp_path / "empty.yaml"
        yaml_file.write_text("")

        config = load_yaml(yaml_file)

        assert config == {}

    def test_load_yaml_invalid_syntax(self, tmp_path):
        """Test loading invalid YAML file."""
        yaml_file = tmp_path / "invalid.yaml"
        yaml_file.write_text("invalid: yaml: syntax: [")

        with pytest.raises(ValueError, match="Invalid YAML"):
            load_yaml(yaml_file)


class TestLoadJSON:
    """Tests for load_json function."""

    def test_load_json_success(self, tmp_path):
        """Test successful JSON loading."""
        json_file = tmp_path / "config.json"
        json_file.write_text('{"simulation": {"time_limit": 1000}}')

        config = load_json(json_file)

        assert config["simulation"]["time_limit"] == 1000

    def test_load_json_file_not_found(self, tmp_path):
        """Test loading nonexistent JSON file."""
        with pytest.raises(FileNotFoundError):
            load_json(tmp_path / "nonexistent.json")

    def test_load_json_invalid_syntax(self, tmp_path):
        """Test loading invalid JSON file."""
        json_file = tmp_path / "invalid.json"
        json_file.write_text("{invalid json}")

        with pytest.raises(ValueError, match="Invalid JSON"):
            load_json(json_file)


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_config_yaml(self, tmp_path):
        """Test loading YAML config file."""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("key: value")

        config = load_config(yaml_file)

        assert config["key"] == "value"

    def test_load_config_yml(self, tmp_path):
        """Test loading .yml config file."""
        yml_file = tmp_path / "config.yml"
        yml_file.write_text("key: value")

        config = load_config(yml_file)

        assert config["key"] == "value"

    def test_load_config_json(self, tmp_path):
        """Test loading JSON config file."""
        json_file = tmp_path / "config.json"
        json_file.write_text('{"key": "value"}')

        config = load_config(json_file)

        assert config["key"] == "value"

    def test_load_config_unsupported_format(self, tmp_path):
        """Test loading unsupported format."""
        txt_file = tmp_path / "config.txt"
        txt_file.write_text("key=value")

        with pytest.raises(ValueError, match="Unsupported configuration format"):
            load_config(txt_file)


class TestSaveYAML:
    """Tests for save_yaml function."""

    def test_save_yaml_success(self, tmp_path):
        """Test successful YAML saving."""
        data = {"simulation": {"time_limit": 1000}}
        output_file = tmp_path / "output.yaml"

        save_yaml(data, output_file)

        assert output_file.exists()
        content = output_file.read_text()
        assert "time_limit" in content

    def test_save_yaml_creates_parent_dirs(self, tmp_path):
        """Test that save_yaml creates parent directories."""
        data = {"key": "value"}
        output_file = tmp_path / "subdir" / "output.yaml"

        save_yaml(data, output_file)

        assert output_file.exists()


class TestSaveJSON:
    """Tests for save_json function."""

    def test_save_json_success(self, tmp_path):
        """Test successful JSON saving."""
        data = {"simulation": {"time_limit": 1000}}
        output_file = tmp_path / "output.json"

        save_json(data, output_file)

        assert output_file.exists()
        loaded = json.loads(output_file.read_text())
        assert loaded["simulation"]["time_limit"] == 1000

    def test_save_json_creates_parent_dirs(self, tmp_path):
        """Test that save_json creates parent directories."""
        data = {"key": "value"}
        output_file = tmp_path / "subdir" / "output.json"

        save_json(data, output_file)

        assert output_file.exists()


class TestMergeConfigs:
    """Tests for merge_configs function."""

    def test_merge_simple(self):
        """Test merging simple configs."""
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}

        result = merge_configs(base, override)

        assert result == {"a": 1, "b": 3, "c": 4}

    def test_merge_nested(self):
        """Test merging nested configs."""
        base = {"sim": {"time": 1000, "users": 10}}
        override = {"sim": {"users": 20}}

        result = merge_configs(base, override)

        assert result["sim"]["time"] == 1000
        assert result["sim"]["users"] == 20

    def test_merge_deeply_nested(self):
        """Test merging deeply nested configs."""
        base = {"level1": {"level2": {"level3": {"a": 1, "b": 2}}}}
        override = {"level1": {"level2": {"level3": {"b": 3}}}}

        result = merge_configs(base, override)

        assert result["level1"]["level2"]["level3"]["a"] == 1
        assert result["level1"]["level2"]["level3"]["b"] == 3

    def test_merge_empty_override(self):
        """Test merging with empty override."""
        base = {"a": 1, "b": 2}

        result = merge_configs(base, {})

        assert result == base

    def test_merge_empty_base(self):
        """Test merging with empty base."""
        override = {"a": 1, "b": 2}

        result = merge_configs({}, override)

        assert result == override


class TestGetDefaultConfigPath:
    """Tests for get_default_config_path function."""

    def test_returns_path(self):
        """Test that function returns a Path object."""
        result = get_default_config_path()

        assert isinstance(result, Path)
