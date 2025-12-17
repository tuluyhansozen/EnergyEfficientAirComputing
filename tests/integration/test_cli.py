import contextlib
import sys
from unittest.mock import MagicMock, patch

import pytest

from aircompsim.cli import main


class TestCLI:
    def test_cli_help(self):
        with (
            patch.object(sys, "argv", ["aircompsim", "--help"]),
            pytest.raises(SystemExit) as excinfo,
        ):
            main()
        assert excinfo.value.code == 0

    def test_cli_run_simulation(self, tmp_path):
        config_path = tmp_path / "config.yaml"
        output_dir = tmp_path / "results"

        # Mock the simulation runner
        # Configure mock to return an object with float attributes
        mock_results = MagicMock()
        mock_results.total_tasks = 100
        mock_results.successful_tasks = 95
        mock_results.success_rate = 0.95
        mock_results.avg_latency = 0.5
        mock_results.avg_qos = 0.8
        mock_results.total_energy = 50.0

        config_path.touch()

        # The CLI uses a flat argument structure, not 'run' subcommand
        with (
            patch("aircompsim.cli.run_single_simulation", return_value=mock_results) as mock_run,
            patch.object(
                sys,
                "argv",
                ["aircompsim", "--config", str(config_path), "--output", str(output_dir)],
            ),
            contextlib.suppress(SystemExit),
        ):
            main()

        mock_run.assert_called_once()

    def test_cli_drl_flag(self):
        # The CLI calls run_drl_training when --drl is passed
        # Configure mock results
        mock_results = MagicMock()
        mock_results.total_tasks = 100
        mock_results.successful_tasks = 90
        mock_results.success_rate = 0.90
        mock_results.avg_latency = 0.6
        mock_results.avg_qos = 0.7
        mock_results.total_energy = 60.0

        with (
            patch("aircompsim.cli.run_drl_training", return_value=mock_results) as mock_drl,
            patch.object(sys, "argv", ["aircompsim", "--drl"]),
            contextlib.suppress(SystemExit),
        ):
            main()
        mock_drl.assert_called_once()
