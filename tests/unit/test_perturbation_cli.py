"""Unit tests for perturbation CLI arguments and entry point."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from video_text_translator.config import build_argparser


class TestPerturbFlagParsing:
    """Test --perturb flag parsing."""

    def test_perturb_flag_default_false(self):
        """--perturb defaults to False when not specified."""
        parser = build_argparser()
        args = parser.parse_args([])
        assert args.perturb is False

    def test_perturb_flag_set_true(self):
        """--perturb sets the flag to True."""
        parser = build_argparser()
        args = parser.parse_args(["--perturb"])
        assert args.perturb is True

    def test_perturb_with_input_output(self):
        """--perturb can be combined with -i and -o."""
        parser = build_argparser()
        args = parser.parse_args([
            "--perturb", "-i", "input.mp4", "-o", "output.mp4"
        ])
        assert args.perturb is True
        assert args.input_path == "input.mp4"
        assert args.output_path == "output.mp4"


class TestPerturbPresetValidation:
    """Test --perturb-preset argument validation."""

    def test_preset_default_none(self):
        """--perturb-preset defaults to None when not specified."""
        parser = build_argparser()
        args = parser.parse_args(["--perturb"])
        assert args.perturb_preset is None

    def test_preset_light(self):
        """--perturb-preset accepts 'light'."""
        parser = build_argparser()
        args = parser.parse_args(["--perturb", "--perturb-preset", "light"])
        assert args.perturb_preset == "light"

    def test_preset_medium(self):
        """--perturb-preset accepts 'medium'."""
        parser = build_argparser()
        args = parser.parse_args(["--perturb", "--perturb-preset", "medium"])
        assert args.perturb_preset == "medium"

    def test_preset_heavy(self):
        """--perturb-preset accepts 'heavy'."""
        parser = build_argparser()
        args = parser.parse_args(["--perturb", "--perturb-preset", "heavy"])
        assert args.perturb_preset == "heavy"

    def test_preset_invalid_rejected(self):
        """--perturb-preset rejects invalid values."""
        parser = build_argparser()
        with pytest.raises(SystemExit):
            parser.parse_args(["--perturb", "--perturb-preset", "extreme"])


class TestPerturbConfigPath:
    """Test --perturb-config argument."""

    def test_config_default_none(self):
        """--perturb-config defaults to None when not specified."""
        parser = build_argparser()
        args = parser.parse_args(["--perturb"])
        assert args.perturb_config is None

    def test_config_accepts_path(self):
        """--perturb-config accepts a file path string."""
        parser = build_argparser()
        args = parser.parse_args([
            "--perturb", "--perturb-config", "custom/config.yaml"
        ])
        assert args.perturb_config == "custom/config.yaml"

    def test_config_accepts_long_path(self):
        """--perturb-config accepts paths up to reasonable length."""
        parser = build_argparser()
        long_path = "a/" * 100 + "config.yaml"
        args = parser.parse_args([
            "--perturb", "--perturb-config", long_path
        ])
        assert args.perturb_config == long_path


class TestPerturbationEntryPoint:
    """Test the perturbation entry point in main.py."""

    def test_missing_input_returns_error(self):
        """--perturb without --input returns exit code 2."""
        # Import main here to avoid import issues
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from main import main

        exit_code = main(["--perturb", "-o", "output.mp4"])
        assert exit_code == 2

    def test_missing_output_returns_error(self):
        """--perturb without --output returns exit code 2."""
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from main import main

        exit_code = main(["--perturb", "-i", "input.mp4"])
        assert exit_code == 2

    def test_missing_both_input_output_returns_error(self):
        """--perturb without --input and --output returns exit code 2."""
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from main import main

        exit_code = main(["--perturb"])
        assert exit_code == 2

    def test_invalid_config_path_returns_error(self):
        """--perturb-config with non-existent path returns exit code 1."""
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from main import main

        exit_code = main([
            "--perturb",
            "-i", "input.mp4",
            "-o", "output.mp4",
            "--perturb-config", "nonexistent/path.yaml",
        ])
        assert exit_code == 1

    def test_valid_config_with_invalid_yaml_returns_error(self):
        """--perturb-config with invalid YAML content returns exit code 1."""
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from main import main

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write("invalid: yaml: content: [[[")
            tmp_path = f.name

        try:
            exit_code = main([
                "--perturb",
                "-i", "input.mp4",
                "-o", "output.mp4",
                "--perturb-config", tmp_path,
            ])
            assert exit_code == 1
        finally:
            os.unlink(tmp_path)

    def test_default_preset_medium_when_not_specified(self):
        """When --perturb is specified without --perturb-preset, preset defaults to medium."""
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from main import _run_perturbation
        from video_text_translator.config import build_argparser

        parser = build_argparser()
        args = parser.parse_args([
            "--perturb", "-i", "input.mp4", "-o", "output.mp4"
        ])

        with patch("main.load_perturbation_config") as mock_load:
            from video_text_translator.perturbation_config import PerturbationConfig
            mock_load.return_value = PerturbationConfig(
                input_path="input.mp4",
                output_path="output.mp4",
                preset="medium",
            )

            with patch("main.PerturbationPipeline") as mock_pipeline:
                mock_pipeline.return_value.run.return_value = 0
                _run_perturbation(args)

            # Verify load_perturbation_config was called with preset_override="medium"
            mock_load.assert_called_once_with(
                yaml_path="configs/perturbation.yaml",
                preset_override="medium",
                param_overrides={
                    "input_path": "input.mp4",
                    "output_path": "output.mp4",
                },
            )

    def test_explicit_preset_passed_through(self):
        """When --perturb-preset is specified, it is passed to config loader."""
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from main import _run_perturbation
        from video_text_translator.config import build_argparser

        parser = build_argparser()
        args = parser.parse_args([
            "--perturb", "--perturb-preset", "heavy",
            "-i", "input.mp4", "-o", "output.mp4"
        ])

        with patch("main.load_perturbation_config") as mock_load:
            from video_text_translator.perturbation_config import PerturbationConfig
            mock_load.return_value = PerturbationConfig(
                input_path="input.mp4",
                output_path="output.mp4",
                preset="heavy",
            )

            with patch("main.PerturbationPipeline") as mock_pipeline:
                mock_pipeline.return_value.run.return_value = 0
                _run_perturbation(args)

            mock_load.assert_called_once_with(
                yaml_path="configs/perturbation.yaml",
                preset_override="heavy",
                param_overrides={
                    "input_path": "input.mp4",
                    "output_path": "output.mp4",
                },
            )
