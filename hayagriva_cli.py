"""Hayagriva CLI integration for format conversion.

This module provides functions to interact with the Hayagriva CLI tool.
"""

import subprocess
import tempfile
import os
import shutil
import re


# Regular expression to remove ANSI escape codes
ANSI_ESCAPE_PATTERN = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")


def strip_ansi_codes(text):
    """
    Remove ANSI escape codes from text.

    Args:
        text (str): Text that may contain ANSI escape codes

    Returns:
        str: Clean text without ANSI codes
    """
    if text is None:
        return ""
    return ANSI_ESCAPE_PATTERN.sub("", text)


def is_hayagriva_cli_available():
    """
    Check if the Hayagriva CLI is available on the system.

    Returns:
        bool: True if the Hayagriva CLI is available, False otherwise
    """
    return shutil.which("hayagriva") is not None


def get_available_styles():
    """
    Get a list of available citation styles from Hayagriva CLI.

    Returns:
        list: List of available citation style names

    Raises:
        ValueError: If the Hayagriva CLI is not available or fails
    """
    if not is_hayagriva_cli_available():
        raise ValueError(
            "Hayagriva CLI not found. Install from github.com/typst/hayagriva"
        )

    try:
        # Create a temporary file for testing
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yml", delete=False
        ) as temp_file:
            temp_file.write("ref_1:\n  type: book\n  title: Test\n")
            temp_file_path = temp_file.name

        # Get styles list
        cmd = ["hayagriva", temp_file_path, "--format", "yaml", "styles"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Parse the output to extract style names
        styles = []
        for line in result.stdout.splitlines():
            if line.startswith("-"):
                style_name = line.strip("- ").strip()
                styles.append(style_name)

        return styles
    except subprocess.CalledProcessError as e:
        raise ValueError(f"Hayagriva CLI error: {e.stderr}")
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)


def format_references(yaml_content, style):
    """
    Format references in a specific citation style using Hayagriva CLI.

    Args:
        yaml_content (str): Hayagriva YAML content to format
        style (str): Citation style to use (e.g., 'vancouver', 'apa')

    Returns:
        str: Formatted references in the requested style

    Raises:
        ValueError: If the formatting fails or the style is unsupported
    """
    # Check if hayagriva CLI is available
    if not is_hayagriva_cli_available():
        raise ValueError(
            "Hayagriva CLI not found. Install from github.com/typst/hayagriva"
        )

    # Create a temporary file for the YAML content
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yml", delete=False
    ) as temp_file:
        temp_file.write(yaml_content)
        temp_file_path = temp_file.name

    try:
        # Run the Hayagriva CLI command for formatting references
        # Add the -n flag to disable ANSI formatting
        cmd = [
            "hayagriva",
            temp_file_path,
            "--format",
            "yaml",
            "-n",
            "reference",
            "--style",
            style,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        # Strip any remaining ANSI codes from the output
        return strip_ansi_codes(result.stdout)
    except subprocess.CalledProcessError as e:
        raise ValueError(f"Hayagriva CLI error: {strip_ansi_codes(e.stderr)}")
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)


def convert_with_hayagriva_cli(yaml_content, output_format):
    """
    Use the Hayagriva CLI to convert YAML to other formats.
    This leverages the native Rust implementation for better performance.

    Args:
        yaml_content (str): Hayagriva YAML content to convert
        output_format (str): Target format (bibtex, csl-json, ris)

    Returns:
        str: Converted content in the requested format

    Raises:
        ValueError: If the conversion fails or the format is unsupported
    """
    # For technical format conversion, we need to use a different approach
    # since hayagriva CLI doesn't directly support export to these formats
    # without a style parameter

    # Check if hayagriva CLI is available
    if not is_hayagriva_cli_available():
        raise ValueError(
            "Hayagriva CLI not found. Install from github.com/typst/hayagriva"
        )

    # Create a temporary file for the YAML content
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".yml", delete=False
    ) as temp_file:
        temp_file.write(yaml_content)
        temp_file_path = temp_file.name

    try:
        # Currently, the Hayagriva CLI doesn't support direct export to these formats
        # So we'll raise an error to fall back to the Python implementation
        raise ValueError(
            "Hayagriva CLI doesn't support direct export to technical formats yet."
        )
    finally:
        # Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
