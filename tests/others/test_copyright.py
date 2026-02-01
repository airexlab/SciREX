# Copyright (c) 2024 Zenteiq Aitech Innovations Private Limited and
# AiREX Lab, Indian Institute of Science, Bangalore.
# All rights reserved.
#
# This file is part of SciREX
# (Scientific Research and Engineering eXcellence Platform),
# developed jointly by Zenteiq Aitech Innovations and AiREX Lab
# under the guidance of Prof. Sashikumaar Ganesan.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# For any clarifications or special considerations,
# please contact: contact@scirex.org

# Author: Thivin Anandh D (https://thivinanandh.github.io)
# Version Info: 06/Jan/2025: Initial version - Thivin Anandh D

from pathlib import Path

import pytest


def get_python_files():
    """Get all Python files that should have copyright headers."""
    root_dir = Path.cwd()
    python_files = []
    scan_dirs = ["scirex", "tests", "examples"]
    exclude_patterns = [
        "__init__.py",
        "experimental",  # Exclude experimental folder
    ]

    for dir_name in scan_dirs:
        dir_path = root_dir / dir_name
        if dir_path.exists():
            for py_file in dir_path.rglob("*.py"):
                # Skip files matching exclude patterns
                if any(pattern in str(py_file) for pattern in exclude_patterns):
                    continue
                python_files.append(py_file)

    return python_files


def get_copyright_header():
    """Load the copyright header template."""
    root_dir = Path.cwd()
    try:
        header_path = root_dir / "tests" / "support_files" / "CopyrightHeader.txt"
        with open(header_path) as f:
            return f.read().strip()
    except FileNotFoundError:
        pytest.fail(f"Copyright header template file not found at {header_path}")


def test_copyright():
    """Test if all Python files have the required copyright header."""
    copyright_header = get_copyright_header()
    python_files = get_python_files()

    if not python_files:
        pytest.skip("No Python files found to check")

    # Check each file for copyright header
    files_missing_header = []
    for file_path in python_files:
        try:
            with open(file_path) as f:
                content = f.read()
                if copyright_header not in content:
                    files_missing_header.append(str(file_path.relative_to(Path.cwd())))
        except Exception as e:
            pytest.fail(f"Error reading file {file_path}: {e!s}")

    if files_missing_header:
        files_list = "\n  - ".join(files_missing_header)
        error_msg = f"\n\nThe following {len(files_missing_header)} file(s) are missing the copyright header:\n  - {files_list}\n"
        pytest.fail(error_msg)


if __name__ == "__main__":
    test_copyright()
