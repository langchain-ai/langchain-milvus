#!/usr/bin/env python3
"""Patch the standalone_embed.sh script to preserve existing user.yaml."""

import re
import sys


def patch_standalone_script(script_path: str) -> None:
    """
    Patch the standalone_embed.sh script to make user.yaml creation conditional.

    This ensures that if a user.yaml file already exists, it won't be overwritten.

    Args:
        script_path: Path to the standalone_embed.sh script to patch
    """
    with open(script_path, "r") as f:
        content = f.read()

    # Replace the unconditional user.yaml creation with a conditional one
    # Original pattern: cat << EOF > user.yaml ... EOF
    # New: if [ ! -s user.yaml ]; then cat << EOF > user.yaml ... EOF; fi
    pattern = (
        r"(    cat << EOF > user\.yaml\n"
        r"# Extra config to override default milvus\.yaml\nEOF)"
    )
    replacement = r"""    if [ ! -s "./user.yaml" ]; then
        cat << EOF > user.yaml
# Extra config to override default milvus.yaml
EOF
    fi"""

    content = re.sub(pattern, replacement, content)

    with open(script_path, "w") as f:
        f.write(content)

    print(f"Successfully patched {script_path}")  # noqa: T201


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(  # noqa: T201
            "Usage: patch_milvus_script.py <path_to_standalone_embed.sh>"
        )
        sys.exit(1)

    patch_standalone_script(sys.argv[1])
