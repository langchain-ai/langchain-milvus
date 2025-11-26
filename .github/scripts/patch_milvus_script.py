#!/usr/bin/env python3
"""Setup Milvus with OpenAI credentials and patch standalone script."""

import os
import re
import sys


def create_user_yaml(openai_api_key: str, output_path: str = "user.yaml") -> None:
    """
    Create user.yaml with OpenAI API credentials.

    Args:
        openai_api_key: OpenAI API key to configure
        output_path: Path where to create the user.yaml file
    """
    yaml_content = f"""credential:
  openai_key:
    apikey: {openai_api_key}

function:
  textEmbedding:
    providers:
      openai:
        credential: openai_key
"""

    with open(output_path, "w") as f:
        f.write(yaml_content)

    print(f"Created {output_path} with OpenAI credentials")  # noqa: T201


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
        print("Environment variable OPENAI_API_KEY must be set")  # noqa: T201
        sys.exit(1)

    # Get OpenAI API key from environment
    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        print("Error: OPENAI_API_KEY environment variable not set")  # noqa: T201
        print(  # noqa: T201
            "Please configure OPENAI_API_KEY in GitHub repository secrets"
        )
        sys.exit(1)

    script_path = sys.argv[1]

    # Create user.yaml with OpenAI credentials
    create_user_yaml(openai_api_key)

    # Patch the standalone script to preserve user.yaml
    patch_standalone_script(script_path)
