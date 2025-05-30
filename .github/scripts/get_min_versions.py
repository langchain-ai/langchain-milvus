import sys

import tomllib
from packaging.version import parse as parse_version
import re

MIN_VERSION_LIBS = ["langchain-core"]


def get_min_version(version: str) -> str:
    # case ^x.x.x
    _match = re.match(r"^\^(\d+(?:\.\d+){0,2})$", version)
    if _match:
        return _match.group(1)

    # case >=x.x.x,<y.y.y
    _match = re.match(r"^>=(\d+(?:\.\d+){0,2}),<(\d+(?:\.\d+){0,2})$", version)
    if _match:
        _min = _match.group(1)
        _max = _match.group(2)
        assert parse_version(_min) < parse_version(_max)
        return _min

    # case x.x.x
    _match = re.match(r"^(\d+(?:\.\d+){0,2})$", version)
    if _match:
        return _match.group(1)

    raise ValueError(f"Unrecognized version format: {version}")


def get_min_version_from_toml(toml_path: str):
    # Parse the TOML file
    with open(toml_path, "rb") as file:
        toml_data = tomllib.load(file)

    # Get the dependencies - check both Poetry and PEP 621 formats
    dependencies = {}

    # Try PEP 621 format first (project.dependencies)
    if "project" in toml_data and "dependencies" in toml_data["project"]:
        # PEP 621 format stores dependencies as a list of strings
        deps_list = toml_data["project"]["dependencies"]
        for dep in deps_list:
            # Parse dependency string like "langchain-core>=0.2.38,<0.4"
            if ">=" in dep:
                name = dep.split(">=")[0].strip()
                version = dep.split(">=")[1].strip()
                dependencies[name] = version
            elif "^" in dep:
                name, version = dep.split("^")
                dependencies[name.strip()] = f"^{version.strip()}"
            elif "==" in dep:
                name, version = dep.split("==")
                dependencies[name.strip()] = version.strip()

    # Fall back to Poetry format (tool.poetry.dependencies)
    elif (
        "tool" in toml_data
        and "poetry" in toml_data["tool"]
        and "dependencies" in toml_data["tool"]["poetry"]
    ):
        dependencies = toml_data["tool"]["poetry"]["dependencies"]

    # Initialize a dictionary to store the minimum versions
    min_versions = {}

    # Iterate over the libs in MIN_VERSION_LIBS
    for lib in MIN_VERSION_LIBS:
        # Check if the lib is present in the dependencies
        if lib in dependencies:
            # Get the version string
            version_string = dependencies[lib]

            # Use parse_version to get the minimum supported version from version_string
            min_version = get_min_version(version_string)

            # Store the minimum version in the min_versions dictionary
            min_versions[lib] = min_version

    return min_versions


# Get the TOML file path from the command line argument
toml_file = sys.argv[1]

# Call the function to get the minimum versions
min_versions = get_min_version_from_toml(toml_file)

print(" ".join([f"{lib}=={version}" for lib, version in min_versions.items()]))
