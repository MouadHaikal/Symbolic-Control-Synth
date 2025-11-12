# Symbolic-Control-Synth



## Recommended quick start (development)

1. Fetch missing submodules, Create and activate a Python virtual environment in the project root directory:

```bash
git submodule update --init --recurse
python -m venv venv
source .venv/bin/activate
```

2. Build the project with CMake from the repository root:

```bash
# from project root
cmake -S . -B build
cmake --build build
```

Notes:
- The top-level `CMakeLists.txt` configures building the C++ extension located under `cpp/` and integrates pybind11.

3. Install the Python package in editable mode so `symControl` points at your source tree:

```bash
# from project root (virtualenv active)
pip install -e .
```

That installs the Python package metadata while allowing edits to source files without reinstallation.

## Run the tests / example

With the venv active and the package built/installed:

```bash
# run a quick import check
python -c "from symControl.ExternalModules import floodFill; print('floodFill is available:', callable(floodFill))"

# run the provided mapping script (example test)
python tests/mapping.py
```
