# Installation (MesoHOPS environment)

This repository targets a MesoHOPS simulation environment. To create a reproducible environment:

1. Create a virtual environment (Python 3.8+):

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

2. Install base dependencies:

```bash
pip install -r requirements.txt
```

3. Install MesoHOPS and mesohops-specific dependencies (system-dependent):

```bash
pip install -r requirements-mesohops.txt
```

4. Run tests (will skip mesohops tests if mesohops not installed):

```bash
pytest -q
```

Notes:
- Pin `mesohops` to a compatible version if your environment requires a different release.
- If `mesohops` installation requires system libraries, consult the MesoHOPS project documentation.
