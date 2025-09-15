### Goal
Help contributors and AI agents become productive quickly in FTO-Sim by describing the architecture, entry points, important conventions, and common run/debug workflows.

### Quick facts (what to know first)
- Primary scripts: `scripts/main.py` (simulation entry), `scripts/evaluation_spatial_visibility.py`, `scripts/evaluation_VRU_specific_detection.py` (post-processing).
- SUMO is required (libsumo/traci/sumolib packages in `requirements.txt`). The simulation talks to SUMO via TraCI. Default example SUMO configs live under `simulation_examples/`.
- Outputs are written to `outputs/{project_name}_{file_tag}_FCO{X}%_FBO{Y}%/` with subfolders `out_logging`, `out_raytracing`, `out_spatial_visibility`, `out_VRU-specific_detection`.

### High-level architecture (essential)
- Input: SUMO (dynamic actors via TraCI), OSM (static geometry via `osmnx`), optional GeoJSON (road space visualization).
- Core processing: `scripts/main.py` performs per-observer 360° ray generation, occlusion checks against static (OSM) and dynamic (SUMO) objects, builds visibility polygons and updates grid-based visibility counts.
- Post-processing: evaluation scripts read CSV logs in `outputs/*/out_*` and produce heatmaps/plots. See `evaluation_spatial_visibility.py` and `evaluation_VRU_specific_detection.py` for auto-detection logic and CLI examples.

### Where to start editing (practical guide)
- To change default scenario/config values, edit top of `scripts/main.py` (configuration block near the top: `file_tag`, `sumo_config_path`, `FCO_share`, `FBO_share`, `numberOfRays`, `radius`, `performance_optimization_level`).
- Visualization and plotting styles are implemented inline in `scripts/main.py` and mirrored in evaluation scripts — keep color maps and grid sizes consistent by reading `grid_size` and `file_tag` from logs or main config.

### Important workflows & commands
- Install environment: use `requirements.txt` (recommended inside a venv). The repo uses many geospatial packages; prefer conda for binary dependencies, or pip from `requirements.txt` if wheels available.
- Run a simulation (example): open `scripts/main.py`, set `file_tag` and `sumo_config_path` to an example in `simulation_examples/`, then run `python scripts/main.py` (the script assumes being run from project root and uses relative paths).
- Post-process outputs: `python scripts/evaluation_spatial_visibility.py --scenario-path outputs/<scenario_folder>` or run `python scripts/evaluation_VRU_specific_detection.py --scenario-path outputs/<scenario_folder>` (each script contains auto-detection of parameters and helpful examples in comments).

### Project-specific conventions & patterns
- Configs are mostly file-scoped: edit top-of-file constants in `scripts/main.py` for experiments — there is no central config file. Many evaluation scripts try to auto-detect parameters from `outputs/.../out_logging/*.json` or infer from folder names (pattern: `_FCO{num}%_FBO{num}%`).
- Outputs use a strict folder naming convention: the scenario folder name encodes file tag and FCO/FBO shares. Evaluation scripts parse these names to find parameters.
- Performance modes: `performance_optimization_level` accepts `none`, `cpu`, `gpu`. GPU code paths require optional dependencies (CuPy / CUDA) and fall back gracefully.

### Integration & dependencies to watch for
- SUMO (libsumo/traci/sumolib) — the runtime depends on a local SUMO installation for `traci.start(...)`. Tests and CI must provide SUMO or mock TraCI.
- OSGeo/geopandas/osmnx stack — heavy native dependencies (use conda or appropriate wheels on Windows).
- Optional GPU acceleration: `cupy` and `numba` are optional but change runtime behaviour.

### Useful code examples to reference
- Entry point: look at `scripts/main.py` top-level config and `initialize_performance_settings()` for how threads/GPU are chosen.
- Auto-detection: `scripts/evaluation_spatial_visibility.py::auto_detect_parameters_from_scenario()` demonstrates how logs and file names are parsed for `grid_size`, `step_length`, and scenario info.
- Outputs: `scripts/evaluation_VRU_specific_detection.py::_detect_step_length()` shows how `summary_log_*.csv` is read for metadata comments.

### Do not assume / gotchas
- The project expects to be executed from the repository root; many paths are relative to `scripts/` and the project parent. When running from other working directories, set absolute paths.
- SUMO config parsing is fragile: evaluation scripts include fallbacks and regex parsing of `main.py` if logs are missing.
- Visualization backends use non-interactive Matplotlib (`Agg`) - animation/video saving requires ffmpeg/FFMpegWriter availability on PATH.

### Quick checklist for PRs from an AI agent
- Keep edits local to `scripts/` unless adding well-scoped utilities. If you change default config constants, update README.md examples.
- Preserve existing folder-name parsing patterns (the `_FCO###%_FBO###%` convention) or update detection logic in evaluation scripts when renaming outputs.
- When adding dependencies, update `requirements.txt` and mention platform-specific installation notes (Windows: prefer conda for geopackages; GPU: CUDA/CuPy versions).

### Where to ask questions / next steps
- If uncertain about expected output folder names or SUMO setup, inspect `scripts/main.py` top config and `simulation_examples/` for canonical examples.
- If you need broader design context or experiments, consult the README sections referenced near top-of-file for methodological intent and evaluation goals.

---
If you'd like, I can iterate: (1) compress to a single-page quick reference, (2) add step-by-step run commands for Windows PowerShell, or (3) include a minimal example `run_example.ps1` wrapper. Tell me which you'd prefer.
