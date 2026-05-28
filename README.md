# FTO-Sim

FTO-Sim is an open-source framework for evaluating cooperative perception in urban traffic using SUMO-based simulation and ray-tracing-based visibility analysis.

## Scope of This README

To match the pushed repository scope, this documentation describes:

- `scripts/main.py`
- `scripts/performance_optimizer.py`
- `scripts/evaluation_spatial_visibility.py`
- `scripts/evaluation_VRU_specific_detection.py`

and only the following simulation example folders:

- `simulation_examples/Intersection-Redesign_Ilic-TR-PartA-2026`
- `simulation_examples/Spatial-Visibility_Ilic-TRB-2025`
- `simulation_examples/VRU-specific-Detection_Ilic-TRA-2026`

## Table of Contents

1. [Papers and Example Mapping](#papers-and-example-mapping)
2. [Framework Architecture](#framework-architecture)
3. [Methodology](#methodology)
   1. [Input Data Model](#31-input-data-model)
   2. [Coordinate and Geometry Pipeline](#32-coordinate-and-geometry-pipeline)
   3. [Observer Assignment](#33-observer-assignment)
   4. [Ray Tracing and Occlusion Logic](#34-ray-tracing-and-occlusion-logic)
   5. [Visibility Polygon and Grid Updates](#35-visibility-polygon-and-grid-updates)
   6. [Relative Visibility and LoV](#36-relative-visibility-and-lov)
   7. [VRU-Specific Detection Evaluation](#37-vru-specific-detection-evaluation)
4. [Script-Level Documentation](#script-level-documentation)
   1. [main.py](#41-mainpy)
   2. [performance_optimizer.py](#42-performance_optimizerpy)
   3. [evaluation_spatial_visibility.py](#43-evaluation_spatial_visibilitypy)
   4. [evaluation_VRU_specific_detection.py](#44-evaluation_vru_specific_detectionpy)
5. [Simulation Examples](#simulation-examples)
   1. [Intersection-Redesign_Ilic-TR-PartA-2026](#51-intersection-redesign_ilic-tr-parta-2026)
   2. [Spatial-Visibility_Ilic-TRB-2025](#52-spatial-visibility_ilic-trb-2025)
   3. [VRU-specific-Detection_Ilic-TRA-2026](#53-vru-specific-detection_ilic-tra-2026)
6. [Typical Workflow](#typical-workflow)
7. [Requirements and Setup Notes](#requirements-and-setup-notes)
8. [License](#license)

## Papers and Example Mapping

This section links the included simulation examples to their related publications or manuscript streams.

- **Spatial-Visibility_Ilic-TRB-2025**
  - Related paper: Ilic et al. (2025), TRB 104th Annual Meeting
  - Title: "An Open-Source Framework for Evaluating Cooperative Perception in Urban Areas"
  - Focus: foundational FTO-Sim workflow and spatial visibility / LoV analysis

- **VRU-specific-Detection_Ilic-TRA-2026**
  - Related paper: Ilic et al. (2026), Transport Research Arena (TRA)
  - Title: "Evaluating VRU Detection Performance in Urban Environments using Cooperative Perception"
  - Focus: VRU-specific detection performance, including speed-limit scenario comparison

- **Intersection-Redesign_Ilic-TR-PartA-2026**
  - Related manuscript stream: Transportation Research Part A case-study line (2026)
  - Focus: infrastructure and traffic-policy scenario comparisons (demand, parking, speed variants)

## Framework Architecture

FTO-Sim combines traffic simulation, ray tracing, and post-processing in a modular sequence.

![FTO-Sim Framework Architecture](readme_images/framework_features.png)

*Figure 1: High-level architecture and data flow.*

Core phases:

1. Scenario loading from SUMO inputs
2. Optional contextual enrichment (OSM and GeoJSON)
3. Observer assignment and per-timestep ray tracing
4. Visibility/detection logging during simulation
5. Post-processing into spatial visibility and VRU-specific metrics

## Methodology

### 3.1 Input Data Model

At runtime, `scripts/main.py` combines:

- Dynamic entities from SUMO via libsumo/TraCI
- Scenario topology and demand from SUMO network/route/additional files
- Optional geospatial context (OSM layers and GeoJSON)

The method distinguishes:

- **Static occluders**: infrastructure and static scene elements
- **Dynamic occluders**: moving road users in the current timestep

### 3.2 Coordinate and Geometry Pipeline

The processing pipeline harmonizes multiple coordinate systems into a consistent projected geometry space for robust intersection calculations.

This enables:

- stable polygon/ray computations
- grid-based counting in metric units
- consistent plotting and map overlays

### 3.3 Observer Assignment

Observer assignment is controlled by penetration parameters:

- `FCO_share` for motorized observers
- `FBO_share` for bicycle observers

For each eligible road user, assignment follows a seeded stochastic decision:

$$
\text{observer} =
\begin{cases}
1 & \text{if } u < p \\
0 & \text{otherwise}
\end{cases}
\quad \text{with } u \sim U(0,1),\; p \in \{\text{FCO\_share},\text{FBO\_share}\}
$$

This makes experiments reproducible while preserving randomized observer placement.

### 3.4 Ray Tracing and Occlusion Logic

For each active observer and timestep (after warm-up):

1. Generate `numberOfRays` rays around 360 degrees
2. Limit ray length to `radius`
3. Intersect each ray with static and dynamic geometry
4. Truncate at the nearest hit

The resulting visible segment endpoints describe effective line-of-sight under occlusion.

![Ray Tracing Workflow](readme_images/ray_tracing_flowchart.png)

*Figure 2: Ray tracing sequence used in simulation.*

![Ray Tracing Visualization](readme_images/ray_tracing.png)

*Figure 3: Example ray-level visibility and occlusion outcome.*

### 3.5 Visibility Polygon and Grid Updates

Ray endpoints are angularly ordered and connected to create one visibility polygon per observer per frame.

A spatial grid (`grid_size`) is updated using two parallel counters:

- **Discrete visibility counter**: binary frame-based increment when a cell is visible
- **Continuous visibility counter**: weighted increment using sensor-accuracy lookup and redundancy

The continuous component models higher confidence for multi-observer visibility at the same timestep.

### 3.6 Relative Visibility and LoV

Post-processing in `scripts/evaluation_spatial_visibility.py` computes:

- **Relative Visibility**: normalized visibility representation for spatial comparison
- **Discrete LoV**
- **Continuous LoV**

LoV normalization follows:

$$
\mathrm{LoV}(i) = \frac{C_i}{N_{\text{steps}} \cdot \Delta t}
$$

where:

- $C_i$ is the visibility count for grid cell $i$
- $N_{\text{steps}}$ is total simulation steps
- $\Delta t$ is simulation step length

Cells with no observations are kept as NaN in visualization to distinguish "unobserved" from "low observed".

![Spatial Visibility Processing](readme_images/spatial_visibility_flowchart.png)

*Figure 4: Spatial visibility post-processing flow.*

![Relative Visibility Example](readme_images/relative_visibility.png)

*Figure 5: Relative visibility map output.*

![Level of Visibility Example](readme_images/level_of_visibility.png)

*Figure 6: LoV class-based spatial output.*

### 3.7 VRU-Specific Detection Evaluation

`scripts/evaluation_VRU_specific_detection.py` evaluates bicycle/VRU detectability over time and space, including:

- trajectory-level detection status
- detection redundancy (number of simultaneous observing agents)
- occlusion-level representations
- optional conflict-focused and 3D views

It supports both scenario-level summaries and fine-grained trajectory diagnostics.

![Critical Interaction Areas](readme_images/critical_interaction_areas.png)

*Figure 7: Critical interaction area context used in VRU-focused analyses.*

![2D VRU Trajectory Detection](readme_images/2D_VRU_trajectory_detection.png)

*Figure 8: Example 2D VRU detection trajectory output.*

![3D VRU Trajectory Detection](readme_images/3D_VRU_trajectory_detection.png)

*Figure 9: Example 3D VRU detection output.*

## Script-Level Documentation

### 4.1 main.py

`main.py` is the simulation entry point and orchestrator.

Main responsibilities:

- scenario and runtime configuration
- SUMO startup and simulation stepping
- observer assignment
- ray tracing and visibility polygon generation
- grid visibility counting (discrete + continuous)
- logging detections, trajectories, conflicts, and summaries

Typical command:

```powershell
python scripts/main.py
```

### 4.2 performance_optimizer.py

`performance_optimizer.py` provides optimization and profiling utilities used by `main.py`.

Main responsibilities:

- profiling support for performance diagnostics
- optimized computational paths for ray-processing hotspots
- CPU/GPU-oriented helper logic used by the simulation runtime

### 4.3 evaluation_spatial_visibility.py

`evaluation_spatial_visibility.py` performs post-processing for spatial visibility metrics.

Main outputs:

- relative visibility heatmaps
- discrete LoV maps and logs
- continuous LoV maps and logs

Typical command:

```powershell
python scripts/evaluation_spatial_visibility.py --scenario-path <scenario_output_folder>
```

### 4.4 evaluation_VRU_specific_detection.py

`evaluation_VRU_specific_detection.py` performs VRU-focused post-processing.

Main outputs:

- 2D trajectory detection plots
- flow-based, redundancy, and occlusion-level outputs
- optional 3D visualizations
- summary statistics files

Typical command:

```powershell
python scripts/evaluation_VRU_specific_detection.py --scenario-path <scenario_output_folder>
```

## Simulation Examples

### 5.1 Intersection-Redesign_Ilic-TR-PartA-2026

Case-study family for infrastructure/policy redesign comparisons.

Available SUMO configs:

- `50_low_demand.sumocfg`
- `50_low_demand_30.sumocfg`
- `50_low_demand_no-parking.sumocfg`
- `high_demand.sumocfg`
- `high_demand_30.sumocfg`
- `high_demand_no-parking.sumocfg`
- `high_demand_singleFCO.sumocfg`

Linked manuscript line:

- Transportation Research Part A case-study stream (2026)

### 5.2 Spatial-Visibility_Ilic-TRB-2025

Spatial visibility benchmark scenario family.

Available SUMO configs:

- `Ilic-2025_config_low-demand.sumocfg`
- `Ilic-2025_config_high-demand.sumocfg`

Related paper:

- Ilic et al. (TRB 2025): foundational FTO-Sim and spatial visibility methodology

### 5.3 VRU-specific-Detection_Ilic-TRA-2026

VRU detection scenario family with speed-limit variants.

Available SUMO configs:

- `Ilic-2026_config_30kmh.sumocfg`
- `Ilic-2026_config_50kmh.sumocfg`

Related paper:

- Ilic et al. (TRA 2026): VRU-specific cooperative perception evaluation

## Typical Workflow

1. Choose one scenario from the included simulation example folders.
2. Configure experiment parameters in `scripts/main.py`.
3. Run simulation:

```powershell
python scripts/main.py
```

4. Run spatial visibility evaluation:

```powershell
python scripts/evaluation_spatial_visibility.py --scenario-path <scenario_output_folder>
```

5. Run VRU-specific detection evaluation:

```powershell
python scripts/evaluation_VRU_specific_detection.py --scenario-path <scenario_output_folder>
```

## Requirements and Setup Notes

- Python 3.10+ recommended
- SUMO installation required and callable via command line
- Install dependencies from `requirements.txt`
- Run commands from repository root to preserve relative-path behavior

## License

See `LICENSE`.
