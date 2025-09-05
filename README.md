# FTO-Sim

## Introduction

*FTO-Sim* is an open-source simulation framework for Floating Traffic Observation (FTO) that integrates SUMO traffic simulation with advanced ray tracing techniques to analyze the visibility coverage of Floating Car Observers (FCOs) and Floating Bike Observers (FBOs) in urban environments. The framework enables comprehensive evaluation of cooperative perception systems and their effectiveness in detecting vulnerable road users.

The FTO concept is adapted from the Floating Car Observer (FCO) method, utilizing extended floating car data (xFCD) for traffic planning and traffic management purposes. Additionally, *FTO-Sim* introduces further observer vehicle types, such as Floating Bike Observers (FBO).

## About This Documentation

This README file serves as comprehensive documentation for *FTO-Sim*, combining methodological explanations of the framework with practical user instructions. It provides detailed descriptions of the ray tracing algorithm and subsequent evaluation metrics, alongside step-by-step configuration guides, usage examples, and implementation details.

## Table of Contents
1. [Citation](#citation)
2. [Features](#features)
3. [Configuration](#configuration)
4. [Ray Tracing](#ray-tracing)
5. [Data Collection and Logging](#data-collection-and-logging)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Installation / Prerequisites](#installation--prerequisites)
8. [Usage](#usage)
9. [Performance Considerations](#performance-considerations)
10. [Dependencies](#dependencies)
11. [Error Handling and Troubleshooting](#error-handling-and-troubleshooting)

## Citation
When using *FTO-Sim*, please cite the following references:

### Primary Reference

* **Ilic, M., et al.** "FTO-Sim: An Open-Source Framework for Evaluating Cooperative Perception in Urban Areas." *European Transport Research Review*, 2024. *(Publication pending - [Link will be available upon publication])*

This paper presents the current version of the FTO-Sim framework with comprehensive methodological foundations and explanations about implemented evaluation metrics. Besides *spatial visibility analysis*, additional evaluation metrics, including *VRU-specific detection*, are implemented and described. Furthermore, the paper introduces simulation examples included in *FTO-Sim*, that enable a fast applpication of the simulation framework for other users.

### Secondary References

* [**Ilic, M., et al.**](https://www.researchgate.net/publication/383272173_An_Open-Source_Framework_for_Evaluating_Cooperative_Perception_in_Urban_Areas) "An Open-Source Framework for Evaluating Cooperative Perception in Urban Areas." *Transportation Research Board 104th Annual Meeting*, Washington D.C., 2025. *(Accepted for presentation)*

This paper introduces the initial version of FTO-Sim with a first implementation of spatial visibility analysis metrics. It presents the foundational occlusion modeling approach and demonstrates the framework's capability for analyzing relative visibility patterns and the Level of Visibility (LoV),  originally introduced by [Pechinger et al.](https://www.researchgate.net/publication/372952261_THRESHOLD_ANALYSIS_OF_STATIC_AND_DYNAMIC_OCCLUSION_IN_URBAN_AREAS_A_CONNECTED_AUTOMATED_VEHICLE_PERSPECTIVE). Through a case study, potential for further calibration of the LoV metric have been identified.

## Features
The following sub-chapters elaborate on the different modules and functionalities of *FTO-Sim*, which are organized in a modular workflow as illustrated in the figure below.

![FTO-Sim Framework Architecture](readme_images/framework_features.png)
*Figure 1: Workflow of FTO-Sim*

### Framework Modules

The FTO-Sim framework consists of six core modules that work together to provide comprehensive cooperative perception analysis:

#### 1. Input Data Processing
*FTO-Sim* integrates three primary data sources to create and visualize a comprehensive urban simulation scene:

- **SUMO Simulation**: *FTO-Sim* communicates with a microscopic traffic simulation using SUMO and its integrated TraCI interface (Traffic Control Interface) to retrieve the positions of all static and dynamic road users at each simulation time step. Parked vehicles are treated as static road users, while vehicular traffic and VRUs are treated as dynamic road users. *FTO-Sim* is expecting the path to a SUMO configuration file (`.sumocfg`), which it will use for live communication with the traffic simulation through the TraCI interface to retrieve information from the SUMO network file (`.net.xml`), traffic demand files (`.rou.xml`) and additional files (`.add.xml`).

- **OpenStreetMap (OSM) Data**: Additionally, *FTO-Sim* loads static infrastructure elements, such as buildings or urban greenery (e.g. parks and trees) from OSM, which will be considered during the ray tracing simulation (see [Ray Tracing](#ray-tracing)).

- **GeoJSON Files** *(optional)*: If available, a GeoJSON file can be used by *FTO-Sim* to visualize the distribution of road space to support a faster understanding of the simulated scene.

#### 2. Coordinate Transformation
After loading all spatial input data, *FTO-Sim* performs automated coordinate system transformations to ensure spatial consistency between different data sources. This module handles conversions between SUMO's local coordinate system, WGS84 geographic coordinates (from OSM and GeoJSON), and projected coordinate systems (UTM) for accurate geometric calculations.

#### 3. Configuration & Initialization
*FTO-Sim* offers a wide range of functionalities and allows customization of, amongst others, essential simulation parameters such as the spatial extent of the simulated scene (bounding box), warm-up duration of the SUMO simulation, observer penetration rates, and the number and length of rays generated during ray tracing. Based on this user-defined configuration (see [Configuration](#configuration)), *FTO-Sim* initializes the simulation and performs the ray tracing method for every observer vehicle.

#### 4. Ray Tracing Engine
Based on the provided input data and configuration settings, *FTO-Sim* performs the ray tracing method for every observer vehicle (FCO and FBO) to determine the final field of view (FoV) of each observer. The module performs 360-degree ray generation around observer vehicles and intersects each ray with static (buildings, trees, etc.) and dynamic objects (other road users) to account for occlusion effects in the simulated scene.

Key configurable features include (see [Usage](#usage)):
- **Ray Parameters**: Number of rays (e.g., 360 rays for 1° resolution) and maximum ray length (typically set to 30 meters)
- **Performance Options**: Multi-threaded processing with optional GPU acceleration
- **Visualization Capabilities**: Real-time ray tracing visualization during run-time or video generation and saving for later demonstration purposes

Connecting the endpoints of all (occluded and non-occluded) rays forms the visibility polygon, which reresents the final FoV of an observer vehicle. For detailed methodological explanations of the ray tracing nodule, ray intersection, and visibility polygon generation, see [Ray Tracing](#ray-tracing).

#### 5. Data Logging
The comprehensive data collection system captures detailed simulation values throughout the ray tracing process. *FTO-Sim* generates multiple structured CSV log files, each containing specific data categories for comprehensive analysis. Data is logged in real-time during simulation with consistent coordinate systems (UTM) and temporal resolution.

**Generated Log Files:**

- **`summary_log_*.csv`**: Simulation overview with configuration parameters, overall simulation statistics and performance metrics
- **`log_fleet_composition_*.csv`**: Vehicle fleet composition with number of generated and currently in the simulation present vehicles per vehicle type (for each simulation time step)
- **`log_vehicle_trajectories_*.csv`**: Motorized vehicle trajectory data including position, speed, acceleration, distance traveled, distance to leader/follower vehicles, distance to next traffic light and detection information (for each time step)
- **`log_bicycle_trajectories_*.csv`**: Bicycle trajectory data including position, speed, acceleration, distance traveled, distance to next traffic light and detection information (for each time step)
- **`log_detections_*.csv`**: Detection events between observers and bicycles including coordinates, distances, and vehicle speeds (of observers and observed road users)
- **`log_traffic_lights_*.csv`**: Traffic light information including phases, queue lengths, waiting times, and signal state changes (for each time step)
- **`log_conflicts_*.csv`**: SUMO-native conflict analysis data including time-to-collision (TTC), post-encroachment-time (PET), and deceleration to avoid crash (DRAC) metrics

For detailed information about data analysis and processing, see [Data Collection and Logging](#data-collection-and-logging).

#### 6. Evaluation Metrics
Post-processing analysis modules that generate evaluation metrics from logged simulation data (see [Evaluation Metrics](#evaluation-metrics)):

- **Spatial Visibility Analysis**: Generates heat maps and statistical measures of visibility coverage across the study area, including relative visibility patterns and level of visibility (LoV) assessments.

- **VRU-Specific Detection**: Analyzes detection performance for VRUs including spatio-temporal detection rates and a specific focus to critical interaction areas.

## Configuration

*FTO-Sim* offers users a wide range of functionalities that can be individually configured before initializing the framework. This enables a customized use of the offered functionalities, depending on the individual needs of users. All configuration is done by editing the [`main.py`](Scripts/main.py) script.

### General Settings

#### Simulation Identification Settings
```python
# Change this tag to distinguish different simulation runs with e.g. same configuration
file_tag = 'individual_tag'  # outputted files will be tagged with "_{file_tag}_FCO{FCO-penetration-rate}_FBO{FBO-penetration-rate}"
```

#### Performance Optimization Settings
```python
# Choose performance optimization level based on your system capabilities:
# - "none": Single-threaded processing (best performing for very small scenarios)
# - "cpu": Multi-threaded CPU processing (best performing for intermediate scenarios)
# - "gpu": CPU multi-threading + GPU acceleration (best performing for large scenarios, requires NVIDIA GPU with CUDA/CuPy)
performance_optimization_level = "cpu"
max_worker_threads = None  # None = auto-detect optimal thread count, or specify number (e.g., 4, 8)
```

#### Path Settings
```python
base_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(base_dir)
# Path to SUMO config-file
sumo_config_path = os.path.join(parent_dir, 'simulation_examples', 'Ilic_ETRR_single-FCO', 'osm_small_3d.sumocfg') # Ilic_ETRR_2025
# Path to GeoJSON fle (optional)
geojson_path = os.path.join(parent_dir, 'simulation_examples', 'Ilic_ETRR_single-FCO', 'TUM_CentralCampus.geojson') # Ilic_ETRR_2025
```

#### Geographic Bounding Box Settings
```python
# Geographic boundaries in longitude / latitude in EEPSG:4326 (WGS84)
north, south, east, west = 48.15050, 48.14905, 11.57100, 11.56790 # Ilic_ETRR_2025
bbox = (north, south, east, west)
```

#### Simulation Warm-up Settings
```python
delay = 0  # Warm-up time in seconds (no ray tracing during this period)
```

### Ray Tracing Settings

#### Observer Penetration Rate Settings
```python
FCO_share = 1.0  # Floating Car Observers penetration rate (0.0 to 1.0)
FBO_share = 0.0  # Floating Bike Observers penetration rate (0.0 to 1.0)
```

#### Ray Tracing Parameter Settings
```python
numberOfRays = 360  # Number of rays emerging from each observer vehicle
radius = 30         # Ray radius in meters
grid_size = 10      # Grid size for spatial visibility analysis (meters) - determines the resolution of LoV and realtive visibility heatmaps
```

### Visualization Settings
```python
useLiveVisualization = True       # Show live visualization during simulation
visualizeRays = True              # Show individual rays in visualization (besides resulting visibility polygon)
useManualFrameForwarding = False  # Manual frame-by-frame progression (for debugging)
saveAnimation = False             # Save animation as video file
```

### Data Collection & Analysis Settings

#### Data Collection Settings
```python
CollectLoggingData = True    # Enable detailed data logging
basic_gap_bridge = 10        # Gap bridging for trajectory smoothing
basic_segment_length = 3     # Minimum segment length for trajectories
```

## Ray Tracing

Based on the input data provided and the configuration settings, the simulation framework is initiated and performs the ray tracing method for every FCO and FBO. In parallel, a binning map approach is used to update the visibility count of each bin included within an observer's FoV at every simulation time step. An overview of the ray tracing method and internal data gathering for subsequent evaluation purposes (spatial visibility analysis and VRU-specific detection) is given in the following figure.

![Ray Tracing Workflow](readme_images/ray_tracing_flowchart.png)
*Figure 2: Workflow of the Ray Tracing Method*

### Methodology

#### Initialization
When initializing the ray tracing method, *FTO-Sim* loads the required input parameters and sets up a binning map that divides the simulated scene into equally sized squares, with each bin's visibility count initially set to zero. Besides the number and length of rays descending from an observer's center point, the bin size, which determines the resolution of the subsequent spatial visibility analyses, can be configured by the user (see [Configuration](#configuration)).

#### Simulation Loop and Observer Assignment
Once the simulation loop starts, *FTO-Sim* checks the vehicle type of each road user within the predefined bounding box at each time step. After the warm-up phase and according to the specified FCO / FBO penetration rates, vehicles or bicycles are assigned the vehicle type 'floating car observer' or 'floating bike observer', thereby activating the ray tracing.

**Observer Types and Assignment Process:**
Every generated vehicle and/or bicycle in the SUMO simulation is assigned a random number from a uniform distribution ranging between [0, 1]. If this number is below the defined FCO/FBO penetration rate, the vehicle or bicycle is assigned the vehicle type 'floating car observer' or 'floating bike observer', respectively.
1. **FCOs (Floating Car Observers)**: Assignment of passenger cars based on configured FCO penetration rate using a fixed seed value for reproducible results
2. **FBOs (Floating Bike Observers)**: Assignment of bicycles based on configured FBO penetration rate using a fixed seed value for reproducible results

### Ray Tracing Algorithm

#### Ray Generation and Intersection Detection
For each observer, the ray tracing module generates the previously configured number of rays descending from the observer's center point and extending up to the previously defined length of the rays (representing the theoretical detection range of an observer). The angle between the rays is equivalently sized to generate a non-occluded FoV around the observer. Rays intersecting with static or dynamic objects are cut to account for occlusion, and the endpoints of all rays are connected to form the visibility polygon representing the observer’s total FoV.

**Key Processing Steps:**
1. **Ray Generation**: Each observer generates rays distributed in a 360° pattern within the specified radius (typically 30 meters)
2. **Occlusion Detection**: Rays are systematically tested for intersections with:
   - **Static objects**: Parked vehicles, obtained from SUMO, as well as buildings, trees, and further infrastructure elements obtained from OSM
   - **Dynamic objects**: Road users (motorized vehicles and VRUs) obtained from SUMO
3. **Ray Intersection**: Rays intersecting with static or dynamic objects are cut to account for occlusion effects
4. **Visibility Polygon Creation**: The endpoints of all rays (both occluded and non-occluded) are connected to form the visibility polygon representing the observer's total FoV

#### Internal Data Collection for Spatial Visibility Analysis
For spatial visibility analyses, the framework updates the binning map by increasing the visibility count of each bin within an observer's total FoV.

#### Internal Data Collection for VRU-specific Detection
If a ray intersects a dynamic object, the framework checks the object's vehicle type. If the object is a bicycle or pedestrian, the corresponding trajectory point is marked as detected for subsequent evaluation of VRU-specific detection.

#### Internal Data Logging
At the end of the simulation loop, the final visibility counts and marked bicycle trajectories are obtained.

### Visualization and Output

In addition to the numerical output, the framework offers flexible visualization options for the ray tracing method. Users can enable live visualization to directly observe the influence of static and dynamic occlusions on observer’s FoVs during runtime. Alternatively, the visualization can be exported as a video file, which allows for later inspection of detection events and facilitates communication of results. (see [Configuration](#configuration))

A visualization of the ray tracing method is provided in Figure 3. The rays emerging from the center point of an observer are colored blue when they are unobstructed, and red when they intersect with objects.

![Ray Tracing Visualization](readme_images/ray_tracing.png)
*Figure 3: Ray Tracing Visualization for an FCO (left) and an FBO (right)*

The left sub-figure illustrates an FCO with its FoV obstructed from the VRU infrastructure by parked vehicles, while the right sub-figure shows the same situation from the perspective of an FBO, whose FoV instead covers the VRU  infrastructure but is obstructed from the vehicular carriageway.

## Data Collection and Logging

*FTO-Sim* implements a comprehensive data collection system that captures detailed simulation parameters throughout the ray tracing process when `CollectLoggingData = True`. The framework generates structured output files organized in a systematic directory structure, enabling analysis of cooperative perception systems and traffic safety implications.

### Output Directory Structure

The simulation creates organized output directories with standardized naming conventions based on simulation configuration. Besides the {project-name} (name of input directory with SUMO simulation used to initialize *FTO-Sim*), the {file-tag} and observer penetration rates ({X} for FCO, {Y} for FBO penetration rate) are used to set the naming conventions:

```
outputs/
└── {project-name}_{file_tag}_FCO{X}%_FBO{Y}%/
    ├── out_logging/                                               # Simulation log files
    │   ├── summary_log_{file_tag}.csv                               # Simulation overview
    │   ├── log_fleet_composition_{file_tag}.csv                     # Vehicle type and fleet composition
    │   ├── log_vehicle_trajectories_{file_tag}.csv                  # Motorized vehicle movement data
    │   ├── log_bicycle_trajectories_{file_tag}.csv                  # Bicycle movement data
    │   ├── log_detections_{file_tag}.csv                            # Observer-VRU detection events
    │   ├── log_conflicts_{file_tag}.csv                             # Conflict analysis
    │   └── log_traffic_lights_{file_tag}.csv                        # Signal control data
    ├── out_raytracing/                                            # Ray tracing output
    │   ├── visibility_counts_{file_tag}.csv                         # Grid-based visibility count data
    │   └── ray_tracing_animation__FCO{X}%_FBO{Y}%.mp4               # Ray tracing animation
    └── out_spatial_visibility/                                    # Spatial visibility analysis output
        ├── relative_visibility_heatmap_FCO{X}%_FBO{Y}%.png          # Relative visibility heatmap
        └── LoV_heatmap_FCO{X}%_FBO{Y}%.png                          # Level of visibility heatmap
    └── out_VRU-specific_detection/                                # VRU-specific detection output
        ├── detection_rates_{file_tag}_data.csv                      # Detailed detection performance
        ├── detection_rates_{file_tag}_summary.txt                   # Statistical summary
        ├── 2D_individual_{file_tag}_{vehicle_id}_summary.txt        # 2D individual bicycle trajectories
        └── 3D_detection_{file_tag}_{vehicle_id}.png                 # 3D individual bicycle detection
```

### Detailed Log File Specifications

#### Core Simulation Logging (`out_logging/`)

##### 1. Summary Log (`out_logging/summary_log_*.csv`)
**Purpose**: Comprehensive simulation overview containing configuration parameters, performance summaries, fleet statistics, and system information in a structured text format.

**Structure**: Text-based summary file with sections about:
- Configuration parameters (FCO/FBO shares, ray settings, performance level)
- Runtime and performance summary
- Fleet composition statistics
- Safety analysis summaries
- Hardware and software specifications

**Key Features**: Broader simulation documentation, reproducibility information, performance statistics

##### 2. Fleet Composition Log (`out_logging/log_fleet_composition_*.csv`)
**Purpose**: Time-series tracking of fleet composition and observer presence throughout the simulation.

**Column Structure**:
time_step, new_DEFAULT_VEHTYPE_count, present_DEFAULT_VEHTYPE_count, new_floating_car_observer_count, present_floating_car_observer_count, new_DEFAULT_BIKETYPE_count, present_DEFAULT_BIKETYPE_count, new_floating_bike_observer_count, present_floating_bike_observer_count

**Key Features**: Vehicle generation monitoring, fleet composition monitoring, observer penetration rate monitoring

##### 3. Vehicle Trajectory Log (`out_logging/log_vehicle_trajectories_*.csv`)
**Purpose**: Movement and kinematic data for all motorized vehicles in the simulation.

**Column Structure**:
time_step, vehicle_id, vehicle_type, x_coord, y_coord, speed, angle, acceleration, lateral_speed, slope, distance, route_id, lane_id, edge_id, lane_position, lane_index, leader_id, leader_distance, follower_id, follower_distance, next_tls_id, distance_to_tls, length, width, max_speed

**Key Features**: UTM coordinate positioning, microscopic traffic flow parameters, inter-vehicle relationship tracking, traffic signal interaction data

##### 4. Bicycle Trajectory Log (`out_logging/log_bicycle_trajectories_*.csv`)
**Purpose**: VRU movement data with integrated detection status and realted traffic light tracking.

**Column Structure**:
time_step, vehicle_id, vehicle_type, x_coord, y_coord, speed, angle, acceleration, lateral_speed, slope, distance, route_id, lane_id, edge_id, lane_position, lane_index, is_detected, detecting_observers, in_test_area, next_tl_id, next_tl_distance, next_tl_state, next_tl_index

**Key Features**: VRU-specific detection tracking, observer identification, critical area monitoring, realted traffic signal state tracking

##### 5. Detection Events Log (`out_logging/log_detections_*.csv`)
**Purpose**: Spatial and temporal documentation of all observer-VRU detection events with contextual information.

**Column Structure**:
time_step, observer_id, observer_type, bicycle_id, x_coord, y_coord, detection_distance, observer_speed, bicycle_speed

**Key Features**: Detection event logging, spatial relationship analysis, relative velocity tracking, observer type differentiation

##### 6. Conflict Analysis Log (`out_logging/log_conflicts_*.csv`)
**Purpose**: Traffic safety assessment using SUMO's Surrogate Safety Measures (SSM) with detection coverage analysis.

**Column Structure**:
time_step, bicycle_id, foe_id, foe_type, x_coord, y_coord, distance, ttc, pet, drac, severity, is_detected, detecting_observer, observer_type

**Key Features**: Time-to-Collision (TTC) calculations, Post-Encroachment-Time (PET) analysis, Deceleration Rate to Avoid Crash (DRAC) metrics, conflict severity assessment

**Comment**: It has shown, that SUMO tends to classify events as potential conflicts, that in fact do not reflect real conflict-prone situations. Therefore, the evaluation of "conflict detection rates" has been dismissed.

##### 7. Traffic Light Data Log (`out_logging/log_traffic_lights_*.csv`)
**Purpose**: Traffic signal information including programs, phases and signal states (including signal-to-lane mapping information). Additionally, queue lengths, number of stops and waiting times are tracked.

**Column Structure**:
time_step, traffic_light_id, program, phase, phase_duration, remaining_duration, signal_states, total_queue_length, vehicles_stopped, average_waiting_time, vehicles_by_type, lane_to_signal_mapping

**Key Features**: Signal phase tracking, queue length monitoring, waiting time analysis, vehicle type distribution at signals

##### Visibility Counts (`out_raytracing/visibility_counts_*.csv`)
**Purpose**: Grid-based visibility counts for generating relative visibility and LoV heatmaps (see [Evaluation Metrics](#evaluation-metrics)).

**Column Structure**:
x_coord, y_coord, visibility_count

**Key Features**: Bin-wise visibility count tracking

#### Post-Processing Outputs

##### Spatial Visibility Analysis (`out_spatial_visibility/`)
**Purpose**: Visual representation of cooperative perception coverage through heat map generation.

**Generated Files**:
- `relative_visibility_heatmap_*.png`: Relative visibility visualization
- `LoV_heatmap_*.png`: Level of Visibility visualization

**Key Features**: Heatmap visualization of spatial perception coverage

##### VRU-Specific Detection Analysis (`out_VRU-specific_detection/`)
**Purpose**: Specialized VRU detection performance evaluation with VRU trajectory analysis.

**Generated Files**:
- `detection_rates_*_data.csv`: Quantitative detection performance metrics
- `detection_rates_*_summary.txt`: Statistical summaries (temporal, spatial, spatio-temporal rates)  
- `2D_individual_*_*.png`: 2D detection plots (space-time-diagrams) for each bicycle [detection events highlighted]
- `3D_detection_*_*.png`: 3D detection plots (space-time-diagrams) for each bicycle with respective observer trajectory / trajectories [detection events highlighted]

**Key Features**: Individual VRU tracking, detection rate calculations, critical interaction area analysis

### Data Processing and Analysis

#### Real-Time Data Collection
*FTO-Sim* implements comprehensive real-time data collection throughout the simulation process. All logging occurs simultaneously with the ray tracing simulation to capture detailed behavioral patterns and system performance metrics.

**Data Collection Characteristics:**
- **Coordinate Systems**: Consistent UTM projection for spatial accuracy and cross-analysis compatibility
- **Temporal Resolution**: Matching SUMO simulation time step (typically 0.1 seconds) for precise temporal tracking
- **Data Integrity**: Automated validation checks ensure completeness and consistency across all log files
- **Memory Efficiency**: Streaming data writing prevents memory overflow during larger simulation scenarios
- **Multi-threaded Logging**: Optimized data structures minimize computational overhead during real-time collection

#### Post-Processing Possibilities
The generated datasets serve as standardized input for specialized evaluation scripts that provide comprehensive analysis capabilities across different research domains.

**Primary Evaluation Scripts:**
- **`evaluation_spatial_visibility.py`**: Generates relative visibility and LoV heatmaps from visibility count data. Creates spatial coverage visualizations and statistical assessments of cooperative perception effectiveness.
- **`evaluation_VRU_specific_detection.py`**: Performs individual VRU trajectory analysis with detection event tracking. Produces temporal, spatial, and spatio-temporal detection rate calculations along with 3D visualization plots for each bicycle trajectory. Special focus in detection rate calculations can be given to critical interaction areas.

**Post-Processing Workflow:**
1. **Data Validation**: Automated checks for simulation completeness and data consistency
2. **Metric Calculation**: Statistical analysis of VRU detection rates and spatial visibility coverage  
3. **Visualization Generation (when activated)**: Ray tracing animation, spatial visibility heatmaps, 2D / 3D VRU trajectory plots

#### Performance Considerations
The comprehensive data collection system is designed for scalability while maintaining computational efficiency during simulation execution.

**Computational Impact:**
- **File Sizes**: Large-scale simulations generate substantial datasets
- **Memory Management**: Streaming data architecture prevents memory bottlenecks and supports extended simulation runs
- **Processing Efficiency**: Optimized data structures and parallel logging minimize performance impact on ray tracing calculations
- **Storage Requirements**: Plan adequate disk space for complete dataset generation, especially for multi-scenario comparative studies

## Evaluation Metrics

*FTO-Sim* implements comprehensive evaluation metrics and analysis capabilities that enable the assessment of cooperative perception systems effectiveness. Based on the ray tracing methodology and comprehensive data collection framework, the evaluation system provides two primary analysis domains that address different aspects of cooperative perception performance.

### Research Applications

The framework supports research across multiple domains:
- **Cooperative Perception Systems**: Coverage analysis, blind spot identification, and cooperative effectiveness assessment
- **Urban Traffic Monitoring**: Strategic sensor placement evaluation and coverage optimization
- **Vulnerable Road User Safety**: Bicycle and pedestrian visibility analysis with focus on critical interaction areas
- **Connected Vehicle Technologies**: V2X communication effectiveness and cooperative awareness studies
- **Traffic Safety Assessment**: Conflict detection capabilities and severity analysis

### Analysis Workflow Overview

The evaluation metrics follow a systematic post-processing workflow that transforms logged simulation data into actionable insights:

1. **Data Collection**: Comprehensive real-time logging during ray tracing simulation
2. **Grid-based Spatial Analysis**: Binning map approach for spatial visibility assessment
3. **Trajectory-based Detection Analysis**: Individual VRU movement tracking with detection event correlation
4. **Statistical Evaluation**: Quantitative metrics calculation for comparative analysis
5. **Visualization Generation**: Heat maps, trajectory plots, and statistical summaries

## Spatial Visibility Analysis

The spatial visibility analysis module evaluates the coverage characteristics of cooperative perception systems through grid-based spatial assessment. This analysis follows the methodology described in chapter 3.3 of the ETRR paper, implementing both relative visibility and Level of Visibility (LoV) assessments.

### Methodology Overview

![Spatial Visibility Workflow](readme_images/spatial_visibility_flowchart.png)
*Figure 4: Workflow of Spatial Visibility Analysis Methods*

The spatial visibility analysis operates in parallel to the ray tracing simulation, utilizing a binning map approach to systematically track visibility coverage across the simulated environment. The methodology encompasses two complementary metrics that provide different perspectives on cooperative perception effectiveness.

### Relative Visibility Assessment

**Conceptual Foundation**: Relative visibility measures the cumulative spatial coverage achieved by cooperative perception systems, providing insights into the collective observation capabilities of FCOs and FBOs throughout the simulation period.

**Implementation Process**:

1. **Initialization Phase**: The framework initializes a binning map that divides the simulated scene into equally sized squares, with each bin's visibility count initially set to zero. The bin size determines the spatial resolution of subsequent analyses and can be configured by users (see [Configuration](#configuration)).

2. **Real-time Visibility Tracking**: During each simulation time step, the relative visibility module updates the binning map by incrementing the visibility count for each bin within an observer's field of view (FoV). Following the methodology proposed by [Pechinger et al.](https://www.researchgate.net/publication/372952261_THRESHOLD_ANALYSIS_OF_STATIC_AND_DYNAMIC_OCCLUSION_IN_URBAN_AREAS_A_CONNECTED_AUTOMATED_VEHICLE_PERSPECTIVE), overlapping FoVs from multiple observers contribute equally (count increment of one per time step).

3. **Data Processing**: Upon simulation completion, the framework obtains final visibility counts and generates both raw and normalized visibility maps. Normalization divides each bin value by the maximum observed visibility count, enabling comparative analysis across different scenarios.

4. **Visualization Generation**: Heat map visualization provides intuitive representation of spatiotemporal visibility characteristics, highlighting areas of high cooperative perception coverage and identifying potential blind spots.

### Level of Visibility (LoV) Assessment

**Conceptual Foundation**: The LoV metric, as introduced by [Pechinger et al.](https://www.researchgate.net/publication/372952261_THRESHOLD_ANALYSIS_OF_STATIC_AND_DYNAMIC_OCCLUSION_IN_URBAN_AREAS_A_CONNECTED_AUTOMATED_VEHICLE_PERSPECTIVE), provides a standardized framework for comparing visibility performance across different scenarios and conditions. By converting raw visibility counts into observation rates, LoV enables time-dependent comparisons and categorical assessment.

**Implementation Process**:

1. **Initialization Phase**: The framework initializes arrays for both LoV calculations and observation rate computations, establishing the analytical foundation for temporal visibility assessment.

2. **Observation Rate Calculation**: For each spatial bin, the observation rate is calculated by dividing the accumulated visibility count by the total simulation time. The maximum possible observation rate is defined as the inverse of the simulation step size, ensuring compatibility across simulations with different temporal resolutions.

3. **LoV Categorization**: The observation rate for each bin is assigned to one of five discrete LoV categories (A through E), with thresholds distributed equidistantly based on the maximum possible observation rate. This categorical approach provides simplified interpretation while maintaining analytical rigor.

4. **Visualization and Analysis**: Heat map visualization represents the assessed LoV for each spatial bin, offering clear visual communication of visibility performance patterns and enabling comparative assessment across scenarios.

### Output Products

**Generated Files**: 
- `relative_visibility_heatmap_*.png`: Normalized visibility coverage visualization
- `LoV_heatmap_*.png`: Categorical visibility performance assessment
- Logging data for quantitative analysis and further processing

**Analysis Capabilities**: Post-processing via `evaluation_spatial_visibility.py` script enables automated generation of spatial visibility metrics, statistical summaries, and publication-ready visualizations.

## VRU-Specific Detection Analysis

The VRU-specific detection analysis module focuses on the assessment of vulnerable road user (VRU) detection performance within cooperative perception systems. This analysis module evaluates the effectiveness of FCOs and FBOs in detecting bicycles and pedestrians, providing insights into traffic safety implications and system performance.

### Methodology Overview

The VRU-specific detection analysis operates through trajectory-based assessment that correlates individual VRU movements with observer detection capabilities. This approach enables detailed analysis of detection performance across different spatial, temporal, and behavioral contexts.

### Detection Performance Metrics

**Temporal Detection Rate**: Measures the percentage of time steps during which a VRU is detected throughout its trajectory, providing insights into continuous monitoring capabilities.

**Spatial Detection Rate**: Evaluates the percentage of traveled distance during which a VRU is detected, focusing on coverage effectiveness across different spatial contexts.

**Spatio-temporal Detection Rate**: Combines temporal and spatial metrics to provide comprehensive detection performance assessment.

**Critical Interaction Area Analysis**: Specialized evaluation of detection performance within predefined critical areas where VRU safety is of particular concern.

### Analysis Implementation

**Individual Trajectory Processing**: Each VRU trajectory is analyzed individually, tracking detection events and correlating them with observer presence and behavior. The analysis includes:

1. **Trajectory Segmentation**: VRU paths are divided into detected and undetected segments for detailed performance assessment
2. **Observer Correlation**: Detection events are linked to specific observer vehicles, enabling analysis of observer type effectiveness (FCO vs. FBO)
3. **Critical Area Focus**: Special attention to detection performance within high-risk interaction zones
4. **Statistical Aggregation**: Individual metrics are aggregated to provide flow-based and system-wide performance indicators

**Visualization Capabilities**: The analysis generates comprehensive visualization outputs including:

- **2D Trajectory Plots**: Space-time diagrams showing detection events along individual VRU trajectories
- **3D Detection Visualization**: Three-dimensional plots combining spatial position, temporal progression, and detection status
- **Statistical Summaries**: Quantitative performance metrics with comparative analysis capabilities

### Output Products

**Generated Files**: 
- `detection_rates_*_data.csv`: Detailed quantitative detection performance metrics
- `detection_rates_*_summary.txt`: Statistical summaries with temporal, spatial, and spatio-temporal rates
- `2D_individual_*_*.png`: Individual VRU trajectory analysis with detection event highlighting
- `3D_detection_*_*.png`: Three-dimensional visualization of VRU-observer interactions

**Analysis Capabilities**: Post-processing via `evaluation_VRU_specific_detection.py` script provides automated analysis of detection performance, comparative assessment across different scenarios, and detailed individual trajectory evaluation.

### Research Applications

The VRU-specific detection analysis supports research in:
- **Traffic Safety Assessment**: Quantifying detection coverage for safety-critical road users
- **System Optimization**: Identifying optimal observer placement and penetration rates
- **Behavioral Analysis**: Understanding detection performance across different VRU movement patterns
- **Technology Evaluation**: Comparative assessment of different cooperative perception strategies

## Performance Considerations

### Computational Requirements
- **Ray Count Impact**: Higher ray counts increase precision but reduce performance
- **Observer Density**: More observers significantly increase computational load
- **Visualization Overhead**: Live visualization reduces simulation speed by 60-80%

### Optimization Strategies
- Use `performance_optimization_level = 'cpu'` for multi-threading benefits
- Disable visualization (`useLiveVisualization = False`) for production runs
- Consider GPU acceleration for very large scenarios (requires CUDA/CuPy)
- Adjust ray count based on required precision vs. performance trade-off

### Memory Usage
The simulation tracks memory consumption and provides detailed performance metrics including:
- Step-by-step timing analysis
- Memory usage patterns
- Component-wise execution time breakdown

## Dependencies

### Required Python Packages
```
libsumo>=1.15.0
matplotlib>=3.5.0
numpy>=1.21.0
pandas>=1.3.0
geopandas>=0.10.0
shapely>=1.8.0
osmnx>=1.2.0
pyproj>=3.3.0
tqdm>=4.62.0
psutil>=5.8.0
adjustText>=0.7.0
```

### Optional Dependencies for Enhanced Performance
```
cupy>=10.0.0         # GPU acceleration (requires NVIDIA GPU + CUDA)
numba>=0.56.0        # JIT compilation
opencv-python>=4.5.0 # Advanced visualization
```

## Error Handling and Troubleshooting

### Common Issues

1. **SUMO Connection Errors**: Verify SUMO installation and file paths in `sumo_config_path`
2. **Memory Issues**: Reduce ray count or observer penetration rates for large simulations
3. **Performance Problems**: Disable visualization and use CPU optimization
4. **Data Export Failures**: Ensure sufficient disk space and write permissions
5. **Coordinate System Issues**: Verify bounding box coordinates match your study area

### Debug Mode
Enable step-by-step debugging with:
```python
useManualFrameForwarding = True
useLiveVisualization = True
```

## Installation / Prerequisites

### Creation of an isolated virtual environment
It is recommended to initially create an isolated virtual environment (venv) that allows users to install Python packages without affecting the global Python installation on their system. Furthermore, it ensures that each project has its own set of dependencies isolated from others.

When creating a virtual environment, a new directory named 'venv' will be created in the current working directory. Inside the 'venv' directory, a copy of the Python interpreter will be placed, along with a 'Scripts' (or 'bin' on Ubuntu) directory that contains executables for Python and pip. The 'venv' directory will also include a 'Lib' directory where installed packages will be stored.

After creating the isolated virtual environment once, this step does not have to be executed again. In order to initially create the isolated virtual environment 'venv', execute the following code in the terminal:
```
python -m venv venv
```

### Activating the isolated virtual environment
Once 'venv' is created, users have to activate the virtual environment. This step should be performed every time, 'venv' is not activated anymore. Once activated, any Python commands will be contained within this virtual environment, preventing conflicts with other projects or system-wide packages. In order to activate the isolated environment 'venv', execute the following code in the terminal:
```
.\venv\Scripts\activate
```

If users encounter problems, when trying to activate the isolated virtual environment, it is often due to the Windows PowerShell's execution policy, which controls the ability to run scripts on the system. By default, PowerShell has a restrictive execution policy to prevent the execution of potentially harmful scripts. To resolve this issue, users can change the execution policy of Windows PowerShell to allow the script to run.

### Installation of required packages
Before using *FTO-Sim*, users have to make sure all necessary Python packages are installed. The file 'requirements.txt' lists all the necessary packages and their corresponding versions that are required to execute *FTO-Sim*. Users can easily install all required packages by executing the following code in the terminal:
```
pip install -r requirements.txt
```

## Usage

### General Usage
Depending on the customized configuration settings (see [Configuration](#configuration)), the use of *FTO-Sim* differs slightly. In general, it can be distinguished between different use modes:

1. **Simulation Mode**: This use mode is available for an execution of *FTO-Sim* without any visualization. While decreasing the computational cost and therefore increasing simulation speed with this use mode, it does not provide any visual aids for checking the simulation's correct performance. Therefore, this use mode is recommended for well-developed simulation scenarios. In order to initialize this use mode, users should set the following general settings, while all other configuration settings can be customized according to the user's needs:
    ```python
    # General Settings
    useLiveVisualization = False            # Live Visualization of Ray Tracing
    visualizeRays = False                   # Visualize rays additionally to the visibility polygon
    useManualFrameForwarding = False        # Visualization of each frame, manual input necessary to forward the visualization
    saveAnimation = False                   # Save the animation
    performance_optimization_level = 'cpu'  # Enable multi-threading for better performance
    ```

2. **Visualization Mode**: This use mode is available for an execution of *FTO-Sim* with a live visualization of the ray tracing method. While increasing the computational cost and therefore decreasing simulation speed with this use mode, it provides visual aids for checking the simulation's correct performance. This use mode is recommended for simulation scenarios that are not yet thoroughly developed or if a live visualization is wanted for e.g. demonstration purposes. In order to initialize this use mode, users should set the following general settings:
    ```python
    # General Settings
    useLiveVisualization = True             # Live Visualization of Ray Tracing
    visualizeRays = True                    # Visualize rays additionally to the visibility polygon (can be set to 'False' in this use mode)
    useManualFrameForwarding = False        # Visualization of each frame, manual input necessary to forward the visualization
    saveAnimation = False                   # Save the animation
    ```

3. **Debugging Mode**: This use mode is available for a step-wise execution of *FTO-Sim*, which, when activated, requests a user's input to proceed to the calculation of the next simulation step / frame. In order to initialize this use mode:
    ```python
    # General Settings
    useLiveVisualization = True             # Live Visualization of Ray Tracing
    visualizeRays = True                    # Visualize rays additionally to the visibility polygon
    useManualFrameForwarding = True         # Visualization of each frame, manual input necessary to forward the visualization
    saveAnimation = False                   # Save the animation
    ```

4. **Saving Mode**: This use mode is available for an execution of *FTO-Sim* that saves the simulation as an animation file. Since live visualization is currently not compatible with saving animations, this mode requires live visualization to be disabled. The saved animation can be reviewed afterwards for analysis or demonstration purposes:
    ```python
    # General Settings
    useLiveVisualization = False            # Live Visualization of Ray Tracing
    visualizeRays = False                   # Visualize rays additionally to the visibility polygon
    useManualFrameForwarding = False        # Visualization of each frame, manual input necessary to forward the visualization
    saveAnimation = True                    # Save the animation
    ```

### Example Scenario
This repository contains an example on the use of *FTO-Sim*, consisting of a SUMO simulation (including network, demand and additional files) and a GeoJSON file covering the simulated scene. The simulated scene is covering the intersection ['Arcisstr. / Theresienstr.'](https://maps.app.goo.gl/UAHCgc9CT8kryamJ7) in Munich, Germany - close to the main campus of the Technical University of Munich (TUM).

The SUMO simulation includes the following files:
* Network (network.net.xml): A SUMO-readable representation of the traffic network - including vehicular carriageways, bike paths, pedestrian walkways and intersection crossings - of the simulated scene.
* Demand (demand.rou.xml): A SUMO-readable representation of the traffic demand, including traffic flows for passenger cars and bicycles. The traffic demand has been derived from real traffic counts provided by the city of Munich and is scalled down to a duration of 270 seconds (4.5 minutes).  Furthermore, parking vehicles are initiated through the SUMO demand file.
* Additionals (parkinglots.add.xml): A SUMO-readable representation of additional infrastructure elements. In this case, the additional file contains parking lots on the northern intersection aproach of the simulated scene.
* Configuration File (SUMO_example.sumocfg): A SUMO configuration file contains all the parameters and input files (e.g. network, demand and additional files) to execute the simulation. 

### Basic Simulation Examples

#### Quick Start
```python
# 1. Configure parameters in Scripts/main.py
FCO_share = 0.25
FBO_share = 0.10
file_tag = "baseline_simulation"
performance_optimization_level = "cpu"

# 2. Execute simulation
python Scripts/main.py
```

#### High-Performance Mode
```python
# Optimize for large-scale simulations
useLiveVisualization = False
performance_optimization_level = 'gpu'  # If CUDA available
max_worker_threads = 8
```

## Related Documentation and Scripts

The main simulation generates standardized data outputs that can be processed by specialized evaluation scripts:

- **[`main.py`](Scripts/main.py)** - Main simulation script with ray tracing and data collection
- **`evaluation_VRU_specific_detection.py`** - Individual bicycle trajectory analysis
- **`evaluation_relative_visibility.py`** - Cooperative perception analysis  
- **`evaluation_lov.py`** - Line-of-visibility heatmap generation
- **Additional evaluation scripts** - Various traffic safety and detection metrics

This modular approach allows for focused analysis of specific aspects while maintaining computational efficiency in the main simulation.