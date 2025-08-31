# FTO-Sim

*FTO-Sim* is an open-source simulation framework for Floating Traffic Observation (FTO) that integrates SUMO traffic simulation with advanced ray tracing techniques to analyze the visibility coverage of Floating Car Observers (FCOs) and Floating Bike Observers (FBOs) in urban environments. The framework enables comprehensive evaluation of cooperative perception systems and their effectiveness in detecting vulnerable road users.

The FTO concept is adapted from the Floating Car Observer (FCO) method, utilizing extended floating car data (xFCD) for traffic planning and traffic management purposes. Additionally, *FTO-Sim* introduces further observer vehicle types, such as Floating Bike Observers (FBO).

## Table of Contents
1. [Citation](#citation)
2. [Features](#features)
3. [Configuration](#configuration)
4. [Ray Tracing Methodology](#ray-tracing-methodology)
5. [Data Collection and Logging](#data-collection-and-logging)
6. [Evaluation Metrics](#evaluation-metrics)
7. [Installation / Prerequisites](#installation--prerequisites)
8. [Usage](#usage)
9. [Performance Considerations](#performance-considerations)
10. [Dependencies](#dependencies)
11. [Error Handling and Troubleshooting](#error-handling-and-troubleshooting)

## Citation
When using *FTO-Sim*, please cite the following references:

### Primary Citation
* **Ilic, L., et al.** "FTO-Sim: An Open-Source Framework for Evaluating Cooperative Perception in Urban Areas." *European Transport Research Review*, 2024. *(Publication pending)*

This paper provides the comprehensive methodological foundation of the FTO-Sim framework, including detailed descriptions of the ray tracing algorithms, cooperative perception analysis, and validation studies.

### Secondary References
* [Introduction of FTO-Sim](https://www.researchgate.net/publication/383272173_An_Open-Source_Framework_for_Evaluating_Cooperative_Perception_in_Urban_Areas) includes a detailed description of the general features of the simulation framework. Furthermore, a first application (visibility analysis) of *FTO-Sim* is included to further calibrate the Level of Visibility (LoV) metric, originally introduced by [Pechinger et al.](https://www.researchgate.net/publication/372952261_THRESHOLD_ANALYSIS_OF_STATIC_AND_DYNAMIC_OCCLUSION_IN_URBAN_AREAS_A_CONNECTED_AUTOMATED_VEHICLE_PERSPECTIVE).

## Features
The following sub-chapters elaborate on the different modules and functionalities of *FTO-Sim*, which are summarized in the figure below.

![Overview of the FTO-Sim Framework Architecture](readme_images/framework_features.png)

### Core Functionality
- **SUMO Integration**: Seamless connection to SUMO traffic simulation via TraCI
- **Ray Tracing Engine**: Advanced visibility analysis with configurable parameters
- **Multi-Observer Support**: FCO and FBO observer types with customizable penetration rates
- **Geospatial Data Integration**: Support for OpenStreetMap and GeoJSON road network data
- **Performance Optimization**: Multi-threading and optional GPU acceleration support
- **Comprehensive Data Logging**: Detailed simulation metrics and trajectory data export

### Input Data

*FTO-Sim* makes use of three different input data types:

1. **SUMO Simulation Files**
   - Network files (`.net.xml`)
   - Route/demand files (`.rou.xml`) 
   - Configuration files (`.sumocfg`)
   - Additional infrastructure files (`.add.xml`)
   
   SUMO and its interface TraCI (Traffic Control Interface) are used to retrieve the location of every static and dynamic road user for each time step of the simulation. Parked vehicles are considered static road users, while vehicular traffic, as well as VRUs (pedestrians and cyclists), are considered dynamic road users.

2. **OpenStreetMap (OSM) Data**
   - Building geometries and locations
   - Urban greenery (parks, trees)
   - Traffic infrastructure elements
   
   Shapes and locations of static infrastructure elements, such as buildings, are retrieved from OSM. Furthermore, shapes and locations of urban greenery, such as parks and trees, are obtained from OSM.

3. **GeoJSON Files** *(optional)*
   - Road space distribution visualization
   - Carriageways, parking areas, bicycle lanes
   - Pedestrian walkways
   
   If available, *FTO-Sim* makes use of GeoJSON files containing the road space distribution of the simulated scene for visualization purposes only.

## Configuration

*FTO-Sim* offers users a wide range of functionalities that can be individually configured before initializing the framework. This enables a customized use of the offered functionalities, depending on the individual needs of users. All configuration is done by editing the [`main.py`](Scripts/main.py) script.

### Essential Settings

#### Observer Configuration
```python
# Observer penetration rates (0.0 to 1.0)
FCO_share = 0.25    # 25% of vehicles become FCOs
FBO_share = 0.10    # 10% of bicycles become FBOs
```

Every generated vehicle and/or bicycle in the SUMO simulation is assigned a random number from a uniform distribution ranging between [0, 1]. If this number is below the defined FCO/FBO penetration rate, the vehicle or bicycle is assigned the vehicle type 'floating car observer' or 'floating bike observer', respectively.

#### Ray Tracing Parameters
```python
numberOfRays = 360          # Number of rays per observer (higher = more precise)
radius = 30                 # Maximum ray tracing distance in meters
grid_size = 10              # Grid size for visibility heat map (meters)
```

#### Study Area Definition
```python
# Bounding box coordinates (WGS84 format)
north, south, east, west = 48.1505, 48.14905, 11.5720, 11.5669
bbox = (north, south, east, west)
```

#### Simulation Timing
```python
delay = 90          # Warm-up period in seconds
```

#### File Paths
```python
# Path Settings
base_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(base_dir)
sumo_config_path = os.path.join(parent_dir, 'simulation_examples', 'example', 'osm.sumocfg')
geojson_path = os.path.join(parent_dir, 'simulation_examples', 'example', 'roads.geojson')
```

### Visualization Options
```python
useLiveVisualization = True         # Enable real-time visualization
visualizeRays = False              # Show individual rays
saveAnimation = False              # Export animation video
useManualFrameForwarding = False   # Step-by-step debugging mode
```

### Performance Settings
```python
performance_optimization_level = 'cpu'  # Options: 'none', 'cpu', 'gpu'
max_worker_threads = 4                   # Multi-threading configuration
```

Available optimization levels:
- **`'none'`**: Single-threaded processing (most compatible, but slower)
- **`'cpu'`**: Multi-threaded CPU processing (recommended default)
- **`'gpu'`**: CPU multi-threading + GPU acceleration (fastest, requires NVIDIA GPU with CUDA/CuPy)

### Data Collection Settings
```python
CollectLoggingData = True    # Enable detailed data logging
file_tag = 'baseline_test'   # Unique identifier for simulation outputs
```

## Ray Tracing Methodology

Based on the provided input data and configuration settings, *FTO-Sim* is initiated and performs the ray tracing method for every FCO and FBO. The following figure shows the working principle of the ray tracing method.

![Ray Tracing Workflow](readme_images/ray_tracing_flowchart.png)

### Observer Detection Process

1. **Observer Assignment**: Vehicles and bicycles are randomly assigned observer roles based on configured penetration rates using a fixed random seed for reproducible results

2. **Ray Generation**: Each observer generates rays distributed in a 360° pattern within a specified radius:
   - FCOs (Floating Car Observers): Passenger vehicles with observer capabilities
   - FBOs (Floating Bike Observers): Bicycles with observer capabilities

3. **Occlusion Detection**: Rays are tested for intersections with:
   - **Static objects**: Buildings, trees, barriers, infrastructure
   - **Dynamic objects**: Moving vehicles, parked cars

4. **Visibility Polygon Creation**: Ray endpoints form visibility polygons representing observable areas

Once the **simulation loop** is initiated, *FTO-Sim* will check the vehicle type of every road user within the previously defined bounding box for each time step of the simulation. After the initially defined warm-up phase and depending on the defined FCO / FBO penetration rates, vehicles or bicycles with the vehicle type 'floating car observer' or 'floating bike observer' will be randomly generated and thus activating the ray tracing.

The **ray tracing module** will generate the previously defined number of rays descending from every observer's center point up to a distance of 30 meters. The angle between the rays will be equivallently sized to generate a non-occluded field of view (FoV) in a circular form around the observer. Subsequently, the rays that intersect with static or dynamic objects are cut to obtain an observer's occluded FoV. Lastly, the endpoints of all rays are connected to create an observer's visibility polygon representing the area within an observer's total FoV.

The following figure shows a visualization of the ray tracing method, both for FCOs (left) and FBOs (right). The rays emerging from the centerpoint of an observer are colored in blue when they are unobstructed and in red when they intersect with objects.

![Ray Tracing Visualization](readme_images/ray_tracing.png)

### Field of View Calculation

The framework calculates circular fields of view around each observer, accounting for:
- **Static Occlusion**: Buildings, infrastructure elements from OpenStreetMap
- **Dynamic Occlusion**: Moving and parked vehicles from SUMO simulation
- **Observer Height**: Different visibility characteristics for cars vs. bicycles

## Data Collection and Logging

The simulation generates comprehensive datasets across multiple categories when `CollectLoggingData = True`:

### Primary Data Outputs

#### Fleet Composition Logs
- Vehicle counts by type and time step
- Observer presence and penetration rates
- Real vs. target observer ratios

#### Detection Data
- Bicycle detection events with timestamps
- Observer-target relationships
- Detection distances and durations
- Visibility coverage metrics

#### Trajectory Data
- Complete movement histories for all vehicles
- Position, speed, acceleration, and heading data
- Traffic light interactions and lane positions
- Cumulative distance and travel time statistics

#### Conflict Analysis
- Time-to-Collision (TTC) measurements using SUMO's SSM device
- Post-Encroachment-Time (PET) calculations
- Deceleration Rate to Avoid Crash (DRAC) metrics
- Conflict severity and detection coverage analysis

#### Traffic Light Data
- Signal phase information and timing
- Queue lengths and waiting times
- Vehicle-traffic light interactions

### Output Directory Structure

The simulation creates organized output directories:

```
outputs/
└── [file_tag]_FCO[X]%_FBO[Y]%/
    ├── out_logging/
    │   ├── summary_log_*.csv           # Comprehensive simulation summary
    │   ├── log_fleet_composition_*.csv # Vehicle type tracking
    │   ├── log_detections_*.csv        # Detection events
    │   ├── log_vehicle_trajectories_*.csv
    │   ├── log_bicycle_trajectories_*.csv
    │   ├── log_conflicts_*.csv         # Safety analysis
    │   └── log_traffic_lights_*.csv    # Traffic signal data
    └── out_raytracing/
        └── visibility_counts_*.csv     # Spatial visibility data
```

## Evaluation Metrics

Based on the ray tracing method, different evaluation metrics and analysis applications of *FTO-Sim* are available and further will be developed and provided in future.

### Visibility Analysis Metrics

The framework provides several key metrics for evaluating cooperative perception systems:
- **Line-of-Visibility (LoV)**: Direct visibility between observer and target
- **Relative Visibility (RelVis)**: Visibility improvement through cooperation
- **Ray-based Occlusion Modeling**: Geometric visibility calculations
- **Multi-layer Detection Analysis**: Static and dynamic occlusion handling

### Research Applications

This framework supports research in:
- **Cooperative Perception Systems**: Analyzing coverage and blind spots
- **Urban Traffic Monitoring**: Evaluating sensor placement strategies
- **Vulnerable Road User Safety**: Bicycle and pedestrian visibility analysis
- **Connected Vehicle Technologies**: V2X communication effectiveness studies
- **Traffic Safety Assessment**: Conflict detection and severity analysis

#### Relative Visibility

In parallel to the ray tracing, a binning map approach is followed to update the visibility count for every bin that is included within the FoV of a FCO / FBO for every time step of the simulation. The following figure shows the working principle of the relative visibility analysis that is performed in parallel to the previously described ray tracing method.

![Relative Visibility Workflow](readme_images/relative_visibility_flowchart.png)

During the **initialization phase**, *FTO-Sim* initializes a binning map that divides the simulated scene into equivalently sized squares and sets the visibility count of each bin to zero. The size of the bins and, with that, the resolution of the following visibility analyses can be individually set by users (see [Configuration Settings](#configuration-settings) under grid map settings).

The **realtive visibility module** updates the initialized binning map by increasing the visibility count for each bin within an observer's FoV by one. In case of overlapping FoV's of multiple observers, the visibility count is still increased by one, thus following the methodology proposed by [Pechinger et al.](https://www.researchgate.net/publication/372952261_THRESHOLD_ANALYSIS_OF_STATIC_AND_DYNAMIC_OCCLUSION_IN_URBAN_AREAS_A_CONNECTED_AUTOMATED_VEHICLE_PERSPECTIVE). The simulation loop is repeated until the simulation end is reached after which the final binning map and visibility counts are obtained. Additionally, the visibility counts are normalized by dividing each bin value by the maximum observed visibility count. Both resulting binning maps (raw visibility counts and normalized visibility counts) are saved for further processing. Finally, a heat map of the normalized visibility counts is generated providing a visual representation of the spatiotemporal characteristics of the potential data collection process of FCOs / FBOs.

#### Level of Visibility (LoV)
The LoV, as introduced by [Pechinger et al.](https://www.researchgate.net/publication/372952261_THRESHOLD_ANALYSIS_OF_STATIC_AND_DYNAMIC_OCCLUSION_IN_URBAN_AREAS_A_CONNECTED_AUTOMATED_VEHICLE_PERSPECTIVE), provides a metric for comparing visibility across different scenarios under varying conditions. By converting the raw visibility counts into an observation rate, defined as the frequency of observations of a bin over time, obtained from the observer's final FoV, it provides a time-dependent scale for the comparison of different scenarios. Subsequently, the observation rate is categorized into one of five discrete LoVs offering a simplified representation of an observer's visibility conditions. The following figure gives an overview of the working principle of the LoV assessment.

#### Placeholder for future applications

![LoV Workflow](readme_images/LoV_flowchart.png)

Through the **initialization phase**, *FTO-Sim* initializes arrays for both the LoV as well as the observation rate.

The **observation rate** is then calculated for each bin by dividing the visibility count by the simulation time. The maximum possible observation rate is defined as the inverse of the simulation step size and, therefore, provides the possibility to account for differences in step sizes between different simulations.

Subsequently, the **level of visibility** for each bin is determined by assigning the observation rate to one of the five discrete LoVs, with the thresholds between different LoVs distributed equidistantly based on the maximum possible observation rate. Finally, a heat map of the simulated scene representing the assessed LoV for each bin is provided as a visual representation of the metric.

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