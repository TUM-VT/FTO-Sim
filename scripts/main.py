#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# type: ignore
# See copilot-instructions.md for agent guidance
"""
FTO-Sim Main Simulation Script with Performance Optimization

This script implements a traffic simulation system with ray tracing capabilities,
GPU/CPU acceleration, and comprehensive performance monitoring.

Note: Some type checking warnings are expected due to dynamic attributes
and third-party library flexibility - they do not affect functionality.
"""

import os
import osmnx as ox
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import libsumo as traci
from shapely.geometry import Point, box, Polygon, MultiPolygon
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Polygon as MatPolygon, Rectangle, Circle
import matplotlib.transforms as transforms
import numpy as np
from shapely.geometry import LineString
from matplotlib.lines import Line2D
import pyproj
from shapely.affinity import rotate, translate
import geopandas as gpd
import xml.etree.ElementTree as ET
import networkx as nx
import logging
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from collections import defaultdict
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import shapely.ops
import imageio.v2 as imageio
import glob
import math
import SumoNetVis
from adjustText import adjust_text
import time
import multiprocessing
import datetime
import time
import psutil
from collections import defaultdict
import subprocess
from tqdm import tqdm
import contextlib
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerPatch
# Performance optimization imports
try:
    from performance_optimizer import PerformanceProfiler, OptimizedRayTracer, profiler
    PERFORMANCE_OPTIMIZER_AVAILABLE = True
    print("Performance optimizer loaded successfully")
except ImportError:
    PERFORMANCE_OPTIMIZER_AVAILABLE = False
    print("Performance optimizer not available - using basic optimization")

# Fallback profiler class (define globally)
class BasicProfiler:
    def start_timer(self, operation): pass
    def end_timer(self): pass
    def update_frame_stats(self, *args): pass
    def print_summary(self): pass

# Initialize profiler
profiler = BasicProfiler()

# GPU/CUDA availability check
try:
    import cupy as cp
    CUDA_AVAILABLE = cp.cuda.is_available()
    if CUDA_AVAILABLE:
        cuda_device_count = cp.cuda.runtime.getDeviceCount()
        device_props = cp.cuda.runtime.getDeviceProperties(0)
        gpu_name = device_props['name'].decode()
        free_bytes, total_bytes = cp.cuda.runtime.memGetInfo()
        gpu_memory_gb = total_bytes / (1024**3)
        print(f"✅ GPU acceleration available: {gpu_name} ({gpu_memory_gb:.1f} GB)")
    else:
        print("⚠️  GPU detected but CUDA not available")
        gpu_name = "Unknown"
        gpu_memory_gb = 0
except ImportError:
    CUDA_AVAILABLE = False
    cp = None
    print("ℹ️  GPU acceleration not available (CuPy not installed)")
try:
    from numba import cuda, jit, prange
    import numba as nb
    NUMBA_AVAILABLE = True
    print("Numba is available for JIT compilation")
except ImportError:
    NUMBA_AVAILABLE = False
    print("Numba not available - using standard Python functions")

# =====================================================================================
# CONFIGURATION
# =====================================================================================

# ═══════════════════════════════════════════════════════════════════════════════════
# GENERAL SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════════

# Simulation Identification Settings:
# ──────────────────────────────────────────────────────────────────────────────────
# Change this tag to distinguish different simulation runs with e.g. same configuration
file_tag = 'test' # simulation identifier

# Performance Optimization Settings:
# ──────────────────────────────────────────────────────────────────────────────────
# Choose performance optimization level based on your system capabilities:
# - "none": Single-threaded processing (most compatible, but slower)
# - "cpu": Multi-threaded CPU processing (recommended default, good balance)
# - "gpu": CPU multi-threading + GPU acceleration (fastest, requires NVIDIA GPU with CUDA/CuPy)
performance_optimization_level = "gpu"
max_worker_threads = None  # None = auto-detect optimal thread count, or specify number (e.g., 4, 8)

# Path Settings:
# ──────────────────────────────────────────────────────────────────────────────────
base_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(base_dir)
# Path to SUMO config-file
# mo_config_path = os.path.join(parent_dir, 'simulation_examples', 'Intersection-Redesign_Ilic-TR-PartA-2026', '50_low_demand.sumocfg')  # Simulation example: spatial visibility analysis (low demand) [Ilic, 2025]
# sumo_config_path = os.path.join(parent_dir, 'simulation_examples', 'Spatial-Visibility_Ilic-TRB-2025', 'Ilic-2025_config_high-demand.sumocfg')  # Simulation example: spatial visibility analysis (high demand) [Ilic, 2025]
# sumo_config_path = os.path.join(parent_dir, 'simulation_examples', 'VRU-specific-Detection_Ilic-TRA-2026', 'Ilic-2026_config_30kmh.sumocfg')  # Simulation example: VRU-specific detection (30 km/h scenario) [Ilic, 2026]
# sumo_config_path = os.path.join(parent_dir, 'simulation_examples', 'VRU-specific-Detection_Ilic-TRA-2026', 'Ilic-2026_config_50kmh.sumocfg')  # Simulation example: VRU-specific detection (50 km/h scenario) [Ilic, 2026]
sumo_config_path = os.path.join(parent_dir, 'simulation_examples', 'Intersection-Redesign_Ilic-TR-PartA-2026', '50_low_demand.sumocfg')  # Simulation example: Intersection Redesign (low demand, status quo) [Ilic, 2026]

# Path to GeoJSON file (optional)
geojson_path = os.path.join(parent_dir, 'simulation_examples', 'Spatial-Visibility_Ilic-TRB-2025', 'Ilic-2025.geojson') # Simulation example: spatial visibility analysis [Ilic, 2025]
# geojson_path = os.path.join(parent_dir, 'simulation_examples', 'VRU-specific-Detection_Ilic-TRA-2026', 'Ilic-2026.geojson') # Simulation example: spatial visibility analysis [Ilic, 2025]

# Geographic Bounding Box Settings:
# ──────────────────────────────────────────────────────────────────────────────────
# Geographic boundaries in longitude / latitude in EEPSG:4326 (WGS84)
north, south, east, west = 48.129996, 48.126756, 11.558936, 11.553166  # Current simulation area (auto-detected from bicycle trajectories)
# north, south, east, west = 48.150500, 48.149050, 11.571000, 11.567900 # Simulation example: spatial visibility analysis [Ilic, 2025]
# north, south, east, west = 48.146200, 48.144400, 11.580650, 11.577150 # Simulation example: VRU-specific detection [Ilic, 2026]
bbox = (north, south, east, west)

# OSM Feature Toggles (enable/disable loading from OpenStreetMap)
# Set to True to load the corresponding layer; False to skip loading entirely
LoadOSM_Buildings   = True
LoadOSM_Parks       = True
LoadOSM_Trees       = True
LoadOSM_Barriers    = True
LoadOSM_PT_Shelters = True

# Simulation Warm-up Settings:
# ──────────────────────────────────────────────────────────────────────────────────
delay = 180 # Warm-up time in seconds (no ray tracing during this period)

# ═══════════════════════════════════════════════════════════════════════════════════
# RAY TRACING SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════════

# Observer Penetration Rate Settings:
# ──────────────────────────────────────────────────────────────────────────────────
FCO_share = 0.1 # Floating Car Observers penetration rate (0.0 to 1.0)
FBO_share = 0.0  # Floating Bike Observers penetration rate (0.0 to 1.0)

# Ray Tracing Parameter Settings:
# ──────────────────────────────────────────────────────────────────────────────────
numberOfRays = 360  # Number of rays emerging from each observer vehicle
radius = 30         # Ray radius in meters
grid_size = 1.0     # Grid size for visibility heat map (meters) - determines the resolution of LoV and RelVis heatmaps

# Sensor Accuracy Settings:
# ──────────────────────────────────────────────────────────────────────────────────
# Single sensor accuracy for continuous visibility counts (affects probability calculations)
# Valid values: 60, 70, 80, or 90 (representing 60%, 70%, 80%, or 90% accuracy)
single_sensor_accuracy = 70  # Single observer detection accuracy percentage

# Visualization Settings:
# ──────────────────────────────────────────────────────────────────────────────────
useLiveVisualization = False      # Show live visualization during simulation
visualizeRays = False             # Show individual rays in visualization (besides resulting visibility polygon)
useManualFrameForwarding = False  # Manual frame-by-frame progression (for debugging)
saveAnimation = False             # Save animation as video file

# Sensor Accuracy Lookup Table:
# ──────────────────────────────────────────────────────────────────────────────────
# Maps single sensor accuracy to continuous visibility values based on number of simultaneous observations
# Formula basis: Combined probability = 1 - (1 - single_accuracy)^num_observers
SENSOR_ACCURACY_VALUES = {
    60: {1: 0.6, 2: 0.84, 3: 0.94, 4: 0.97, 5: 0.99},
    70: {1: 0.7, 2: 0.91, 3: 0.97, 4: 0.99, 5: 1.0},
    80: {1: 0.8, 2: 0.96, 3: 0.99, 4: 1.0, 5: 1.0},
    90: {1: 0.9, 2: 0.99, 3: 1.0, 4: 1.0, 5: 1.0}
}

# Validate sensor accuracy configuration
if single_sensor_accuracy not in SENSOR_ACCURACY_VALUES:
    raise ValueError(f"Invalid single_sensor_accuracy: {single_sensor_accuracy}. Must be one of {list(SENSOR_ACCURACY_VALUES.keys())}")

# ═══════════════════════════════════════════════════════════════════════════════════
# DATA COLLECTION & ANALYSIS SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════════

# Data Collection Settings:
# ──────────────────────────────────────────────────────────────────────────────────
basic_gap_bridge = 10        # Gap bridging for trajectory smoothing
basic_segment_length = 3     # Minimum segment length for trajectories

# Logging Configuration (Performance Tuning):
# ──────────────────────────────────────────────────────────────────────────────────
# Control which data is collected during simulation to optimize performance.
# Disabling unused logs can significantly reduce computation time and memory usage.
COLLECT_DETECTION_LOGS = True           # Required by evaluation scripts (keep enabled)
COLLECT_BICYCLE_TRAJECTORIES = True     # Required by evaluation scripts (keep enabled)
COLLECT_VEHICLE_TRAJECTORIES = False    # Disabled by default - saves ~40-50% time (only needed for observer visualization)
COLLECT_CONFLICT_DATA = True           # Disabled by default - only enable for safety analysis
COLLECT_FLEET_COMPOSITION = False       # Disabled by default - not used by any evaluation script
COLLECT_TRAFFIC_LIGHT_DATA = True      # Disabled by default - not used by any evaluation script

# Analysis Applications Settings:
# ──────────────────────────────────────────────────────────────────────────────────
AnimatedThreeDimensionalDetectionPlots = False  # Generate 3D animated detection plots

# =====================================================================================
# PERFORMANCE SYSTEM INITIALIZATION
# =====================================================================================

def initialize_performance_settings():
    """
    Initialize performance settings based on the configured optimization level.
    Validates system capabilities and sets up appropriate processing modes.
    """
    global use_multithreading, use_gpu_acceleration, max_worker_threads
    
    # Validate and set thread count
    if max_worker_threads is None:
        if performance_optimization_level == "none":
            max_worker_threads = 1
        elif performance_optimization_level == "cpu":
            max_worker_threads = min(multiprocessing.cpu_count(), 8)  # Conservative for compatibility
        elif performance_optimization_level == "gpu":
            max_worker_threads = min(multiprocessing.cpu_count(), 16)  # Higher cap with GPU
        else:
            max_worker_threads = min(multiprocessing.cpu_count(), 4)  # Safe fallback
    
    # Set processing modes based on optimization level
    if performance_optimization_level == "none":
        use_multithreading = False
        use_gpu_acceleration = False
        print("Performance optimization disabled - using single-threaded processing")
        
    elif performance_optimization_level == "cpu":
        use_multithreading = True
        use_gpu_acceleration = False
        print(f"CPU optimization enabled - using {max_worker_threads} worker threads")
        
    elif performance_optimization_level == "gpu":
        use_multithreading = True
        # Check if GPU acceleration is actually available
        if CUDA_AVAILABLE:
            use_gpu_acceleration = True
            gpu_impl = "Numba CUDA kernels" if NUMBA_AVAILABLE else "CuPy vectorized"
            print(f"GPU optimization enabled - using {max_worker_threads} worker threads + GPU acceleration")
            print(f"  GPU Implementation: {gpu_impl}")
        else:
            use_gpu_acceleration = False
            print(f"GPU requested but not available - falling back to CPU optimization with {max_worker_threads} threads")
            
    else:
        # Invalid optimization level - use safe defaults
        print(f"Warning: Invalid performance_optimization_level '{performance_optimization_level}'. Using 'cpu' mode.")
        use_multithreading = True
        use_gpu_acceleration = False
        print(f"Fallback CPU optimization - using {max_worker_threads} worker threads")

# Initialize performance settings
initialize_performance_settings()

# Print logging configuration
print("\n" + "="*80)
print("LOGGING CONFIGURATION")
print("="*80)
print(f"Detection logs:          {'✅ ENABLED' if COLLECT_DETECTION_LOGS else '❌ DISABLED'}")
print(f"Bicycle trajectories:    {'✅ ENABLED' if COLLECT_BICYCLE_TRAJECTORIES else '❌ DISABLED'}")
print(f"Vehicle trajectories:    {'✅ ENABLED' if COLLECT_VEHICLE_TRAJECTORIES else '❌ DISABLED (saves ~40% time)'}")
print(f"Conflict data:           {'✅ ENABLED' if COLLECT_CONFLICT_DATA else '❌ DISABLED'}")
print(f"Fleet composition:       {'✅ ENABLED' if COLLECT_FLEET_COMPOSITION else '❌ DISABLED (not used)'}")
print(f"Traffic lights:          {'✅ ENABLED' if COLLECT_TRAFFIC_LIGHT_DATA else '❌ DISABLED (not used)'}")
print("="*80 + "\n")

def print_configuration_help():
    """
    Print help information about configuration options.
    Call this function to see available settings and their descriptions.
    """
    print("=" * 80)
    print("FTO-Sim Configuration Help")
    print("=" * 80)
    
    print("\n🏷️  SIMULATION IDENTIFICATION:")
    print("  file_tag = 'test'            # Unique identifier for this simulation run")
    print("                               # Examples: 'baseline', 'scenario_1', 'high_density'")
    
    print("\n🚀 PERFORMANCE OPTIMIZATION LEVELS:")
    print("  performance_optimization_level = 'none'  # Single-threaded (most compatible)")
    print("  performance_optimization_level = 'cpu'   # Multi-threaded CPU (recommended)")
    print("  performance_optimization_level = 'gpu'   # CPU + GPU acceleration (fastest)")
    
    print("\n⚙️  PERFORMANCE SETTINGS:")
    print("  max_worker_threads = None     # Auto-detect optimal thread count")
    print("  max_worker_threads = 4        # Specify exact number of threads")
    
    print("\n🎯 SIMULATION PARAMETERS:")
    print("  FCO_share = 1.0              # Floating Car Observer penetration (0.0-1.0)")
    print("  FBO_share = 0.0              # Floating Bike Observer penetration (0.0-1.0)")
    print("  numberOfRays = 360           # Ray count per observer vehicle")
    print("  radius = 30                  # Ray radius in meters")
    print("  delay = 0                    # Warm-up time in seconds")
    
    print("\n🎨 VISUALIZATION OPTIONS:")
    print("  useLiveVisualization = True  # Show live animation")
    print("  visualizeRays = True         # Show individual rays")
    print("  saveAnimation = False        # Save as video file")
    
    print("\n📊 DATA COLLECTION:")
    print("  basic_gap_bridge = 10        # Gap bridging for trajectory smoothing")
    print("  basic_segment_length = 3     # Minimum segment length for trajectories")
    
    print("\n📍 STUDY AREA (Geographic coordinates):")
    print(f"  north, south, east, west = {north}, {south}, {east}, {west}")
    
    print("\n" + "=" * 80)
    print("💡 Tips:")
    print("  • ALWAYS change file_tag before running different experiments!")
    print("  • Use descriptive tags: 'baseline_360rays', 'reduced_density', 'gpu_test'")
    print("  • For first-time users: Keep 'cpu' optimization level")
    print("  • For GPU acceleration: Install CUDA Toolkit + CuPy")
    print("  • For maximum compatibility: Use 'none' optimization level")
    print("  • Edit values directly in this script's configuration section")
    print("=" * 80)

# =====================================================================================

# Output Directory Setup
def setup_scenario_output_directory():
    """
    Creates scenario-specific output directory structure.
    Returns the base scenario directory path.
    """
    # Create scenario identifier
    scenario_name = f"{file_tag}_FCO{FCO_share*100:.0f}%_FBO{FBO_share*100:.0f}%"
    scenario_dir = os.path.join(parent_dir, 'outputs', scenario_name)
    
    # Create main scenario directory
    os.makedirs(scenario_dir, exist_ok=True)
    
    # Create subdirectories for different output types
    subdirs = [
        'out_logging',
        'out_raytracing'
    ]
    
    for subdir in subdirs:
        os.makedirs(os.path.join(scenario_dir, subdir), exist_ok=True)
    
    return scenario_dir

# General Visualization Settings
fig, ax = plt.subplots(figsize=(12, 8))

# 3D Visualization Settings (for AnimatedThreeDimensionalDetectionPlots)
fig_3d = None
ax_3d = None

# Create namespace object for 3D detection plots
class ThreeDDetectionPlotsNamespace:
    def __init__(self):
        self.observer_trajectories = {}

three_dimensional_detection_plots = ThreeDDetectionPlotsNamespace()

# Projection Settings
proj_from = pyproj.Proj('epsg:4326')   # Source projection: WGS 84
proj_to = pyproj.Proj('epsg:32632')    # Target projection: UTM zone 32N
project = pyproj.Transformer.from_proj(proj_from, proj_to, always_xy=True).transform

# Initialization of empty lists
vehicle_patches = []
ray_lines = []
visibility_polygons = []
# Additional SUMO polygons (from additional-files .add.xml)
additional_polygons = []

# Initialization of empty dictionaries
bicycle_flow_data = {}

# 3D detection plot data storage
bicycle_trajectories = {}
flow_ids = set()
bicycle_detection_data = {}
detection_gif_trajectories = {}

# Logging Settings
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
unique_vehicles = set()
vehicle_type_set = set()
log_columns = ['time_step']

# Use lists for efficient data collection (convert to DataFrame at end)
# This avoids expensive pd.concat() operations during simulation
fleet_composition_logs_list = []
traffic_light_logs_list = []
detection_logs_list = []
vehicle_trajectory_logs_list = []
bicycle_trajectory_logs_list = []
conflict_logs_list = []

# Performance tracking counters for ray tracing
_ray_tracing_stats = {
    'total_rays': 0,
    'total_segments': 0,
    'total_calls': 0,
    'total_time': 0.0,
    'gpu_calls': 0,
    'cpu_calls': 0
}

# DataFrame references (populated at end of simulation)
fleet_composition_logs = None
traffic_light_logs = None
detection_logs = None
vehicle_trajectory_logs = None
bicycle_trajectory_logs = None
conflict_logs = None
# Track SSM device errors to report only once at the end
ssm_device_errors = set()  # Store unique vehicle IDs that have SSM errors
ssm_device_available = None  # None = unknown, True = available, False = not available
performance_stats = pd.DataFrame(columns=[
    'time_step', 'step_duration', 'memory_usage'
])
operation_times = defaultdict(float)
class TimingContext:  # Timing context manager
    _active_contexts = []  # Class variable to track active contexts
    def __init__(self, operation_name):
        self.operation_name = operation_name
        self.start_time = None
        self.child_time = 0  # Time spent in nested contexts
    def __enter__(self):
        self.start_time = time.perf_counter()
        TimingContext._active_contexts.append(self)
        return self
    def __exit__(self, *args):
        end_time = time.perf_counter()
        TimingContext._active_contexts.pop()
        # Calculate duration excluding nested contexts
        if self.start_time is not None:
            total_duration = end_time - self.start_time
        actual_duration = total_duration - self.child_time
        # Add time to parent context
        if TimingContext._active_contexts:  # If there's a parent context
            parent = TimingContext._active_contexts[-1]
            parent.child_time += total_duration
        operation_times[self.operation_name] += actual_duration

# Global variables to store bicycle data (for "Visibility & Bicycle Safety" application functions)
bicycle_data = defaultdict(list)
bicycle_start_times = {}
traffic_light_ids = {}
traffic_light_positions = {}
bicycle_tls = {}
individual_detection_data = {}
flow_detection_data = {}
bicycle_detection_data = {}
bicycle_conflicts = defaultdict(list)
bicycle_conflicts_ind = {}
bicycle_conflicts_flow = {}
foe_trajectories = {}
conflict_bicycle_trajectories = {}
detection_bicycle_trajectories = {}
detection_gif_trajectories = {}
conflict_gif_trajectories = {}

# ---------------------

# ---------------------
# INITIALIZATION
# ---------------------

def load_sumo_simulation():
    """
    Initializes and starts SUMO traffic simulation with error logging and warnings disabled.
    """
    import sys
    import os
    # Suppress SUMO informational messages by redirecting to null
    if sys.platform == 'win32':
        devnull = 'NUL'
    else:
        devnull = '/dev/null'
    sumoCmd = ["sumo", "-c", sumo_config_path, "--message-log", devnull, "--no-warnings", "true", "--no-step-log", "--seed", "18"]
    traci.start(sumoCmd)

def load_geospatial_data():
    """
    Loads road space distribution from the GeoJSON file, buildings, and parks data from OpenStreetMap for the simulated scene.
    """
    # Suppress OGR and projection library warnings
    import warnings
    import logging
    warnings.filterwarnings('ignore', message='.*unsupported OGR type.*')
    warnings.filterwarnings('ignore', message='.*pj_obj_create.*')
    logging.getLogger('pyproj').setLevel(logging.ERROR)
    logging.getLogger('fiona').setLevel(logging.ERROR)
    logging.getLogger('rasterio').setLevel(logging.ERROR)
    
    gdf1 = gpd.read_file(geojson_path) # road space distribution
    # Filter for line elements only (curbs) - exclude Junction polygons (intersection areas)
    gdf1 = gdf1[
        (gdf1['Type'].isin(['LaneBoundary', 'Gate', 'Signal'])) &
        (gdf1.geometry.type.isin(['LineString', 'MultiLineString']))
    ]
    G = ox.graph_from_bbox(bbox=bbox, network_type='all') # NetworkX graph (bounding box)

    # Initialize as None, then load conditionally based on toggles
    buildings = None
    parks = None
    trees = None
    leaves = None
    barriers = None
    PT_shelters = None

    # Buildings
    if LoadOSM_Buildings:
        try:
            buildings = ox.features_from_bbox(bbox=bbox, tags={'building': True}) # buildings
        except Exception:
            buildings = None

    # Parks
    if LoadOSM_Parks:
        try:
            parks = ox.features_from_bbox(bbox=bbox, tags={'leisure': 'park'}) # parks
        except Exception:
            parks = None

    # Trees (and canopy leaves buffer)
    if LoadOSM_Trees:
        try:
            trees = ox.features_from_bbox(bbox=bbox, tags={'natural': 'tree'}) # trees
            leaves = ox.features_from_bbox(bbox=bbox, tags={'natural': 'tree'}) # leaves
        except Exception:
            trees = None
            leaves = None

    # Barriers
    if LoadOSM_Barriers:
        try:
            barriers = ox.features_from_bbox(bbox=bbox, tags={'barrier': 'retaining_wall'}) # barriers (walls)
        except Exception:
            barriers = None

    # Public transport shelters
    if LoadOSM_PT_Shelters:
        try:
            PT_shelters = ox.features_from_bbox(bbox=bbox, tags={'shelter_type': 'public_transport'}) # PT shelters
        except Exception:
            PT_shelters = None
    
    return gdf1, G, buildings, parks, trees, leaves, barriers, PT_shelters

def project_geospatial_data(gdf1, G, buildings, parks, trees, leaves, barriers, PT_shelters):
    """
    Projects all geospatial data (NetworkX graph, road space distribution, buildings, parks) to UTM zone 32N for consistent spatial analysis.
    """
    gdf1_proj = gdf1.to_crs("EPSG:32632")  # road space distribution
    G_proj = ox.project_graph(G, to_crs="EPSG:32632") # NetworkX graph (bounding box)
    # Project buildings if they exist
    buildings_proj = buildings.to_crs("EPSG:32632") if buildings is not None else None
    # Project parks if they exist
    parks_proj = parks.to_crs("EPSG:32632") if parks is not None else None
    # Project trees if they exist
    trees_proj = trees.to_crs("EPSG:32632") if trees is not None else None
    leaves_proj = leaves.to_crs("EPSG:32632") if leaves is not None else None
    # Project barriers if they exist
    barriers_proj = barriers.to_crs("EPSG:32632") if barriers is not None else None
    # Project PT shelters if they exist
    PT_shelters_proj = PT_shelters.to_crs("EPSG:32632") if PT_shelters is not None else None
    
    return gdf1_proj, G_proj, buildings_proj, parks_proj, trees_proj, leaves_proj, barriers_proj, PT_shelters_proj

def initialize_grid(buildings_proj, grid_size=1.0):
    """
    Creates a grid of cells over the simulation area for tracking visibility.
    Each cell is a square of size grid_size and is initiated with a visibility count of 0.
    """
    # Always initialize grid for consistent data collection
    # Use the same bounding box as defined in the configuration (same as ray tracing)
    north, south, east, west = bbox
    # Convert geographic coordinates to UTM for consistency with projected data
    x_min, y_min = project(west, south)
    x_max, y_max = project(east, north)
    
    x_coords = np.arange(x_min, x_max, grid_size)  # array of x-coordinates with specified grid size
    y_coords = np.arange(y_min, y_max, grid_size)  # array of y-coordinates with specified grid size
    grid_points = [(x, y) for x in x_coords for y in y_coords]  # grid points as (x, y) tuples
    grid_cells = [box(x, y, x + grid_size, y + grid_size) for x, y in grid_points]  # box geometries for each grid cell
    discrete_visibility_counts = {cell: 0 for cell in grid_cells}  # initialization of discrete (integer) visibility count for each cell to 0
    continuous_visibility_counts = {cell: 0.0 for cell in grid_cells}  # initialization of continuous (float) visibility count for each cell to 0.0
    return x_coords, y_coords, grid_cells, discrete_visibility_counts, continuous_visibility_counts  # returning grid information and both visibility count types

def get_total_simulation_steps(sumo_config_file):
    """
    Extracts simulation duration parameters from SUMO config file and calculates total steps.
    Sets global step_length and returns total steps as integer.
    """
    global step_length 

    tree = ET.parse(sumo_config_file)  # parsing the XML file
    root = tree.getroot()  # root element of the XML tree

    begin = 0  # default start time (can be adjusted)
    end = 3600  # default end time (1 hour, can be adjusted)
    step_length = 0.1  # default step length (can be adjusted)

    for time in root.findall('time'):  # finding all 'time' elements in the XML
        # Handle potential None returns from XML parsing
        begin_elem = time.find('begin')
        if begin_elem is not None:
            begin = float(begin_elem.get('value', begin))
        
        end_elem = time.find('end')  
        if end_elem is not None:
            end = float(end_elem.get('value', end))
        
        step_elem = time.find('step-length')
        if step_elem is not None:
            step_length = float(step_elem.get('value', step_length))
    total_steps = int((end - begin) / step_length)  # calculating total steps
    return total_steps  # returning the total number of simulation steps

def get_step_length(sumo_config_file):
    """
    Gets the simulation time step length from the SUMO config file.
    Returns the step length as a float.
    """
    tree = ET.parse(sumo_config_file)  # parsing the XML file
    root = tree.getroot()  # root element of the XML tree
    step_length = -1  # default step length (can be adjusted, -1 will lead to an error if not found)
    for time in root.findall('time'):  # finding all 'time' elements in the XML
        # Handle potential None returns from XML parsing
        step_elem = time.find('step-length')
        if step_elem is not None:
            step_length = float(step_elem.get('value', step_length))
    return step_length  # returning the extracted step length

# ---------------------
# SIMULATION SETUP
# ---------------------

def setup_plot():
    """
    Configures the ray tracing visualization plot with title and legend showing buildings, parks, trees, barriers, PT shelters, and vehicle types.
    """
    global fig, ax
    ax.set_aspect('equal')

    ax.set_title(f'Ray Tracing Visualization for penetration rates FCO {FCO_share*100:.0f}% and FBO {FBO_share*100:.0f}%')
    
    # Custom legend proxy and handler for Trees
    class TreeRect(Rectangle):
        pass

    # Custom legend handler for Trees
    class HandlerTree(HandlerPatch):
        def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
            cx = xdescent + width * 0.5
            cy = ydescent + height * 0.5
            canopy = Circle((cx, cy), height * 0.9, facecolor='forestgreen', edgecolor='none', alpha=0.5, transform=trans)
            stem = Circle((cx, cy), height * 0.15, facecolor='forestgreen', edgecolor='none', transform=trans)
            return [canopy, stem]

    # Initialize static elements list - only include elements that are loaded from OSM
    static_elements = []
    
    # Add buildings if enabled and exist
    if LoadOSM_Buildings and buildings_proj is not None:
        static_elements.append(
            Rectangle((0, 0), 1, 1, facecolor='darkgray', edgecolor='black', linewidth=0.5, label='Buildings')
        )
    # Add parks if enabled and exist
    if LoadOSM_Parks and parks_proj is not None:
        static_elements.append(
            Rectangle((0, 0), 1, 1, facecolor='lightgreen', edgecolor='black', linewidth=0.5, alpha=0.5, label='Parks')
        )
    # Add trees if enabled and exist (match concentric stem + canopy visualization)
    if LoadOSM_Trees and trees_proj is not None:
        # Use a dummy Patch as handle; HandlerTree draws concentric circles
        static_elements.append(
            TreeRect((0, 0), 1, 1, facecolor='none', edgecolor='none', label='Trees')
        )
    # Add barriers if enabled and exist - using Rectangle to match expected type
    if LoadOSM_Barriers and barriers_proj is not None:
        static_elements.append(
            Rectangle((0, 0), 0, 0, facecolor='black', linewidth=1.0, label='Barriers')  # type: ignore
        )
    # Add PT shelters if enabled and exist
    if LoadOSM_PT_Shelters and PT_shelters_proj is not None:
        static_elements.append(
            Rectangle((0, 0), 1, 1, facecolor='teal', edgecolor='black', linewidth=0.5, alpha=0.5, label='PT Shelters')
        )
    # Create vehicle type elements based on FCO and FBO presence
    if FCO_share > 0 and FBO_share > 0:
        vehicle_elements = [
            Rectangle((0, 0), 0.36, 1, facecolor='lightgray', edgecolor='gray', label='Parked Vehicle'),
            Rectangle((0, 0), 0.36, 1, facecolor='none', edgecolor='black', label='Passenger Car'),
            Rectangle((0, 0), 0.13, 0.32, facecolor='none', edgecolor='blue', label='Bicycle'),
            Rectangle((0, 0), 0.36, 1, facecolor='none', edgecolor='red', label='FCO'),
            Rectangle((0, 0), 0.13, 0.32, facecolor='none', edgecolor='orange', label='FBO')
        ]
    elif FCO_share > 0 and FBO_share == 0:
        vehicle_elements = [
            Rectangle((0, 0), 0.36, 1, facecolor='lightgray', edgecolor='gray', label='Parked Vehicle'),
            Rectangle((0, 0), 0.36, 1, facecolor='none', edgecolor='black', label='Passenger Car'),
            Rectangle((0, 0), 0.13, 0.32, facecolor='none', edgecolor='blue', label='Bicycle'),
            Rectangle((0, 0), 0.36, 1, facecolor='none', edgecolor='red', label='FCO')
        ]
    elif FCO_share == 0 and FBO_share > 0:
        vehicle_elements = [
            Rectangle((0, 0), 0.36, 1, facecolor='lightgray', edgecolor='gray', label='Parked Vehicle'),
            Rectangle((0, 0), 0.36, 1, facecolor='none', edgecolor='black', label='Passenger Car'),
            Rectangle((0, 0), 0.13, 0.32, facecolor='none', edgecolor='blue', label='Bicycle'),
            Rectangle((0, 0), 0.13, 0.32, facecolor='none', edgecolor='orange', label='FBO')
        ]
    else:
        vehicle_elements = [
            Rectangle((0, 0), 0.36, 1, facecolor='lightgray', edgecolor='gray', label='Parked Vehicle'),
            Rectangle((0, 0), 0.36, 1, facecolor='none', edgecolor='black', label='Passenger Car'),
            Rectangle((0, 0), 0.13, 0.32, facecolor='none', edgecolor='blue', label='Bicycle')
        ]
    
    # Primary legend: static elements + vehicle legend (upper right)
    from matplotlib.lines import Line2D as _Line2D
    from matplotlib.patches import Patch as _Patch
    primary_handles = static_elements + vehicle_elements
    primary_legend = ax.legend(
        handles=primary_handles,
        loc='upper right',
        fontsize=12,
        handler_map={TreeRect: HandlerTree()}
    )
    # Add the primary legend as an artist so it remains when a second legend is created
    try:
        ax.add_artist(primary_legend)
    except Exception:
        pass

    # Secondary legend: rays and critical interaction areas (bottom right), conditional
    secondary_handles = []
    if FCO_share > 0 or FBO_share > 0:
        unobstructed_ray_handle = _Line2D([0], [0], color=(0.53, 0.81, 0.98, 1.0), lw=2, label='Unobstructed Ray')
        intersected_ray_handle = _Line2D([0], [0], color=(1.0, 0.27, 0, 1.0), lw=2, label='Intersected Ray')
        secondary_handles.extend([unobstructed_ray_handle, intersected_ray_handle])

    try:
        if additional_polygons:
            critical_area_handle = _Patch(facecolor='yellow', edgecolor='yellow', alpha=0.3, label='Critical Interaction Areas')
            secondary_handles.append(critical_area_handle)
    except Exception:
        pass

    if secondary_handles:
        second_legend = ax.legend(handles=secondary_handles, loc='lower right', fontsize=10)
        ax.add_artist(second_legend)

    # Add initial warm-up text box
    # Dynamic attribute assignment for matplotlib axes - ignore type checking
    ax.warm_up_text = ax.text(0.02, 0.98, f'Warm-up phase\nRemaining: {delay}s',  # type: ignore 
                             transform=ax.transAxes,
                             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
                             verticalalignment='top',
                             fontsize=10)
    ax.warm_up_text.set_visible(True)

def plot_geospatial_data(gdf1_proj, G_proj, buildings_proj, parks_proj, trees_proj, leaves_proj, barriers_proj, PT_shelters_proj):
    """
    Plots the geospatial data (base map, road network, buildings and parks) onto the current axes.
    """
    gdf1_proj.plot(ax=ax, color='lightgray', alpha=0.5, edgecolor='lightgray', zorder=1)  # Plot the road space distribution
    ox.plot_graph(G_proj, ax=ax, bgcolor='none', edge_color='none', node_size=0, show=False, close=False)  # Plot the NetworkX graph
    # Plot parks if they exist
    if parks_proj is not None:
        parks_proj.plot(ax=ax, facecolor='lightgreen', alpha=0.5, edgecolor='black', linewidth=0.5, zorder=2)  # Plot parks
    # Plot buildings if they exist
    if buildings_proj is not None:
        buildings_proj.plot(ax=ax, facecolor='darkgray', edgecolor='black', linewidth=0.5, zorder=3)  # Plot buildings
    # Plot trees if they exist
    if trees_proj is not None:
        trees_circle = trees_proj.buffer(0.5)
        trees_circle.plot(ax=ax, facecolor='forestgreen', edgecolor='black', linewidth=0.5, zorder=5)  # Plot trees
        leaves_circle = leaves_proj.buffer(2.5)
        leaves_circle.plot(ax=ax, facecolor='forestgreen', alpha=0.5, edgecolor='black', linewidth=0.5, zorder=5)  # Plot leaves
    # Plot barriers if they exist
    if barriers_proj is not None:
        barriers_proj.plot(ax=ax, edgecolor='black', linewidth=1.0, zorder=4)  # Plot barriers
    # Plot PT shelters if they exist
    if PT_shelters_proj is not None:
        PT_shelters_proj.plot(ax=ax, facecolor='teal', alpha=0.5, edgecolor='black', linewidth=0.5, zorder=6)  # Plot PT shelters

    # Plot any additional polygons loaded from SUMO additional-files
    try:
        if additional_polygons:
            for poly, attrs in additional_polygons:
                coords = list(poly.exterior.coords)
                # use yellow border and yellow fill with 30% transparency for critical areas
                facecolor = 'yellow'
                edgecolor = 'yellow'
                lw = attrs.get('linewidth', 0.5)
                patch = MatPolygon(coords, closed=True, facecolor=facecolor, edgecolor=edgecolor, linewidth=lw, alpha=0.3, zorder=7)
                ax.add_patch(patch)
    except Exception:
        pass

def convert_simulation_coordinates(x, y):
    """
    Converts coordinates from SUMO's internal system to UTM zone 32N.
    """
    lon, lat = traci.simulation.convertGeo(x, y)  # Convert SUMO coordinates to longitude and latitude
    x_32632, y_32632 = project(lon, lat)  # Project longitude and latitude to UTM zone 32N
    return x_32632, y_32632  # Return the converted coordinates

def load_additional_polygons_from_sumocfg(sumocfg_path):
    """
    Parse the SUMO .sumocfg and any referenced additional-files (.add.xml).
    Extract <poly> elements, convert their SUMO coordinates to projected UTM
    coordinates using `convert_simulation_coordinates`, and return a list of
    tuples (shapely_polygon, attrs_dict).
    Requires TraCI connection to be active for coordinate conversion.
    """
    polys = []
    try:
        tree = ET.parse(sumocfg_path)
        root = tree.getroot()
    except Exception:
        return polys

    # Find all additional-files entries (value may contain multiple files)
    add_files = []
    for add in root.findall('.//additional-files'):
        val = add.get('value')
        if not val:
            continue
        for part in val.split():
            add_files.append(part)

    # Resolve relative paths and parse each additional file
    cfg_dir = os.path.dirname(sumocfg_path)
    for af in add_files:
        af_path = af
        if not os.path.isabs(af_path):
            af_path = os.path.join(cfg_dir, af_path)
        if not os.path.exists(af_path):
            continue
        try:
            atree = ET.parse(af_path)
            aroot = atree.getroot()
        except Exception:
            continue

        # Extract all <poly> elements
        for pel in aroot.findall('.//poly'):
            shape = pel.get('shape')
            if not shape:
                continue
            color = pel.get('color', 'yellow')
            fill = pel.get('fill', '1')
            lineWidth = pel.get('lineWidth', '0.5')

            # Parse SUMO coordinates (shape format: "x1,y1 x2,y2 ...")
            coords = []
            for xy in shape.strip().split():
                try:
                    sx, sy = xy.split(',')
                    sx = float(sx)
                    sy = float(sy)
                except Exception:
                    continue
                # Convert to projected coordinates; skip if conversion fails
                try:
                    px, py = convert_simulation_coordinates(sx, sy)
                except Exception:
                    px, py = None, None
                if px is None or py is None:
                    continue
                coords.append((px, py))

            if len(coords) >= 3:
                try:
                    shp = Polygon(coords)
                    attrs = {
                        'color': color,
                        'fill': (fill != '0'),
                        'linewidth': float(lineWidth)
                    }
                    polys.append((shp, attrs))
                except Exception:
                    continue

    return polys

def vehicle_attributes(vehicle_type):
    """
    Returns visualization attributes (shape, color, dimensions) for different vehicle types.
    Uses default passenger car attributes if vehicle type is not recognized.
    """
    # Define a dictionary to store vehicle attributes
    vehicle_types = {
        # Observer vehicles
        "floating_car_observer": (Rectangle, 'red', (1.8, 5)),
        "floating_bike_observer": (Rectangle, 'orange', (0.65, 1.6)),
        # Regular passenger vehicles
        "veh_passenger": (Rectangle, 'gray', (1.8, 5)),
        "parked_vehicle": (Rectangle, 'gray', (1.8, 5)),
        "DEFAULT_VEHTYPE": (Rectangle, 'gray', (1.8, 5)),
        # Public transport vehicles
        "pt_bus": (Rectangle, 'gray', (2.5, 12)),
        "bus_bus": (Rectangle, 'gray', (2.5, 12)),
        "bus": (Rectangle, 'gray', (2.5, 12)),
        "pt_tram": (Rectangle, 'gray', (2.5, 12)),
        # Trucks
        "truck_truck": (Rectangle, 'gray', (2.4, 7.1)),
        # Bicycles
        "bike_bicycle": (Rectangle, 'blue', (0.65, 1.6)),
        "DEFAULT_BIKETYPE": (Rectangle, 'blue', (0.65, 1.6)),
        # Pedestrians
        "ped_pedestrian": (Rectangle, 'green', (0.5, 0.5))
    }
    # Return the attributes for the given vehicle type, or default if not found
    return vehicle_types.get(vehicle_type, (Rectangle, 'gray', (1.8, 5)))

def sumo_position_to_center(x, y, length, angle):
    """
    Converts SUMO's position (center of front bumper) to geometric center of vehicle.
    
    SUMO returns the position of the center of the front bumper. To get the geometric
    center, we need to offset backwards by half the vehicle length in the direction
    opposite to the vehicle's heading.
    
    Args:
        x, y: SUMO position (front bumper center)
        length: Vehicle length in meters
        angle: Vehicle angle in degrees (0 = North, 90 = East, SUMO convention)
    
    Returns:
        (x_center, y_center): Geometric center of the vehicle
    """
    # Convert angle to radians for trigonometric calculations
    # SUMO angle: 0° = North (up), 90° = East (right), increases clockwise
    angle_rad = np.radians(angle)
    
    # Calculate offset: move backwards by half vehicle length
    # sin and cos are swapped because 0° is North (y-axis) in SUMO, not East (x-axis)
    offset_x = -np.sin(angle_rad) * (length / 2)
    offset_y = -np.cos(angle_rad) * (length / 2)
    
    return x + offset_x, y + offset_y

def create_vehicle_polygon(x, y, width, length, angle):
    """
    Creates a rectangular polygon representing a vehicle at the given position and orientation.
    
    Note: This function expects the geometric center position (x, y).
    Use sumo_position_to_center() to convert from SUMO's front bumper position first.
    """
    adjusted_angle = (-angle) % 360  # Adjust angle for correct rotation
    rect = Polygon([(-width / 2, -length / 2), (-width / 2, length / 2), (width / 2, length / 2), (width / 2, -length / 2)])  # Create initial rectangle
    rotated_rect = rotate(rect, adjusted_angle, use_radians=False, origin=(0, 0))  # Rotate rectangle
    translated_rect = translate(rotated_rect, xoff=x, yoff=y)  # Move rectangle to correct position
    return translated_rect  # Return final polygon

# ---------------------
# RAY TRACING
# ---------------------

def generate_rays(center):
    """
    Generates evenly spaced rays radiating from the center point of an observer vehicle (FCO/FBO).
    """
    angles = np.linspace(0, 2 * np.pi, numberOfRays, endpoint=False)  # Create evenly spaced angles
    rays = [(center, (center[0] + np.cos(angle) * radius, center[1] + np.sin(angle) * radius)) for angle in angles]  # Generate ray endpoints
    return rays  # Return the list of rays

def detect_intersections(ray, objects):
    """
    Detects intersections between a ray and a list of objects.
    Returns the closest intersection point to the ray's origin.
    Logs the start and completion of the thread for each ray.
    """
    logging.info(f"Thread started for ray: {ray}")  # Log the start of the thread
    closest_intersection = None  # Initialize the closest intersection point
    min_distance = float('inf')  # Set initial minimum distance to infinity
    ray_line = LineString(ray)  # Create a LineString object from the ray
    for obj in objects:  # Iterate through all objects
        if ray_line.intersects(obj):  # Check if the ray intersects with the object
            intersection_point = ray_line.intersection(obj)  # Get the intersection point
            if intersection_point.is_empty:  # Skip if intersection is empty
                continue
            if intersection_point.geom_type.startswith('Multi'):  # Handle multi-part intersections
                for part in intersection_point.geoms:  # Iterate through each part
                    if part.is_empty or not hasattr(part, 'coords'):  # Skip invalid parts
                        continue
                    for coord in part.coords:  # Check each coordinate
                        distance = Point(ray[0]).distance(Point(coord))  # Calculate distance
                        if distance < min_distance:  # Update if closer
                            min_distance = distance
                            closest_intersection = coord
            else:  # Handle single-part intersections
                if not hasattr(intersection_point, 'coords'):  # Skip if no coordinates
                    continue
                for coord in intersection_point.coords:  # Check each coordinate
                    distance = Point(ray[0]).distance(Point(coord))  # Calculate distance
                    if distance < min_distance:  # Update if closer
                        min_distance = distance
                        closest_intersection = coord
    logging.info(f"Thread completed for ray: {ray}")  # Log the completion of the thread
    return closest_intersection  # Return the closest intersection point

def count_rays_hitting_object(rays, object_polygon, occluding_objects, apply_occlusion=True):
    """
    Count how many rays from an observer hit a specific object.
    
    Args:
        rays: List of ray tuples (origin, endpoint)
        object_polygon: Shapely polygon representing the object to check
        occluding_objects: List of objects that can occlude (should exclude target object)
        apply_occlusion: If True, cut rays at first intersection; if False, use full rays
    
    Returns:
        int: Number of rays hitting the object
    """
    hit_count = 0
    
    for ray in rays:
        ray_line = LineString(ray)
        
        if apply_occlusion:
            # Find first intersection with occluding objects
            closest_intersection = detect_intersections_optimized(ray, occluding_objects)
            if closest_intersection:
                # Cut ray at first intersection
                ray_line = LineString([ray[0], closest_intersection])
        
        # Check if (possibly cut) ray intersects target object
        if ray_line.intersects(object_polygon):
            hit_count += 1
    
    return hit_count

def detect_intersections_optimized(ray, objects):
    """
    Optimized version of intersection detection with reduced logging and better data structures.
    Uses distance calculation without Point object creation for better performance.
    """
    closest_intersection = None
    min_distance_sq = float('inf')  # Use squared distance to avoid sqrt
    ray_line = LineString(ray)
    ray_origin = ray[0]  # Tuple (x, y)
    
    # Early exit if no objects
    if not objects:
        return None
    
    for obj in objects:
        try:
            if not ray_line.intersects(obj):
                continue
                
            intersection_point = ray_line.intersection(obj)
            if intersection_point.is_empty:
                continue
                
            # Handle different geometry types more efficiently
            coords_to_check = []
            if hasattr(intersection_point, 'geoms'):  # Multi-part geometry
                for part in intersection_point.geoms:
                    if hasattr(part, 'coords'):
                        coords_to_check.extend(part.coords)
            elif hasattr(intersection_point, 'coords'):  # Single geometry
                coords_to_check.extend(intersection_point.coords)
            
            # Find closest intersection using squared distance (faster)
            for coord in coords_to_check:
                dx = coord[0] - ray_origin[0]
                dy = coord[1] - ray_origin[1]
                distance_sq = dx*dx + dy*dy
                if distance_sq < min_distance_sq:
                    min_distance_sq = distance_sq
                    closest_intersection = coord
                    
        except Exception:
            continue  # Skip problematic geometries
    
    return closest_intersection


def count_rays_with_occlusion_source(rays, object_polygon, static_objects, dynamic_objects):
    """
    Count rays hitting an object and separately track occlusions by static vs dynamic objects.
    
    Args:
        rays: List of ray tuples (origin, endpoint)
        object_polygon: Shapely polygon representing the target object
        static_objects: List of static occluding objects (buildings, trees, etc.)
        dynamic_objects: List of dynamic occluding objects (vehicles)
    
    Returns:
        dict: {
            'theoretical_rays': rays that would hit without occlusion,
            'actual_rays': rays that hit after occlusion,
            'static_blocked': rays blocked by static objects,
            'dynamic_blocked': rays blocked by dynamic objects
        }
    """
    theoretical_rays = 0
    actual_rays = 0
    static_blocked = 0
    dynamic_blocked = 0
    
    for ray in rays:
        ray_line = LineString(ray)
        
        # Check if ray would hit target without occlusion
        if ray_line.intersects(object_polygon):
            theoretical_rays += 1
            
            # Find first intersection with all occluding objects
            closest_intersection = None
            min_distance = float('inf')
            blocking_object_type = None  # 'static' or 'dynamic'
            ray_origin = Point(ray[0])
            
            # Check static objects
            for obj in static_objects:
                try:
                    if not ray_line.intersects(obj):
                        continue
                    intersection_point = ray_line.intersection(obj)
                    if intersection_point.is_empty:
                        continue
                    
                    coords_to_check = []
                    if hasattr(intersection_point, 'geoms'):
                        for part in intersection_point.geoms:
                            if hasattr(part, 'coords'):
                                coords_to_check.extend(part.coords)
                    elif hasattr(intersection_point, 'coords'):
                        coords_to_check.extend(intersection_point.coords)
                    
                    for coord in coords_to_check:
                        distance = ray_origin.distance(Point(coord))
                        if distance < min_distance:
                            min_distance = distance
                            closest_intersection = coord
                            blocking_object_type = 'static'
                except Exception:
                    continue
            
            # Check dynamic objects
            for obj in dynamic_objects:
                try:
                    if not ray_line.intersects(obj):
                        continue
                    intersection_point = ray_line.intersection(obj)
                    if intersection_point.is_empty:
                        continue
                    
                    coords_to_check = []
                    if hasattr(intersection_point, 'geoms'):
                        for part in intersection_point.geoms:
                            if hasattr(part, 'coords'):
                                coords_to_check.extend(part.coords)
                    elif hasattr(intersection_point, 'coords'):
                        coords_to_check.extend(intersection_point.coords)
                    
                    for coord in coords_to_check:
                        distance = ray_origin.distance(Point(coord))
                        if distance < min_distance:
                            min_distance = distance
                            closest_intersection = coord
                            blocking_object_type = 'dynamic'
                except Exception:
                    continue
            
            # Determine if ray reaches target
            if closest_intersection:
                # Ray is blocked - check if it still reaches target
                cut_ray = LineString([ray[0], closest_intersection])
                if cut_ray.intersects(object_polygon):
                    actual_rays += 1
                else:
                    # Ray was blocked before reaching target
                    if blocking_object_type == 'static':
                        static_blocked += 1
                    elif blocking_object_type == 'dynamic':
                        dynamic_blocked += 1
            else:
                # No blocking, ray reaches target
                actual_rays += 1
    
    return {
        'theoretical_rays': theoretical_rays,
        'actual_rays': actual_rays,
        'static_blocked': static_blocked,
        'dynamic_blocked': dynamic_blocked
    }


def generate_rays_vectorized(centers, num_rays, ray_radius):
    """
    Vectorized ray generation for multiple observer centers.
    """
    if not centers:
        return []
    
    angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
    rays = []
    
    for center in centers:
        center_rays = [
            (center, (center[0] + np.cos(angle) * ray_radius, center[1] + np.sin(angle) * ray_radius))
            for angle in angles
        ]
        rays.extend(center_rays)
    
    return rays

# GPU-accelerated function (if CUDA is available)
def extract_segment_endpoints(objects):
    """
    Extract all line segments from geometric objects for GPU processing.
    Returns segments as arrays of start and end points.
    """
    segments_start = []
    segments_end = []
    
    for obj in objects:
        try:
            # Handle different geometry types
            if hasattr(obj, 'exterior'):  # Polygon
                coords = list(obj.exterior.coords)
                for i in range(len(coords) - 1):
                    segments_start.append(coords[i][:2])  # (x, y)
                    segments_end.append(coords[i+1][:2])
            elif hasattr(obj, 'coords'):  # LineString
                coords = list(obj.coords)
                for i in range(len(coords) - 1):
                    segments_start.append(coords[i][:2])
                    segments_end.append(coords[i+1][:2])
        except Exception:
            continue
    
    return segments_start, segments_end

# Numba CUDA kernel for ray-segment intersection (fastest GPU implementation)
if NUMBA_AVAILABLE and CUDA_AVAILABLE:
    @cuda.jit
    def ray_segment_intersection_kernel(ray_origins, ray_dirs, seg_starts, seg_ends, 
                                       min_t_out, hit_out):
        """
        CUDA kernel for parallel ray-segment intersection.
        Each thread processes one ray against all segments.
        
        Args:
            ray_origins: (n_rays, 2) array of ray start points
            ray_dirs: (n_rays, 2) array of normalized ray directions
            seg_starts: (n_segs, 2) array of segment start points
            seg_ends: (n_segs, 2) array of segment end points
            min_t_out: (n_rays,) output array for closest intersection distance
            hit_out: (n_rays,) output array for hit flags (1=hit, 0=miss)
        """
        ray_idx = cuda.grid(1)
        
        if ray_idx >= ray_origins.shape[0]:
            return
        
        # Load ray data into registers (fast)
        ro_x = ray_origins[ray_idx, 0]
        ro_y = ray_origins[ray_idx, 1]
        rd_x = ray_dirs[ray_idx, 0]
        rd_y = ray_dirs[ray_idx, 1]
        
        min_t = 1e10  # Large number (infinity)
        hit_found = 0
        
        # Test against all segments
        for seg_idx in range(seg_starts.shape[0]):
            # Load segment data
            ss_x = seg_starts[seg_idx, 0]
            ss_y = seg_starts[seg_idx, 1]
            se_x = seg_ends[seg_idx, 0]
            se_y = seg_ends[seg_idx, 1]
            
            # Segment direction
            sd_x = se_x - ss_x
            sd_y = se_y - ss_y
            
            # Vector from segment start to ray origin
            dx = ss_x - ro_x
            dy = ss_y - ro_y
            
            # Cross product (determinant)
            det = rd_x * sd_y - rd_y * sd_x
            
            # Check if lines are parallel (det ~ 0)
            if abs(det) < 1e-10:
                continue
            
            # Calculate intersection parameters
            t = (dx * sd_y - dy * sd_x) / det
            s = (dx * rd_y - dy * rd_x) / det
            
            # Valid intersection: t > 0 (in front) and 0 <= s <= 1 (on segment)
            if t > 1e-10 and s >= 0.0 and s <= 1.0:
                if t < min_t:
                    min_t = t
                    hit_found = 1
        
        # Write results
        min_t_out[ray_idx] = min_t
        hit_out[ray_idx] = hit_found

def ray_segment_intersection_gpu_numba(ray_origins, ray_dirs, seg_starts, seg_ends):
    """
    GPU ray-segment intersection using Numba CUDA kernels (fastest implementation).
    
    Returns closest intersection point for each ray.
    """
    n_rays = len(ray_origins)
    
    # Allocate output arrays on GPU
    min_t_out = cuda.device_array(n_rays, dtype=np.float32)
    hit_out = cuda.device_array(n_rays, dtype=np.int32)
    
    # Configure kernel launch for better GPU occupancy
    # Quadro P520 has 384 CUDA cores - aim for good utilization
    threads_per_block = 256
    blocks_per_grid = (n_rays + threads_per_block - 1) // threads_per_block
    
    # For small ray counts, adjust block size to ensure better occupancy
    # Aim for at least 8-16 blocks to keep GPU busy
    if blocks_per_grid < 8:
        # Reduce threads per block to create more blocks
        threads_per_block = max(32, (n_rays + 7) // 8)  # Target ~8 blocks
        blocks_per_grid = (n_rays + threads_per_block - 1) // threads_per_block
    
    # Launch kernel
    ray_segment_intersection_kernel[blocks_per_grid, threads_per_block](
        ray_origins, ray_dirs, seg_starts, seg_ends, min_t_out, hit_out
    )
    
    # Copy results back to host
    min_t_cpu = min_t_out.copy_to_host()
    hit_cpu = hit_out.copy_to_host()
    
    # Calculate intersection points
    ro_cpu = ray_origins.copy_to_host()
    rd_cpu = ray_dirs.copy_to_host()
    
    intersections = np.zeros((n_rays, 2), dtype=np.float32)
    for i in range(n_rays):
        if hit_cpu[i]:
            intersections[i, 0] = ro_cpu[i, 0] + min_t_cpu[i] * rd_cpu[i, 0]
            intersections[i, 1] = ro_cpu[i, 1] + min_t_cpu[i] * rd_cpu[i, 1]
        else:
            intersections[i, 0] = np.nan
            intersections[i, 1] = np.nan
    
    return intersections, hit_cpu.astype(bool)

def ray_segment_intersection_gpu(ray_origins, ray_dirs, seg_starts, seg_ends):
    """
    Vectorized GPU ray-segment intersection using CuPy.
    Uses parametric line intersection formula for maximum parallelism.
    
    For each ray: P(t) = origin + t * direction
    For each segment: Q(s) = seg_start + s * (seg_end - seg_start), where 0 <= s <= 1
    
    Returns closest intersection point for each ray (or None).
    """
    n_rays = len(ray_origins)
    n_segs = len(seg_starts)
    
    # Reshape for broadcasting: (n_rays, 1, 2) and (1, n_segs, 2)
    ro = ray_origins[:, cp.newaxis, :]  # (n_rays, 1, 2)
    rd = ray_dirs[:, cp.newaxis, :]      # (n_rays, 1, 2)
    ss = seg_starts[cp.newaxis, :, :]    # (1, n_segs, 2)
    se = seg_ends[cp.newaxis, :, :]      # (1, n_segs, 2)
    
    # Segment direction vectors
    seg_dirs = se - ss  # (1, n_segs, 2)
    
    # Cross product components for intersection calculation
    # Solving: ray_origin + t*ray_dir = seg_start + s*seg_dir
    dx = ss[:, :, 0] - ro[:, :, 0]  # (n_rays, n_segs)
    dy = ss[:, :, 1] - ro[:, :, 1]
    
    det = rd[:, :, 0] * seg_dirs[:, :, 1] - rd[:, :, 1] * seg_dirs[:, :, 0]
    
    # Avoid division by zero (parallel lines)
    eps = 1e-10
    valid = cp.abs(det) > eps
    
    # Calculate parameters t (for ray) and s (for segment)
    t = cp.where(valid, (dx * seg_dirs[:, :, 1] - dy * seg_dirs[:, :, 0]) / (det + eps), cp.inf)
    s = cp.where(valid, (dx * rd[:, :, 1] - dy * rd[:, :, 0]) / (det + eps), -1.0)
    
    # Valid intersections: t > 0 (in front of ray) and 0 <= s <= 1 (on segment)
    valid_intersections = valid & (t > eps) & (s >= 0) & (s <= 1)
    
    # Set invalid intersections to infinity
    t = cp.where(valid_intersections, t, cp.inf)
    
    # Find closest intersection for each ray
    min_t_idx = cp.argmin(t, axis=1)  # Index of closest segment for each ray
    min_t = cp.min(t, axis=1)          # Distance to closest intersection
    
    # Compute intersection points for rays that hit something
    hit_rays = min_t < cp.inf
    
    # Calculate intersection coordinates VECTORIZED (no CPU loop!)
    # intersections[i] = ray_origins[i] + min_t[i] * ray_dirs[i]
    intersections = ray_origins + (min_t[:, cp.newaxis] * ray_dirs)
    
    # Set non-hit rays to NaN
    intersections[~hit_rays] = cp.nan
    
    return intersections, hit_rays

# GPU-accelerated function (if CUDA is available)
def gpu_intersection_detection(rays, objects):
    """
    GPU-accelerated intersection detection using Numba CUDA kernels (when available)
    or CuPy vectorized operations as fallback.
    
    Intelligently chooses between GPU and CPU based on workload size:
    - Small workloads (<5000 total ops): Use CPU (lower overhead)
    - Large workloads (>=5000 ops): Use GPU (better parallelism)
    
    Falls back to CPU if GPU processing fails.
    """
    import time
    global _gpu_first_run
    
    try:
        dispatch_start = time.perf_counter()
        
        if not rays or not objects:
            return [None] * len(rays)
        
        # GPU WORKLOAD PROCESSING
        # Extract segments  FIRST to know real workload (extraction is cheap: ~20ms)
        extraction_start = time.perf_counter()
        seg_starts_cpu, seg_ends_cpu = extract_segment_endpoints(objects)
        extraction_time = (time.perf_counter() - extraction_start) * 1000
        
        if not seg_starts_cpu:
            return [None] * len(rays)
        
        # Calculate workload characteristics with ACTUAL segment count
        n_rays = len(rays)
        n_segments = len(seg_starts_cpu)
        total_ops = n_rays * n_segments
        
        dispatch_time = (time.perf_counter() - dispatch_start) * 1000
        
        # GPU mode: Use GPU acceleration (no CPU fallback)
        # User explicitly chose GPU - respect their choice
        _ray_tracing_stats['gpu_calls'] += 1  # Track GPU usage
        
        # Use Numba CUDA kernel if available (fastest for very large workloads)
        if NUMBA_AVAILABLE:
            # Print diagnostic on first run
            if not hasattr(gpu_intersection_detection, '_gpu_logged'):
                gpu_intersection_detection._gpu_logged = True
                print(f"\n{'='*80}")
                print(f"GPU ACCELERATION: Numba CUDA Kernels")
                print(f"{'='*80}")
                print(f"Workload: {n_rays} rays × {n_segments} segments = {total_ops:,} operations")
                print(f"Segment extraction: {extraction_time:.2f}ms")
                print(f"{'='*80}")
            
            # Extract ray data as numpy arrays
            prep_start = time.perf_counter()
            ray_data = [(ray[0][0], ray[0][1], ray[1][0] - ray[0][0], ray[1][1] - ray[0][1]) for ray in rays]
            ray_origins_np = np.array([[r[0], r[1]] for r in ray_data], dtype=np.float32)
            ray_dirs_np = np.array([[r[2], r[3]] for r in ray_data], dtype=np.float32)
            
            # Normalize ray directions
            ray_lengths = np.sqrt(ray_dirs_np[:, 0]**2 + ray_dirs_np[:, 1]**2)
            ray_dirs_np = ray_dirs_np / ray_lengths[:, np.newaxis]
            
            # Convert to Numba device arrays
            seg_starts_np = np.array(seg_starts_cpu, dtype=np.float32)
            seg_ends_np = np.array(seg_ends_cpu, dtype=np.float32)
            prep_time = (time.perf_counter() - prep_start) * 1000
            
            # Transfer to GPU
            transfer_start = time.perf_counter()
            ray_origins_gpu = cuda.to_device(ray_origins_np)
            ray_dirs_gpu = cuda.to_device(ray_dirs_np)
            seg_starts_gpu = cuda.to_device(seg_starts_np)
            seg_ends_gpu = cuda.to_device(seg_ends_np)
            transfer_to_gpu_time = (time.perf_counter() - transfer_start) * 1000
            
            # Calculate transfer sizes
            transfer_size_mb = (ray_origins_np.nbytes + ray_dirs_np.nbytes + 
                               seg_starts_np.nbytes + seg_ends_np.nbytes) / (1024*1024)
            
            # Run Numba CUDA kernel
            kernel_start = time.perf_counter()
            intersections, hit_rays = ray_segment_intersection_gpu_numba(
                ray_origins_gpu, ray_dirs_gpu, seg_starts_gpu, seg_ends_gpu
            )
            kernel_time = (time.perf_counter() - kernel_start) * 1000
            
            # Fast list comprehension
            postproc_start = time.perf_counter()
            results = [
                (float(intersections[i, 0]), float(intersections[i, 1])) if hit_rays[i] else None
                for i in range(len(rays))
            ]
            postproc_time = (time.perf_counter() - postproc_start) * 1000
            
            gpu_total = prep_time + transfer_to_gpu_time + kernel_time + postproc_time
            
            # Log detailed timing on first run
            if not hasattr(gpu_intersection_detection, '_gpu_timing_logged'):
                gpu_intersection_detection._gpu_timing_logged = True
                print(f"\nGPU TIMING BREAKDOWN (Numba CUDA):")
                print(f"  Data preparation: {prep_time:.2f}ms")
                print(f"  Transfer to GPU: {transfer_to_gpu_time:.2f}ms ({transfer_size_mb:.2f} MB)")
                print(f"  Kernel execution: {kernel_time:.2f}ms")
                print(f"  Result processing: {postproc_time:.2f}ms")
                print(f"  TOTAL: {gpu_total:.2f}ms")
                print(f"  Throughput: {total_ops / (kernel_time/1000):,.0f} ops/sec (kernel only)")
                print(f"  Effective: {n_rays / (gpu_total/1000):,.0f} rays/sec (including overhead)")
            
            return results
        
        # Fallback to CuPy if Numba not available
        else:
            # Print diagnostic on first run
            if not hasattr(gpu_intersection_detection, '_gpu_logged'):
                gpu_intersection_detection._gpu_logged = True
                print(f"\n{'='*80}")
                print(f"GPU ACCELERATION: CuPy Vectorized")
                print(f"{'='*80}")
                print(f"Workload: {n_rays} rays × {n_segments} segments = {total_ops:,} operations")
                print(f"Segment extraction: {extraction_time:.2f}ms")
                print(f"Note: Numba not available, using CuPy vectorized operations")
                print(f"{'='*80}")
            
            # Extract ray data (list comprehension is faster than loops)
            prep_start = time.perf_counter()
            ray_data = [(ray[0][0], ray[0][1], ray[1][0] - ray[0][0], ray[1][1] - ray[0][1]) for ray in rays]
            ray_origins = cp.array([[r[0], r[1]] for r in ray_data], dtype=cp.float32)
            ray_dirs = cp.array([[r[2], r[3]] for r in ray_data], dtype=cp.float32)
            
            # Normalize ray directions
            ray_lengths = cp.sqrt(ray_dirs[:, 0]**2 + ray_dirs[:, 1]**2)
            ray_dirs = ray_dirs / ray_lengths[:, cp.newaxis]
            
            # Transfer to GPU
            seg_starts = cp.array(seg_starts_cpu, dtype=cp.float32)
            seg_ends = cp.array(seg_ends_cpu, dtype=cp.float32)
            prep_time = (time.perf_counter() - prep_start) * 1000
            
            # Perform vectorized intersection on GPU
            compute_start = time.perf_counter()
            intersections, hit_rays = ray_segment_intersection_gpu(ray_origins, ray_dirs, seg_starts, seg_ends)
            compute_time = (time.perf_counter() - compute_start) * 1000
            
            # Convert results back to CPU format - vectorized
            transfer_start = time.perf_counter()
            intersections_cpu = cp.asnumpy(intersections)
            hit_rays_cpu = cp.asnumpy(hit_rays)
            transfer_time = (time.perf_counter() - transfer_start) * 1000
            
            # Fast list comprehension instead of loop
            postproc_start = time.perf_counter()
            results = [
                (float(intersections_cpu[i, 0]), float(intersections_cpu[i, 1])) if hit_rays_cpu[i] else None
                for i in range(len(rays))
            ]
            postproc_time = (time.perf_counter() - postproc_start) * 1000
            
            gpu_total = prep_time + compute_time + transfer_time + postproc_time
            
            # Log detailed timing on first run
            if not hasattr(gpu_intersection_detection, '_gpu_timing_logged'):
                gpu_intersection_detection._gpu_timing_logged = True
                print(f"\nGPU TIMING BREAKDOWN (CuPy):")
                print(f"  Data prep + transfer to GPU: {prep_time:.2f}ms")
                print(f"  GPU computation: {compute_time:.2f}ms")
                print(f"  Transfer from GPU: {transfer_time:.2f}ms")
                print(f"  Result processing: {postproc_time:.2f}ms")
                print(f"  TOTAL: {gpu_total:.2f}ms")
                print(f"  Throughput: {n_rays / (gpu_total/1000):,.0f} rays/sec")
            
            return results
        
    except Exception as e:
        # Fallback to CPU processing on any GPU error
        print(f"\n{'='*80}")
        print(f"GPU ERROR - EMERGENCY CPU FALLBACK")
        print(f"{'='*80}")
        print(f"Error: {str(e)}")
        print(f"Falling back to CPU single-threaded processing")
        print(f"{'='*80}")
        return [detect_intersections_optimized(ray, objects) for ray in rays]

def update_with_ray_tracing(frame):
    """
    Updates the simulation for each frame, performing ray tracing for FCOs and FBOs.
    Handles vehicle creation, ray generation, intersection detection, and visibility polygon creation.
    Updates vehicle patches, ray lines, and visibility counts for visualization.
    Also updates bicycle diagrams and logs simulation data.
    """
    import time  # Import at function level for performance measurements
    global vehicle_patches, ray_lines, visibility_polygons, FCO_share, FBO_share, discrete_visibility_counts, continuous_visibility_counts, numberOfRays, useRTREEmethod, visualizeRays, useManualFrameForwarding, delay, bicycle_detection_data, progress_bar
    detected_color = (1.0, 0.27, 0, 0.5)
    undetected_color = (0.53, 0.81, 0.98, 0.5)

    # Initialize new lists
    new_vehicle_patches = []
    new_ray_lines = []

    stepLength = get_step_length(sumo_config_path)
    step_start_time = time.time() # Start time for performance metrics

    with TimingContext("simulation_step"):
        traci.simulationStep()  # Advance the simulation by one step
        if useManualFrameForwarding and frame > delay / stepLength:
            input("Press Enter to continue...")  # Wait for user input if manual forwarding is enabled

    # Set vehicle types (FCO and FBO) based on probability
    # Set seed for consistent FCO/FBO assignment across runs
    if frame == 0:  # Only set seed once at the beginning
        np.random.seed(75)
    
    FCO_type = "floating_car_observer"
    FBO_type = "floating_bike_observer"
    for vehicle_id in traci.simulation.getDepartedIDList():
        if traci.vehicle.getTypeID(vehicle_id) == "DEFAULT_VEHTYPE" and np.random.uniform() < FCO_share:
            traci.vehicle.setType(vehicle_id, FCO_type)
        if traci.vehicle.getTypeID(vehicle_id) == "DEFAULT_BIKETYPE" and np.random.uniform() < FBO_share:
            traci.vehicle.setType(vehicle_id, FBO_type)

    # Initialize progress tracking on first frame
    # if frame == 0:
    if 'progress_bar' not in globals():
        progress_bar = tqdm(total=total_steps-1, 
                          desc='Simulation Progress',
                          bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} steps [elapsed time: {elapsed} min] ',
                          ncols=100,  # Fixed width
                          position=0,  # Keep bar at the bottom
                          leave=True)  # Don't clear the bar when done
        
    if frame > 0:
        progress_bar.update(1)

    current_vehicles = set(traci.vehicle.getIDList())
    
    # Store which vehicles are finished this frame
    finished_vehicles = set()
    for vehicle_id in bicycle_trajectories.keys():
        if vehicle_id not in current_vehicles:
            finished_vehicles.add(vehicle_id)

    if AnimatedThreeDimensionalDetectionPlots:
        if frame == 0:
            print('Animated 3D detection plots initiated.')
        with TimingContext("3d_animated_detections"):
            three_dimensional_detection_plots_gif(frame)

    # Close progress bar and plot on last frame
    if frame == total_steps - 1:
        progress_bar.close()
        
        # Close the plot window if live visualization is enabled
        # This allows the script to continue to logging without user interaction
        if useLiveVisualization:
            plt.close(fig)
            print('Visualization window closed automatically.')

        if frame == delay / stepLength:
            print(f'\nWarm-up phase completed after {delay/stepLength:.0f} steps.')

    # Update warm-up text box
    if frame <= delay / stepLength:
        remaining_time = int(delay - frame * stepLength)
        ax.warm_up_text.set_text(f'Warm-up phase\nremaining: {remaining_time}s')
    elif frame == (delay / stepLength) + 1:
        ax.warm_up_text.set_visible(False)  # Hide text box after warm-up

    # Main simulation loop (after warm-up period)
    if frame > delay / stepLength:
        updated_cells = {}  # Changed from set to dict to track observer count per cell

        # Optimized static objects creation with caching
        if not hasattr(update_with_ray_tracing, 'static_objects_cache'):
            # Initialize static objects cache on first run
            static_objects = []
            if buildings_proj is not None:
                static_objects.extend(building.geometry for building in buildings_proj.itertuples())
            if trees_proj is not None:
                trees_circle = trees_proj.buffer(0.5)  # 1 meter radius for trees
                static_objects.extend(tree for tree in trees_circle.geometry)
            if barriers_proj is not None:
                static_objects.extend(barriers.geometry for barriers in barriers_proj.itertuples())
            if PT_shelters_proj is not None:
                static_objects.extend(PT_shelters.geometry for PT_shelters in PT_shelters_proj.itertuples())
            
            update_with_ray_tracing.static_objects_cache = static_objects
        else:
            static_objects = update_with_ray_tracing.static_objects_cache.copy()
        
        # Add parked vehicles to static objects (these can change each frame)
        parked_vehicle_count = 0
        for vehicle_id in traci.vehicle.getIDList():
            vehicle_type = traci.vehicle.getTypeID(vehicle_id)
            if vehicle_type == "parked_vehicle":
                x, y = traci.vehicle.getPosition(vehicle_id)
                width, length = vehicle_attributes(vehicle_type)[2]
                angle = traci.vehicle.getAngle(vehicle_id)
                # Convert from SUMO position (front bumper) to geometric center
                x_center, y_center = sumo_position_to_center(x, y, length, angle)
                x_32632, y_32632 = convert_simulation_coordinates(x_center, y_center)
                parked_vehicle_geom = create_vehicle_polygon(x_32632, y_32632, width, length, angle)
                static_objects.append(parked_vehicle_geom)
                parked_vehicle_count += 1

        # Safely remove existing dynamic elements
        for patch in vehicle_patches[:]:  # Create a copy of the list to iterate over
            try:
                if patch in ax.patches:
                    patch.remove()
            except:
                pass  # Ignore if patch can't be removed
        for line in ray_lines[:]:  # Create a copy of the list to iterate over
            try:
                if line in ax.lines:
                    line.remove()
            except:
                pass  # Ignore if line can't be removed
        for polygon in visibility_polygons[:]:  # Create a copy of the list to iterate over
            try:
                if polygon in ax.patches:
                    polygon.remove()
            except:
                pass  # Ignore if polygon can't be removed

        vehicle_patches.clear()
        ray_lines.clear()
        visibility_polygons.clear()

        # Initialize observer-polygon mapping to track which observer created each visibility polygon
        observer_visibility_polygons = {}  # Maps observer_id to their visibility polygon
        
        # Process each vehicle and perform ray tracing
        for vehicle_id in traci.vehicle.getIDList():
            vehicle_type = traci.vehicle.getTypeID(vehicle_id)
            Shape, edgecolor, (width, length) = vehicle_attributes(vehicle_type)
            x, y = traci.vehicle.getPosition(vehicle_id)
            angle = traci.vehicle.getAngle(vehicle_id)
            # Convert from SUMO position (front bumper) to geometric center
            x_center, y_center = sumo_position_to_center(x, y, length, angle)
            x_32632, y_32632 = convert_simulation_coordinates(x_center, y_center)
            
            # Create and update vehicle patches
            adjusted_angle = (-angle) % 360
            lower_left_corner = (x_32632 - width / 2, y_32632 - length / 2)
            patch = Rectangle(lower_left_corner, width, length, 
                            facecolor='lightgray' if vehicle_type == "parked_vehicle" else 'white',
                            edgecolor='gray' if vehicle_type == "parked_vehicle" else edgecolor)
            
            # Create dynamic objects - helper function for angle handling
            def get_vehicle_angle_safe(vid):
                try:
                    return traci.vehicle.getAngle(vid)
                except Exception:
                    return 0  # Fallback if TRACI signature changed
                    
            dynamic_objects_geom = []
            for vid in traci.vehicle.getIDList():
                if vid != vehicle_id:
                    # Get position and convert coordinates for the other vehicle
                    pos_x, pos_y = traci.vehicle.getPosition(vid)
                    # Get vehicle dimensions for the other vehicle
                    o_width, o_length = vehicle_attributes(traci.vehicle.getTypeID(vid))[2]
                    # Get vehicle angle safely for the other vehicle
                    o_angle = get_vehicle_angle_safe(vid)
                    # Convert from SUMO position (front bumper) to geometric center
                    o_center_x, o_center_y = sumo_position_to_center(pos_x, pos_y, o_length, o_angle)
                    other_x_32632, other_y_32632 = convert_simulation_coordinates(o_center_x, o_center_y)
                    # Create polygon for the other vehicle (do NOT overwrite outer x_32632/y_32632)
                    polygon = create_vehicle_polygon(other_x_32632, other_y_32632, o_width, o_length, o_angle)
                    dynamic_objects_geom.append(polygon)

            t = transforms.Affine2D().rotate_deg_around(x_32632, y_32632, adjusted_angle) + ax.transData
            patch.set_transform(t)
            patch.set_zorder(10)
            new_vehicle_patches.append(patch)

            # Ray tracing for observers
            if vehicle_type in ["floating_car_observer", "floating_bike_observer"]:
                # Start performance timing
                profiler.start_timer("ray_tracing_total")
                
                center = (x_32632, y_32632)
                rays = generate_rays(center)
                all_objects = static_objects + dynamic_objects_geom
                ray_endpoints = []
                
                # Track performance metrics
                num_rays = len(rays)
                num_objects = len(all_objects)
                intersections_found = 0

                # Ray tracing processing based on performance level
                if use_gpu_acceleration and CUDA_AVAILABLE:
                    # GPU-accelerated processing
                    rt_start = time.perf_counter()
                    profiler.start_timer("gpu_ray_processing")
                    intersections = gpu_intersection_detection(rays, all_objects)
                    profiler.end_timer()
                    rt_elapsed = time.perf_counter() - rt_start
                    
                    # Track performance stats
                    _ray_tracing_stats['total_rays'] += len(rays)
                    _ray_tracing_stats['total_segments'] += len(all_objects)
                    _ray_tracing_stats['total_calls'] += 1
                    _ray_tracing_stats['total_time'] += rt_elapsed
                    
                    # Print timing on first observer (show actual method used)
                    if not hasattr(update_with_ray_tracing, '_gpu_timing_logged'):
                        update_with_ray_tracing._gpu_timing_logged = True
                        # Check if CPU fallback was used
                        was_cpu = _ray_tracing_stats.get('cpu_calls', 0) > 0
                        was_gpu = _ray_tracing_stats.get('gpu_calls', 0) > 0
                        if was_cpu and not was_gpu:
                            method_name = "CPU (adaptive fallback)"
                        elif was_gpu:
                            method_name = "GPU (Numba CUDA)" if NUMBA_AVAILABLE else "GPU (CuPy)"
                        else:
                            method_name = "Unknown"
                        print(f"\n{'='*80}")
                        print(f"FIRST OBSERVER COMPLETE")
                        print(f"{'='*80}")
                        print(f"Method: {method_name}")
                        print(f"Total time: {rt_elapsed*1000:.2f}ms for {len(rays)} rays")
                        print(f"Throughput: {len(rays) / rt_elapsed:,.0f} rays/sec")
                        print(f"{'='*80}")
                    
                    for i, ray in enumerate(rays):
                        intersection = intersections[i]
                        if intersection:
                            end_point = intersection
                            ray_color = detected_color
                            intersections_found += 1
                        else:
                            angle = np.arctan2(ray[1][1] - ray[0][1], ray[1][0] - ray[0][0])
                            end_point = (ray[0][0] + np.cos(angle) * radius, ray[0][1] + np.sin(angle) * radius)
                            ray_color = undetected_color

                        ray_endpoints.append(end_point)
                        ray_line = Line2D([ray[0][0], end_point[0]], [ray[0][1], end_point[1]], 
                                        color=ray_color, linewidth=1)
                        if visualizeRays:
                            ray_line.set_zorder(5)
                            ax.add_line(ray_line)
                        new_ray_lines.append(ray_line)
                elif use_multithreading:
                    # CPU multithreaded processing with batching for better efficiency
                    cpu_start = time.perf_counter()
                    profiler.start_timer("cpu_ray_processing")
                    
                    # Print diagnostic on first run
                    if not hasattr(update_with_ray_tracing, '_cpu_timing_logged'):
                        update_with_ray_tracing._cpu_timing_logged = True
                        print(f"\n→ Using CPU multi-threading with {max_worker_threads} workers")
                        print(f"  Processing {len(rays)} rays against {len(all_objects)} objects")
                    
                    # Batch rays to reduce thread overhead (process multiple rays per thread)
                    batch_size = max(1, len(rays) // (max_worker_threads * 4))  # 4 batches per worker
                    ray_batches = [rays[i:i+batch_size] for i in range(0, len(rays), batch_size)]
                    
                    def process_ray_batch(batch):
                        """Process a batch of rays on a single thread"""
                        results = []
                        for ray in batch:
                            intersection = detect_intersections_optimized(ray, all_objects)
                            results.append((ray, intersection))
                        return results
                    
                    with ThreadPoolExecutor(max_workers=max_worker_threads) as executor:
                        # Submit batches instead of individual rays
                        futures = [executor.submit(process_ray_batch, batch) for batch in ray_batches]
                        
                        for future in as_completed(futures):
                            batch_results = future.result()
                            for ray, intersection in batch_results:
                                if intersection:
                                    end_point = intersection
                                    ray_color = detected_color
                                    intersections_found += 1
                                else:
                                    angle = np.arctan2(ray[1][1] - ray[0][1], ray[1][0] - ray[0][0])
                                    end_point = (ray[0][0] + np.cos(angle) * radius, ray[0][1] + np.sin(angle) * radius)
                                    ray_color = undetected_color

                                ray_endpoints.append(end_point)
                                ray_line = Line2D([ray[0][0], end_point[0]], [ray[0][1], end_point[1]], 
                                                color=ray_color, linewidth=1)
                                if visualizeRays:
                                    ray_line.set_zorder(5)
                                    ax.add_line(ray_line)
                                new_ray_lines.append(ray_line)
                    profiler.end_timer()
                    cpu_elapsed = time.perf_counter() - cpu_start
                    
                    # Track performance stats
                    _ray_tracing_stats['total_rays'] += len(rays)
                    _ray_tracing_stats['total_segments'] += len(all_objects)
                    _ray_tracing_stats['total_calls'] += 1
                    _ray_tracing_stats['total_time'] += cpu_elapsed
                    
                    # Print timing on first observer
                    if not hasattr(update_with_ray_tracing, '_cpu_batch_timing_logged'):
                        update_with_ray_tracing._cpu_batch_timing_logged = True
                        print(f"\n{'='*80}")
                        print(f"FIRST OBSERVER COMPLETE")
                        print(f"{'='*80}")
                        print(f"Method: CPU multi-threading ({max_worker_threads} workers)")
                        print(f"Total time: {cpu_elapsed*1000:.2f}ms for {len(rays)} rays")
                        print(f"Batches: {len(ray_batches)} batches")
                        print(f"Throughput: {len(rays) / cpu_elapsed:,.0f} rays/sec")
                        print(f"{'='*80}")
                else:
                    # Single-threaded processing (most compatible)
                    profiler.start_timer("single_thread_ray_processing")
                    for ray in rays:
                        intersection = detect_intersections_optimized(ray, all_objects)
                        if intersection:
                            end_point = intersection
                            ray_color = detected_color
                            intersections_found += 1
                        else:
                            angle = np.arctan2(ray[1][1] - ray[0][1], ray[1][0] - ray[0][0])
                            end_point = (ray[0][0] + np.cos(angle) * radius, ray[0][1] + np.sin(angle) * radius)
                            ray_color = undetected_color

                        ray_endpoints.append(end_point)
                        ray_line = Line2D([ray[0][0], end_point[0]], [ray[0][1], end_point[1]], 
                                        color=ray_color, linewidth=1)
                        if visualizeRays:
                            ray_line.set_zorder(5)
                            ax.add_line(ray_line)
                        new_ray_lines.append(ray_line)
                    profiler.end_timer()
                
                # Update performance statistics
                profiler.update_frame_stats(num_rays, num_objects, intersections_found)
                profiler.end_timer()  # End ray_tracing_total timing

                # Create and update visibility polygons
                if len(ray_endpoints) > 2:
                    ray_endpoints.sort(key=lambda point: np.arctan2(point[1] - center[1], point[0] - center[0]))
                    visibility_polygon = MatPolygon(ray_endpoints, color='green', alpha=0.5, fill=None)
                    ax.add_patch(visibility_polygon)
                    visibility_polygons.append(visibility_polygon)
                    
                    # Store mapping of this observer to their visibility polygon
                    observer_visibility_polygons[vehicle_id] = visibility_polygon

                    # Update visibility counts (both discrete and continuous)
                    visibility_polygon_shape = Polygon(ray_endpoints)
                    for cell in discrete_visibility_counts.keys():
                        if visibility_polygon_shape.contains(cell):
                            # Track number of observers seeing this cell in current frame
                            if cell not in updated_cells:
                                updated_cells[cell] = 0
                            updated_cells[cell] += 1

        # Apply visibility count updates after all observers processed in this frame
        # Convert updated_cells from set to dict tracking observer counts per cell
        for cell, num_observers in updated_cells.items():
            # Update discrete counts (binary: +1 if any observer sees the cell)
            discrete_visibility_counts[cell] += 1
            
            # Update continuous counts (weighted by number of simultaneous observers)
            # Use lookup table based on configured single_sensor_accuracy
            accuracy_lookup = SENSOR_ACCURACY_VALUES[single_sensor_accuracy]
            # Cap at 5 observers (values for 5+ are the same)
            capped_observers = min(num_observers, 5)
            continuous_value = accuracy_lookup[capped_observers]
            continuous_visibility_counts[cell] += continuous_value

        # Process bicycle detections
        for vehicle_id in traci.vehicle.getIDList():
            if vehicle_id not in bicycle_detection_data:
                bicycle_detection_data[vehicle_id] = []
            vehicle_type = traci.vehicle.getTypeID(vehicle_id)
            if vehicle_type in ["bicycle", "DEFAULT_BIKETYPE", "floating_bike_observer"]:
                is_detected = False
                detecting_observers = []  # Track ALL observers that detect this bicycle
                
                # Get vehicle dimensions and angle
                width, length = vehicle_attributes(vehicle_type)[2]
                angle = traci.vehicle.getAngle(vehicle_id)
                # Get SUMO position and convert to geometric center
                x, y = traci.vehicle.getPosition(vehicle_id)
                x_center, y_center = sumo_position_to_center(x, y, length, angle)
                x_32632, y_32632 = convert_simulation_coordinates(x_center, y_center)
                vehicle_polygon = create_vehicle_polygon(x_32632, y_32632, width, length, angle)
                
                # Check against each observer's visibility polygon
                for obs_id, vis_polygon in observer_visibility_polygons.items():
                    if vis_polygon and vehicle_polygon.intersects(Polygon(vis_polygon.get_xy())):
                        # Calculate occlusion level for this observer-bicycle pair
                        # Get observer position
                        obs_x, obs_y = traci.vehicle.getPosition(obs_id)
                        obs_type = traci.vehicle.getTypeID(obs_id)
                        obs_length = vehicle_attributes(obs_type)[2][1]
                        obs_angle = traci.vehicle.getAngle(obs_id)
                        obs_x_center, obs_y_center = sumo_position_to_center(obs_x, obs_y, obs_length, obs_angle)
                        obs_x_utm, obs_y_utm = convert_simulation_coordinates(obs_x_center, obs_y_center)
                        
                        # Generate rays for this observer
                        observer_center = (obs_x_utm, obs_y_utm)
                        observer_rays = generate_rays(observer_center)
                        
                        # Separate static and dynamic occluding objects
                        # Remove bicycle from both lists
                        static_occluders = [obj for obj in static_objects 
                                          if not obj.equals(vehicle_polygon)]
                        dynamic_occluders = [obj for obj in dynamic_objects_geom 
                                            if not obj.equals(vehicle_polygon)]
                        
                        # Count rays with detailed occlusion source tracking
                        occlusion_stats = count_rays_with_occlusion_source(
                            observer_rays, vehicle_polygon, static_occluders, dynamic_occluders
                        )
                        
                        theoretical_rays = occlusion_stats['theoretical_rays']
                        actual_rays = occlusion_stats['actual_rays']
                        static_blocked_rays = occlusion_stats['static_blocked']
                        dynamic_blocked_rays = occlusion_stats['dynamic_blocked']
                        
                        # Only mark as detected if at least one ray actually reaches the bicycle
                        if actual_rays > 0:
                            is_detected = True
                            
                            # Calculate occlusion level
                            if theoretical_rays > 0:
                                visible_percentage = (actual_rays / theoretical_rays) * 100
                                occlusion_level = 100 - visible_percentage
                            else:
                                occlusion_level = 0  # No rays hit, so no occlusion to measure
                            
                            detecting_observers.append({
                                'id': obs_id,
                                'theoretical_rays': theoretical_rays,
                                'actual_rays': actual_rays,
                                'occlusion_level': occlusion_level,
                                'static_blocked_rays': static_blocked_rays,
                                'dynamic_blocked_rays': dynamic_blocked_rays
                            })
                
                # Store detection with ALL detecting observers and their occlusion data
                bicycle_detection_data[vehicle_id].append(
                    (traci.simulation.getTime(), is_detected, detecting_observers)
                )

        # Update visualization
        # if useLiveVisualization:
        for patch in vehicle_patches:
            patch.remove()
        vehicle_patches = new_vehicle_patches
        ray_lines = new_ray_lines
        for patch in vehicle_patches:
            ax.add_patch(patch)
    
    # Data collection for logging (always active)
    with TimingContext("data_collection"):
        collect_fleet_composition(frame)
        collect_bicycle_trajectories(frame)
        collect_bicycle_detection_data(frame)
        collect_bicycle_conflict_data(frame)
        collect_traffic_light_data(frame)
        collect_vehicle_trajectories(frame)
        collect_performance_data(frame, step_start_time)

    return vehicle_patches + ray_lines + visibility_polygons + [ax]

def run_animation(total_steps):
    """
    Runs and displays a matplotlib animation of the ray tracing simulation.
    """
    global fig, ax

    # Always create a new figure and plot the static elements
    plt.close(fig)
    if useLiveVisualization:
        matplotlib.use('TkAgg', force=True)
        print('Ray tracing live visualization initiated.')
    else:
        matplotlib.use('Agg')  # Use non-interactive backend for saving
    
    # Create new figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot static elements
    plot_geospatial_data(gdf1_proj, G_proj, buildings_proj, parks_proj, trees_proj, leaves_proj, barriers_proj, PT_shelters_proj)
    setup_plot()
    
    # Draw the figure to ensure static elements are rendered
    fig.canvas.draw()
    
    # Create animation
    anim = FuncAnimation(fig, update_with_ray_tracing, 
                        frames=range(1, total_steps),
                        interval=33,
                        blit=False)

    if saveAnimation:
        # Calculate fps based on simulation step length
        step_length = get_step_length(sumo_config_path)
        fps = int(1 / step_length)  # Convert step length to fps
        # Create writer
        writer = FFMpegWriter(fps=fps, metadata=dict(artist='Me'), bitrate=2400)
        
        filename = os.path.join(scenario_output_dir, 'out_raytracing', f'ray_tracing_visualization_{file_tag}_FCO{FCO_share*100:.0f}%_FBO{FBO_share*100:.0f}%.mp4')
        print('Saving of ray tracing animation initiated.')
        
        try:
            plt.tight_layout()
            fig.canvas.draw()
            anim.save(filename, writer=writer)
            print('Ray tracing animation saved successfully.')
        except Exception as e:
            print(f'\nError saving animation: {str(e)}')
            print('Please ensure FFmpeg is installed and accessible.')
    
    if useLiveVisualization:
        plt.show()
    
    return anim

# ---------------------
# DATA COLLECTION & LOGGING
# ---------------------

def collect_fleet_composition(time_step):
    """
    Collects fleet composition data at each simulation time step.
    Tracks new and present vehicles by type, updates global variables for unique vehicles 
    and vehicle types. Only logs timesteps with vehicle activity.
    
    Note: time_step parameter is the frame counter from the main loop.
    We use traci.simulation.getTime() to get the actual SUMO simulation time.
    """
    if not COLLECT_FLEET_COMPOSITION:
        return  # Skip if disabled for performance
    
    global unique_vehicles, vehicle_type_set, fleet_composition_logs_list

    # Get the actual SUMO simulation time (in seconds)
    current_sim_time = traci.simulation.getTime()

    # Initialize counters for new and present vehicles
    new_vehicle_counts = {}
    present_vehicle_counts = {}
    has_activity = False
    
    # Count new vehicles that have entered the simulation
    for vehicle_id in traci.simulation.getDepartedIDList():
        if vehicle_id not in unique_vehicles:
            has_activity = True
            unique_vehicles.add(vehicle_id)
            vehicle_type = traci.vehicle.getTypeID(vehicle_id)
            vehicle_type_set.add(vehicle_type)

            # Increment count for new vehicles of this type
            if vehicle_type not in new_vehicle_counts:
                new_vehicle_counts[vehicle_type] = 0
            new_vehicle_counts[vehicle_type] += 1

    # Count vehicles currently present in the simulation
    for vehicle_id in traci.vehicle.getIDList():
        has_activity = True
        vehicle_type = traci.vehicle.getTypeID(vehicle_id)
        if vehicle_type not in present_vehicle_counts:
            present_vehicle_counts[vehicle_type] = 0
        present_vehicle_counts[vehicle_type] += 1

    # Only create and add log entry if there was vehicle activity
    if has_activity:
        # Prepare log entry for this time step
        log_entry = {'time_step': current_sim_time}  # Use actual SUMO simulation time
        for vehicle_type in vehicle_type_set:
            log_entry[f'new_{vehicle_type}_count'] = new_vehicle_counts.get(vehicle_type, 0)
            log_entry[f'present_{vehicle_type}_count'] = present_vehicle_counts.get(vehicle_type, 0)

        # Add log entry to the list
        fleet_composition_logs_list.append(log_entry)

def collect_traffic_light_data(frame):
    """Collects traffic light data at each simulation time step."""
    if not COLLECT_TRAFFIC_LIGHT_DATA:
        return  # Skip if disabled for performance
    
    global traffic_light_logs_list
    
    current_time = traci.simulation.getTime()
    entries = []  # Collect all entries first
    
    # Process each traffic light intersection
    for tl_id in traci.trafficlight.getIDList():
        # Get all controlled lanes and links
        controlled_lanes = traci.trafficlight.getControlledLanes(tl_id)
        controlled_links = traci.trafficlight.getControlledLinks(tl_id)
        signal_states = traci.trafficlight.getRedYellowGreenState(tl_id)
        
        # Create mapping of lane to signal index only at the start
        lane_to_signal = {}
        if frame == 0:  # Only create mapping at simulation start
            for i, links in enumerate(controlled_links):
                for connection in links:
                    if connection:  # Some might be None
                        from_lane = connection[0]
                        lane_to_signal[from_lane] = i
        
        # Track unique vehicles to avoid counting them multiple times
        unique_vehicles = set()
        vehicles_stopped = 0
        total_waiting_time = 0
        vehicles_by_type = {}
        total_queue_length = 0
        
        # Process each lane
        for lane in controlled_lanes:
            vehicles = traci.lane.getLastStepVehicleIDs(lane)
            lane_queue_length = 0
            
            for vehicle in vehicles:
                # Skip parked vehicles and vehicles we've already counted
                if (vehicle not in unique_vehicles and 
                    not traci.vehicle.isAtBusStop(vehicle) and
                    not traci.vehicle.getTypeID(vehicle).startswith('parked')):
                    
                    unique_vehicles.add(vehicle)
                    veh_type = traci.vehicle.getTypeID(vehicle)
                    vehicles_by_type[veh_type] = vehicles_by_type.get(veh_type, 0) + 1
                    
                    if traci.vehicle.getSpeed(vehicle) < 0.1:  # Stopped vehicles
                        vehicles_stopped += 1
                        total_waiting_time += traci.vehicle.getAccumulatedWaitingTime(vehicle)
                        lane_queue_length += 1
            
            total_queue_length += lane_queue_length
    
        # Get current program and phase information
        current_program = traci.trafficlight.getProgram(tl_id)
        current_phase = traci.trafficlight.getPhase(tl_id)
        phase_duration = traci.trafficlight.getPhaseDuration(tl_id)
        remaining_duration = traci.trafficlight.getNextSwitch(tl_id) - current_time
        
        log_entry = {
            'time_step': frame,
            'traffic_light_id': tl_id,
            'program': current_program,
            'phase': current_phase,
            'phase_duration': phase_duration,
            'remaining_duration': remaining_duration,
            'signal_states': signal_states,
            'total_queue_length': total_queue_length,
            'vehicles_stopped': vehicles_stopped,
            'average_waiting_time': total_waiting_time / max(vehicles_stopped, 1),
            'vehicles_by_type': str(vehicles_by_type),
            'lane_to_signal_mapping': str(lane_to_signal) if frame == 0 else ""  # Always include this column
        }
        
        entries.append(log_entry)
    
    # Only extend list if we have entries
    if entries:
        traffic_light_logs_list.extend(entries)

def collect_bicycle_detection_data(time_step):
    """
    Collects detection data at each simulation time step.
    Records when and where bicycles are detected by observers.
    Only records entries when actual detections occur.
    
    Note: time_step parameter is the frame counter from the main loop.
    We use traci.simulation.getTime() to get the actual SUMO simulation time.
    """
    if not COLLECT_DETECTION_LOGS:
        return  # Skip if disabled for performance
    
    global detection_logs_list
    detection_entries = []
    has_detections = False
    
    # Get the actual SUMO simulation time (in seconds)
    current_sim_time = traci.simulation.getTime()
    
    # Get all current detections
    for vehicle_id in traci.vehicle.getIDList():
        vehicle_type = traci.vehicle.getTypeID(vehicle_id)
        
        # Only process FCOs and FBOs
        if vehicle_type in ["floating_car_observer", "floating_bike_observer"]:
            x_obs, y_obs = traci.vehicle.getPosition(vehicle_id)
            obs_length = vehicle_attributes(vehicle_type)[2][1]
            obs_angle = traci.vehicle.getAngle(vehicle_id)
            # Convert from SUMO position (front bumper) to geometric center
            x_obs_center, y_obs_center = sumo_position_to_center(x_obs, y_obs, obs_length, obs_angle)
            x_obs_utm, y_obs_utm = convert_simulation_coordinates(x_obs_center, y_obs_center)
                            
            # Check which bicycles this observer detects
            for bicycle_id in traci.vehicle.getIDList():
                bicycle_type = traci.vehicle.getTypeID(bicycle_id)
                
                if bicycle_type in ["bicycle", "DEFAULT_BIKETYPE", "floating_bike_observer"]:
                    # Check if this bicycle is detected by the observer's visibility polygon
                    if bicycle_id in bicycle_detection_data and bicycle_detection_data[bicycle_id]:
                        latest_detection = bicycle_detection_data[bicycle_id][-1]
                        
                        # Unpack detection data
                        if len(latest_detection) >= 3:
                            detection_time, is_detected, observers = latest_detection
                        else:
                            detection_time = latest_detection[0]
                            is_detected = latest_detection[1]
                            observers = []
                        
                        if detection_time == traci.simulation.getTime() and is_detected:
                            # Find this observer's specific occlusion data
                            observer_occlusion = 0
                            theoretical_rays = 0
                            actual_rays = 0
                            observer_found = False
                            
                            for obs in observers:
                                if obs['id'] == vehicle_id:
                                    observer_occlusion = obs.get('occlusion_level', 0)
                                    theoretical_rays = obs.get('theoretical_rays', 0)
                                    actual_rays = obs.get('actual_rays', 0)
                                    static_blocked_rays = obs.get('static_blocked_rays', 0)
                                    dynamic_blocked_rays = obs.get('dynamic_blocked_rays', 0)
                                    observer_found = True
                                    break
                            
                            # Only log if this specific observer actually detected the bicycle
                            if not observer_found:
                                continue
                            
                            has_detections = True
                            # Get bicycle position and calculate distance
                            x_bike, y_bike = traci.vehicle.getPosition(bicycle_id)
                            bike_length = vehicle_attributes(bicycle_type)[2][1]
                            bike_angle = traci.vehicle.getAngle(bicycle_id)
                            # Convert from SUMO position (front bumper) to geometric center
                            x_bike_center, y_bike_center = sumo_position_to_center(x_bike, y_bike, bike_length, bike_angle)
                            x_bike_utm, y_bike_utm = convert_simulation_coordinates(x_bike_center, y_bike_center)
                            detection_distance = np.sqrt((x_obs_utm - x_bike_utm)**2 + (y_obs_utm - y_bike_utm)**2)
                            
                            # Create detection entry with occlusion data
                            detection_entries.append({
                                'time_step': current_sim_time,  # Use actual SUMO simulation time
                                'observer_id': vehicle_id,
                                'observer_type': vehicle_type,
                                'bicycle_id': bicycle_id,
                                'x_coord': x_bike_utm,
                                'y_coord': y_bike_utm,
                                'detection_distance': detection_distance,
                                'observer_speed': traci.vehicle.getSpeed(vehicle_id),
                                'bicycle_speed': traci.vehicle.getSpeed(bicycle_id),
                                'theoretical_rays': theoretical_rays,
                                'actual_rays': actual_rays,
                                'occlusion_level': observer_occlusion,
                                'static_blocked_rays': static_blocked_rays,
                                'dynamic_blocked_rays': dynamic_blocked_rays
                            })
    
    # Extend list only if we have actual detections
    if detection_entries:
        detection_logs_list.extend(detection_entries)

def collect_vehicle_trajectories(time_step):
    """
    Collects trajectory data for all motorized vehicles (including parked vehicles) at each simulation time step.
    Only logs data when vehicles are present in the simulation.
    
    Note: time_step parameter is the frame counter from the main loop.
    We use traci.simulation.getTime() to get the actual SUMO simulation time.
    """
    if not COLLECT_VEHICLE_TRAJECTORIES:
        return  # Skip if disabled for performance (MAJOR SPEEDUP)
    
    global vehicle_trajectory_logs_list
    trajectory_entries = []
    has_vehicles = False
    
    # Get the actual SUMO simulation time (in seconds)
    current_sim_time = traci.simulation.getTime()
    
    # Get all vehicles currently in simulation
    for vehicle_id in traci.vehicle.getIDList():
        try:
            vehicle_type = traci.vehicle.getTypeID(vehicle_id)
            
            # Skip only bicycles
            if vehicle_type in ["bicycle", "DEFAULT_BIKETYPE", "floating_bike_observer"]:
                continue
                
            has_vehicles = True
            
            # Get position and convert to UTM
            x, y = traci.vehicle.getPosition(vehicle_id)
            # Get vehicle dimensions and angle for position correction
            vehicle_length = vehicle_attributes(vehicle_type)[2][1]
            vehicle_angle = traci.vehicle.getAngle(vehicle_id)
            # Convert from SUMO position (front bumper) to geometric center
            x_center, y_center = sumo_position_to_center(x, y, vehicle_length, vehicle_angle)
            x_utm, y_utm = convert_simulation_coordinates(x_center, y_center)
            
            # Get vehicle dynamics
            speed = traci.vehicle.getSpeed(vehicle_id)
            angle = traci.vehicle.getAngle(vehicle_id)
            acceleration = traci.vehicle.getAcceleration(vehicle_id)
            lateral_speed = traci.vehicle.getLateralSpeed(vehicle_id)
            slope = traci.vehicle.getSlope(vehicle_id)
            distance = traci.vehicle.getDistance(vehicle_id)
            
            # Get route and lane information
            route_id = traci.vehicle.getRouteID(vehicle_id)
            lane_id = traci.vehicle.getLaneID(vehicle_id)
            edge_id = traci.vehicle.getRoadID(vehicle_id)
            distance = traci.vehicle.getDistance(vehicle_id)
            lane_position = traci.vehicle.getLanePosition(vehicle_id)
            lane_index = traci.vehicle.getLaneIndex(vehicle_id)
            
            # Get leader and follower information (skip for parked vehicles)
            leader_id = ''
            leader_distance = None
            follower_id = ''
            follower_distance = None
            
            if not vehicle_type.startswith('parked'):
                leader = traci.vehicle.getLeader(vehicle_id)
                if leader:
                    leader_id = leader[0]
                    leader_distance = leader[1]
                
                follower = traci.vehicle.getFollower(vehicle_id)
                if follower:
                    follower_id = follower[0]
                    follower_distance = follower[1]
            
            # Traffic light interaction (skip for parked vehicles)
            next_tls_id = ''
            distance_to_tls = None
            if not vehicle_type.startswith('parked'):
                next_tls = traci.vehicle.getNextTLS(vehicle_id)
                if next_tls:
                    next_tls_id = next_tls[0][0]
                    distance_to_tls = next_tls[0][2]
            
            # Vehicle dimensions and type info
            length = traci.vehicle.getLength(vehicle_id)
            width = traci.vehicle.getWidth(vehicle_id)
            max_speed = traci.vehicle.getMaxSpeed(vehicle_id)
            
            # Skip if essential data is invalid
            if distance == -1073741824.0 or lane_position == -1073741824.0:
                continue
            
            trajectory_entry = {
                'time_step': current_sim_time,  # Use actual SUMO simulation time, not frame counter
                'vehicle_id': vehicle_id,
                'vehicle_type': vehicle_type,
                'x_coord': x_utm,
                'y_coord': y_utm,
                'speed': speed,
                'angle': angle,
                'acceleration': acceleration,
                'lateral_speed': lateral_speed,
                'slope': slope,
                'distance': distance,
                'route_id': route_id,
                'lane_id': lane_id,
                'edge_id': edge_id,
                'lane_position': lane_position,
                'lane_index': lane_index,
                'leader_id': leader_id,
                'leader_distance': leader_distance,
                'follower_id': follower_id,
                'follower_distance': follower_distance,
                'next_tls_id': next_tls_id,
                'distance_to_tls': distance_to_tls,
                'length': length,
                'width': width,
                'max_speed': max_speed
            }
            
            trajectory_entries.append(trajectory_entry)
            
        except traci.exceptions.TraCIException as e:
            logging.warning(f"TraCI error for vehicle {vehicle_id}: {str(e)}")
        except Exception as e:
            logging.error(f"Error collecting trajectory data for vehicle {vehicle_id}: {str(e)}")
    
    # Extend list only if we have entries
    if trajectory_entries:
        vehicle_trajectory_logs_list.extend(trajectory_entries)

def collect_bicycle_trajectories(time_step):
    """
    Collects trajectory data for bicycles at each simulation time step.
    Only logs data when bicycles are present in the simulation.
    
    Note: time_step parameter is the frame counter from the main loop.
    We use traci.simulation.getTime() to get the actual SUMO simulation time.
    """
    if not COLLECT_BICYCLE_TRAJECTORIES:
        return  # Skip if disabled for performance
    
    global bicycle_trajectory_logs_list
    trajectory_entries = []
    has_bicycles = False
    
    # Get the actual SUMO simulation time (in seconds)
    # This is the absolute simulation time, not the frame counter
    current_sim_time = traci.simulation.getTime()
    
    # Get list of test polygons and their shapes
    test_polygons = []
    for poly_id in traci.polygon.getIDList():
        if traci.polygon.getType(poly_id) == "test":
            shape = traci.polygon.getShape(poly_id)
            # Convert shape to shapely polygon
            poly = Polygon(shape)
            test_polygons.append(poly)
    
    # Get all bicycles currently in simulation
    for vehicle_id in traci.vehicle.getIDList():
        try:
            vehicle_type = traci.vehicle.getTypeID(vehicle_id)
            
            # Only process bicycles
            if vehicle_type not in ["bicycle", "DEFAULT_BIKETYPE", "floating_bike_observer"]:
                continue
                
            has_bicycles = True
            
            # Inside the vehicle loop, add this debug for one specific bicycle
            if vehicle_id == "flow_10.0" and time_step < 10:  # Just check first few frames of one bicycle
                x_sumo, y_sumo = traci.vehicle.getPosition(vehicle_id)
                point = Point(x_sumo, y_sumo)
                print(f"Time step {time_step}, Bicycle {vehicle_id} at {point.wkt}")
                print(f"In test area: {any(poly.contains(point) for poly in test_polygons)}")

            # Get position in SUMO coordinates for test area check
            x_sumo, y_sumo = traci.vehicle.getPosition(vehicle_id)
            point = Point(x_sumo, y_sumo)
            
            # Check if position is in any test polygon
            in_test_area = any(poly.contains(point) for poly in test_polygons)
            
            # Get vehicle dimensions and angle for position correction
            bicycle_length = vehicle_attributes(vehicle_type)[2][1]
            bicycle_angle = traci.vehicle.getAngle(vehicle_id)
            # Convert from SUMO position (front bumper) to geometric center
            x_center, y_center = sumo_position_to_center(x_sumo, y_sumo, bicycle_length, bicycle_angle)
            # Convert coordinates to UTM for logging
            x_utm, y_utm = convert_simulation_coordinates(x_center, y_center)
            
            # Get bicycle dynamics
            speed = traci.vehicle.getSpeed(vehicle_id)
            angle = traci.vehicle.getAngle(vehicle_id)
            acceleration = traci.vehicle.getAcceleration(vehicle_id)
            lateral_speed = traci.vehicle.getLateralSpeed(vehicle_id)
            slope = traci.vehicle.getSlope(vehicle_id)
            distance = traci.vehicle.getDistance(vehicle_id)
            
            # Get route and lane information
            route_id = traci.vehicle.getRouteID(vehicle_id)
            lane_id = traci.vehicle.getLaneID(vehicle_id)
            edge_id = traci.vehicle.getRoadID(vehicle_id)
            lane_position = traci.vehicle.getLanePosition(vehicle_id)
            lane_index = traci.vehicle.getLaneIndex(vehicle_id)
            
            # Check detection status
            is_detected = False
            detecting_observers = []
            num_detecting_observers = 0
            if vehicle_id in bicycle_detection_data and bicycle_detection_data[vehicle_id]:
                latest_detection = bicycle_detection_data[vehicle_id][-1]
                if latest_detection[0] == traci.simulation.getTime() and latest_detection[1]:
                    is_detected = True
                    detecting_observers = latest_detection[2] if len(latest_detection) > 2 else []
                    num_detecting_observers = len(detecting_observers)
            
            # Skip if essential data is invalid
            if distance == -1073741824.0 or lane_position == -1073741824.0:
                continue
            
            # Get next traffic light information
            next_tl_id = ''
            next_tl_distance = -1
            next_tl_state = ''
            next_tl_index = -1
            
            try:
                next_tls = traci.vehicle.getNextTLS(vehicle_id)
                if next_tls:
                    # getNextTLS returns list of tuples: (tl_id, tl_index, distance, state)
                    tl_data = next_tls[0]
                    
                    next_tl_id = tl_data[0]          # traffic light ID
                    # Replace # character to avoid CSV parsing issues (pandas treats # as comment)
                    if '#' in next_tl_id:
                        next_tl_id = next_tl_id.replace('#', '_')
                    next_tl_index = tl_data[1]       # traffic light link index
                    next_tl_distance = tl_data[2]    # distance to traffic light
                    next_tl_state = tl_data[3]       # traffic light state
            except Exception as e:
                # If getNextTLS fails, keep default values
                logging.debug(f"Failed to get traffic light data for {vehicle_id}: {str(e)}")
                pass
            
            trajectory_entry = {
                'time_step': current_sim_time,  # Use actual SUMO simulation time, not frame counter
                'vehicle_id': vehicle_id,
                'vehicle_type': vehicle_type,
                'x_coord': x_utm,
                'y_coord': y_utm,
                'speed': speed,
                'angle': angle,
                'acceleration': acceleration,
                'lateral_speed': lateral_speed,
                'slope': slope,
                'distance': distance,
                'route_id': route_id,
                'lane_id': lane_id,
                'edge_id': edge_id,
                'lane_position': lane_position,
                'lane_index': lane_index,
                'is_detected': int(is_detected),
                'detecting_observers': ','.join([obs['id'] for obs in detecting_observers]) if detecting_observers else '-',
                'num_detecting_observers': num_detecting_observers,
                'in_test_area': int(in_test_area),
                'next_tl_id': next_tl_id,
                'next_tl_distance': next_tl_distance,
                'next_tl_state': next_tl_state,
                'next_tl_index': next_tl_index
            }
            
            trajectory_entries.append(trajectory_entry)
            
        except traci.exceptions.TraCIException as e:
            logging.warning(f"TraCI error for bicycle {vehicle_id}: {str(e)}")
        except Exception as e:
            logging.error(f"Error collecting trajectory data for bicycle {vehicle_id}: {str(e)}")
    
    # Extend list only if we have entries
    if trajectory_entries:
        bicycle_trajectory_logs_list.extend(trajectory_entries)

def collect_bicycle_conflict_data(frame):
    """
    Collects conflict data at each simulation time step using SUMO's SSM device.
    Note: SSM devices must be configured in SUMO configuration files before simulation start.
    Only records entries when actual conflicts occur.
    """
    if not COLLECT_CONFLICT_DATA:
        return  # Skip if disabled for performance
    
    global conflict_logs_list, ssm_device_errors, ssm_device_available
    
    current_time = traci.simulation.getTime()
    conflict_entries = []
    has_conflicts = False
    
    # Process each bicycle
    for vehicle_id in traci.vehicle.getIDList():
        vehicle_type = traci.vehicle.getTypeID(vehicle_id)
        
        
        if vehicle_type in ["bicycle", "DEFAULT_BIKETYPE", "floating_bike_observer"]:
            try:
                # Get bicycle position data
                distance = traci.vehicle.getDistance(vehicle_id)
                x, y = traci.vehicle.getPosition(vehicle_id)
                # Get vehicle dimensions and angle for position correction
                bicycle_length = vehicle_attributes(vehicle_type)[2][1]
                bicycle_angle = traci.vehicle.getAngle(vehicle_id)
                # Convert from SUMO position (front bumper) to geometric center
                x_center, y_center = sumo_position_to_center(x, y, bicycle_length, bicycle_angle)
                x_utm, y_utm = convert_simulation_coordinates(x_center, y_center)
                
                # Check both leader and follower vehicles
                leader = traci.vehicle.getLeader(vehicle_id)
                follower = traci.vehicle.getFollower(vehicle_id)
                
                potential_foes = []
                if leader and leader[0] != '':
                    potential_foes.append(('leader', *leader))
                if follower and follower[0] != '':
                    potential_foes.append(('follower', *follower))
                
                for position, foe_id, foe_distance in potential_foes:
                    # Check foe vehicle type
                    foe_type = traci.vehicle.getTypeID(foe_id)
                    
                    # Skip if foe is also a bicycle
                    if foe_type in ["bicycle", "DEFAULT_BIKETYPE", "floating_bike_observer"]:
                        continue
                    
                    # Get SSM values - check if SSM device is available
                    try:
                        ttc_str = traci.vehicle.getParameter(vehicle_id, "device.ssm.minTTC")
                        pet_str = traci.vehicle.getParameter(vehicle_id, "device.ssm.minPET")
                        drac_str = traci.vehicle.getParameter(vehicle_id, "device.ssm.maxDRAC")
                        
                        # Mark SSM device as available on first successful read
                        if ssm_device_available is None:
                            ssm_device_available = True
                        
                        # Convert to float with error handling
                        ttc = float(ttc_str) if ttc_str and ttc_str.strip() else float('inf')
                        pet = float(pet_str) if pet_str and pet_str.strip() else float('inf')
                        drac = float(drac_str) if drac_str and drac_str.strip() else 0.0
                    except traci.exceptions.TraCIException as ssm_error:
                        # SSM device still not available for this vehicle
                        if ssm_device_available is None:
                            ssm_device_available = False
                        ssm_device_errors.add(vehicle_id)
                        # Skip this conflict check since we can't get SSM data
                        continue
                    
                    # Define thresholds
                    TTC_THRESHOLD = 3.0  # seconds
                    PET_THRESHOLD = 2.0  # seconds
                    DRAC_THRESHOLD = 3.0  # m/s²
                    
                    # Check for conflict using thresholds
                    if (ttc < TTC_THRESHOLD or pet < PET_THRESHOLD or drac > DRAC_THRESHOLD):
                        # Calculate severity
                        ttc_severity = 1 - (ttc / TTC_THRESHOLD) if ttc < TTC_THRESHOLD else 0
                        pet_severity = 1 - (pet / PET_THRESHOLD) if pet < PET_THRESHOLD else 0
                        drac_severity = min(drac / DRAC_THRESHOLD, 1.0) if drac > 0 else 0
                        
                        conflict_severity = max(ttc_severity, pet_severity, drac_severity)
                        
                        # Check if bicycle is detected using bicycle_detection_data
                        is_detected = False
                        detecting_observers = []
                        
                        if vehicle_id in bicycle_detection_data and bicycle_detection_data[vehicle_id]:
                            latest_detection = bicycle_detection_data[vehicle_id][-1]
                            # Check if detection time matches current time (within tolerance) and bicycle is detected
                            if abs(latest_detection[0] - current_time) < 0.01 and latest_detection[1]:
                                is_detected = True
                                # Get observers from the detection data (format: (time, is_detected, [{'id': observer_id}]))
                                if len(latest_detection) > 2 and latest_detection[2]:
                                    # Extract observer info from detection data
                                    for obs_info in latest_detection[2]:
                                        obs_id = obs_info.get('id')
                                        if obs_id:
                                            try:
                                                obs_type = traci.vehicle.getTypeID(obs_id)
                                                detecting_observers.append({
                                                    'id': obs_id,
                                                    'type': obs_type
                                                })
                                            except traci.exceptions.TraCIException:
                                                # Observer may have left simulation
                                                pass
                        
                        has_conflicts = True
                        conflict_entries.append({
                            'time_step': current_time,
                            'bicycle_id': vehicle_id,
                            'foe_id': foe_id,
                            'foe_type': foe_type,
                            'x_coord': x_utm,
                            'y_coord': y_utm,
                            'distance': distance,
                            'ttc': ttc,
                            'pet': pet,
                            'drac': drac,
                            'severity': conflict_severity,
                            'is_detected': int(is_detected),
                            'detecting_observer': ','.join([obs['id'] for obs in detecting_observers]) if detecting_observers else '',
                            'observer_type': ','.join([obs['type'] for obs in detecting_observers]) if detecting_observers else ''
                        })
                        
            except traci.exceptions.TraCIException as e:
                # Handle other TraCI errors (not SSM-related)
                if "device.ssm" not in str(e):
                    logging.warning(f"TraCI error in conflict detection for {vehicle_id}: {str(e)}")
            except Exception as e:
                logging.error(f"Error in conflict detection for {vehicle_id}: {str(e)}")
    
    # Extend list only if we have actual conflicts
    if conflict_entries:
        conflict_logs_list.extend(conflict_entries)

def collect_performance_data(frame, step_start_time):
    """Collect performance metrics for the current simulation step."""
    step_duration = time.time() - step_start_time
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024  # Convert to MB
    
    # Add to performance stats DataFrame
    performance_stats.loc[len(performance_stats)] = {
        'time_step': frame,
        'step_duration': step_duration,
        'memory_usage': memory_mb
    }

def print_ssm_device_summary():
    """
    Prints a summary of SSM device status at the end of simulation.
    Only prints if SSM devices were not available.
    """
    global ssm_device_errors, ssm_device_available
    
    if ssm_device_available == False and len(ssm_device_errors) > 0:
        print("\n" + "="*60)
        print("SSM DEVICE STATUS")
        print("="*60)
        print("⚠️  SSM (Surrogate Safety Measures) devices not configured")
        print(f"   Affected vehicles: {len(ssm_device_errors)} bicycle(s)")
        print(f"   Example vehicles: {', '.join(list(ssm_device_errors)[:5])}")
        if len(ssm_device_errors) > 5:
            print(f"   ... and {len(ssm_device_errors) - 5} more")
        print("\nNote: To enable conflict detection, add SSM devices to your")
        print("      SUMO configuration file. See SUMO documentation for details:")
        print("      https://sumo.dlr.de/docs/Simulation/Output/SSM_Device.html")
        print("="*60)
    elif ssm_device_available == True:
        # SSM devices are working, no need to print anything
        pass

def save_simulation_logs():
    """
    Saves all collected simulation data to log files.
    Generates and saves both detailed log files (with data for each time step) and a summary log file.
    """
    global fleet_composition_logs, traffic_light_logs, detection_logs
    global vehicle_trajectory_logs, bicycle_trajectory_logs, conflict_logs
    
    print("\n" + "="*80)
    print("CONVERTING COLLECTED DATA TO DATAFRAMES...")
    print("="*80)
    
    # Convert lists to DataFrames (single operation - very fast)
    if COLLECT_FLEET_COMPOSITION and fleet_composition_logs_list:
        fleet_composition_logs = pd.DataFrame(fleet_composition_logs_list)
        print(f"✓ Fleet composition: {len(fleet_composition_logs):,} rows")
    else:
        fleet_composition_logs = pd.DataFrame()
        print(f"  Fleet composition: DISABLED")
    
    if COLLECT_TRAFFIC_LIGHT_DATA and traffic_light_logs_list:
        traffic_light_logs = pd.DataFrame(traffic_light_logs_list)
        print(f"✓ Traffic lights: {len(traffic_light_logs):,} rows")
    else:
        traffic_light_logs = pd.DataFrame()
        print(f"  Traffic lights: DISABLED")
    
    if COLLECT_DETECTION_LOGS and detection_logs_list:
        detection_logs = pd.DataFrame(detection_logs_list)
        print(f"✓ Detection logs: {len(detection_logs):,} rows")
    else:
        detection_logs = pd.DataFrame()
        print(f"  Detection logs: DISABLED")
    
    if COLLECT_VEHICLE_TRAJECTORIES and vehicle_trajectory_logs_list:
        vehicle_trajectory_logs = pd.DataFrame(vehicle_trajectory_logs_list)
        print(f"✓ Vehicle trajectories: {len(vehicle_trajectory_logs):,} rows")
    else:
        vehicle_trajectory_logs = pd.DataFrame()
        print(f"  Vehicle trajectories: DISABLED (performance optimization)")
    
    if COLLECT_BICYCLE_TRAJECTORIES and bicycle_trajectory_logs_list:
        bicycle_trajectory_logs = pd.DataFrame(bicycle_trajectory_logs_list)
        print(f"✓ Bicycle trajectories: {len(bicycle_trajectory_logs):,} rows")
    else:
        bicycle_trajectory_logs = pd.DataFrame()
        print(f"  Bicycle trajectories: DISABLED")
    
    if COLLECT_CONFLICT_DATA and conflict_logs_list:
        conflict_logs = pd.DataFrame(conflict_logs_list)
        print(f"✓ Conflict logs: {len(conflict_logs):,} rows")
    else:
        conflict_logs = pd.DataFrame()
        print(f"  Conflict logs: DISABLED")
    
    print("="*80 + "\n")

    # Detailed logging -----------------------------------------------------------------------------------------

    # Fleet composition data
    with open(os.path.join(scenario_output_dir, 'out_logging', f'log_fleet_composition_{file_tag}_FCO{FCO_share*100:.0f}%_FBO{FBO_share*100:.0f}%.csv'), 'w', newline='') as f:
        # First order header
        f.write('# =========================================\n')
        f.write('# Summary of Simulation Results (Fleet Composition)\n')
        f.write(f'# Generated on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'# Step length: {step_length} seconds\n')
        f.write('# =========================================\n')
        f.write('#\n')
        # Second order header
        f.write('# -----------------------------------------\n')
        f.write('# Units explanation:\n')
        f.write('# -----------------------------------------\n')
        f.write('# time_step: current simulation time step\n')
        f.write('# new_*_count: number of new vehicles of this type entering in this time step\n')
        f.write('# present_*_count: total number of vehicles of this type present in this time step\n')
        f.write('# -----------------------------------------\n')
        f.write('\n')
        fleet_composition_logs.to_csv(f, index=False)

    # Traffic light data
    with open(os.path.join(scenario_output_dir, 'out_logging', f'log_traffic_lights_{file_tag}_FCO{FCO_share*100:.0f}%_FBO{FBO_share*100:.0f}%.csv'), 'w', newline='') as f:
        f.write('# -----------------------------------------\n')
        f.write('# Units explanation:\n')
        f.write('# -----------------------------------------\n')
        f.write('# time_step: Simulation time step\n')
        f.write('# phase_duration: Duration of current traffic light phase (seconds)\n')
        f.write('# remaining_duration: Time until next phase change (seconds)\n')
        f.write('# signal_states: Current state of all signals (g=green, y=yellow, r=red, G=priority green)\n')
        f.write('# total_queue_length: Number of stopped vehicles at intersection (vehicles)\n')
        f.write('# vehicles_stopped: Number of unique vehicles stopped at intersection (vehicles)\n')
        f.write('# average_waiting_time: Average time vehicles have been waiting (seconds)\n')
        f.write('# vehicles_by_type: Dictionary of vehicle counts by vehicle type\n')
        f.write('# lane_to_signal_mapping: Dictionary mapping lanes to their controlling signals\n')
        f.write('# -----------------------------------------\n')
        f.write('\n')
        traffic_light_logs.to_csv(f, index=False)

    # Detection data
    with open(os.path.join(scenario_output_dir, 'out_logging', f'log_detections_{file_tag}_FCO{FCO_share*100:.0f}%_FBO{FBO_share*100:.0f}%.csv'), 'w', newline='') as f:
        # First order header
        f.write('# =========================================\n')
        f.write('# Summary of Simulation Results (Bicycle Detections)\n')
        f.write(f'# Generated on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'# Step length: {step_length} seconds\n')
        f.write('# =========================================\n')
        f.write('#\n')
        # Second order header
        f.write('# -----------------------------------------\n')
        f.write('# Units explanation:\n')
        f.write('# -----------------------------------------\n')
        f.write('# time_step: current simulation time step\n')
        f.write('# x_coord, y_coord: UTM coordinates in meters (EPSG:32632)\n')
        f.write('# detection_distance: meters\n')
        f.write('# observer_speed, bicycle_speed: meters per second (m/s)\n')
        f.write('# theoretical_rays: rays that would hit bicycle without occlusion\n')
        f.write('# actual_rays: rays that hit bicycle after occlusion\n')
        f.write('# occlusion_level: percentage of bicycle surface occluded (0-100%)\n')
        f.write('# static_blocked_rays: rays blocked by static objects (buildings, trees, barriers, etc.)\n')
        f.write('# dynamic_blocked_rays: rays blocked by dynamic objects (vehicles)\n')
        f.write('# -----------------------------------------\n')
        f.write('\n')
        detection_logs.to_csv(f, index=False)

    # Vehicle trajectory data
    with open(os.path.join(scenario_output_dir, 'out_logging', f'log_vehicle_trajectories_{file_tag}_FCO{FCO_share*100:.0f}%_FBO{FBO_share*100:.0f}%.csv'), 'w', newline='') as f:
        # First order header
        f.write('# =========================================\n')
        f.write('# Summary of Simulation Results (Vehicle Trajectories)\n')
        f.write(f'# Generated on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'# Step length: {step_length} seconds\n')
        f.write('# =========================================\n')
        f.write('#\n')
        # Second order header
        f.write('# -----------------------------------------\n')
        f.write('# Units explanation:\n')
        f.write('# -----------------------------------------\n')
        f.write('# time_step: current simulation time step\n')
        f.write('# x_coord, y_coord: UTM coordinates in meters (EPSG:32632)\n')
        f.write('# speed: meters per second (m/s)\n')
        f.write('# angle: degrees (0-360, clockwise from north)\n')
        f.write('# acceleration: meters per second squared (m/s^2)\n')
        f.write('# lateral_speed: meters per second (m/s)\n')
        f.write('# slope: road gradient in degrees\n')
        f.write('# distance: cumulative distance traveled in meters\n')
        f.write('# lane_position: ...\n')
        f.write('# leader/follower_distance: meters to leader/follower vehicle\n')
        f.write('# distance_to_tls: meters to next traffic light\n')
        f.write('# length, width: meters\n')
        f.write('# -----------------------------------------\n')
        f.write('\n')
        vehicle_trajectory_logs.to_csv(f, index=False)

    # Bicycle trajectory data
    with open(os.path.join(scenario_output_dir, 'out_logging', f'log_bicycle_trajectories_{file_tag}_FCO{FCO_share*100:.0f}%_FBO{FBO_share*100:.0f}%.csv'), 'w', newline='') as f:
        # First order header
        f.write('# =========================================\n')
        f.write('# Summary of Simulation Results (Bicycle Trajectories)\n')
        f.write(f'# Generated on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'# Step length: {step_length} seconds\n')
        f.write('# =========================================\n')
        f.write('#\n')
        # Second order header
        f.write('# -----------------------------------------\n')
        f.write('# Units explanation:\n')
        f.write('# -----------------------------------------\n')
        f.write('# time_step: current simulation time step\n')
        f.write('# x_coord, y_coord: UTM coordinates in meters (EPSG:32632)\n')
        f.write('# speed: meters per second (m/s)\n')
        f.write('# angle: degrees (0-360, clockwise from north)\n')
        f.write('# acceleration: meters per second squared (m/s^2)\n')
        f.write('# lateral_speed: meters per  second (m/s)\n')
        f.write('# slope: road gradient in degrees\n')
        f.write('# distance: cumulative distance traveled in meters\n')
        f.write('# lane_position: ...\n')
        f.write('# is_detected: if bicycle is detected by any observer (0: no, 1: yes)\n')
        f.write('# detecting_observers: list of observer IDs that detected the bicycle\n')
        f.write('# num_detecting_observers: number of observers detecting the bicycle\n')
        f.write('# in_test_area: if bicycle is in test area (0: no, 1: yes)\n')
        f.write('# next_tl_id: ID of the next traffic light on the route\n')
        f.write('# next_tl_distance: distance to the next traffic light in meters\n')
        f.write('# next_tl_state: current state of the relevant signal (r/y/g/G)\n')
        f.write('# next_tl_index: index of the relevant signal within the traffic light\n')
        f.write('# -----------------------------------------\n')
        f.write('\n')
        bicycle_trajectory_logs.to_csv(f, index=False)

    # Conflict data
    with open(os.path.join(scenario_output_dir, 'out_logging', f'log_conflicts_{file_tag}_FCO{FCO_share*100:.0f}%_FBO{FBO_share*100:.0f}%.csv'), 'w', newline='') as f:
        # First order header
        f.write('# =========================================\n')
        f.write('# Summary of Simulation Results (Bicycle Conflicts)\n')
        f.write(f'# Generated on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'# Step length: {step_length} seconds\n')
        f.write('# =========================================\n')
        f.write('#\n')
        # Second order header
        f.write('# -----------------------------------------\n')
        f.write('# Units explanation:\n')
        f.write('# -----------------------------------------\n')
        f.write('# time_step: current simulation time step\n')
        f.write('# x_coord, y_coord: UTM coordinates in meters (EPSG:32632)\n')
        f.write('# distance: cumulative distance traveled in meters\n')
        f.write('# ttc: Time-To-Collision in seconds\n')
        f.write('# pet: Post-Encroachment-Time in seconds\n')
        f.write('# drac: Deceleration Rate to Avoid Crash in m/s^2\n')
        f.write('# severity: calculated conflict severity (0-1)\n')
        f.write('# is_detected: if bicycle / conflict is detected by any observer (0: no, 1: yes)\n')
        f.write('# detecting_observer: list of observer IDs that detected the bicycle / conflict\n')
        f.write('# -----------------------------------------\n')
        f.write('\n')
        conflict_logs.to_csv(f, index=False)

    # add further detailed logging here

    # ----------------------------------------------------------------------------------------------------------

    # Statistics calculations ----------------------------------------------------------------------------------

    # Traffic light statistics ----------------------------------------------
    non_zero_queues = traffic_light_logs['total_queue_length'][traffic_light_logs['total_queue_length'] > 0]
    tl_stats = {
        'total_traffic_lights': len(traffic_light_logs['traffic_light_id'].unique()),
        'avg_queue_length': non_zero_queues.mean() if len(non_zero_queues) > 0 else 0,
        'max_queue_length': traffic_light_logs['total_queue_length'].max(),
        'min_queue_length': non_zero_queues.min() if len(non_zero_queues) > 0 else 0,
        'avg_waiting_time': traffic_light_logs['average_waiting_time'].mean(),
        'max_waiting_time': traffic_light_logs['average_waiting_time'].max(),
        'min_waiting_time': traffic_light_logs['average_waiting_time'][traffic_light_logs['average_waiting_time'] > 0].min() if any(traffic_light_logs['average_waiting_time'] > 0) else 0
    }
    # -----------------------------------------------------------------------

    # Observer penetration rates --------------------------------------------
    total_vehicle_counts = {vehicle_type: 0 for vehicle_type in vehicle_type_set}
    # Define relevant vehicle types for cars and bikes
    relevant_car_types = {"floating_car_observer", "DEFAULT_VEHTYPE", "pt_bus", "passenger"}
    relevant_bike_types = {"floating_bike_observer", "DEFAULT_BIKETYPE"}
    # Initialize counters for relevant vehicles
    total_relevant_cars = 0
    total_relevant_bikes = 0
    total_floating_car_observers = 0
    total_floating_bike_observers = 0
    # Calculate total counts for each vehicle type
    for vehicle_type in vehicle_type_set:
        total_vehicle_counts[vehicle_type] = fleet_composition_logs[f'new_{vehicle_type}_count'].sum()
        # Sum up relevant car and bike counts
        if vehicle_type in relevant_car_types:
            total_relevant_cars += total_vehicle_counts[vehicle_type]
            if vehicle_type == "floating_car_observer":
                total_floating_car_observers += total_vehicle_counts[vehicle_type]
        if vehicle_type in relevant_bike_types:
            total_relevant_bikes += total_vehicle_counts[vehicle_type]
            if vehicle_type == "floating_bike_observer":
                total_floating_bike_observers += total_vehicle_counts[vehicle_type]
    # Calculate observer penetration rates
    fco_penetration_rate = total_floating_car_observers / total_relevant_cars if total_relevant_cars > 0 else 0
    fbo_penetration_rate = total_floating_bike_observers / total_relevant_bikes if total_relevant_bikes > 0 else 0
    # -----------------------------------------------------------------------

    # Detection statistics --------------------------------------------------
    detection_stats = {}
    if not detection_logs.empty:
        # Basic statistics
        detection_stats['total_detections'] = len(detection_logs)
        detection_stats['avg_detection_distance'] = detection_logs['detection_distance'].mean()
        # Multiple detections analysis
        bicycle_detection_counts = detection_logs['bicycle_id'].value_counts()
        detection_stats['bicycles_detected'] = len(bicycle_detection_counts)
        detection_stats['bicycles_multiple_detections'] = len(bicycle_detection_counts[bicycle_detection_counts > 1])
        # Detection quality analysis
        bicycle_detection_durations = {}
        for bicycle_id in bicycle_detection_counts.index:
            detections = detection_logs[detection_logs['bicycle_id'] == bicycle_id]
            total_time = len(detections) * step_length
            bicycle_detection_durations[bicycle_id] = total_time
        detection_stats['avg_detection_duration'] = np.mean(list(bicycle_detection_durations.values()))
        # Blind spots analysis
        all_bicycles = set(bicycle_trajectory_logs['vehicle_id'].unique())
        detected_bicycles = set(detection_logs['bicycle_id'].unique())
        never_detected = all_bicycles - detected_bicycles
        detection_stats['never_detected_count'] = len(never_detected)
        detection_stats['never_detected_percentage'] = (len(never_detected) / len(all_bicycles)) * 100 if all_bicycles else 0
    # -----------------------------------------------------------------------

    # Vehicle trajectory statistics -----------------------------------------
    vehicle_trajectory_stats = {}
    if not vehicle_trajectory_logs.empty:
        # Define all bicycle types to exclude
        bicycle_types = {'DEFAULT_BIKETYPE', 'floating_bike_observer', 'bicycle'}
        
        # Get all vehicle types excluding bicycles and parked vehicles
        motorized_types = {vtype for vtype in vehicle_trajectory_logs['vehicle_type'].unique() 
                         if vtype not in bicycle_types and 'parked' not in vtype.lower()}
        
        for vehicle_type in motorized_types:
            # Filter for current vehicle type and exclude stopped vehicles
            active_vehicles = vehicle_trajectory_logs[
                (vehicle_trajectory_logs['vehicle_type'] == vehicle_type) &
                (vehicle_trajectory_logs['speed'] > 0.1)
            ]
            
            if not active_vehicles.empty:
                active_vehicles = active_vehicles.copy()  # Create an explicit copy
                active_vehicles.loc[:, 'time_diff'] = active_vehicles.groupby('vehicle_id')['time_step'].diff()
                active_vehicles.loc[:, 'acceleration_diff'] = active_vehicles.groupby('vehicle_id')['speed'].diff()

                # Only consider acceleration changes within the same vehicle and consecutive timesteps
                valid_acc = active_vehicles[
                    (active_vehicles['time_diff'] == 1) &  # Consecutive timesteps
                    (active_vehicles['vehicle_id'] == active_vehicles['vehicle_id'].shift())  # Same vehicle
                ]['acceleration_diff'] / step_length

                pos_acc = valid_acc[valid_acc > 0]
                neg_acc = valid_acc[valid_acc < 0]
                
                stats = {
                    'unique_vehicles': active_vehicles['vehicle_id'].nunique(),
                    # Time in simulation
                    'time_in_sim': {
                        'mean': (active_vehicles.groupby('vehicle_id')['time_step'].agg(['min', 'max']).diff(axis=1)['max'] * step_length).mean(),
                        'max': (active_vehicles.groupby('vehicle_id')['time_step'].agg(['min', 'max']).diff(axis=1)['max'] * step_length).max(),
                        'min': (active_vehicles.groupby('vehicle_id')['time_step'].agg(['min', 'max']).diff(axis=1)['max'] * step_length).min()
                    },
                    # Distance traveled
                    'distance': {
                        'mean': active_vehicles.groupby('vehicle_id')['distance'].max().mean(),
                        'max': active_vehicles.groupby('vehicle_id')['distance'].max().max(),
                        'min': active_vehicles.groupby('vehicle_id')['distance'].max().min()
                    },
                    # Speed statistics
                    'speed': {
                        'mean': active_vehicles['speed'].mean(),
                        'max': active_vehicles['speed'].max(),
                        'min': active_vehicles[active_vehicles['speed'] > 0.1]['speed'].min(),
                        'std': active_vehicles['speed'].std(),
                        'percentile_15': active_vehicles['speed'].quantile(0.15),
                        'median': active_vehicles['speed'].median(),
                        'percentile_85': active_vehicles['speed'].quantile(0.85)
                    }
                }
                
                stats['acceleration'] = {
                    'mean': pos_acc.mean() if not pos_acc.empty else 0,
                    'max': pos_acc.max() if not pos_acc.empty else 0,
                    'min': pos_acc.min() if not pos_acc.empty else 0
                }
                stats['deceleration'] = {
                    'mean': abs(neg_acc.mean()) if not neg_acc.empty else 0,
                    'max': abs(neg_acc.min()) if not neg_acc.empty else 0,
                    'min': abs(neg_acc.max()) if not neg_acc.empty else 0,
                    'hard_braking_events': (abs(neg_acc) > 4.5).sum()
                }
                
                # Following distance statistics
                following_distances = active_vehicles[active_vehicles['leader_distance'] >= 0]['leader_distance']
                if not following_distances.empty:
                    stats['following_distance'] = {
                        'mean': following_distances.mean(),
                        'max': following_distances.max(),
                        'min': following_distances.min()
                    }
                vehicle_trajectory_stats[vehicle_type] = stats
    # -----------------------------------------------------------------------

    # Bicycle trajectory statistics -----------------------------------------
    bicycle_stats = {}
    if not bicycle_trajectory_logs.empty:
        bicycle_types = {'DEFAULT_BIKETYPE', 'floating_bike_observer', 'bicycle'}
        
        for bicycle_type in bicycle_types:
            # Filter for current bicycle type and exclude stopped bicycles
            active_bicycles = bicycle_trajectory_logs[
                (bicycle_trajectory_logs['vehicle_type'] == bicycle_type) &
                (bicycle_trajectory_logs['speed'] > 0.1)
            ]
            
            if not active_bicycles.empty:
                active_bicycles = active_bicycles.copy()  # Create an explicit copy
                active_bicycles.loc[:, 'time_diff'] = active_bicycles.groupby('vehicle_id')['time_step'].diff()
                active_bicycles.loc[:, 'acceleration_diff'] = active_bicycles.groupby('vehicle_id')['speed'].diff()

                # Only consider acceleration changes within the same bicycle and consecutive timesteps
                valid_acc = active_bicycles[
                    (active_bicycles['time_diff'] == 1) &  # Consecutive timesteps
                    (active_bicycles['vehicle_id'] == active_bicycles['vehicle_id'].shift())  # Same bicycle
                ]['acceleration_diff'] / step_length

                pos_acc = valid_acc[valid_acc > 0]
                neg_acc = valid_acc[valid_acc < 0]
                
                stats = {
                    'unique_bicycles': active_bicycles['vehicle_id'].nunique(),
                    # Time in simulation
                    'time_in_sim': {
                        'mean': (active_bicycles.groupby('vehicle_id')['time_step'].agg(['min', 'max']).diff(axis=1)['max'] * step_length).mean(),
                        'max': (active_bicycles.groupby('vehicle_id')['time_step'].agg(['min', 'max']).diff(axis=1)['max'] * step_length).max(),
                        'min': (active_bicycles.groupby('vehicle_id')['time_step'].agg(['min', 'max']).diff(axis=1)['max'] * step_length).min()
                    },
                    # Distance traveled
                    'distance': {
                        'mean': active_bicycles.groupby('vehicle_id')['distance'].max().mean(),
                        'max': active_bicycles.groupby('vehicle_id')['distance'].max().max(),
                        'min': active_bicycles.groupby('vehicle_id')['distance'].max().min()
                    },
                    # Speed statistics
                    'speed': {
                        'mean': active_bicycles['speed'].mean(),
                        'max': active_bicycles['speed'].max(),
                        'min': active_bicycles[active_bicycles['speed'] > 0.1]['speed'].min(),
                        'std': active_bicycles['speed'].std(),
                        'percentile_15': active_bicycles['speed'].quantile(0.15),
                        'median': active_bicycles['speed'].median(),
                        'percentile_85': active_bicycles['speed'].quantile(0.85)
                    }
                }
                
                stats['acceleration'] = {
                    'mean': pos_acc.mean() if not pos_acc.empty else 0,
                    'max': pos_acc.max() if not pos_acc.empty else 0,
                    'min': pos_acc.min() if not pos_acc.empty else 0
                }
                stats['deceleration'] = {
                    'mean': abs(neg_acc.mean()) if not neg_acc.empty else 0,
                    'max': abs(neg_acc.min()) if not neg_acc.empty else 0,
                    'min': abs(neg_acc.max()) if not neg_acc.empty else 0,
                    'hard_braking_events': (abs(neg_acc) > 2.5).sum()
                }
                
                bicycle_stats[bicycle_type] = stats
    else:
        bicycle_stats = None
    # -----------------------------------------------------------------------

    # Conflict statistics ---------------------------------------------------
    conflict_stats = {}
    if not conflict_logs.empty:
        # Basic conflict statistics
        conflict_stats['conflict_frames'] = len(conflict_logs)
        conflict_stats['unique_bicycles'] = len(conflict_logs['bicycle_id'].unique())
        # Count unique conflicts
        conflict_groups = conflict_logs.groupby(['bicycle_id', 'foe_id'])
        unique_conflicts = 0
        unique_conflict_frames = []
        for name, group in conflict_groups:
            group = group.sort_values('time_step')
            time_diffs = group['time_step'].diff()
            # Handle pandas Series comparison - fixed version
            try:
                # Use .iloc to get numeric values and handle NaN properly
                time_diff_values = time_diffs.iloc[1:].values  # Skip first NaN value
                conflict_starts = [True] + [bool(diff > 1) for diff in time_diff_values if not pd.isna(diff)]
            except Exception:
                # Fallback for any pandas/type issues
                conflict_starts = [True] * len(time_diffs)  # Conservative fallback
            group_conflicts = sum(conflict_starts)
            unique_conflicts += group_conflicts
            # Store frames for each unique conflict
            current_frames = []
            for idx, row in group.iterrows():
                current_frames.append(row)
                if len(current_frames) > 1 and (row['time_step'] - current_frames[-2]['time_step'] > 1):
                    if len(current_frames) > 1:
                        unique_conflict_frames.append(current_frames[:-1])
                    current_frames = [row]
            if current_frames:
                unique_conflict_frames.append(current_frames)
        conflict_stats['unique_conflicts'] = unique_conflicts
        conflict_stats['unique_conflict_frames'] = unique_conflict_frames
        # Per bicycle statistics
        conflict_stats['conflicts_per_bicycle'] = unique_conflicts / conflict_stats['unique_bicycles']
        conflict_stats['conflict_frames_per_bicycle'] = conflict_stats['conflict_frames'] / conflict_stats['unique_bicycles']
        conflict_stats['max_conflict_frames'] = conflict_logs['bicycle_id'].value_counts().max()
        # Calculate maximum conflicts per bicycle
        conflicts_by_bicycle = {}
        conflict_frames_by_bicycle = {}
        for frames in unique_conflict_frames:
            bicycle_id = frames[0]['bicycle_id']
            conflicts_by_bicycle[bicycle_id] = conflicts_by_bicycle.get(bicycle_id, 0) + 1
        for bicycle_id in conflict_logs['bicycle_id'].unique():
            conflict_frames_by_bicycle[bicycle_id] = len(conflict_logs[conflict_logs['bicycle_id'] == bicycle_id])
        conflict_stats['max_conflicts_per_bicycle'] = max(conflicts_by_bicycle.values())
        conflict_stats['min_conflicts_per_bicycle'] = min(conflicts_by_bicycle.values())
        conflict_stats['max_conflict_frames'] = max(conflict_frames_by_bicycle.values())
        conflict_stats['min_conflict_frames'] = min(conflict_frames_by_bicycle.values())
        # Conflict durations
        conflict_durations = []
        for frames in unique_conflict_frames:
            duration = (len(frames) - 1) * step_length
            if duration > 0:
                conflict_durations.append(duration)
        conflict_stats['avg_duration'] = np.mean(conflict_durations) if conflict_durations else 0
        conflict_stats['min_duration'] = np.min(conflict_durations) if conflict_durations else 0
        conflict_stats['max_duration'] = np.max(conflict_durations) if conflict_durations else 0
        conflict_stats['prolonged_conflicts'] = sum(1 for d in conflict_durations if d > 3.0)
        # Conflict types
        conflict_stats['ttc_conflicts'] = sum(1 for frames in unique_conflict_frames 
                                            if any(frame['ttc'] < 3.0 for frame in frames))
        conflict_stats['pet_conflicts'] = sum(1 for frames in unique_conflict_frames 
                                            if any(frame['pet'] < 2.0 for frame in frames))
        conflict_stats['drac_conflicts'] = sum(1 for frames in unique_conflict_frames 
                                             if any(frame['drac'] > 3.0 for frame in frames))
        # Severity metrics
        ttc_filtered = conflict_logs[conflict_logs['ttc'] < 3.0]['ttc']
        pet_filtered = conflict_logs[conflict_logs['pet'] < 2.0]['pet']
        drac_filtered = conflict_logs[conflict_logs['drac'] > 3.0]['drac']
        conflict_stats['ttc_stats'] = ttc_filtered.agg(['mean', 'min', 'max']).to_dict() if not ttc_filtered.empty else {'mean': 0, 'min': 0, 'max': 0}
        conflict_stats['pet_stats'] = pet_filtered.agg(['mean', 'min', 'max']).to_dict() if not pet_filtered.empty else {'mean': 0, 'min': 0, 'max': 0}
        conflict_stats['drac_stats'] = drac_filtered.agg(['mean', 'min', 'max']).to_dict() if not drac_filtered.empty else {'mean': 0, 'min': 0, 'max': 0}
        # Conflict partners
        conflicts_by_partner = {}
        for frames in unique_conflict_frames:
            foe_type = frames[0]['foe_type']
            conflicts_by_partner[foe_type] = conflicts_by_partner.get(foe_type, 0) + 1
        conflict_stats['conflicts_by_partner'] = conflicts_by_partner
        # Detection statistics
        conflict_stats['detected_conflicts'] = sum(1 for frames in unique_conflict_frames 
                                                 if any(frame['is_detected'] for frame in frames))
        conflict_stats['fco_detected'] = sum(1 for frames in unique_conflict_frames 
                                           if any(frame['observer_type'] and 
                                                 'floating_car_observer' in str(frame['observer_type']) 
                                                 for frame in frames))
        conflict_stats['fbo_detected'] = sum(1 for frames in unique_conflict_frames 
                                           if any(frame['observer_type'] and 
                                                 'floating_bike_observer' in str(frame['observer_type']) 
                                                 for frame in frames))
    # -----------------------------------------------------------------------

    # Performance statistics --------------------------------------------------
    perf_stats = {}
    if not performance_stats.empty:
        total_time = sum(operation_times.values())
        perf_stats['total_runtime'] = total_time
        perf_stats['step_timing'] = {
            'mean': performance_stats['step_duration'].mean(),
            'max': performance_stats['step_duration'].max(),
            'min': performance_stats['step_duration'].min()
        }
        perf_stats['memory_usage'] = {
            'mean': performance_stats['memory_usage'].mean(),
            'max': performance_stats['memory_usage'].max()
        }
        # Add operation timings from TimingContext
        perf_stats['operation_timing'] = {
            name: total_time for name, total_time in operation_times.items()
        }
        total_runtime = perf_stats['total_runtime']
        setup_time = operation_times['simulation_setup']
        data_collection_time = operation_times['data_collection']
        logging_time = operation_times['logging']
        ray_tracing_time = operation_times['ray_tracing']
        visualization_time = operation_times['visualization']
        application_time = operation_times.get('flow_trajectories', 0) + operation_times.get('important_trajectories', 0) + operation_times.get('visibility_data_export', 0)
        if 'visualization' in operation_times and visualization_time > 0:
            component_sum = setup_time + visualization_time + data_collection_time + logging_time + application_time
        else:
            component_sum = setup_time + ray_tracing_time + data_collection_time + logging_time + application_time
        timing_offset = total_runtime - component_sum
    # -----------------------------------------------------------------------

    # ----------------------------------------------------------------------------------------------------------

    # Summary logging ------------------------------------------------------------------------------------------
    
    with open(os.path.join(scenario_output_dir, 'out_logging', f'summary_log_{file_tag}_FCO{str(FCO_share*100)}%_FBO{str(FBO_share*100)}%.csv'), mode='w', newline='') as f:
        writer = csv.writer(f)
        
        # First order header
        f.write('# =========================================\n')
        f.write('# Summary of Simulation Results\n')
        f.write(f'# Generated on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'# Step length: {step_length} seconds\n')
        f.write('# =========================================\n')
        f.write('\n')
        
        # Simulation parameters section
        writer.writerow(['========================================='])
        writer.writerow(['1. SIMULATION PARAMETERS'])
        writer.writerow(['-----------------------------------------'])
        writer.writerow(['Parameter', 'Value'])
        writer.writerow([])
        writer.writerow(['Total simulation steps', total_steps])
        writer.writerow(['Step length', step_length])
        writer.writerow(['Simulation duration', f'{total_steps * step_length:.1f} seconds'])
        writer.writerow(['Warm-up period', f'{delay} seconds'])
        writer.writerow([])
        writer.writerow(['Live visualization', useLiveVisualization])
        writer.writerow(['Visualize rays', visualizeRays])
        writer.writerow(['Save animation', saveAnimation])
        writer.writerow([])
        writer.writerow(['FCO share (input)', f'{(FCO_share):.2%}'])
        writer.writerow(['FBO share (input)', f'{(FBO_share):.2%}'])
        writer.writerow([])
        writer.writerow(['Number of rays (Ray Tracing)', numberOfRays])
        writer.writerow(['Ray tracing radius', f'{radius} meters'])
        writer.writerow([])
        writer.writerow(['Grid size (Heat Map)', grid_size])
        writer.writerow([])
        writer.writerow(['Bounding box (north)', north])
        writer.writerow(['Bounding box (south)', south])
        writer.writerow(['Bounding box (east)', east])
        writer.writerow(['Bounding box (west)', west])
        writer.writerow([])
        writer.writerow(['SUMO configuration file', sumo_config_path])
        writer.writerow(['GeoJSON path', os.path.relpath(geojson_path, parent_dir) if geojson_path else 'None'])
        writer.writerow(['File tag', file_tag])
        writer.writerow([])
        writer.writerow(['========================================='])
        
        # Fleet composition section
        writer.writerow(['2. FLEET COMPOSITION'])
        writer.writerow(['-----------------------------------------'])
        writer.writerow(['Vehicle Type', 'Total Count', 'Average Present'])
        writer.writerow([])
        # Driving vehicles
        for vehicle_type in sorted(vtype for vtype in vehicle_type_set if 'parked' not in vtype.lower()):
            total = total_vehicle_counts[vehicle_type]
            if fleet_composition_logs is not None and not fleet_composition_logs.empty:
                avg_present = fleet_composition_logs[f'present_{vehicle_type}_count'].mean()
                writer.writerow([f'Total {vehicle_type} vehicles', total, f'{avg_present:.1f}'])
            else:
                writer.writerow([f'Total {vehicle_type} vehicles', total, 'N/A'])
        # Parked vehicles, only if there were any
        parked_vehicles = [vtype for vtype in vehicle_type_set if 'parked' in vtype.lower()]
        if parked_vehicles:
            writer.writerow([])
            writer.writerow(['Parked Vehicles:'])
            for vehicle_type in sorted(parked_vehicles):
                total = int(total_vehicle_counts[vehicle_type])
                writer.writerow([f'Total {vehicle_type}', total])
        writer.writerow([])
        writer.writerow(['========================================='])
        
        #Traffic light section
        writer.writerow(['3. TRAFFIC LIGHT STATISTICS'])
        writer.writerow(['-----------------------------------------'])
        writer.writerow(['Metric', 'Value'])
        writer.writerow([])
        writer.writerow(['Total traffic lights', tl_stats['total_traffic_lights']])
        writer.writerow([])
        writer.writerow(['Average queue length', f"{tl_stats['avg_queue_length']:.1f} vehicles"])
        writer.writerow(['Maximum queue length', f"{tl_stats['max_queue_length']:.0f} vehicles"])
        writer.writerow(['Minimum queue length', f"{tl_stats['min_queue_length']:.0f} vehicles"])
        writer.writerow([])
        writer.writerow(['Average waiting time', f"{tl_stats['avg_waiting_time']:.1f} seconds"])
        writer.writerow(['Maximum waiting time', f"{tl_stats['max_waiting_time']:.1f} seconds"])
        writer.writerow(['Minimum waiting time', f"{tl_stats['min_waiting_time']:.1f} seconds"])
        writer.writerow([])

        # Observer penetration rates section
        writer.writerow(['4. OBSERVER PENETRATION RATES'])
        writer.writerow(['-----------------------------------------'])
        writer.writerow(['Category', 'Total Count', 'Average Present', 'Rate'])
        writer.writerow([])
        
        # Only write fleet composition data if it was collected
        if fleet_composition_logs is not None and not fleet_composition_logs.empty:
            writer.writerow(['Total relevant cars', total_relevant_cars, 
                            f'{fleet_composition_logs["present_DEFAULT_VEHTYPE_count"].mean():.1f}', '100%'])
            fco_present = fleet_composition_logs.get("present_floating_car_observer_count", pd.Series([0])).mean()
            fco_present = 0.0 if pd.isna(fco_present) else fco_present
            writer.writerow(['Floating Car Observers', total_floating_car_observers, 
                            f'{fco_present:.1f}', f'{fco_penetration_rate:.2%}'])
            writer.writerow([])
            
            writer.writerow(['Total relevant bikes', total_relevant_bikes, 
                            f'{fleet_composition_logs["present_DEFAULT_BIKETYPE_count"].mean():.1f}', '100%'])
            fbo_present = fleet_composition_logs.get("present_floating_bike_observer_count", pd.Series([0])).mean()
            fbo_present = 0.0 if pd.isna(fbo_present) else fbo_present
            writer.writerow(['Floating Bike Observers', total_floating_bike_observers, 
                            f'{fbo_present:.1f}', f'{fbo_penetration_rate:.2%}'])
        else:
            writer.writerow(['Total relevant cars', total_relevant_cars, 'N/A', '100%'])
            writer.writerow(['Floating Car Observers', total_floating_car_observers, 'N/A', f'{fco_penetration_rate:.2%}'])
            writer.writerow([])
            writer.writerow(['Total relevant bikes', total_relevant_bikes, 'N/A', '100%'])
            writer.writerow(['Floating Bike Observers', total_floating_bike_observers, 'N/A', f'{fbo_penetration_rate:.2%}'])

        # Detection statistics section
        writer.writerow([])
        writer.writerow(['========================================='])
        writer.writerow(['5. BICYCLE DETECTION STATISTICS'])
        writer.writerow(['-----------------------------------------'])
        writer.writerow(['Metric', 'Value'])
        writer.writerow([])
        if detection_stats:
            writer.writerow(['Total detections', detection_stats['total_detections']])
            writer.writerow(['Unique bicycles detected', detection_stats['bicycles_detected']])
            writer.writerow([])
            writer.writerow(['Average detection distance', f'{detection_stats["avg_detection_distance"]:.1f} m'])
            writer.writerow(['Average detection duration', f'{detection_stats["avg_detection_duration"]:.1f} s'])
            writer.writerow([])
            writer.writerow(['Bicycles with multiple detections', detection_stats['bicycles_multiple_detections']])
            writer.writerow(['Bicycles never detected', detection_stats['never_detected_count']])
            writer.writerow(['Percentage never detected', f'{detection_stats["never_detected_percentage"]:.1f}%'])
        else:
            writer.writerow(['Note', 'No detection data available'])

        # Motorized vehicle trajectory statistics section
        writer.writerow([])
        writer.writerow(['========================================='])
        writer.writerow(['6. MOTORIZED VEHICLE TRAJECTORY STATISTICS'])
        writer.writerow(['-----------------------------------------'])
        writer.writerow(['Metric', 'Value'])
        writer.writerow([])
        if vehicle_trajectory_stats:
            total_unique_vehicles = sum(stats['unique_vehicles'] for stats in vehicle_trajectory_stats.values())
            writer.writerow(['Total unique motorized vehicles', total_unique_vehicles])
            writer.writerow([])
            for vehicle_type, stats in vehicle_trajectory_stats.items():
                writer.writerow([f'Statistics for {vehicle_type}:'])
                writer.writerow(['Total unique vehicles', stats['unique_vehicles']])
                writer.writerow([])
                writer.writerow(['Time in simulation:'])
                writer.writerow(['- Average time', f"{stats['time_in_sim']['mean']:.1f} s"])
                writer.writerow(['- Maximum time', f"{stats['time_in_sim']['max']:.1f} s"])
                writer.writerow(['- Minimum time', f"{stats['time_in_sim']['min']:.1f} s"])
                writer.writerow([])
                writer.writerow(['Distance traveled:'])
                writer.writerow(['- Average distance', f"{stats['distance']['mean']:.1f} m = {stats['distance']['mean']/1000:.2f} km"])
                writer.writerow(['- Maximum distance', f"{stats['distance']['max']:.1f} m = {stats['distance']['max']/1000:.2f} km"])
                writer.writerow(['- Minimum distance', f"{stats['distance']['min']:.1f} m = {stats['distance']['min']/1000:.2f} km"])
                writer.writerow([])
                writer.writerow(['Speed statistics:'])
                writer.writerow(['- Average speed', f"{stats['speed']['mean']:.2f} m/s = {stats['speed']['mean']*3.6:.1f} km/h"])
                writer.writerow(['- Maximum speed', f"{stats['speed']['max']:.2f} m/s = {stats['speed']['max']*3.6:.1f} km/h"])
                writer.writerow(['- Speed standard deviation', f"{stats['speed']['std']:.2f} m/s = {stats['speed']['std']*3.6:.1f} km/h"])
                writer.writerow(['- 15th percentile speed', f"{stats['speed']['percentile_15']:.2f} m/s = {stats['speed']['percentile_15']*3.6:.1f} km/h"])
                writer.writerow(['- Median speed', f"{stats['speed']['median']:.2f} m/s = {stats['speed']['median']*3.6:.1f} km/h"])
                writer.writerow(['- 85th percentile speed', f"{stats['speed']['percentile_85']:.2f} m/s = {stats['speed']['percentile_85']*3.6:.1f} km/h"])
                writer.writerow([])
                writer.writerow(['Acceleration statistics:'])
                writer.writerow(['- Average acceleration', f"{stats['acceleration']['mean']:.2f} m/s^2"])
                writer.writerow(['- Maximum acceleration', f"{stats['acceleration']['max']:.2f} m/s^2"])
                writer.writerow([])
                writer.writerow(['Deceleration statistics:'])
                writer.writerow(['- Average deceleration', f"{stats['deceleration']['mean']:.2f} m/s^2"])
                writer.writerow(['- Maximum deceleration', f"{stats['deceleration']['max']:.2f} m/s^2"])
                writer.writerow(['- Hard braking events (>4.5 m/s^2)', stats['deceleration']['hard_braking_events']])
                writer.writerow([])
                if 'following_distance' in stats:
                    writer.writerow(['Following distance statistics:'])
                    writer.writerow(['- Average following distance', f"{stats['following_distance']['mean']:.1f} m = {stats['following_distance']['mean']/1000:.3f} km"])
                    writer.writerow(['- Maximum following distance', f"{stats['following_distance']['max']:.1f} m = {stats['following_distance']['max']/1000:.3f} km"])
                    writer.writerow(['- Minimum following distance', f"{stats['following_distance']['min']:.1f} m = {stats['following_distance']['min']/1000:.3f} km"])
                writer.writerow([])
        else:
            writer.writerow(['Note', 'No vehicle trajectory data available'])

        # Bicycle trajectory statistics section
        writer.writerow([])
        writer.writerow(['========================================='])
        writer.writerow(['7. BICYCLE TRAJECTORY STATISTICS'])
        writer.writerow(['-----------------------------------------'])
        writer.writerow(['Metric', 'Value'])
        writer.writerow([])
        if bicycle_stats:
            total_unique_bicycles = sum(stats['unique_bicycles'] for stats in bicycle_stats.values())
            writer.writerow(['Total unique bicycles', total_unique_bicycles])
            writer.writerow([])
            for bicycle_type, stats in bicycle_stats.items():
                writer.writerow([f'Statistics for {bicycle_type}:'])
                writer.writerow(['Total unique bicycles', stats['unique_bicycles']])
                writer.writerow([])
                writer.writerow(['Time in simulation:'])
                writer.writerow(['- Average time', f"{stats['time_in_sim']['mean']:.1f} s"])
                writer.writerow(['- Maximum time', f"{stats['time_in_sim']['max']:.1f} s"])
                writer.writerow(['- Minimum time', f"{stats['time_in_sim']['min']:.1f} s"])
                writer.writerow([])
                writer.writerow(['Distance traveled:'])
                writer.writerow(['- Average distance', f"{stats['distance']['mean']:.1f} m = {stats['distance']['mean']/1000:.2f} km"])
                writer.writerow(['- Maximum distance', f"{stats['distance']['max']:.1f} m = {stats['distance']['max']/1000:.2f} km"])
                writer.writerow(['- Minimum distance', f"{stats['distance']['min']:.1f} m = {stats['distance']['min']/1000:.2f} km"])
                writer.writerow([])
                writer.writerow(['Speed statistics:'])
                writer.writerow(['- Average speed', f"{stats['speed']['mean']:.2f} m/s = {stats['speed']['mean']*3.6:.1f} km/h"])
                writer.writerow(['- Maximum speed', f"{stats['speed']['max']:.2f} m/s = {stats['speed']['max']*3.6:.1f} km/h"])
                writer.writerow(['- Speed standard deviation', f"{stats['speed']['std']:.2f} m/s = {stats['speed']['std']*3.6:.1f} km/h"])
                writer.writerow(['- 15th percentile speed', f"{stats['speed']['percentile_15']:.2f} m/s = {stats['speed']['percentile_15']*3.6:.1f} km/h"])
                writer.writerow(['- Median speed', f"{stats['speed']['median']:.2f} m/s = {stats['speed']['median']*3.6:.1f} km/h"])
                writer.writerow(['- 85th percentile speed', f"{stats['speed']['percentile_85']:.2f} m/s = {stats['speed']['percentile_85']*3.6:.1f} km/h"])
                writer.writerow([])
                writer.writerow(['Acceleration statistics:'])
                writer.writerow(['- Average acceleration', f"{stats['acceleration']['mean']:.2f} m/s^2"])
                writer.writerow(['- Maximum acceleration', f"{stats['acceleration']['max']:.2f} m/s^2"])
                writer.writerow([])
                writer.writerow(['Deceleration statistics:'])
                writer.writerow(['- Average deceleration', f"{stats['deceleration']['mean']:.2f} m/s^2"])
                writer.writerow(['- Maximum deceleration', f"{stats['deceleration']['max']:.2f} m/s^2"])
                writer.writerow(['- Hard braking events (>2.5 m/s^2)', stats['deceleration']['hard_braking_events']])
        else:
            writer.writerow(['Note', 'No bicycle trajectory data available'])

        # Bicycle conflict statistics section
        writer.writerow([])
        writer.writerow(['========================================='])
        writer.writerow(['8. BICYCLE CONFLICT STATISTICS'])
        writer.writerow(['-----------------------------------------'])
        writer.writerow(['Metric', 'Value'])
        writer.writerow([])
        if conflict_stats:
            writer.writerow(['Conflict frequency:'])
            writer.writerow(['- Total unique conflicts', conflict_stats['unique_conflicts']])
            writer.writerow(['- Total conflict frames', conflict_stats['conflict_frames']])
            writer.writerow(['- Unique bicycles involved', conflict_stats['unique_bicycles']])
            writer.writerow([])
            writer.writerow(['Per bicycle statistics:'])
            writer.writerow(['- Average unique conflicts per bicycle', f"{conflict_stats['conflicts_per_bicycle']:.1f}"])
            writer.writerow(['- Maximum unique conflicts per bicycle', conflict_stats['max_conflicts_per_bicycle']])
            writer.writerow(['- Minimum unique conflicts per bicycle', conflict_stats['min_conflicts_per_bicycle']])
            writer.writerow(['- Average conflict frames per bicycle', f"{conflict_stats['conflict_frames_per_bicycle']:.1f}"])
            writer.writerow(['- Maximum conflict frames per bicycle', conflict_stats['max_conflict_frames']])
            writer.writerow(['- Minimum conflict frames per bicycle', conflict_stats['min_conflict_frames']])
            writer.writerow([])
            writer.writerow(['Conflict durations:'])
            writer.writerow(['- Average duration', f"{conflict_stats['avg_duration']:.1f} s"])
            writer.writerow(['- Minimum duration', f"{conflict_stats['min_duration']:.1f} s"])
            writer.writerow(['- Maximum duration', f"{conflict_stats['max_duration']:.1f} s"])
            writer.writerow(['- Prolonged conflicts (>3s)', conflict_stats['prolonged_conflicts']])
            writer.writerow([])
            writer.writerow(['Conflict types (unique conflicts):'])
            writer.writerow(['- TTC-based conflicts', conflict_stats['ttc_conflicts']])
            writer.writerow(['- PET-based conflicts', conflict_stats['pet_conflicts']])
            writer.writerow(['- DRAC-based conflicts', conflict_stats['drac_conflicts']])
            writer.writerow([])
            writer.writerow(['Severity metrics:'])
            writer.writerow(['TTC (Time-To-Collision):'])
            writer.writerow(['- Average TTC', f"{conflict_stats['ttc_stats']['mean']:.2f} s"])
            writer.writerow(['- Minimum TTC', f"{conflict_stats['ttc_stats']['min']:.2f} s"])
            writer.writerow(['- Maximum TTC', f"{conflict_stats['ttc_stats']['max']:.2f} s"])
            writer.writerow([])
            writer.writerow(['PET (Post-Encroachment-Time):'])
            writer.writerow(['- Average PET', f"{conflict_stats['pet_stats']['mean']:.2f} s"])
            writer.writerow(['- Minimum PET', f"{conflict_stats['pet_stats']['min']:.2f} s"])
            writer.writerow(['- Maximum PET', f"{conflict_stats['pet_stats']['max']:.2f} s"])
            writer.writerow([])
            writer.writerow(['DRAC (Deceleration Rate to Avoid Crash):'])
            writer.writerow(['- Average DRAC', f"{conflict_stats['drac_stats']['mean']:.2f} m/s^2"])
            writer.writerow(['- Minimum DRAC', f"{conflict_stats['drac_stats']['min']:.2f} m/s^2"])
            writer.writerow(['- Maximum DRAC', f"{conflict_stats['drac_stats']['max']:.2f} m/s^2"])
            writer.writerow([])
            writer.writerow(['Conflict partners (unique conflicts):'])
            for foe_type, count in conflict_stats['conflicts_by_partner'].items():
                writer.writerow([f'- Conflicts with {foe_type}', count])
            writer.writerow([])
            writer.writerow(['Detection coverage (unique conflicts):'])
            detected = conflict_stats['detected_conflicts']
            total = conflict_stats['unique_conflicts']
            writer.writerow(['- Conflicts detected by observers', f"{detected} ({detected/total*100:.1f}%)"])
            writer.writerow(['- Conflicts detected by FCOs', 
                        f"{conflict_stats['fco_detected']} ({conflict_stats['fco_detected']/total*100:.1f}%)"])
            writer.writerow(['- Conflicts detected by FBOs', 
                        f"{conflict_stats['fbo_detected']} ({conflict_stats['fbo_detected']/total*100:.1f}%)"])
            writer.writerow(['- Undetected conflicts', 
                        f"{total-detected} ({(total-detected)/total*100:.1f}%)"])
        else:
            writer.writerow(['Note', 'No conflict data available'])

        # Performance metrics section
        if perf_stats:
            writer.writerow([])
            writer.writerow(['========================================='])
            writer.writerow(['9. PERFORMANCE METRICS'])
            writer.writerow(['-----------------------------------------'])
            writer.writerow(['Total runtime', f"{total_runtime:.2f} seconds"])
            writer.writerow([])
            writer.writerow(['Runtime breakdown:'])
            writer.writerow(['- Simulation setup', f"{setup_time:.2f} seconds"])
            if 'visualization' in operation_times and visualization_time > 0:
                writer.writerow(['- Visualization (incl. Ray tracing)', f"{visualization_time:.2f} seconds"])
            else:
                writer.writerow(['- Ray tracing (without visualization)', f"{ray_tracing_time:.2f} seconds"])
            if AnimatedThreeDimensionalDetectionPlots and '3d_animated_detections' in operation_times:
                writer.writerow(['- Animated 3D detections', f"{operation_times['3d_animated_detections']:.2f} seconds"])
            if 'visibility_data_export' in operation_times:
                writer.writerow(['- Visibility data export', f"{operation_times['visibility_data_export']:.2f} seconds"])
            if not AnimatedThreeDimensionalDetectionPlots:
                writer.writerow(['- Note:', 'No additional trajectory applications were activated'])
            writer.writerow(['- Data collection (for Logging)', f"{data_collection_time:.2f} seconds"])
            writer.writerow(['- Final logging and cleanup', f"{logging_time:.2f} seconds"])
            writer.writerow([])
            writer.writerow(['Timing verification:'])
            writer.writerow(['- Sum of components', f"{component_sum:.2f} seconds"])
            writer.writerow(['- Timing offset', f"{timing_offset:.2f} seconds"])
            writer.writerow(['Note: Offset includes'])
            writer.writerow(['- Python garbage collection'])
            writer.writerow(['- Operating system scheduling'])
            writer.writerow(['- File I/O operations'])
            writer.writerow(['- Inter-process communication with SUMO'])
            writer.writerow([])
            writer.writerow(['Step timing:'])
            writer.writerow(['- Average step duration', f"{perf_stats['step_timing']['mean']*1000:.2f} ms"])
            writer.writerow(['- Maximum step duration', f"{perf_stats['step_timing']['max']*1000:.2f} ms"])
            writer.writerow(['- Minimum step duration', f"{perf_stats['step_timing']['min']*1000:.2f} ms"])
            writer.writerow([])
            writer.writerow(['Memory usage:'])
            writer.writerow(['- Average memory usage', f"{perf_stats['memory_usage']['mean']:.1f} MB"])
            writer.writerow(['- Peak memory usage', f"{perf_stats['memory_usage']['max']:.1f} MB"])
        else:
            writer.writerow(['Note', 'No performance data available'])

    print('Summary logging completed.')

# ---------------------
# APPLICATIONS - VISIBILITY & BICYCLE SAFETY
# ---------------------

def three_dimensional_detection_plots_gif(frame):
    """
    Creates a 3D visualization of bicycle trajectories where the z=0 plane shows the static scene.
    The time dimension is represented on the z-axis. Observer vehicles (FCOs/FBOs) are shown
    with their detection ranges, and bicycles are colored based on their detection status.
    """
    global fig_3d, ax_3d, total_steps, bicycle_trajectories, transformer, flow_ids, bicycle_detection_data, detection_gif_trajectories
    
    # Initialize transformer at frame 0
    if frame == 0:
        transformer = pyproj.Transformer.from_crs('EPSG:4326', 'EPSG:32632', always_xy=True)
        detection_gif_trajectories.clear()
        flow_ids.clear()
        bicycle_detection_data = {}  # Store detection status over time
        
        # Add new dictionary for observer trajectories
        if not hasattr(three_dimensional_detection_plots, 'observer_trajectories'):
            three_dimensional_detection_plots.observer_trajectories = {}
        three_dimensional_detection_plots.observer_trajectories.clear()

    # Ensure transformer is initialized (still needed for coordinate transformation)
    if 'transformer' not in globals() or transformer is None:
        transformer = pyproj.Transformer.from_crs('EPSG:4326', 'EPSG:32632', always_xy=True)
    # Create bounding box for clipping
    bbox = box(west, south, east, north)
    # Transform bounding box with proper lambda signature
    bbox_transformed = shapely.ops.transform(
        lambda x, y, z=None: transformer.transform(x, y),  # type: ignore # shapely transform quirk
        bbox
    )    # Get bounds of transformed bounding box for relative coordinates
    minx, miny, maxx, maxy = bbox_transformed.bounds
            
    # Define functions for relative coordinates
    rel_x = lambda x: x - minx
    rel_y = lambda y: y - miny

    # Get current vehicles and collect data
    current_vehicles = set(traci.vehicle.getIDList())
    current_time = frame * step_length
    for vehicle_id in current_vehicles:
        vehicle_type = traci.vehicle.getTypeID(vehicle_id)
        x_sumo, y_sumo = traci.vehicle.getPosition(vehicle_id)
        # Get vehicle dimensions and angle for position correction
        vehicle_length = vehicle_attributes(vehicle_type)[2][1]
        vehicle_angle = traci.vehicle.getAngle(vehicle_id)
        # Convert from SUMO position (front bumper) to geometric center
        x_center, y_center = sumo_position_to_center(x_sumo, y_sumo, vehicle_length, vehicle_angle)
        lon, lat = traci.simulation.convertGeo(x_center, y_center)
        x_utm, y_utm = transformer.transform(lon, lat)
        point = Point(x_utm, y_utm)
        if bbox_transformed.contains(point):
            # Store positions for bicycles
            if vehicle_type in ["bicycle", "DEFAULT_BIKETYPE", "floating_bike_observer"]:
                # Initialize default values
                is_detected = False
                current_observers = []
                # Get detection info from bicycle_detection_data if it exists
                if vehicle_id in bicycle_detection_data:
                    detection_info = next((info for info in bicycle_detection_data[vehicle_id] 
                                        if abs(info[0] - current_time) < step_length), None)
                    
                    if detection_info:
                        # Always convert to 3-tuple format
                        if len(detection_info) == 2:  # Old format
                            is_detected = detection_info[1]
                        else:  # New format
                            is_detected = detection_info[1]
                            current_observers = detection_info[2]
                        
                        # If detected, add the current observer to the list
                        if is_detected:
                            # Find the nearest observer at this time
                            for obs_id, obs_data in three_dimensional_detection_plots.observer_trajectories.items():
                                if obs_data['trajectory'] and abs(obs_data['trajectory'][-1][2] - current_time) < step_length:
                                    current_observers.append({'id': obs_id})
                # Always store in new format
                if vehicle_id not in bicycle_detection_data:
                    bicycle_detection_data[vehicle_id] = []
                bicycle_detection_data[vehicle_id].append((current_time, is_detected, current_observers))
                # Store trajectory data
                flow_id = vehicle_id.rsplit('.', 1)[0]
                flow_ids.add(flow_id)
                if vehicle_id not in detection_gif_trajectories:
                    detection_gif_trajectories[vehicle_id] = []
                detection_gif_trajectories[vehicle_id].append((x_utm, y_utm, current_time))
            
            # Store observer vehicle trajectories
            elif vehicle_type in ["floating_car_observer", "floating_bike_observer"]:
                if vehicle_id not in three_dimensional_detection_plots.observer_trajectories:
                    three_dimensional_detection_plots.observer_trajectories[vehicle_id] = {
                        'type': vehicle_type,
                        'trajectory': [],
                        'detections': []
                    }
                
                # Store trajectory point
                three_dimensional_detection_plots.observer_trajectories[vehicle_id]['trajectory'].append(
                    (x_utm, y_utm, current_time)
                )

    # Check for bicycles that have finished their trajectory
    # finished_bicycles = set(detection_gif_trajectories.keys()) - current_vehicles

    # Generate plots for finished bicycles
    for vehicle_id in list(detection_gif_trajectories.keys()):
        if vehicle_id not in current_vehicles:
        # if len(detection_gif_trajectories[vehicle_id]) > 0:  # Only plot if we have trajectory data
            trajectory = detection_gif_trajectories[vehicle_id]
            
            for detection_info in bicycle_detection_data[vehicle_id]:
                # Safely unpack the detection info
                if len(detection_info) == 3:  # New format (time, is_detected, observers)
                    time, is_detected, observers = detection_info
                else:  # Old format (time, is_detected)
                    time, is_detected = detection_info
                    observers = []  # Empty list for old format
            # Create new figure for this bicycle
            fig_3d = plt.figure(figsize=(15, 12))
            ax_3d = fig_3d.add_subplot(111, projection='3d')
            
            # Get bounds of transformed bounding box
            minx, miny, maxx, maxy = bbox_transformed.bounds
            
            # Set axis labels and limits
            ax_3d.set_xlabel('Distance (m)')
            ax_3d.set_ylabel('Distance (m)')
            ax_3d.set_zlabel('Time (s)')
            ax_3d.set_xlim(0, maxx - minx)
            ax_3d.set_ylim(0, maxy - miny)
            
            # Calculate z range with padding
            z_min = min(t for _, _, t in trajectory)
            z_max = max(t for _, _, t in trajectory)
            z_padding = (z_max - z_min) * 0.05
            base_z = z_min - z_padding
            ax_3d.set_zlim(base_z, z_max + z_padding)
            # Set background color of the planes and grid style
            ax_3d.xaxis.pane.fill = False
            ax_3d.yaxis.pane.fill = False
            ax_3d.zaxis.pane.fill = False
            ax_3d.xaxis._axinfo['grid'].update(linestyle="--")
            ax_3d.yaxis._axinfo['grid'].update(linestyle="--")
            ax_3d.zaxis._axinfo['grid'].update(linestyle="--")
            
            # Calculate aspect ratios
            dx = maxx - minx
            dy = maxy - miny
            dz = (z_max + z_padding) - base_z
            max_xy = max(dx, dy)
            aspect_ratios = [dx/max_xy, dy/max_xy, dz/max_xy * 2.0]
            # Set 3D plot aspect ratio - handle type checking
            ax_3d.set_box_aspect(aspect_ratios)  # type: ignore  # matplotlib 3D specific method
            
            # Set view angle
            ax_3d.view_init(elev=35, azim=285)
            ax_3d.set_axisbelow(True)
            # Create base plane
            base_vertices = [
                [0, 0, base_z],
                [maxx - minx, 0, base_z],
                [maxx - minx, maxy - miny, base_z],
                [0, maxy - miny, base_z]
            ]
            base_poly = Poly3DCollection([base_vertices], alpha=0.1)
            base_poly.set_facecolor('white')
            base_poly.set_edgecolor('gray')
            base_poly.set_sort_zpos(-2)
            ax_3d.add_collection3d(base_poly)
            # Plot roads
            for _, road in gdf1_proj.iterrows():
                if road.geometry.intersects(bbox_transformed):
                    clipped_geom = road.geometry.intersection(bbox_transformed)
                    if isinstance(clipped_geom, (MultiPolygon, Polygon)):
                        if isinstance(clipped_geom, MultiPolygon):
                            polygons = clipped_geom.geoms
                        else:
                            polygons = [clipped_geom]
                        
                        for polygon in polygons:
                            xs, ys = polygon.exterior.xy
                            xs = np.clip([rel_x(x) for x in xs], 0, maxx - minx)
                            ys = np.clip([rel_y(y) for y in ys], 0, maxy - miny)
                            verts = [(x, y, base_z) for x, y in zip(xs, ys)]
                            poly = Poly3DCollection([verts], alpha=0.5)
                            poly.set_facecolor('lightgray')
                            poly.set_edgecolor('darkgray')
                            poly.set_linewidth(1.0)
                            poly.set_sort_zpos(-1)
                            ax_3d.add_collection3d(poly)
                    
                    elif isinstance(clipped_geom, LineString):
                        xs, ys = clipped_geom.xy
                        xs = np.clip([rel_x(x) for x in xs], 0, maxx - minx)
                        ys = np.clip([rel_y(y) for y in ys], 0, maxy - miny)
                        ax_3d.plot(xs, ys, [base_z]*len(xs),
                                 color='darkgray', linewidth=1.0, alpha=0.5,
                                 zorder=-1)
            # Plot buildings and parks
            for collection in [buildings_proj, parks_proj]:
                if collection is not None:
                    for _, element in collection.iterrows():
                        if element.geometry.intersects(bbox_transformed):
                            clipped_geom = element.geometry.intersection(bbox_transformed)
                            if isinstance(clipped_geom, (MultiPolygon, Polygon)):
                                if isinstance(clipped_geom, MultiPolygon):
                                    polygons = clipped_geom.geoms
                                else:
                                    polygons = [clipped_geom]
                                
                                for polygon in polygons:
                                    xs, ys = polygon.exterior.xy
                                    xs = np.clip([rel_x(x) for x in xs], 0, maxx - minx)
                                    ys = np.clip([rel_y(y) for y in ys], 0, maxy - miny)
                                    verts = [(x, y, base_z) for x, y in zip(xs, ys)]
                                    poly = Poly3DCollection([verts])
                                    if collection is buildings_proj:
                                        poly.set_facecolor('darkgray')
                                        poly.set_alpha(1.0)
                                    else:  # parks
                                        poly.set_facecolor('forestgreen')
                                        poly.set_alpha(1.0)
                                    poly.set_edgecolor('black')
                                    poly.set_sort_zpos(0)
                                    ax_3d.add_collection3d(poly)
            # First, collect all unique observers that detected this bicycle at any point
            detecting_observers = set()
            for detection_info in bicycle_detection_data[vehicle_id]:
                # Safely unpack the detection info
                if len(detection_info) == 3:  # New format (time, is_detected, observers)
                    time, is_detected, observers = detection_info
                else:  # Old format (time, is_detected)
                    time, is_detected = detection_info
                    observers = []  # Empty list for old format
                
                if is_detected:
                    for observer in observers:
                        detecting_observers.add(observer['id'])
            # Initialize segments dictionary and detection buffer for smoothing
            segments = {'detected': [], 'undetected': []}
            detection_buffer = []
            current_points = []
            current_detected = None
            segment_observers = set()  # Add this to track observers for current segment
            # Process trajectory points with smoothing
            for x, y, t in trajectory:
                # Get detection status for this time
                detection_info = next((info for info in bicycle_detection_data[vehicle_id] 
                                    if abs(info[0] - t) < step_length), None)
                
                if detection_info:
                    if len(detection_info) == 3:  # New format
                        is_detected = detection_info[1]
                        current_observers = detection_info[2]
                    else:  # Old format
                        is_detected = detection_info[1]
                        current_observers = []
                else:
                    is_detected = False
                    current_observers = []
                # Update detection buffer for smoothing
                detection_buffer.append(is_detected)
                if len(detection_buffer) > basic_gap_bridge:
                    detection_buffer.pop(0)
                # Apply smoothing logic
                recent_detection = any(detection_buffer[-3:]) if len(detection_buffer) >= 3 else is_detected
                if not recent_detection and len(detection_buffer) >= basic_gap_bridge:
                    if any(detection_buffer[:3]) and any(detection_buffer[-3:]):
                        smoothed_detection = True
                    else:
                        smoothed_detection = False
                else:
                    smoothed_detection = recent_detection
                
                if current_detected is None:
                    current_detected = smoothed_detection
                    current_points = [(rel_x(x), rel_y(y), t)]
                    segment_observers = set(obs['id'] for obs in current_observers)  # Track observers for segment
                elif smoothed_detection != current_detected:
                    if len(current_points) >= basic_segment_length:
                        segments['detected' if current_detected else 'undetected'].append(
                            (current_points, [{'id': obs_id} for obs_id in segment_observers]))
                        current_points = [(rel_x(x), rel_y(y), t)]
                        current_detected = smoothed_detection
                        segment_observers = set(obs['id'] for obs in current_observers)  # Reset observers for new segment
                    else:
                        current_points.append((rel_x(x), rel_y(y), t))
                        if smoothed_detection:
                            segment_observers.update(obs['id'] for obs in current_observers)
                else:
                    current_points.append((rel_x(x), rel_y(y), t))
                    if smoothed_detection:
                        segment_observers.update(obs['id'] for obs in current_observers)
            if current_points:
                segments['detected' if current_detected else 'undetected'].append(
                    (current_points, [{'id': obs_id} for obs_id in segment_observers]))
            # First, sort segments by time
            all_segments = [(state, (points, obs)) for state in ['undetected', 'detected'] 
                            for points, obs in segments[state]]
            all_segments.sort(key=lambda x: x[1][0][0][2])  # Sort by first time point
            # Then plot with adjusted zorder and alpha
            for i, (state, (segment_points, observers)) in enumerate(all_segments):
                if len(segment_points) > 1:
                    x_coords, y_coords, times = zip(*segment_points)
                    is_detected = state == 'detected'
                    color = 'cornflowerblue' if is_detected else 'darkslateblue'
                    
                    # Plot 3D trajectory with adjusted parameters
                    ax_3d.plot(x_coords, y_coords, times, 
                            color=color, linewidth=2, 
                            alpha=1.0,
                            zorder=1000)  # Higher zorder for detected segments
                    
                    # Plot ground projection
                    ax_3d.plot(x_coords, y_coords, [base_z]*len(x_coords),
                            color=color, linestyle='--', 
                            linewidth=2, 
                            alpha=1.0,
                            zorder=1000)
                    
                    # Add projection plane with adjusted transparency
                    plane_vertices = []
                    for j in range(len(x_coords)-1):
                        quad = [
                            (x_coords[j], y_coords[j], times[j]),
                            (x_coords[j+1], y_coords[j+1], times[j+1]),
                            (x_coords[j+1], y_coords[j+1], base_z),
                            (x_coords[j], y_coords[j], base_z)
                        ]
                        plane_vertices.append(quad)
                        
                        # If this is the last point of a segment and there's a next segment,
                        # add an extra quad to connect to the next segment
                        if j == len(x_coords)-2 and i < len(all_segments)-1:
                            next_state, (next_points, _) = all_segments[i+1]
                            next_x, next_y, next_t = next_points[0]
                            transition_quad = [
                                (x_coords[j+1], y_coords[j+1], times[j+1]),
                                (next_x, next_y, next_t),
                                (next_x, next_y, base_z),
                                (x_coords[j+1], y_coords[j+1], base_z)
                            ]
                            plane_vertices.append(transition_quad)  # Fixed: append transition_quad instead of quad
                    # Create single projection plane for all vertices
                    proj_plane = Poly3DCollection(plane_vertices, alpha=0.2)
                    proj_plane.set_facecolor(color)
                    proj_plane.set_edgecolor('none')
                    proj_plane.set_sort_zpos(900)
                    ax_3d.add_collection3d(proj_plane)
            # Plot observer trajectories
            for observer_id in set(obs['id'] for segment in segments['detected'] for obs in segment[1]):
                if observer_id in three_dimensional_detection_plots.observer_trajectories:
                    obs_traj = three_dimensional_detection_plots.observer_trajectories[observer_id]['trajectory']
                    filtered_traj = [(rel_x(x), rel_y(y), t) for x, y, t in obs_traj 
                                if base_z <= t <= z_max + z_padding]
                    
                    if filtered_traj:
                        obs_x, obs_y, obs_t = zip(*filtered_traj)
                        
                        # Get all detection time periods and segments
                        detection_times = set()
                        detection_segments = []
                        for state, (segment_points, segment_observers) in all_segments:
                            if state == 'detected' and any(obs['id'] == observer_id for obs in segment_observers):
                                detection_times.update(t for _, _, t in segment_points)
                                detection_segments.append(segment_points)
                        
                        # Plot 3D trajectory with varying colors
                        for i in range(len(obs_x)-1):
                            color = 'darkred' if obs_t[i] in detection_times else 'indianred'
                            ax_3d.plot(obs_x[i:i+2], obs_y[i:i+2], obs_t[i:i+2],
                                    color=color, linewidth=2, alpha=1.0,
                                    zorder=1000)
                        
                        # Plot ground projection as segments
                        ground_segments = []
                        current_color = 'indianred'
                        current_segment = []
                        
                        for i in range(len(obs_x)):
                            is_detecting = obs_t[i] in detection_times
                            color = 'darkred' if is_detecting else 'indianred'
                            
                            if color != current_color and current_segment:
                                ground_segments.append((current_segment, current_color))
                                current_segment = [(obs_x[i], obs_y[i])]
                                current_color = color
                            else:
                                current_segment.append((obs_x[i], obs_y[i]))
                        
                        if current_segment:
                            ground_segments.append((current_segment, current_color))
                        
                        # Plot each ground segment
                        for segment, color in ground_segments:
                            seg_x, seg_y = zip(*segment)
                            ax_3d.plot(seg_x, seg_y, [base_z]*len(seg_x),
                                    color=color, linestyle='--', linewidth=2, alpha=0.7,
                                    zorder=1000)
                        
                        # Add projection plane for observer
                        obs_plane_vertices = []
                        for j in range(len(obs_x)-1):
                            quad = [
                                (obs_x[j], obs_y[j], obs_t[j]),
                                (obs_x[j+1], obs_y[j+1], obs_t[j+1]),
                                (obs_x[j+1], obs_y[j+1], base_z),
                                (obs_x[j], obs_y[j], base_z)
                            ]
                            obs_plane_vertices.append(quad)
                        
                        obs_proj_plane = Poly3DCollection(obs_plane_vertices, alpha=0.1)
                        obs_proj_plane.set_facecolor('indianred')
                        obs_proj_plane.set_edgecolor('none')
                        obs_proj_plane.set_sort_zpos(900)
                        ax_3d.add_collection3d(obs_proj_plane)
            
            # Add bicycle label - fix argument order
            ax_3d.text(rel_x(x_coords[-1]), rel_y(y_coords[-1]), base_z,
                      f'bicycle {vehicle_id}',  # type: ignore  # matplotlib 3D text signature issue
                      color='darkslateblue',
                      horizontalalignment='right',
                      verticalalignment='bottom',
                      rotation=90,
                      bbox=dict(facecolor='white', alpha=1.0, edgecolor='none'),
                      zorder=1000)
            # Create legend
            handles = [
                Line2D([0], [0], color='darkslateblue', linewidth=2, label='Bicycle Undetected'),
                Line2D([0], [0], color='cornflowerblue', linewidth=2, label='Bicycle Detected'),
                Line2D([0], [0], color='indianred', linewidth=2, label='Observer Vehicle'),
                Line2D([0], [0], color='darkred', linewidth=2, label='Observer Vehicle (Detecting)'),
                Line2D([0], [0], color='black', linestyle='--', label='Ground Projections')
            ]
            ax_3d.legend(handles=handles, loc='upper left')

            base_filename_detection = f'bicycle_{vehicle_id}_FCO{FCO_share*100:.0f}_FBO{FBO_share*100:.0f}'
            save_rotating_view_frames(ax_3d, base_filename_detection=base_filename_detection)
            create_rotating_view_gif(base_filename_detection=base_filename_detection)

            plt.close(fig_3d)

            # Clean up trajectories
            del detection_gif_trajectories[vehicle_id]
            # if vehicle_id in bicycle_detection_data:
            #     del bicycle_detection_data[vehicle_id]

# Helper functions for gif creation
# -----
def save_rotating_view_frames(ax_3d, base_filename_conflict=None, base_filename_detection=None, n_frames=30):
    """Helper function to save frames for rotating view animation"""
    # Directory paths already created in setup_scenario_output_directory()
        
    # Start from bird's eye view and smoothly transition both angles
    # Start: (90° elevation, 270° azimuth)    - bird's eye view
    # End: (35° elevation, 285° azimuth)    - final view
    t = np.linspace(0, 1, n_frames)
    t_azim = t**2
    elevations = np.linspace(90, 35, n_frames)
    azimuths = 270 + (t_azim * 15)
    
    # Save frames based on which type is active and has a filename
    if AnimatedThreeDimensionalDetectionPlots and base_filename_detection:
        for i, (elev, azim) in enumerate(zip(elevations, azimuths)):
            ax_3d.view_init(elev=elev, azim=azim)
            detection_frame_path = os.path.join(scenario_output_dir, 'out_3d_detections_gif', 'rotation_frames', f'{base_filename_detection}_frame_{i:03d}.png')
            plt.savefig(detection_frame_path, dpi=300)
            
def create_rotating_view_gif(base_filename_conflict=None, base_filename_detection=None, duration=0.1):
    """Helper function to create GIF from saved frames"""
    # Get all frames for this plot
    frames_detection = None
    
    # Check for detection plots
    if AnimatedThreeDimensionalDetectionPlots and base_filename_detection:
        detection_frames_pattern = os.path.join(scenario_output_dir, 'out_3d_detections_gif', 'rotation_frames', f'{base_filename_detection}_frame_*.png')
        frames_detection = sorted(glob.glob(detection_frames_pattern))
    
    # Create GIF for available frames
    if frames_detection:
        images_detection = [imageio.imread(frame) for frame in frames_detection]
        output_file_detection = os.path.join(scenario_output_dir, 'out_3d_detections_gif', f'{base_filename_detection}_rotation.gif')
        # Create GIF animation with proper type handling
        imageio.mimsave(output_file_detection, images_detection, format='GIF', duration=duration)  # type: ignore # imageio v2 type annotations
        print(f'\nCreated rotating view animation: {output_file_detection}')
        # Clean up detection frames
        for frame in frames_detection:
            os.remove(frame)

# ------

# ---------------------
# MAIN EXECUTION
# ---------------------

if __name__ == "__main__":  
    with TimingContext("simulation_setup"):
        # Setup scenario-specific output directory
        scenario_output_dir = setup_scenario_output_directory()
        print(f'Scenario output directory created: {scenario_output_dir}')
        
        print("\n" + "="*80)
        print("STEP 1/4: Loading input data and connecting to SUMO...")
        print("="*80)
        load_sumo_simulation()
        gdf1, G, buildings, parks, trees, leaves, barriers, PT_shelters = load_geospatial_data()
        
        print("\n" + "="*80)
        print("STEP 2/4: Transforming coordinates to UTM projection...")
        print("="*80)
        gdf1_proj, G_proj, buildings_proj, parks_proj, trees_proj, leaves_proj, barriers_proj, PT_shelters_proj = project_geospatial_data(gdf1, G, buildings, parks, trees, leaves, barriers, PT_shelters)
        
        # Load SUMO additional-files polygons (if any) and convert to projected coords
        try:
            additional_polygons = load_additional_polygons_from_sumocfg(sumo_config_path)
        except Exception:
            additional_polygons = []
        # (debug sampling removed) continue initialization
        setup_plot()
        plot_geospatial_data(gdf1_proj, G_proj, buildings_proj, parks_proj, trees_proj, leaves_proj, barriers_proj, PT_shelters_proj)
        # Always initialize visibility grid for consistent data collection
        x_coords, y_coords, grid_cells, discrete_visibility_counts, continuous_visibility_counts = initialize_grid(buildings_proj, grid_size)
        total_steps = get_total_simulation_steps(sumo_config_path)
        
        print("\n" + "="*80)
        print("STEP 3/4: Running simulation (ray tracing & visibility binning)...")
        print("="*80)
        # Continue initialization; do not perform debug sampling here
    if useLiveVisualization or saveAnimation:
        with TimingContext("visualization"):
            anim = run_animation(total_steps)
    else:
        for frame in range(total_steps):
            with TimingContext("ray_tracing"):
                update_with_ray_tracing(frame)
    
    print("\n" + "="*80)
    print("STEP 4/4: Saving logs and generating outputs...")
    print("="*80)
    with TimingContext("logging"):
        # Save simulation logs (always active)
        save_simulation_logs()
        
        # Print SSM device status summary
        print_ssm_device_summary()
        
        # Print performance summary
        profiler.print_summary()
        
        # Print ray tracing performance summary
        if _ray_tracing_stats['total_calls'] > 0:
            print("\n" + "="*80)
            print("RAY TRACING PERFORMANCE SUMMARY")
            print("="*80)
            total_rays = _ray_tracing_stats['total_rays']
            total_segs = _ray_tracing_stats['total_segments']
            total_calls = _ray_tracing_stats['total_calls']
            total_time = _ray_tracing_stats['total_time']
            
            print(f"Total ray tracing calls: {total_calls:,}")
            print(f"Total rays processed: {total_rays:,}")
            print(f"Average rays per observer: {total_rays/total_calls:.0f}")
            print(f"Total segments tested: {total_segs:,}")
            print(f"Average segments per call: {total_segs/total_calls:.0f}")
            print(f"Total ray tracing time: {total_time:.2f}s")
            print(f"Throughput: {total_rays/total_time:,.0f} rays/second")
            print(f"Average time per observer: {total_time/total_calls*1000:.2f}ms")
            
            # Show which mode was used
            gpu_calls = _ray_tracing_stats.get('gpu_calls', 0)
            cpu_calls = _ray_tracing_stats.get('cpu_calls', 0)
            
            if use_gpu_acceleration and CUDA_AVAILABLE:
                gpu_impl = "Numba CUDA kernels" if NUMBA_AVAILABLE else "CuPy vectorized"
                if gpu_calls > 0 and cpu_calls > 0:
                    print(f"\nMode: Adaptive (requested GPU, used both)")
                    print(f"  GPU calls: {gpu_calls:,} ({gpu_calls/total_calls*100:.1f}%)")
                    print(f"  CPU calls: {cpu_calls:,} ({cpu_calls/total_calls*100:.1f}%) - workload optimization")
                elif gpu_calls > 0:
                    print(f"\nMode: GPU acceleration ({gpu_impl})")
                else:
                    print(f"\nMode: CPU (adaptive fallback from GPU mode)")
                    print(f"  Reason: Workload better suited for CPU")
            elif use_multithreading:
                print(f"\nMode: CPU multi-threading ({max_worker_threads} workers)")
            else:
                print(f"\nMode: Single-threaded CPU")
        
        traci.close()
        print("\n✓ Simulation completed successfully!")
        
    # Always save visibility data for standalone evaluation scripts
    with TimingContext("visibility_data_export"):
        output_prefix = f'{file_tag}_FCO{FCO_share*100:.0f}%_FBO{FBO_share*100:.0f}%'
        
        # Save discrete visibility counts to CSV
        discrete_counts_path = os.path.join(scenario_output_dir, 'out_raytracing', f'discrete_visibility_counts_{file_tag}_FCO{FCO_share*100:.0f}%_FBO{FBO_share*100:.0f}%.csv')
        with open(discrete_counts_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['x_coord', 'y_coord', 'discrete_visibility_count'])
            for cell, count in discrete_visibility_counts.items():
                # Get cell center coordinates
                cell_x = cell.bounds[0] + (cell.bounds[2] - cell.bounds[0]) / 2
                cell_y = cell.bounds[1] + (cell.bounds[3] - cell.bounds[1]) / 2
                csvwriter.writerow([cell_x, cell_y, count])
        
        # Save continuous visibility counts to CSV
        continuous_counts_path = os.path.join(scenario_output_dir, 'out_raytracing', f'continuous_visibility_counts_{file_tag}_FCO{FCO_share*100:.0f}%_FBO{FBO_share*100:.0f}%_SSA{single_sensor_accuracy}%.csv')
        with open(continuous_counts_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['x_coord', 'y_coord', 'continuous_visibility_count'])
            for cell, count in continuous_visibility_counts.items():
                # Get cell center coordinates
                cell_x = cell.bounds[0] + (cell.bounds[2] - cell.bounds[0]) / 2
                cell_y = cell.bounds[1] + (cell.bounds[3] - cell.bounds[1]) / 2
                csvwriter.writerow([cell_x, cell_y, count])
    
    # Print final summary with scenario output directory
    print(f'\n=== SIMULATION COMPLETE ===')
    print(f'All outputs saved to: {scenario_output_dir}')
    if FCO_share == 0 and FBO_share == 0:
        print('Note: All visibility counts are 0 (no observers present)')
    print('Use separate evaluation scripts for further analysis.')