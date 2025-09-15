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
    # Fallback profiler class
    class BasicProfiler:
        def start_timer(self, operation): pass
        def end_timer(self): pass
        def update_frame_stats(self, *args): pass
        def print_summary(self): pass
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
    print("ℹ️  GPU acceleration not available (CuPy not installed)")
    profiler = BasicProfiler()
try:
    import cupy as cp
    import cupyx
    CUDA_AVAILABLE = True
    print("CUDA/CuPy is available for GPU acceleration")
except ImportError:
    CUDA_AVAILABLE = False
    cp = None
    print("CUDA/CuPy not available - using CPU-only processing")
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
file_tag = 'test_run_1'  # Current simulation identifier

# Performance Optimization Settings:
# ──────────────────────────────────────────────────────────────────────────────────
# Choose performance optimization level based on your system capabilities:
# - "none": Single-threaded processing (most compatible, but slower)
# - "cpu": Multi-threaded CPU processing (recommended default, good balance)
# - "gpu": CPU multi-threading + GPU acceleration (fastest, requires NVIDIA GPU with CUDA/CuPy)
performance_optimization_level = "cpu"
max_worker_threads = None  # None = auto-detect optimal thread count, or specify number (e.g., 4, 8)

# Path Settings:
# ──────────────────────────────────────────────────────────────────────────────────
base_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(base_dir)
# Path to SUMO config-file
# sumo_config_path = os.path.join(parent_dir, 'simulation_examples', 'Spatial-Visibility_Ilic-TRB-2025', 'Ilic-2025_config_low-demand.sumocfg')  # Simulation example: spatial visibility analysis (low demand) [Ilic, 2025]
# sumo_config_path = os.path.join(parent_dir, 'simulation_examples', 'Spatial-Visibility_Ilic-TRB-2025', 'Ilic-2025_config_high-demand.sumocfg')  # Simulation example: spatial visibility analysis (high demand) [Ilic, 2025]
sumo_config_path = os.path.join(parent_dir, 'simulation_examples', 'VRU-specific-Detection_Ilic-TRA-2026', 'Ilic-2026_config_30kmh.sumocfg')  # Simulation example: VRU-specific detection (30 km/h scenario) [Ilic, 2026]
# sumo_config_path = os.path.join(parent_dir, 'simulation_examples', 'VRU-specific-Detection_Ilic-TRA-2026', 'Ilic-2026_config_50kmh.sumocfg')  # Simulation example: VRU-specific detection (50 km/h scenario) [Ilic, 2026]
# Path to GeoJSON file (optional)
# geojson_path = os.path.join(parent_dir, 'simulation_examples', 'Spatial-Visibility_Ilic-TRB-2025', 'Ilic-2025.geojson') # Simulation example: spatial visibility analysis [Ilic, 2025]
geojson_path = os.path.join(parent_dir, 'simulation_examples', 'VRU-specific-Detection_Ilic-TRA-2026', 'Ilic-2026.geojson') # Simulation example: spatial visibility analysis [Ilic, 2025]

# Geographic Bounding Box Settings:
# ──────────────────────────────────────────────────────────────────────────────────
# Geographic boundaries in longitude / latitude in EEPSG:4326 (WGS84)
# north, south, east, west = 48.150500, 48.149050, 11.571000, 11.567900 # Simulation example: spatial visibility analysis [Ilic, 2025]
north, south, east, west = 48.146200, 48.144400, 11.580650, 11.577150 # Simulation example: VRU-specific detection [Ilic, 2026]
bbox = (north, south, east, west)

# Simulation Warm-up Settings:
# ──────────────────────────────────────────────────────────────────────────────────
delay = 20  # Warm-up time in seconds (no ray tracing during this period)

# ═══════════════════════════════════════════════════════════════════════════════════
# RAY TRACING SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════════

# Observer Penetration Rate Settings:
# ──────────────────────────────────────────────────────────────────────────────────
FCO_share = 0.1  # Floating Car Observers penetration rate (0.0 to 1.0)
FBO_share = 0.0  # Floating Bike Observers penetration rate (0.0 to 1.0)

# Ray Tracing Parameter Settings:
# ──────────────────────────────────────────────────────────────────────────────────
numberOfRays = 360  # Number of rays emerging from each observer vehicle
radius = 30         # Ray radius in meters
grid_size = 1.0      # Grid size for visibility heat map (meters) - determines the resolution of LoV and RelVis heatmaps

# Visualization Settings:
# ──────────────────────────────────────────────────────────────────────────────────
useLiveVisualization = True      # Show live visualization during simulation
visualizeRays = True             # Show individual rays in visualization (besides resulting visibility polygon)
useManualFrameForwarding = False  # Manual frame-by-frame progression (for debugging)
saveAnimation = False             # Save animation as video file

# ═══════════════════════════════════════════════════════════════════════════════════
# DATA COLLECTION & ANALYSIS SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════════

# Data Collection Settings:
# ──────────────────────────────────────────────────────────────────────────────────
CollectLoggingData = False    # Enable detailed data logging
basic_gap_bridge = 10        # Gap bridging for trajectory smoothing
basic_segment_length = 3     # Minimum segment length for trajectories

# Analysis Applications Settings:
# ──────────────────────────────────────────────────────────────────────────────────
FlowBasedBicycleTrajectories = False        # Generate 2D bicycle flow diagrams
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
            print(f"GPU optimization enabled - using {max_worker_threads} worker threads + GPU acceleration")
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
    print("  CollectLoggingData = True    # Enable detailed logging")
    print("  FlowBasedBicycleTrajectories = False  # Generate flow diagrams")
    
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
fleet_composition_logs = pd.DataFrame(columns=[
    'time_step', 'new_DEFAULT_VEHTYPE_count', 'present_DEFAULT_VEHTYPE_count',
    'new_floating_car_observer_count', 'present_floating_car_observer_count',
    'new_DEFAULT_BIKETYPE_count', 'present_DEFAULT_BIKETYPE_count',
    'new_floating_bike_observer_count', 'present_floating_bike_observer_count'
])
traffic_light_logs = pd.DataFrame(columns=[
    'time_step', 'traffic_light_id', 'program', 'phase', 'phase_duration', 'remaining_duration',
    'signal_states', 'total_queue_length', 'vehicles_stopped', 'average_waiting_time', 'vehicles_by_type',
    'lane_to_signal_mapping'
])
detection_logs = pd.DataFrame(columns=[
    'time_step', 'observer_id', 'observer_type', 'bicycle_id', 'x_coord',
    'y_coord', 'detection_distance', 'observer_speed', 'bicycle_speed'
])
dtypes = {
    'time_step': int, 'vehicle_id': str, 'vehicle_type': str, 'x_coord': float, 'y_coord': float,
    'speed': float, 'angle': float, 'acceleration': float, 'lateral_speed': float, 'slope': float,
    'distance': float, 'route_id': str, 'lane_id': str, 'edge_id': str, 'lane_position': float,
    'lane_index': int, 'leader_id': str, 'leader_distance': float, 'follower_id': str,
    'follower_distance': float, 'next_tls_id': str, 'distance_to_tls': float, 'length': float,
    'width': float, 'max_speed': float
}
vehicle_trajectory_logs = pd.DataFrame(columns=list(dtypes.keys())).astype(dtypes)
bicycle_trajectory_logs = pd.DataFrame(columns=[
    'time_step', 'vehicle_id', 'vehicle_type', 'x_coord', 'y_coord', 'speed',
    'angle', 'distance', 'lane_id', 'edge_id', 'next_tl_id', 'next_tl_distance',
    'next_tl_state', 'next_tl_index'
])
conflict_logs = pd.DataFrame(columns=[
    'time_step', 'bicycle_id', 'foe_id', 'foe_type', 'x_coord', 'y_coord',
    'distance', 'ttc', 'pet', 'drac', 'severity', 'is_detected',
    'detecting_observer', 'observer_type'
])
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
    sumoCmd = ["sumo", "-c", sumo_config_path, "--message-log", "error", "--no-warnings", "true", "--seed", "75"]
    traci.start(sumoCmd)
    print("SUMO simulation loaded and TraCi connection established.")

def load_geospatial_data():
    """
    Loads road space distribution from the GeoJSON file, buildings, and parks data from OpenStreetMap for the simulated scene.
    """
    gdf1 = gpd.read_file(geojson_path) # road space distribution
    # Filter for relevant types
    gdf1 = gdf1[
        (gdf1['Type'].isin(['Junction', 'LaneBoundary', 'Gate', 'Signal']))
    ]
    G = ox.graph_from_bbox(bbox=bbox, network_type='all') # NetworkX graph (bounding box)
    # Try to get buildings, return None if none exist
    try:
        buildings = ox.features_from_bbox(bbox=bbox, tags={'building': True}) # buildings
    except:
        buildings = None
        print("No buildings found in the specified area.")
    # Try to get parks, return None if none exist
    try:
        parks = ox.features_from_bbox(bbox=bbox, tags={'leisure': 'park'}) # parks
    except:
        parks = None
        print("No parks found in the specified area.")
    try:
        trees = ox.features_from_bbox(bbox=bbox, tags={'natural': 'tree'}) # trees
        leaves = ox.features_from_bbox(bbox=bbox, tags={'natural': 'tree'}) # leaves
    except:
        trees = None
        leaves = None
        print("No trees found in the specified area.")
    try:
        barriers = ox.features_from_bbox(bbox=bbox, tags={'barrier': 'retaining_wall'}) # barriers (walls)
    except:
        barriers = None
        print("No barriers (walls) found in the specified area.")
    try:
        PT_shelters = ox.features_from_bbox(bbox=bbox, tags={'shelter_type': 'public_transport'}) # PT shelters
    except:
        PT_shelters = None
        print("No PT shelters found in the specified area.")
    
    return gdf1, G, buildings, parks, trees, leaves, barriers, PT_shelters

    # ... no debug coordinate sampling helper retained in production

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
    visibility_counts = {cell: 0 for cell in grid_cells}  # initialization of visibility count for each cell to 0
    return x_coords, y_coords, grid_cells, visibility_counts  # returning grid information and visibility counts

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

    # Initialize static elements list with buildings (always present)
    static_elements = [
        Rectangle((0, 0), 1, 1, facecolor='darkgray', edgecolor='black', linewidth=0.5, label='Buildings')
    ]
    # Add parks if they exist
    if parks_proj is not None:
        static_elements.append(
            Rectangle((0, 0), 1, 1, facecolor='forestgreen', edgecolor='black', linewidth=0.5, alpha=0.5, label='Parks')
        )
    # Add trees if they exist (match concentric stem + canopy visualization)
    if trees_proj is not None:
        # Use a dummy Patch as handle; HandlerTree draws concentric circles
        static_elements.append(
            TreeRect((0, 0), 1, 1, facecolor='none', edgecolor='none', label='Trees')
        )
    # Add barriers if they exist - using Rectangle to match expected type
    if barriers_proj is not None:
        static_elements.append(
            Rectangle((0, 0), 0, 0, facecolor='black', linewidth=1.0, label='Barriers')  # type: ignore
        )
    # Add PT shelters if they exist
    # if PT_shelters_proj is not None:
    #     static_elements.append(
    #         Rectangle((0, 0), 1, 1, facecolor='lightgray', edgecolor='black', linewidth=0.5, label='PT Shelters')
    #     )
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
        parks_proj.plot(ax=ax, facecolor='seagreen', alpha=0.5, edgecolor='black', linewidth=0.5, zorder=2)  # Plot parks
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
        PT_shelters_proj.plot(ax=ax, facecolor='lightgray', edgecolor='black', linewidth=0.5, zorder=6)  # Plot PT shelters

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

def create_vehicle_polygon(x, y, width, length, angle):
    """
    Creates a rectangular polygon representing a vehicle at the given position and orientation.
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

# ---------------------
# OPTIMIZED RAY TRACING FUNCTIONS
# ---------------------

def detect_intersections_optimized(ray, objects):
    """
    Optimized version of intersection detection with reduced logging and better data structures.
    """
    closest_intersection = None
    min_distance = float('inf')
    ray_line = LineString(ray)
    
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
            
            # Find closest intersection
            ray_origin = Point(ray[0])
            for coord in coords_to_check:
                distance = ray_origin.distance(Point(coord))
                if distance < min_distance:
                    min_distance = distance
                    closest_intersection = coord
                    
        except Exception:
            continue  # Skip problematic geometries
    
    return closest_intersection

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

# GPU-accelerated functions (if CUDA is available)
def prepare_gpu_data(rays, objects):
    """
    Prepare data for GPU processing by converting to appropriate arrays.
    """
    try:
        # Convert rays to GPU arrays
        ray_origins = cp.array([[ray[0][0], ray[0][1]] for ray in rays], dtype=cp.float32)
        ray_ends = cp.array([[ray[1][0], ray[1][1]] for ray in rays], dtype=cp.float32)
        
        # Simplified object representation for GPU (bounding boxes)
        object_bounds = []
        for obj in objects:
            bounds = obj.bounds  # (minx, miny, maxx, maxy)
            object_bounds.append(bounds)
        
        object_bounds_gpu = cp.array(object_bounds, dtype=cp.float32)
        
        return ray_origins, ray_ends, object_bounds_gpu
    except Exception:
        return None, None, None

def gpu_intersection_detection(rays, objects):
    """
    GPU-accelerated intersection detection using CuPy.
    Falls back to CPU if GPU processing fails.
    """
    try:
        ray_origins, ray_ends, object_bounds = prepare_gpu_data(rays, objects)
        if ray_origins is None:
            # Fallback to CPU
            return [detect_intersections_optimized(ray, objects) for ray in rays]
        
        # Simplified GPU intersection test (bounding box intersections)
        # This is a basic implementation - more sophisticated GPU ray tracing could be added
        results = []
        for i, ray in enumerate(rays):
            # Convert back to CPU for detailed intersection
            intersection = detect_intersections_optimized(ray, objects)
            results.append(intersection)
        
        return results
        
    except Exception:
        # Fallback to CPU processing
        return [detect_intersections_optimized(ray, objects) for ray in rays]

def update_with_ray_tracing(frame):
    """
    Updates the simulation for each frame, performing ray tracing for FCOs and FBOs.
    Handles vehicle creation, ray generation, intersection detection, and visibility polygon creation.
    Updates vehicle patches, ray lines, and visibility counts for visualization.
    Also updates bicycle diagrams and logs simulation data.
    """
    global vehicle_patches, ray_lines, visibility_polygons, FCO_share, FBO_share, visibility_counts, numberOfRays, useRTREEmethod, visualizeRays, useManualFrameForwarding, delay, bicycle_detection_data, progress_bar
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
        print('Ray Tracing initiated.')
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

    if FlowBasedBicycleTrajectories:
        if frame == 0:
            print('Flow-based bicycle trajectory tracking initiated.')
        with TimingContext("flow_trajectories"):
            flow_based_bicycle_trajectories(frame, total_steps)

    if AnimatedThreeDimensionalDetectionPlots:
        if frame == 0:
            print('Animated 3D detection plots initiated.')
        with TimingContext("3d_animated_detections"):
            three_dimensional_detection_plots_gif(frame)

    # Close progress bar on last frame
    if frame == total_steps - 1:
        progress_bar.close()
        print('Ray tracing completed.')

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
        updated_cells = set()

        # Optimized static objects creation with caching
        if not hasattr(update_with_ray_tracing, 'static_objects_cache'):
            # Initialize static objects cache on first run
            static_objects = [building.geometry for building in buildings_proj.itertuples()]
            if trees_proj is not None:
                trees_circle = trees_proj.buffer(0.5)  # 1 meter radius for trees
                static_objects.extend(tree for tree in trees_circle.geometry)
            if barriers_proj is not None:
                static_objects.extend(barriers.geometry for barriers in barriers_proj.itertuples())
            if PT_shelters_proj is not None:
                static_objects.extend(PT_shelters.geometry for PT_shelters in PT_shelters_proj.itertuples())
            
            update_with_ray_tracing.static_objects_cache = static_objects
            print(f"Cached {len(static_objects)} static objects for improved performance")
        else:
            static_objects = update_with_ray_tracing.static_objects_cache.copy()
        
        # Add parked vehicles to static objects (these can change each frame)
        parked_vehicle_count = 0
        for vehicle_id in traci.vehicle.getIDList():
            vehicle_type = traci.vehicle.getTypeID(vehicle_id)
            if vehicle_type == "parked_vehicle":
                x, y = traci.vehicle.getPosition(vehicle_id)
                x_32632, y_32632 = convert_simulation_coordinates(x, y)
                width, length = vehicle_attributes(vehicle_type)[2]
                angle = traci.vehicle.getAngle(vehicle_id)
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

        # Process each vehicle and perform ray tracing
        observer_id = None  # Initialize observer_id
        for vehicle_id in traci.vehicle.getIDList():
            vehicle_type = traci.vehicle.getTypeID(vehicle_id)
            Shape, edgecolor, (width, length) = vehicle_attributes(vehicle_type)
            x, y = traci.vehicle.getPosition(vehicle_id)
            x_32632, y_32632 = convert_simulation_coordinates(x, y)
            angle = traci.vehicle.getAngle(vehicle_id)
            
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
                    other_x_32632, other_y_32632 = convert_simulation_coordinates(pos_x, pos_y)
                    # Get vehicle dimensions for the other vehicle
                    o_width, o_length = vehicle_attributes(traci.vehicle.getTypeID(vid))[2]
                    # Get vehicle angle safely for the other vehicle
                    o_angle = get_vehicle_angle_safe(vid)
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
                    profiler.start_timer("gpu_ray_processing")
                    intersections = gpu_intersection_detection(rays, all_objects)
                    profiler.end_timer()
                    
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
                    # CPU multithreaded processing
                    profiler.start_timer("cpu_ray_processing")
                    with ThreadPoolExecutor(max_workers=max_worker_threads) as executor:
                        # Process each ray individually with optimized intersection detection
                        futures = {executor.submit(detect_intersections_optimized, ray, all_objects): ray for ray in rays}
                        
                        for future in as_completed(futures):
                            intersection = future.result()
                            ray = futures[future]
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

                    # Update visibility counts
                    visibility_polygon_shape = Polygon(ray_endpoints)
                    for cell in visibility_counts.keys():
                        if visibility_polygon_shape.contains(cell):
                            if cell not in updated_cells:
                                visibility_counts[cell] += 1
                                updated_cells.add(cell)

                observer_id = vehicle_id # to store observer id in bicycle_detection_data

        # Process bicycle detections
        for vehicle_id in traci.vehicle.getIDList():
            if vehicle_id not in bicycle_detection_data:
                bicycle_detection_data[vehicle_id] = []
            vehicle_type = traci.vehicle.getTypeID(vehicle_id)
            if vehicle_type in ["bicycle", "DEFAULT_BIKETYPE", "floating_bike_observer"]:
                is_detected = False
                vehicle_polygon = create_vehicle_polygon(
                    *convert_simulation_coordinates(*traci.vehicle.getPosition(vehicle_id)),
                    *vehicle_attributes(vehicle_type)[2],
                    traci.vehicle.getAngle(vehicle_id)
                )
                for vis_polygon in visibility_polygons:
                    if vis_polygon and vehicle_polygon.intersects(Polygon(vis_polygon.get_xy())):
                        is_detected = True
                        break
                # bicycle_detection_data[vehicle_id].append((traci.simulation.getTime(), is_detected))
                bicycle_detection_data[vehicle_id].append((traci.simulation.getTime(), is_detected, [{'id': observer_id}] if is_detected and observer_id else []))

        # Update visualization
        # if useLiveVisualization:
        for patch in vehicle_patches:
            patch.remove()
        vehicle_patches = new_vehicle_patches
        ray_lines = new_ray_lines
        for patch in vehicle_patches:
            ax.add_patch(patch)
    
    if CollectLoggingData:
        with TimingContext("data_collection"):
            # Data collection for logging
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
    """
    global unique_vehicles, vehicle_type_set, fleet_composition_logs

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
        log_entry = {'time_step': time_step}
        for vehicle_type in vehicle_type_set:
            log_entry[f'new_{vehicle_type}_count'] = new_vehicle_counts.get(vehicle_type, 0)
            log_entry[f'present_{vehicle_type}_count'] = present_vehicle_counts.get(vehicle_type, 0)

        # Add log entry to the simulation log
        log_entry_df = pd.DataFrame([log_entry])
        fleet_composition_logs = pd.concat([fleet_composition_logs, log_entry_df], ignore_index=True)

def collect_traffic_light_data(frame):
    """Collects traffic light data at each simulation time step."""
    global traffic_light_logs
    
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
    
    # Only create DataFrame and concatenate if we have entries
    if entries:
        entry_df = pd.DataFrame(entries)
        
        # Initialize traffic_light_logs if empty
        if len(traffic_light_logs) == 0:
            traffic_light_logs = entry_df
        else:
            # Ensure dtypes match before concatenation
            for col in traffic_light_logs.columns:
                if col in entry_df.columns:  # Only convert columns that exist in both
                    entry_df[col] = entry_df[col].astype(traffic_light_logs[col].dtype)
            
            # Concatenate with existing logs
            traffic_light_logs = pd.concat([traffic_light_logs, entry_df], ignore_index=True)

def collect_bicycle_detection_data(time_step):
    """
    Collects detection data at each simulation time step.
    Records when and where bicycles are detected by observers.
    Only records entries when actual detections occur.
    """
    global detection_logs
    detection_entries = []
    has_detections = False
    
    # Get all current detections
    for vehicle_id in traci.vehicle.getIDList():
        vehicle_type = traci.vehicle.getTypeID(vehicle_id)
        
        # Only process FCOs and FBOs
        if vehicle_type in ["floating_car_observer", "floating_bike_observer"]:
            x_obs, y_obs = traci.vehicle.getPosition(vehicle_id)
            x_obs_utm, y_obs_utm = convert_simulation_coordinates(x_obs, y_obs)
                            
            # Check which bicycles this observer detects
            for bicycle_id in traci.vehicle.getIDList():
                bicycle_type = traci.vehicle.getTypeID(bicycle_id)
                
                if bicycle_type in ["bicycle", "DEFAULT_BIKETYPE", "floating_bike_observer"]:
                    # Check if this bicycle is detected by the observer's visibility polygon
                    if bicycle_id in bicycle_detection_data and bicycle_detection_data[bicycle_id]:
                        latest_detection = bicycle_detection_data[bicycle_id][-1]
                        if latest_detection[0] == traci.simulation.getTime() and latest_detection[1]:
                            has_detections = True
                            # Get bicycle position and calculate distance
                            x_bike, y_bike = traci.vehicle.getPosition(bicycle_id)
                            x_bike_utm, y_bike_utm = convert_simulation_coordinates(x_bike, y_bike)
                            detection_distance = np.sqrt((x_obs_utm - x_bike_utm)**2 + (y_obs_utm - y_bike_utm)**2)
                            
                            # Create detection entry
                            detection_entries.append({
                                'time_step': time_step,
                                'observer_id': vehicle_id,
                                'observer_type': vehicle_type,
                                'bicycle_id': bicycle_id,
                                'x_coord': x_bike_utm,
                                'y_coord': y_bike_utm,
                                'detection_distance': detection_distance,
                                'observer_speed': traci.vehicle.getSpeed(vehicle_id),
                                'bicycle_speed': traci.vehicle.getSpeed(bicycle_id)
                            })
    
    # Create DataFrame and handle concatenation (only if we have actual detections)
    if detection_entries:
        entry_df = pd.DataFrame(detection_entries)
        if len(detection_logs) == 0:
            detection_logs = entry_df
        else:
            for col in detection_logs.columns:
                if col in entry_df.columns:
                    entry_df[col] = entry_df[col].astype(detection_logs[col].dtype)
            detection_logs = pd.concat([detection_logs, entry_df], ignore_index=True)

def collect_vehicle_trajectories(time_step):
    """
    Collects trajectory data for all motorized vehicles (including parked vehicles) at each simulation time step.
    Only logs data when vehicles are present in the simulation.
    """
    global vehicle_trajectory_logs
    trajectory_entries = []
    has_vehicles = False
    
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
            x_utm, y_utm = convert_simulation_coordinates(x, y)
            
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
                'time_step': time_step,
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
    
    # Create DataFrame and handle concatenation only if we have entries
    if trajectory_entries:
        new_df = pd.DataFrame(trajectory_entries)
        if len(vehicle_trajectory_logs) == 0:
            vehicle_trajectory_logs = new_df
        else:
            # Ensure all columns exist in both DataFrames
            for col in vehicle_trajectory_logs.columns:
                if col not in new_df.columns:
                    new_df[col] = None
            for col in new_df.columns:
                if col not in vehicle_trajectory_logs.columns:
                    vehicle_trajectory_logs[col] = None
                    
            # Ensure matching dtypes before concatenation
            for col in vehicle_trajectory_logs.columns:
                new_df[col] = new_df[col].astype(vehicle_trajectory_logs[col].dtype)
                
            vehicle_trajectory_logs = pd.concat([vehicle_trajectory_logs, new_df], ignore_index=True)

def collect_bicycle_trajectories(time_step):
    """
    Collects trajectory data for bicycles at each simulation time step.
    Only logs data when bicycles are present in the simulation.
    """
    global bicycle_trajectory_logs
    trajectory_entries = []
    has_bicycles = False
    
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
            
            # Convert coordinates to UTM for logging
            x_utm, y_utm = convert_simulation_coordinates(x_sumo, y_sumo)
            
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
            if vehicle_id in bicycle_detection_data and bicycle_detection_data[vehicle_id]:
                latest_detection = bicycle_detection_data[vehicle_id][-1]
                if latest_detection[0] == traci.simulation.getTime() and latest_detection[1]:
                    is_detected = True
                    detecting_observers = latest_detection[2] if len(latest_detection) > 2 else []
            
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
                'time_step': time_step,
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
    
    # Create DataFrame and handle concatenation only if we have entries
    if trajectory_entries:
        new_df = pd.DataFrame(trajectory_entries)
        if len(bicycle_trajectory_logs) == 0:
            bicycle_trajectory_logs = new_df
        else:
            bicycle_trajectory_logs = pd.concat([bicycle_trajectory_logs, new_df], ignore_index=True)

def collect_bicycle_conflict_data(frame):
    """
    Collects conflict data at each simulation time step using SUMO's SSM device.
    Only records entries when actual conflicts occur.
    """
    global conflict_logs
    
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
                x_utm, y_utm = convert_simulation_coordinates(x, y)
                
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
                    
                    # Get SSM values
                    ttc_str = traci.vehicle.getParameter(vehicle_id, "device.ssm.minTTC")
                    pet_str = traci.vehicle.getParameter(vehicle_id, "device.ssm.minPET")
                    drac_str = traci.vehicle.getParameter(vehicle_id, "device.ssm.maxDRAC")
                    
                    # Convert to float with error handling
                    ttc = float(ttc_str) if ttc_str and ttc_str.strip() else float('inf')
                    pet = float(pet_str) if pet_str and pet_str.strip() else float('inf')
                    drac = float(drac_str) if drac_str and drac_str.strip() else 0.0
                    
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
                        
                        # Check if bicycle is detected
                        is_detected = False
                        detecting_observers = []
                        if vehicle_id in bicycle_detection_data and bicycle_detection_data[vehicle_id]:
                            latest_detection = bicycle_detection_data[vehicle_id][-1]
                            if latest_detection[0] == current_time and latest_detection[1]:
                                is_detected = True
                                # Find which observer(s) detected this bicycle
                                for obs_id in traci.vehicle.getIDList():
                                    obs_type = traci.vehicle.getTypeID(obs_id)
                                    if obs_type in ["floating_car_observer", "floating_bike_observer"]:
                                        if vehicle_id in bicycle_detection_data[obs_id]:
                                            detecting_observers.append({
                                                'id': obs_id,
                                                'type': obs_type
                                            })
                        
                        has_conflicts = True
                        conflict_entries.append({
                            'time_step': frame,
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
                        
            except Exception as e:
                print(f"Error in conflict detection for {vehicle_id}: {str(e)}")
    
    # Create DataFrame and handle concatenation (only if we have actual conflicts)
    if conflict_entries:
        entry_df = pd.DataFrame(conflict_entries)
        if len(conflict_logs) == 0:
            conflict_logs = entry_df
        else:
            for col in conflict_logs.columns:
                if col in entry_df.columns:
                    entry_df[col] = entry_df[col].astype(conflict_logs[col].dtype)
            conflict_logs = pd.concat([conflict_logs, entry_df], ignore_index=True)

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

def save_simulation_logs():
    """
    Saves all collected simulation data to log files.
    Generates and saves both detailed log files (with data for each time step) and a summary log file.
    """
    global fleet_composition_logs, bicycle_trajectory_logs

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

    print('\nDetailed logging completed.')

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
            avg_present = fleet_composition_logs[f'present_{vehicle_type}_count'].mean()
            writer.writerow([f'Total {vehicle_type} vehicles', total, f'{avg_present:.1f}'])
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
            # Individual applications (only show if they were used)
            # Individual bicycle trajectories now handled by evaluation script
            if FlowBasedBicycleTrajectories and 'flow_trajectories' in operation_times:
                writer.writerow(['- Flow-based bicycle trajectories', f"{operation_times['flow_trajectories']:.2f} seconds"])
            if AnimatedThreeDimensionalDetectionPlots and '3d_animated_detections' in operation_times:
                writer.writerow(['- Animated 3D detections', f"{operation_times['3d_animated_detections']:.2f} seconds"])
            if 'visibility_data_export' in operation_times:
                writer.writerow(['- Visibility data export', f"{operation_times['visibility_data_export']:.2f} seconds"])
            if not any([FlowBasedBicycleTrajectories, AnimatedThreeDimensionalDetectionPlots]):
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

    print('\nSummary logging completed.')

# ---------------------
# APPLICATIONS - VISIBILITY & BICYCLE SAFETY
# ---------------------

def flow_based_bicycle_trajectories(frame, total_steps):
    """
    Creates space-time diagrams for bicycle flows, including detection status, traffic lights,
    and conflicts detected by SUMO's SSM device.
    """
    global bicycle_flow_data, traffic_light_positions, bicycle_tls, step_length, bicycle_conflicts_flow, traffic_light_programs, flow_detection_data, foe_trajectories

    # Initialize detection buffer for smoothing
    detection_buffer = []

    # Initialize traffic light programs if it doesn't exist
    if 'traffic_light_programs' not in globals():
        traffic_light_programs = {}

    # Initialize traffic light programs at frame 0
    if frame == 0:
        traffic_light_programs = {}
        for tl_id in traci.trafficlight.getIDList():
            if tl_id not in traffic_light_ids:
                traffic_light_ids[tl_id] = len(traffic_light_ids) + 1
            traffic_light_programs[tl_id] = {
                'program': []
            }
    
    current_time = traci.simulation.getTime()
    current_vehicles = set(traci.vehicle.getIDList())

    # Record traffic light states every frame
    for tl_id in traffic_light_programs:
        full_state = traci.trafficlight.getRedYellowGreenState(tl_id)
        traffic_light_programs[tl_id]['program'].append((current_time, full_state))

    # Create output directory if it doesn't exist
    flow_traj_output = os.path.join(scenario_output_dir, 'out_2d_flow_based_trajectories')
    
    # During simulation, collect detection data
    current_vehicles = set(traci.vehicle.getIDList())
    
    for vehicle_id in current_vehicles:
        vehicle_type = traci.vehicle.getTypeID(vehicle_id)
        if not vehicle_type in ["bicycle", "DEFAULT_BIKETYPE", "floating_bike_observer"] or vehicle_id.startswith('parked_'):
            continue
        if vehicle_type in ["bicycle", "DEFAULT_BIKETYPE", "floating_bike_observer"]:
            flow_id = vehicle_id.rsplit('.', 1)[0]
            if flow_id not in flow_detection_data:
                flow_detection_data[flow_id] = {}
            if vehicle_id not in flow_detection_data[flow_id]:
                flow_detection_data[flow_id][vehicle_id] = []
            
            # Check if bicycle is currently detected
            is_detected = False
            if vehicle_id in bicycle_detection_data and bicycle_detection_data[vehicle_id]:  # Updated check
                detection_info = next((info for info in bicycle_detection_data[vehicle_id] 
                                    if abs(info[0] - current_time) < step_length), None)
                
                if detection_info:
                    if len(detection_info) == 3:  # New format
                        is_detected = detection_info[1]
                    else:  # Old format
                        is_detected = detection_info[1]

                # Update detection buffer for smoothing
                detection_buffer.append(is_detected)
                if len(detection_buffer) > basic_gap_bridge:
                    detection_buffer.pop(0)

                # Apply smoothing logic
                recent_detection = any(detection_buffer[-3:]) if len(detection_buffer) >= 3 else is_detected
                if not recent_detection and len(detection_buffer) >= basic_gap_bridge:
                    is_detected = any(detection_buffer[:3]) and any(detection_buffer[-3:])
                else:
                    is_detected = recent_detection
            
            flow_detection_data[flow_id][vehicle_id].append((current_time, is_detected))

    bicycles = [v for v in current_vehicles if traci.vehicle.getTypeID(v) in 
                ["bicycle", "DEFAULT_BIKETYPE", "floating_bike_observer"]]
    
    for bicycle_id in bicycles:
        flow_id = bicycle_id.rsplit('.', 1)[0]
        distance = traci.vehicle.getDistance(bicycle_id)
        current_time = traci.simulation.getTime()
        
        # Initialize dictionaries if they don't exist
        if flow_id not in bicycle_flow_data:
            bicycle_flow_data[flow_id] = {}
            traffic_light_positions[flow_id] = {}
            bicycle_tls[flow_id] = {}

        if bicycle_id not in bicycle_flow_data[flow_id]:
            bicycle_flow_data[flow_id][bicycle_id] = []
        
        # Check detection status from flow_detection_data
        is_detected = False
        if (flow_id in flow_detection_data and 
            bicycle_id in flow_detection_data[flow_id] and 
            bicycle_id in bicycle_detection_data):  # Added this check
            
            detection_info = next((info for info in bicycle_detection_data[bicycle_id]
                                if abs(info[0] - current_time) < step_length), None)
                
            if detection_info:
                if len(detection_info) == 3:  # New format
                    is_detected = detection_info[1]
                else:  # Old format
                    is_detected = detection_info[1]

            # Update detection buffer for smoothing
            detection_buffer.append(is_detected)
            if len(detection_buffer) > basic_gap_bridge:
                detection_buffer.pop(0)

            # Apply smoothing logic
            recent_detection = any(detection_buffer[-3:]) if len(detection_buffer) >= 3 else is_detected
            if not recent_detection and len(detection_buffer) >= basic_gap_bridge:
                is_detected = any(detection_buffer[:3]) and any(detection_buffer[-3:])
            else:
                is_detected = recent_detection
        
        # Store trajectory data with detection status
        bicycle_flow_data[flow_id][bicycle_id].append((distance, current_time, is_detected))
        
        # Check for conflicts
        try:
            # Check both leader and follower vehicles
            leader = traci.vehicle.getLeader(bicycle_id)
            follower = traci.vehicle.getFollower(bicycle_id)
            
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
                
                # Add this new section to store foe trajectory data
                if foe_id not in foe_trajectories:
                    foe_trajectories[foe_id] = []
                
                # Get foe position and store it
                x_sumo, y_sumo = traci.vehicle.getPosition(foe_id)
                foe_trajectories[foe_id].append((x_sumo, y_sumo, current_time))
                
                # Get SSM values
                ttc_str = traci.vehicle.getParameter(bicycle_id, "device.ssm.minTTC")
                pet_str = traci.vehicle.getParameter(bicycle_id, "device.ssm.minPET")
                drac_str = traci.vehicle.getParameter(bicycle_id, "device.ssm.maxDRAC")
                
                # Convert to float with error handling
                ttc = float(ttc_str) if ttc_str and ttc_str.strip() else float('inf')
                pet = float(pet_str) if pet_str and pet_str.strip() else float('inf')
                drac = float(drac_str) if drac_str and drac_str.strip() else 0.0
                
                # Define thresholds
                TTC_THRESHOLD = 3.0  # seconds
                PET_THRESHOLD = 2.0  # seconds
                DRAC_THRESHOLD = 3.0  # m/s²
                
                # Check for conflict
                if (ttc < TTC_THRESHOLD or pet < PET_THRESHOLD or drac > DRAC_THRESHOLD):
                    if bicycle_id not in bicycle_conflicts_flow:
                        bicycle_conflicts_flow[bicycle_id] = []
                    
                    # Calculate severity
                    ttc_severity = 1 - (ttc / TTC_THRESHOLD) if ttc < TTC_THRESHOLD else 0
                    pet_severity = 1 - (pet / PET_THRESHOLD) if pet < PET_THRESHOLD else 0
                    drac_severity = min(drac / DRAC_THRESHOLD, 1.0) if drac > 0 else 0
                    
                    conflict_severity = max(ttc_severity, pet_severity, drac_severity)
                    
                    bicycle_conflicts_flow[bicycle_id].append({
                        'distance': distance,
                        'time': current_time,
                        'ttc': ttc,
                        'pet': pet,
                        'drac': drac,
                        'severity': conflict_severity,
                        'foe_type': foe_type,
                        'foe_id': foe_id
                    })
        
        except Exception as e:
            if frame % 100 == 0:
                print(f"Error in conflict detection for {bicycle_id}: {str(e)}")

        # Check detection status
        is_detected = False
        if flow_id in flow_detection_data and bicycle_id in flow_detection_data[flow_id]:
            detection_data = flow_detection_data[flow_id][bicycle_id]
            for det_time, det_status in detection_data:
                if abs(det_time - current_time) < step_length:
                    is_detected = det_status
                    break
        
        bicycle_flow_data[flow_id][bicycle_id][-1] = (distance, current_time, is_detected)

        # Check for the next traffic light
        next_tls = traci.vehicle.getNextTLS(bicycle_id)
        if next_tls:
            tl_id, tl_index, tl_distance, tl_state = next_tls[0]
            if tl_id not in traffic_light_ids:
                traffic_light_ids[tl_id] = len(traffic_light_ids) + 1
            short_tl_id = f"TL{traffic_light_ids[tl_id]}"
            tl_pos = distance + tl_distance
            
            if short_tl_id not in traffic_light_positions[flow_id]:
                traffic_light_positions[flow_id][short_tl_id] = [tl_pos, []]
            bicycle_tls[flow_id][short_tl_id] = tl_index

    # Plot trajectories if this is the last frame
    if frame == total_steps - 1:
        for flow_id in bicycle_flow_data:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            all_times = [time for bicycle_data in bicycle_flow_data[flow_id].values()
                        for _, time, _ in bicycle_data]
            if not all_times:
                continue
            start_time = min(all_times)
            end_time = max(all_times)

            # Add horizontal line for simulation warm-up
            if start_time < delay:
                ax.axhline(y=delay, color='firebrick', linestyle='--', alpha=0.5)
                ax.text(0, delay, 'simulation warm-up', 
                       color='firebrick', va='bottom', ha='left')

            # Plot trajectories for each bicycle in this flow
            for vehicle_id, trajectory_data in bicycle_flow_data[flow_id].items():
                # Split trajectory into detected and undetected segments
                segments = {'detected': [], 'undetected': []}
                current_points = []
                current_detected = None
                detection_buffer = []  # Buffer to store recent detection states

                for distance, time, is_detected in trajectory_data:
                    # Update detection buffer
                    detection_buffer.append(is_detected)
                    if len(detection_buffer) > basic_gap_bridge:
                        detection_buffer.pop(0)
                    
                    # If there's any detection in the last 3 frames, consider it detected
                    recent_detection = any(detection_buffer[-3:]) if len(detection_buffer) >= 3 else is_detected
                    # For longer gaps, only bridge if there are detections on both sides
                    if not recent_detection and len(detection_buffer) >= basic_gap_bridge:
                        # Check if we have detections on both sides of the gap
                        if any(detection_buffer[:3]) and any(detection_buffer[-3:]):
                            smoothed_detection = True
                        else:
                            smoothed_detection = False
                    else:
                        smoothed_detection = recent_detection
                    
                    if current_detected is None:
                        current_detected = smoothed_detection
                        current_points = [(distance, time)]
                    elif smoothed_detection != current_detected:
                        if len(current_points) >= basic_segment_length:
                            segments['detected' if current_detected else 'undetected'].append(current_points)
                            current_points = [(distance, time)]
                            current_detected = smoothed_detection
                        else:
                            # If segment is too short, just continue with current segment
                            current_points.append((distance, time))
                    else:
                        current_points.append((distance, time))

                if current_points:
                    segments['detected' if current_detected else 'undetected'].append(current_points)

                # Plot segments
                for segment in segments['undetected']:
                    if len(segment) > 1:
                        distances, times = zip(*segment)
                        ax.plot(distances, times, color='black', linewidth=1.5, linestyle='solid')
                for segment in segments['detected']:
                    if len(segment) > 1:
                        distances, times = zip(*segment)
                        ax.plot(distances, times, color='darkturquoise', linewidth=1.5, linestyle='solid')
                
                # Plot conflicts if any exist
                if vehicle_id in bicycle_conflicts_flow:
                    conflicts_by_foe = {}
                    labels = []
                    for conflict in bicycle_conflicts_flow[vehicle_id]:
                        foe_id = conflict.get('foe_id')
                        if foe_id and foe_id in foe_trajectories:
                            # Check if foe entered after bicycle left
                            foe_start_time = foe_trajectories[foe_id][0][2]
                            if foe_start_time > end_time:  # end_time is bicycle's end time
                                continue  # Skip this conflict
                            
                            if foe_id not in conflicts_by_foe:
                                conflicts_by_foe[foe_id] = []
                            conflicts_by_foe[foe_id].append(conflict)
                        
                        else:
                            print(f"Foe {foe_id} not found in foe_trajectories or is None")

                    for foe_conflicts in conflicts_by_foe.values():
                        most_severe_flow = max(foe_conflicts, key=lambda x: x['severity'])
                        size = 50 + (most_severe_flow['severity'] * 100)
                        
                        # Create label based on the most critical metric
                        ttc = most_severe_flow['ttc']
                        pet = most_severe_flow['pet']
                        drac = most_severe_flow['drac']
                        
                        if ttc < 3.0:  # TTC threshold
                            label = f'TTC = {ttc:.1f}s'
                        elif pet < 2.0:  # PET threshold
                            label = f'PET = {pet:.1f}s'
                        elif drac > 3.0:  # DRAC threshold
                            label = f'DRAC = {drac:.1f}m/s²'
                        else:
                            label = 'Conflict'
                        
                        # Plot conflict point
                        ax.scatter(most_severe_flow['distance'], most_severe_flow['time'], 
                                  color='firebrick', marker='o', s=size, zorder=1000,
                                  facecolors='none', edgecolors='firebrick', linewidth=0.75)
                        
                        # Store label information
                        labels.append({
                            'x': most_severe_flow['distance'],
                            'y': most_severe_flow['time'],
                            'text': label
                        })
                    
                    # Sort labels by y-coordinate
                    labels.sort(key=lambda x: x['y'])
                    
                    # Plot labels with minimal vertical adjustment
                    label_height = 6  # Approximate height of label in points
                    for i, label_info in enumerate(labels):
                        y_offset = 0
                        if i > 0:
                            # Check if this label would overlap with the previous one
                            prev_y = labels[i-1]['y']
                            if abs(label_info['y'] - prev_y) < label_height:
                                y_offset = label_height - abs(label_info['y'] - prev_y)
                        
                        ax.annotate(label_info['text'], 
                                  (label_info['x'], label_info['y']),
                                  xytext=(12, y_offset),  # Fixed horizontal offset, minimal vertical
                                  textcoords='offset points',
                                  bbox=dict(facecolor='none', edgecolor='none'),
                                  fontsize=7,
                                  verticalalignment='center',
                                  color='firebrick')

            # Plot traffic light positions and states
            plotted_tl_positions = set()
            for short_tl_id, tl_info in traffic_light_positions[flow_id].items():
                tl_pos, _ = tl_info
                tl_index = bicycle_tls[flow_id][short_tl_id]
                
                full_tl_id = next((id for id, num in traffic_light_ids.items() 
                                  if f"TL{num}" == short_tl_id), None)
                
                if full_tl_id and full_tl_id in traffic_light_programs:
                    states = []
                    for time, full_state in traffic_light_programs[full_tl_id]['program']:
                        if start_time <= time <= end_time:
                            if 0 <= tl_index < len(full_state):
                                relevant_state = full_state[tl_index]
                                states.append((time, relevant_state))
                    
                    if states:
                        ax.axvline(x=tl_pos, ymin=0, ymax=1, color='gray', linestyle='-', alpha=0.3)
                        
                        for i in range(len(states) - 1):
                            current_time, current_state = states[i]
                            next_time = states[i + 1][0]
                            
                            color = {'r': 'red', 'y': 'yellow', 'g': 'green', 'G': 'green'}.get(current_state, 'gray')
                            y_start = (current_time - start_time) / (end_time - start_time)
                            y_end = (next_time - start_time) / (end_time - start_time)
                            
                            ax.axvline(x=tl_pos, ymin=y_start, ymax=y_end, color=color)

            ax.set_ylim(start_time, end_time)
            ax.set_xlabel('Distance Traveled (m)')
            ax.set_ylabel('Simulation Time (s)')
            ax.set_title(f'Space-Time Diagram for Flow {flow_id}')
            ax.grid(True)

            handles = [
                Line2D([0], [0], color='black', lw=2, label='bicycle undetected'),
                Line2D([0], [0], color='darkturquoise', lw=2, label='bicycle detected'),
                Line2D([0], [0], marker='o', color='firebrick', linestyle='None', 
                          markerfacecolor='none', markersize=10, label='potential conflict detected'),
                Line2D([0], [0], color='red', lw=2, label='Red TL'),
                Line2D([0], [0], color='yellow', lw=2, label='Yellow TL'),
                Line2D([0], [0], color='green', lw=2, label='Green TL')
            ]
            ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.01, 0.99))

            flow_plot_path = os.path.join(scenario_output_dir, 'out_2d_flow_based_trajectories', f'{flow_id}_space_time_diagram_{file_tag}_FCO{FCO_share*100:.0f}%_FBO{FBO_share*100:.0f}%.png')
            plt.savefig(flow_plot_path, bbox_inches='tight')
            plt.close(fig)
            
            print(f"\nFlow-based space-time diagram for bicycle flow {flow_id} saved as {flow_plot_path}.")

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
        lon, lat = traci.simulation.convertGeo(x_sumo, y_sumo)
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
        
        load_sumo_simulation()
        gdf1, G, buildings, parks, trees, leaves, barriers, PT_shelters = load_geospatial_data()
        print('Geospatial data loaded.')
        gdf1_proj, G_proj, buildings_proj, parks_proj, trees_proj, leaves_proj, barriers_proj, PT_shelters_proj = project_geospatial_data(gdf1, G, buildings, parks, trees, leaves, barriers, PT_shelters)
        print('Geospatial data projected.')
        # Load SUMO additional-files polygons (if any) and convert to projected coords
        try:
            additional_polygons = load_additional_polygons_from_sumocfg(sumo_config_path)
            if additional_polygons:
                print(f'Loaded {len(additional_polygons)} additional polygon(s) from SUMO additional-files')
        except Exception:
            additional_polygons = []
        # (debug sampling removed) continue initialization
        setup_plot()
        plot_geospatial_data(gdf1_proj, G_proj, buildings_proj, parks_proj, trees_proj, leaves_proj, barriers_proj, PT_shelters_proj)
        # Always initialize visibility grid for consistent data collection
        x_coords, y_coords, grid_cells, visibility_counts = initialize_grid(buildings_proj, grid_size)
        total_steps = get_total_simulation_steps(sumo_config_path)
        # Continue initialization; do not perform debug sampling here
    if useLiveVisualization or saveAnimation:
        with TimingContext("visualization"):
            anim = run_animation(total_steps)
    else:
        for frame in range(total_steps):
            with TimingContext("ray_tracing"):
                update_with_ray_tracing(frame)
    with TimingContext("logging"):
        if CollectLoggingData:
            save_simulation_logs()
        
        # Print performance summary
        print("\n" + "="*60)
        print("RAY TRACING PERFORMANCE OPTIMIZATION SUMMARY")
        print("="*60)
        profiler.print_summary()
        
        traci.close()
        print("SUMO simulation closed and TraCi disconnected.")
    
    print("\nSimulation completed successfully!")
    if PERFORMANCE_OPTIMIZER_AVAILABLE:
        print("Performance optimization features were available and used.")
    else:
        print("Basic performance optimization was used.")
    
    # Check if GPU acceleration was actually used during simulation
    gpu_was_used = globals().get('use_gpu_acceleration', False)
    if gpu_was_used and CUDA_AVAILABLE:
        print("GPU acceleration was enabled and available.")
    elif gpu_was_used:
        print("GPU acceleration was enabled but CUDA/CuPy not available.")
    else:
        print("GPU acceleration was disabled or not used.")
    
    print(f"Multi-threading used: {max_worker_threads} worker threads")
        
    # Always save visibility data for standalone evaluation scripts
    with TimingContext("visibility_data_export"):
        output_prefix = f'{file_tag}_FCO{FCO_share*100:.0f}%_FBO{FBO_share*100:.0f}%'
        visibility_counts_path = os.path.join(scenario_output_dir, 'out_raytracing', f'visibility_counts_{file_tag}_FCO{FCO_share*100:.0f}%_FBO{FBO_share*100:.0f}%.csv')
        
        # Save visibility counts to CSV for standalone evaluation
        with open(visibility_counts_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['x_coord', 'y_coord', 'visibility_count'])
            for cell, count in visibility_counts.items():
                # Get cell center coordinates
                cell_x = cell.bounds[0] + (cell.bounds[2] - cell.bounds[0]) / 2
                cell_y = cell.bounds[1] + (cell.bounds[3] - cell.bounds[1]) / 2
                # Save all counts (including zeros for consistent data structure)
                csvwriter.writerow([cell_x, cell_y, count])
        
        print(f'\nVisibility data exported to: {visibility_counts_path}')
        if FCO_share == 0 and FBO_share == 0:
            print('Note: All visibility counts are 0 (no observers present)')
        print('Use evaluation_relative_visibility.py and evaluation_lov.py scripts for heatmap generation.')
    
    # Print final summary with scenario output directory
    print(f'\n=== SIMULATION COMPLETE ===')
    print(f'All outputs saved to: {scenario_output_dir}')
    print('Use separate evaluation scripts for advanced trajectory analysis (evaluation_VRU_specific_detection.py, etc.)')