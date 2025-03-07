import os
import osmnx as ox
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import libsumo as traci
from shapely.geometry import Point, box, Polygon, MultiPolygon
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Rectangle, Polygon as MatPolygon
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
import shapely
import imageio.v2 as imageio
import glob
import math
import SumoNetVis
from adjustText import adjust_text
import datetime
import time
import psutil
from collections import defaultdict
import subprocess
from tqdm import tqdm
import contextlib
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

# ---------------------
# CONFIGURATION
# ---------------------

# General Settings:
useLiveVisualization = True # Live Visualization of Ray Tracing
visualizeRays = True # Visualize rays additionaly to the visibility polygon
useManualFrameForwarding = False # Visualization of each frame, manual input necessary to forward the visualization
saveAnimation = False # Save the animation (currently not compatible with live visualization)

# Bounding Box Settings:
# north, south, east, west = 48.146200, 48.144400, 11.580650, 11.577150 # OJ-ITS
north, south, east, west = 48.178492, 48.176557, 11.587396, 11.583802 # MTh Bene
bbox = (north, south, east, west)

# Path Settings:
base_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(base_dir)
# sumo_config_path = os.path.join(parent_dir, 'Additionals', 'OJ_T-ITS', 'small_example_signalized.sumocfg') # Path to SUMO config-file
sumo_config_path = os.path.join(parent_dir, 'Additionals', 'Bene_MTh', 'osm.sumocfg') # Path to SUMO config-file
# sumo_config_path = os.path.join(parent_dir, 'SUMO_example', 'SUMO_example.sumocfg') # LoV example (TRB 2025)
geojson_path = os.path.join(parent_dir, 'SUMO_example', 'SUMO_example.geojson') # Path to GEOjson file
file_tag = 'test' # File tag will be included in filenames of output files (additionally to the FCO and FBO shares) - e.g. '..._{file_tag}_FCO{FCO_share*100}%_FBO{FBO_share*100}%' --> '..._small_FCO50%_FBO0%'

# FCO / FBO Settings:
FCO_share = 1 # Penetration rate of FCOs
FBO_share = 0 # Penetration rate of FBOs
numberOfRays = 360 # Number of rays emerging from the observer vehicle's (FCO/FBO) center point
radius = 30 # Radius of the rays emerging from the observer vehicle's (FCO/FBO) center point
min_segment_length = 3  # Base minimum segment length (for bicycle trajectory analysis)
max_gap_bridge = 10  # Maximum number of undetected frames to bridge between detected segments (for bicycle trajectory analysis)

# Warm Up Settings:
delay = 0 #warm-up time in seconds (during this time in the beginning of the simulation, no ray tracing is performed)

# Grid Map Settings:
grid_size =  0.5 # Grid Size for Heat Map Visualization (the smaller the grid size, the higher the resolution)

# Application Settings:
RelativeVisibility = False # Generate relative visibility heatmaps
LoVheatmap = False # Generate LoV heatmap
IndividualBicycleTrajectories = False # Generate 2D space-time diagrams of bicycle trajectories (individual trajectory plots)
ImportantTrajectories = False # For now only testing purposes
FlowBasedBicycleTrajectories = False # Generate 2D space-time diagrams of bicycle trajectories (flow-based trajectory plots)
ThreeDimensionalConflictPlots = False # Generate 3D space-time diagrams of bicycle trajectories (3D conflict plots with foe vehicle trajectories)
ThreeDimensionalDetectionPlots = False # Generate 3D space-time diagrams of bicycle trajectories (3D detection plots with observer vehicles' trajectories)
AnimatedThreeDimensionalConflictPlots = False # Generate animated 3D space-time diagrams of bicycle trajectories (3D conflict plots with foe vehicles' trajectories)
AnimatedThreeDimensionalDetectionPlots = False # Generate animated 3D space-time diagrams of bicycle trajectories (3D detection plots with observer vehicles' trajectories)

# Evaluation Settings:
CollectLoggingData = False # Collect logging data for further analysis and evaluation
BicycleSafetyEvaluation = False # Evaluate bicycle safety (CollectLoggingData must be set to True)

# ---------------------

# General Visualization Settings
fig, ax = plt.subplots(figsize=(12, 8))

# 3D Visualization Settings
bicycle_trajectories = {}
flow_ids = set()
transformer = None
fig_3d = None
ax_3d = None

# Projection Settings
proj_from = pyproj.Proj('epsg:4326')   # Source projection: WGS 84
proj_to = pyproj.Proj('epsg:32632')    # Target projection: UTM zone 32N
project = pyproj.Transformer.from_proj(proj_from, proj_to, always_xy=True).transform

# Initialization of empty lists
vehicle_patches = []
ray_lines = []
visibility_polygons = []

# Initialization of empty dictionaries
bicycle_trajectory_data = {}
bicycle_flow_data = {}

# Logging Settings
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
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
    'time_step', 'traffic_light_id', 'phase', 'phase_duration', 'remaining_duration',
    'total_queue_length', 'vehicles_stopped', 'average_waiting_time', 'vehicles_by_type'
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
vehicle_trajectory_logs = pd.DataFrame(columns=dtypes.keys()).astype(dtypes)
bicycle_trajectory_logs = pd.DataFrame(columns=[
    'time_step', 'vehicle_id', 'vehicle_type', 'x_coord', 'y_coord', 'speed',
    'angle', 'distance', 'lane_id', 'edge_id'
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
    if RelativeVisibility:
        x_min, y_min, x_max, y_max = buildings_proj.total_bounds  # bounding box
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
        begin = float(time.find('begin').get('value', begin))  # getting 'begin' value, use default if not found
        end = float(time.find('end').get('value', end))  # getting 'end' value, use default if not found
        step_length = float(time.find('step-length').get('value', step_length))  # getting 'step-length', use default if not found
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
        step_length = float(time.find('step-length').get('value', step_length))  # getting 'step-length' value, use default if not found
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
    
    # Initialize static elements list with buildings (always present)
    static_elements = [
        Rectangle((0, 0), 1, 1, facecolor='darkgray', edgecolor='black', linewidth=0.5, label='Buildings')
    ]
    # Add parks if they exist
    if parks_proj is not None:
        static_elements.append(
            Rectangle((0, 0), 1, 1, facecolor='forestgreen', edgecolor='black', linewidth=0.5, alpha=0.5, label='Parks')
        )
    # Add barriers if they exist
    if barriers_proj is not None:
        static_elements.append(
            Line2D([0], [0], color='black', linewidth=1.0, label='Barriers')
        )
    # Add PT shelters if they exist
    if PT_shelters_proj is not None:
        static_elements.append(
            Rectangle((0, 0), 1, 1, facecolor='lightgray', edgecolor='black', linewidth=0.5, label='PT Shelters')
        )
    # Create vehicle type elements based on FCO and FBO presence
    if FCO_share > 0 and FBO_share > 0:
        vehicle_elements = [
            Rectangle((0, 0), 0.36, 1, facecolor='none', edgecolor='black', label='Passenger Car'),
            Rectangle((0, 0), 0.13, 0.32, facecolor='none', edgecolor='blue', label='Bicycle'),
            Rectangle((0, 0), 0.36, 1, facecolor='none', edgecolor='red', label='FCO'),
            Rectangle((0, 0), 0.13, 0.32, facecolor='none', edgecolor='orange', label='FBO')
        ]
    elif FCO_share > 0 and FBO_share == 0:
        vehicle_elements = [
            Rectangle((0, 0), 0.36, 1, facecolor='none', edgecolor='black', label='Passenger Car'),
            Rectangle((0, 0), 0.13, 0.32, facecolor='none', edgecolor='blue', label='Bicycle'),
            Rectangle((0, 0), 0.36, 1, facecolor='none', edgecolor='red', label='FCO')
        ]
    elif FCO_share == 0 and FBO_share > 0:
        vehicle_elements = [
            Rectangle((0, 0), 0.36, 1, facecolor='none', edgecolor='black', label='Passenger Car'),
            Rectangle((0, 0), 0.13, 0.32, facecolor='none', edgecolor='blue', label='Bicycle'),
            Rectangle((0, 0), 0.13, 0.32, facecolor='none', edgecolor='orange', label='FBO')
        ]
    else:
        vehicle_elements = [
            Rectangle((0, 0), 0.36, 1, facecolor='none', edgecolor='black', label='Passenger Car'),
            Rectangle((0, 0), 0.13, 0.32, facecolor='none', edgecolor='blue', label='Bicycle')
        ]
    
    # Combine static and vehicle elements
    legend_handles = static_elements + vehicle_elements
    
    ax.legend(handles=legend_handles, loc='upper right', fontsize=7)

    # Add initial warm-up text box
    ax.warm_up_text = ax.text(0.02, 0.98, f'Warm-up phase\nRemaining: {delay}s', 
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

def convert_simulation_coordinates(x, y):
    """
    Converts coordinates from SUMO's internal system to UTM zone 32N.
    """
    lon, lat = traci.simulation.convertGeo(x, y)  # Convert SUMO coordinates to longitude and latitude
    x_32632, y_32632 = project(lon, lat)  # Project longitude and latitude to UTM zone 32N
    return x_32632, y_32632  # Return the converted coordinates

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

    stepLength = get_step_length(sumo_config_path)
    step_start_time = time.time() # Start time for performance metrics

    with TimingContext("simulation_step"):
        traci.simulationStep()  # Advance the simulation by one step
        if useManualFrameForwarding:
            input("Press Enter to continue...")  # Wait for user input if manual forwarding is enabled

    # Set vehicle types (FCO and FBO) based on probability
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

    if IndividualBicycleTrajectories:
        if frame == 0:
            print('Individual bicycle trajectory tracking initiated.')
        with TimingContext("individual_trajectories"):
            individual_bicycle_trajectories(frame)
    if FlowBasedBicycleTrajectories:
        if frame == 0:
            print('Flow-based bicycle trajectory tracking initiated.')
        with TimingContext("flow_trajectories"):
            flow_based_bicycle_trajectories(frame, total_steps)
    if ThreeDimensionalConflictPlots:
        if frame == 0:
            print('3D bicycle conflict plots initiated.')
        with TimingContext("3d_conflicts"):
            three_dimensional_conflict_plots(frame)
    if ThreeDimensionalDetectionPlots:
        if frame == 0:
            print('3D bicycle detection plots initiated.')
        with TimingContext("3d_detections"):
            three_dimensional_detection_plots(frame)
    if AnimatedThreeDimensionalConflictPlots:
        if frame == 0:
            print('3D bicycle conflict tracking and animation initiated.')
        with TimingContext("3d_animated_conflicts"):
            three_dimensional_conflict_plots_gif(frame)
    if AnimatedThreeDimensionalDetectionPlots:
        if frame == 0:
            print('3D bicycle detection tracking and animation initiated.')
        with TimingContext("3d_animated_detections"):
            three_dimensional_detection_plots_gif(frame)
    if ImportantTrajectories:
        if frame == 0:
            print('Important trajectories initiated:')
        with TimingContext("important_trajectories"):
            important_trajectory_parts(frame)

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
        new_vehicle_patches = []
        new_ray_lines = []
        updated_cells = set()

    # Create static objects
    static_objects = [building.geometry for building in buildings_proj.itertuples()]
    if trees_proj is not None:
        trees_circle = trees_proj.buffer(0.5)  # 1 meter radius for trees
        static_objects.extend(tree for tree in trees_circle.geometry)
    if barriers_proj is not None:
        static_objects.extend(barriers.geometry for barriers in barriers_proj.itertuples())
    if PT_shelters_proj is not None:
        static_objects.extend(PT_shelters.geometry for PT_shelters in PT_shelters_proj.itertuples())
        for vehicle_id in traci.vehicle.getIDList():
            vehicle_type = traci.vehicle.getTypeID(vehicle_id)
            if vehicle_type == "parked_vehicle":
                x, y = traci.vehicle.getPosition(vehicle_id)
                x_32632, y_32632 = convert_simulation_coordinates(x, y)
                width, length = vehicle_attributes(vehicle_type)[2]
                angle = traci.vehicle.getAngle(vehicle_id)
                parked_vehicle_geom = create_vehicle_polygon(x_32632, y_32632, width, length, angle)
                static_objects.append(parked_vehicle_geom)

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
            
            # Create dynamic objects
            dynamic_objects_geom = [
                create_vehicle_polygon(
                    *convert_simulation_coordinates(*traci.vehicle.getPosition(vid)),
                    *vehicle_attributes(traci.vehicle.getTypeID(vid))[2],
                    traci.vehicle.getAngle(vid)
                ) for vid in traci.vehicle.getIDList() if vid != vehicle_id
            ]

            t = transforms.Affine2D().rotate_deg_around(x_32632, y_32632, adjusted_angle) + ax.transData
            patch.set_transform(t)
            new_vehicle_patches.append(patch)

            # Ray tracing for observers
            if vehicle_type in ["floating_car_observer", "floating_bike_observer"]:
                center = (x_32632, y_32632)
                rays = generate_rays(center)
                all_objects = static_objects + dynamic_objects_geom
                ray_endpoints = []

                # Multithreaded ray tracing
                with ThreadPoolExecutor() as executor:
                    futures = {executor.submit(detect_intersections, ray, all_objects): ray for ray in rays}
                    for future in as_completed(futures):
                        intersection = future.result()
                        ray = futures[future]
                        if intersection:
                            end_point = intersection
                            ray_color = detected_color
                        else:
                            angle = np.arctan2(ray[1][1] - ray[0][1], ray[1][0] - ray[0][0])
                            end_point = (ray[0][0] + np.cos(angle) * radius, ray[0][1] + np.sin(angle) * radius)
                            ray_color = undetected_color

                        ray_endpoints.append(end_point)
                        ray_line = Line2D([ray[0][0], end_point[0]], [ray[0][1], end_point[1]], 
                                        color=ray_color, linewidth=1)
                        if visualizeRays:
                            ax.add_line(ray_line)
                        new_ray_lines.append(ray_line)

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
                bicycle_detection_data[vehicle_id].append((traci.simulation.getTime(), is_detected, [{'id': observer_id}] if is_detected else []))

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
        # Ensure the output directory exists
        os.makedirs('out_raytracing', exist_ok=True)
        # Calculate fps based on simulation step length
        step_length = get_step_length(sumo_config_path)
        fps = int(1 / step_length)  # Convert step length to fps
        # Create writer
        writer = FFMpegWriter(fps=fps, metadata=dict(artist='Me'), bitrate=2400)
        
        filename = f'out_raytracing/ray_tracing_animation_{file_tag}_FCO{FCO_share*100:.0f}%_FBO{FBO_share*100:.0f}%.mp4'
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
        if frame == 0:  # Only create mapping at simulation start
            lane_to_signal = {}
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
            'vehicles_by_type': str(vehicles_by_type)
        }
        
        # Only add lane_to_signal_mapping at frame 0
        if frame == 0:
            log_entry['lane_to_signal_mapping'] = str(lane_to_signal)
        
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
            if vehicle_id == "flow_10.0" and frame < 10:  # Just check first few frames of one bicycle
                x_sumo, y_sumo = traci.vehicle.getPosition(vehicle_id)
                point = Point(x_sumo, y_sumo)
                print(f"Frame {frame}, Bicycle {vehicle_id} at {point.wkt}")
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
                'in_test_area': int(in_test_area)
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
    with open(f'out_logging/log_fleet_composition_{file_tag}_FCO{FCO_share*100:.0f}%_FBO{FBO_share*100:.0f}%.csv', 'w', newline='') as f:
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
    with open(f'out_logging/log_traffic_lights_{file_tag}_FCO{FCO_share*100:.0f}%_FBO{FBO_share*100:.0f}%.csv', 'w', newline='') as f:
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
    with open(f'out_logging/log_detections_{file_tag}_FCO{FCO_share*100:.0f}%_FBO{FBO_share*100:.0f}%.csv', 'w', newline='') as f:
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
    with open(f'out_logging/log_vehicle_trajectories_{file_tag}_FCO{FCO_share*100:.0f}%_FBO{FBO_share*100:.0f}%.csv', 'w', newline='') as f:
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
    with open(f'out_logging/log_bicycle_trajectories_{file_tag}_FCO{FCO_share*100:.0f}%_FBO{FBO_share*100:.0f}%.csv', 'w', newline='') as f:
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
        f.write('# -----------------------------------------\n')
        f.write('\n')
        bicycle_trajectory_logs.to_csv(f, index=False)

    # Conflict data
    with open(f'out_logging/log_conflicts_{file_tag}_FCO{FCO_share*100:.0f}%_FBO{FBO_share*100:.0f}%.csv', 'w', newline='') as f:
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
            conflict_starts = [True] + list(time_diffs > 1)
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
        application_time = operation_times['individual_trajectories'] + operation_times['flow_trajectories'] + operation_times['3d_conflicts'] + operation_times['3d_detections'] + operation_times['3d_animated_conflicts'] + operation_times['3d_animated_detections'] + operation_times['important_trajectories'] + operation_times['visibility_heatmap'] + operation_times['LoV_heatmap']
        if 'visualization' in operation_times and visualization_time > 0:
            component_sum = setup_time + visualization_time + data_collection_time + logging_time + application_time
        else:
            component_sum = setup_time + ray_tracing_time + data_collection_time + logging_time + application_time
        timing_offset = total_runtime - component_sum
    # -----------------------------------------------------------------------

    # ----------------------------------------------------------------------------------------------------------

    # Summary logging ------------------------------------------------------------------------------------------
    
    with open(f'out_logging/summary_log_{file_tag}_FCO{str(FCO_share*100)}%_FBO{str(FBO_share*100)}%.csv', mode='w', newline='') as f:
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
            if IndividualBicycleTrajectories and 'individual_trajectories' in operation_times:
                writer.writerow(['- Individual bicycle trajectories', f"{operation_times['individual_trajectories']:.2f} seconds"])
            if FlowBasedBicycleTrajectories and 'flow_trajectories' in operation_times:
                writer.writerow(['- Flow-based bicycle trajectories', f"{operation_times['flow_trajectories']:.2f} seconds"])
            if ThreeDimensionalConflictPlots and '3d_conflicts' in operation_times:
                writer.writerow(['- 3D bicycle conflict plots', f"{operation_times['3d_conflicts']:.2f} seconds"])
            if ThreeDimensionalDetectionPlots and '3d_detections' in operation_times:
                writer.writerow(['- 3D bicycle detection plots', f"{operation_times['3d_detections']:.2f} seconds"])
            if AnimatedThreeDimensionalConflictPlots and '3d_animated_conflicts' in operation_times:
                writer.writerow(['- Animated 3D conflicts', f"{operation_times['3d_animated_conflicts']:.2f} seconds"])
            if AnimatedThreeDimensionalDetectionPlots and '3d_animated_detections' in operation_times:
                writer.writerow(['- Animated 3D detections', f"{operation_times['3d_animated_detections']:.2f} seconds"])
            if AnimatedThreeDimensionalConflictPlots and AnimatedThreeDimensionalDetectionPlots:
                writer.writerow(['- Animated 3D conflicts and detections', f"{operation_times['3d_animated_conflicts'] + operation_times['3d_animated_detections']:.2f} seconds"])
            if ImportantTrajectories and 'important_trajectories' in operation_times:
                writer.writerow(['- Test: important trajectories', f"{operation_times['important_trajectories']:.2f} seconds"])
            if RelativeVisibility and 'visibility_heatmap' in operation_times:
                writer.writerow(['- Relative visibility heatmap', f"{operation_times['visibility_heatmap']:.2f} seconds"])
            if LoVheatmap and 'LoV_heatmap' in operation_times:
                writer.writerow(['- Level of Visibility (LoV) heatmap', f"{operation_times['LoV_heatmap']:.2f} seconds"])
            if not any([IndividualBicycleTrajectories, FlowBasedBicycleTrajectories, 
                   ThreeDimensionalConflictPlots, ThreeDimensionalDetectionPlots,
                   AnimatedThreeDimensionalConflictPlots, AnimatedThreeDimensionalDetectionPlots,
                   ImportantTrajectories, RelativeVisibility, LoVheatmap]):
                writer.writerow(['- Note:', 'No ray tracing applications were activated'])
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

def create_relative_visibility_heatmap(x_coords, y_coords, visibility_counts, buildings_proj, parks_proj, trees_proj):
    """
    Generates a CSV file with raw visibility data (visibility counts) and plots a normalized heatmap.
    Saves the heatmap as a PNG file if 'relative visibility' is enabled in the Application Settings.
    """
    if RelativeVisibility: # Only create heatmap if relative visibility is enabled
        if FCO_share == 0 and FBO_share == 0:
            print("No visibility data to plot: FCO and FBO penetration rates are both set to 0%. The relative visibility heatmap could not be created.")
            return

        # Get bounds from buildings for translation
        x_min, y_min, x_max, y_max = buildings_proj.total_bounds

        # Initialize heatmap data array
        heatmap_data = np.zeros((len(x_coords), len(y_coords)))
        
        # Populate heatmap data from visibility counts
        for cell, count in visibility_counts.items():
            x_idx = np.searchsorted(x_coords, cell.bounds[0])
            y_idx = np.searchsorted(y_coords, cell.bounds[1])
            if x_idx < len(x_coords) and y_idx < len(y_coords):
                heatmap_data[x_idx, y_idx] = count
        
        # Set zero counts to NaN for better visualization
        heatmap_data[heatmap_data == 0] = np.nan

        # Save raw visibility data to CSV
        output_prefix = f'{file_tag}_FCO{FCO_share*100:.0f}%_FBO{FBO_share*100:.0f}%'
        visibility_counts_path = os.path.join(parent_dir, 'out_visibility', 'visibility_counts', f'visibility_counts_{output_prefix}.csv')
        with open(visibility_counts_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['x_coord', 'y_coord', 'visibility_count'])
            for i, x in enumerate(x_coords):
                for j, y in enumerate(y_coords):
                    if not np.isnan(heatmap_data[i, j]):
                        csvwriter.writerow([x, y, heatmap_data[i, j]])

        # Normalize heatmap data
        heatmap_data = heatmap_data / np.nanmax(heatmap_data)

        # Plot and save heatmap
        fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')
        ax.set_facecolor('white')
        
        # Translate buildings and parks
        buildings_proj_translated = buildings_proj.translate(-x_min, -y_min)
        parks_proj_translated = parks_proj.translate(-x_min, -y_min)
        trees_proj_translated = trees_proj.translate(-x_min, -y_min)
        
        # Plot translated geometries
        buildings_proj_translated.plot(ax=ax, facecolor='darkgray', edgecolor='black', linewidth=0.5)
        parks_proj_translated.plot(ax=ax, facecolor='forestgreen', edgecolor='black', linewidth=0.5)
        trees_proj_translated.plot(ax=ax, facecolor='forestgreen', edgecolor='black', linewidth=0.5)
        # Plot heatmap with translated coordinates
        translated_extent = [0, x_max - x_min, 0, y_max - y_min]
        cax = ax.imshow(heatmap_data.T, origin='lower', cmap='hot', extent=translated_extent, alpha=0.6)
        
        ax.set_title('Relative Visibility Heatmap')
        ax.set_xlabel('Distance [m]')
        ax.set_ylabel('Distance [m]')
        fig.colorbar(cax, ax=ax, label='Relative Visibility')
        plt.savefig(f'out_visibility/relative_visibility_heatmap_FCO{str(FCO_share*100)}%_FBO{str(FBO_share*100)}%.png')
        print('\nRelative visibility heatmap generated and saved.')

def create_lov_heatmap(visibility_counts_path, output_prefix, buildings_proj, parks_proj, trees_proj):
    """
    Creates a Level of Visibility (LoV) heatmap using the saved visibility counts.
    
    Args:
        visibility_counts_path (str): Path to the CSV file containing visibility counts
        output_prefix (str): Prefix for output files (e.g., 'FCO50.0%_FBO0%')
    """

    if FCO_share == 0 and FBO_share == 0:
        print("No visibility data to plot: FCO and FBO penetration rates are both set to 0%. The relative LoV heatmap could not be created.")
        return

    # Load visibility counts from CSV
    x_coords = []
    y_coords = []
    visibility_counts = []
    with open(visibility_counts_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # Skip header
        for row in csvreader:
            x_coords.append(float(row[0]))
            y_coords.append(float(row[1]))
            visibility_counts.append(float(row[2]))
    
    if not visibility_counts:
        if FCO_share == 0 and FBO_share == 0:
            print("\nNo visibility data found: FCO and FBO penetration rates are both set to 0%. The LoV heatmap could not be created.")
        else:
            print("\nNo visibility data found. The LoV heatmap could not be created.")
        return

    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)
    visibility_counts = np.array(visibility_counts)

    # Calculate LoV data
    total_steps = get_total_simulation_steps(sumo_config_path)
    stepLength = get_step_length(sumo_config_path)
    lov_data = visibility_counts / total_steps
    max_lov = 1 / step_length

    # Save logging information
    logging_info = [
        ['Max. visibility count', np.max(visibility_counts)],
        ['Total simulation steps', total_steps],
        ['Step Size', step_length],
        ['LoV scale', f'0 - {max_lov}'],
        ['Max. LoV value', np.max(lov_data)],
        ['Mean LoV value', np.mean(lov_data)]
    ]
    
    log_path = os.path.join(parent_dir, 'out_visibility', 'LoV_logging', f'log_LoV_{output_prefix}.csv')
    with open(log_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Description', 'Value'])
        csvwriter.writerows(logging_info)

    # Get bounds from buildings (already projected in main.py)
    x_min, y_min, x_max, y_max = buildings_proj.total_bounds
    
    # Translate coordinates
    x_coords_translated = x_coords - x_min
    y_coords_translated = y_coords - y_min

    # Create color map
    alpha = 0.5
    colors = [
        (0.698, 0.133, 0.133, alpha),  # Firebrick
        (1.000, 0.271, 0.000, alpha),  # Orange-Red
        (1.000, 0.647, 0.000, alpha),  # Orange
        (1.000, 1.000, 0.000, alpha),  # Yellow
        (0.678, 1.000, 0.184, alpha)   # Green-Yellow
    ]
    cmap = ListedColormap(colors)
    bounds = [0, max_lov * 0.2, max_lov * 0.4, max_lov * 0.6, max_lov * 0.8, max_lov]
    norm = BoundaryNorm(bounds, cmap.N)

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')
    ax.set_facecolor('white')
    
    # Plot buildings and parks (translated)
    buildings_proj_translated = buildings_proj.translate(-x_min, -y_min)
    parks_proj_translated = parks_proj.translate(-x_min, -y_min)
    trees_proj_translated = trees_proj.translate(-x_min, -y_min)
    buildings_proj_translated.plot(ax=ax, facecolor='gray', edgecolor='black', linewidth=0.5, alpha=0.7)
    parks_proj_translated.plot(ax=ax, facecolor='green', edgecolor='black', linewidth=0.5, alpha=0.7)
    trees_proj_translated.plot(ax=ax, facecolor='forestgreen', edgecolor='black', linewidth=0.5, alpha=0.7)

    # Plot heatmap rectangles
    for i, x in enumerate(x_coords_translated):
        y = y_coords_translated[i]
        value = lov_data[i]
        color = cmap(norm([value])[0])
        ax.add_patch(plt.Rectangle((x, y), 1.0, 1.0, facecolor=color, edgecolor='none'))

    # Create legend
    legend_patches = [
        Patch(color=colors[0], label='LoV E'),
        Patch(color=colors[1], label='LoV D'),
        Patch(color=colors[2], label='LoV C'),
        Patch(color=colors[3], label='LoV B'),
        Patch(color=colors[4], label='LoV A')
    ]
    legend = ax.legend(handles=legend_patches, loc='upper right', title='Level of Visibility (LoV)')
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(1.0)
    legend.get_frame().set_edgecolor('black')

    # Set title and labels
    ax.set_title('Level of Visibility (LoV) Heatmap')
    ax.set_xlabel('Distance [m]')
    ax.set_ylabel('Distance [m]')

    # Set plot limits
    translated_x_max = x_max - x_min
    translated_y_max = y_max - y_min
    ax.set_xlim(0, translated_x_max)
    ax.set_ylim(0, translated_y_max)
    
    # Save plot
    plt.savefig(os.path.join(parent_dir, 'out_visibility', f'LoV_heatmap_{output_prefix}.png'))
    print('\nLevel of Visibility (LoV) heatmap generated and saved.')
    plt.close()

def individual_bicycle_trajectories(frame):
    """
    Creates space-time diagrams for individual bicycles, including detection status, traffic lights,
    and conflicts detected by SUMO's SSM device.
    """
    global bicycle_data, bicycle_start_times, traffic_light_positions, bicycle_tls, bicycle_waiting_times, bicycle_conflicts_ind

    # Initialize detection buffer for smoothing
    detection_buffer = []

    # Create output directory if it doesn't exist
    os.makedirs('out_2d_individual_trajectories', exist_ok=True)
    
    bicycle_waiting_times = {}

    current_vehicles = set(traci.vehicle.getIDList())
    current_time = traci.simulation.getTime()

    # Collect detection data for current bicycles
    for vehicle_id in current_vehicles:
        vehicle_type = traci.vehicle.getTypeID(vehicle_id)
        if vehicle_type in ["bicycle", "DEFAULT_BIKETYPE", "floating_bike_observer"]:
            if vehicle_id not in individual_detection_data:
                individual_detection_data[vehicle_id] = []
            
            # Check if bicycle is currently detected
            is_detected = False
            if vehicle_id in bicycle_detection_data:  # Add this check
                detection_info = next((info for info in bicycle_detection_data[vehicle_id] 
                                    if abs(info[0] - current_time) < step_length), None)
                
                if detection_info:
                    if len(detection_info) == 3:  # New format
                        is_detected = detection_info[1]
                    else:  # Old format
                        is_detected = detection_info[1]

                # Detection buffer for smoothing
                detection_buffer.append(is_detected)
                if len(detection_buffer) > max_gap_bridge:
                    detection_buffer.pop(0)

                # Apply smoothing logic
                recent_detection = any(detection_buffer[-3:]) if len(detection_buffer) >= 3 else is_detected
                if not recent_detection and len(detection_buffer) >= max_gap_bridge:
                    is_detected = any(detection_buffer[:3]) and any(detection_buffer[-3:])
                else:
                    is_detected = recent_detection
            
            individual_detection_data[vehicle_id].append((current_time, is_detected))
    
    # Check for bicycles that have left the simulation
    for vehicle_id in list(bicycle_data.keys()):
        if vehicle_id not in current_vehicles:
            # This bicycle has left the simulation, plot its trajectory
            data = bicycle_data[vehicle_id]
            detection_data = individual_detection_data.get(vehicle_id, [])
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Split trajectory into detected and undetected segments
            current_points = []
            current_detected = None
            segments = {'detected': [], 'undetected': []}
            detection_buffer = []  # Buffer to store recent detection states
            
            for i, (distance, time) in enumerate(data):
                # Find corresponding detection status
                is_detected = False
                for detection_info in detection_data:
                    det_time = detection_info[0]
                    det_status = detection_info[1]
                    if abs(det_time - (time + bicycle_start_times[vehicle_id])) < step_length:
                        is_detected = det_status
                        break
                
                # Update detection buffer
                detection_buffer.append(is_detected)
                if len(detection_buffer) > max_gap_bridge:
                    detection_buffer.pop(0)
                
                recent_detection = any(detection_buffer[-3:]) if len(detection_buffer) >= 3 else is_detected
                if not recent_detection and len(detection_buffer) >= max_gap_bridge:
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
                    if len(current_points) >= min_segment_length:
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

            # Plot segments with appropriate colors
            for segment in segments['undetected']:
                if len(segment) > 1:
                    distances, times = zip(*segment)
                    ax.plot(distances, times, color='black', linewidth=1.5, linestyle='solid')
            for segment in segments['detected']:
                if len(segment) > 1:
                    distances, times = zip(*segment)
                    ax.plot(distances, times, color='darkturquoise', linewidth=1.5, linestyle='solid')
            
            # Plot conflicts if any exist
            if vehicle_id in bicycle_conflicts_ind and bicycle_conflicts_ind[vehicle_id]:
                # Group conflicts by foe vehicle ID
                conflicts_by_foe = {}
                labels = []
                for conflict in bicycle_conflicts_ind[vehicle_id]:
                    if not conflict:  # Skip if conflict is empty
                        continue
                    
                    try:
                        # Convert absolute simulation time to elapsed time for this bicycle
                        conflict_elapsed_time = conflict['time'] - bicycle_start_times[vehicle_id]
                        
                        foe_id = conflict.get('foe_id')
                        if foe_id and 'distance' in conflict:  # Only process if we have both foe_id and distance
                            if foe_id not in conflicts_by_foe:
                                conflicts_by_foe[foe_id] = []
                            # Store conflict with elapsed time instead of simulation time
                            conflicts_by_foe[foe_id].append({
                                'distance': conflict['distance'],
                                'time': conflict_elapsed_time,
                                'ttc': conflict['ttc'],
                                'pet': conflict['pet'],
                                'drac': conflict['drac'],
                                'severity': conflict['severity'],
                                'foe_type': conflict['foe_type']
                            })
                    except KeyError as e:
                        print(f"Skipping conflict due to missing key: {e}")
                        continue
                
                # Plot conflict points
                for foe_conflicts in conflicts_by_foe.values():
                    if foe_conflicts:  # Make sure we have conflicts to plot
                        
                        # for plotting all conflict points ---------------------------
                        # for conflict in foe_conflicts:
                        #     size = 50 + (conflict['severity'] * 100)
                        #     ax.scatter(conflict['distance'], conflict['time'], 
                        #                 color='firebrick', marker='o', s=size, zorder=5,
                        #                 facecolors='none', edgecolors='firebrick', linewidth=0.75)
                        # ------------------------------------------------------------

                        # for plotting markers that span the whole conflict event duration -----
                        # conflict_times = [c['time'] for c in foe_conflicts]
                        # conflict_distances = [c['distance'] for c in foe_conflicts]
                        # event_start = min(conflict_times)
                        # event_end = max(conflict_times)
                        # event_distance = np.mean(conflict_distances)
                        # angle = np.degrees(np.arctan2(
                        #     event_end - event_start,
                        #     max(conflict_distances) - min(conflict_distances)
                        # ))
                        # ----------------------------------------------------------------------    

                        most_severe_ind = max(foe_conflicts, key=lambda x: x['severity'])
                        size = 50 + (most_severe_ind['severity'] * 100) # comment out to plot all conflict points
                        
                        # Create label based on the most critical metric
                        ttc = most_severe_ind['ttc']
                        pet = most_severe_ind['pet']
                        drac = most_severe_ind['drac']
                        
                        if ttc < 3.0:  # TTC threshold
                            label = f'TTC = {ttc:.1f}s'
                        elif pet < 2.0:  # PET threshold
                            label = f'PET = {pet:.1f}s'
                        elif drac > 3.0:  # DRAC threshold
                            label = f'DRAC = {drac:.1f}m/s²'
                        else:
                            label = 'Conflict'
                        
                        # for plotting only the most severe conflict point --------------------
                        ax.scatter(most_severe_ind['distance'], most_severe_ind['time'], 
                                color='firebrick', marker='o', s=size, zorder=5,
                                facecolors='none', edgecolors='firebrick', linewidth=0.75)
                        # ----------------------------------------------------------------------
                        
                        # for plotting markers that span the whole conflict event duration -----
                        # marker_width = np.sqrt(size) / 2  # Scale marker width based on severity
                        # duration = event_end - event_start
                        # min_height = marker_width * 0.5  # Adjust this factor as needed
                        # adjusted_height = max(duration, min_height)
                        # ax.add_patch(plt.matplotlib.patches.Ellipse(
                        #     (event_distance, (event_start + event_end) / 2),  # center point
                        #     width=marker_width,  # width of the ellipse
                        #     height=adjusted_height,  # height spans the event duration
                        #     angle=angle,  # rotate to match trajectory
                        #     facecolor='none',
                        #     edgecolor='firebrick',
                        #     linewidth=0.75,
                        #     zorder=5))
                        # ----------------------------------------------------------------------

                        # Store label information
                        labels.append({
                            'x': most_severe_ind['distance'],
                            'y': most_severe_ind['time'],
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

                    # Add label        
                    ax.annotate(label_info['text'], 
                                (label_info['x'], label_info['y']),
                                xytext=(12, y_offset),  # Fixed horizontal offset, minimal vertical
                                textcoords='offset points',
                                bbox=dict(facecolor='none', edgecolor='none'),
                                fontsize=7,
                                verticalalignment='center',
                                color='firebrick')

            # Keep track of plotted traffic light positions
            plotted_tl_positions = set()

            # Plot traffic light positions with their states
            for tl_id, tl_info in traffic_light_positions[vehicle_id].items():
                tl_pos, tl_states = tl_info
                if tl_pos not in plotted_tl_positions:
                    for i, state in enumerate(tl_states):
                        color = {'r': 'red', 'y': 'yellow', 'g': 'green', 'G': 'green'}.get(state, 'gray')
                        ax.axvline(x=tl_pos, ymin=i/len(tl_states), ymax=(i+1)/len(tl_states),
                                   color=color, linestyle='-')
                    plotted_tl_positions.add(tl_pos)
                
                tl_index = bicycle_tls[vehicle_id].get(tl_id, 'N/A')
                ax.text(tl_pos, ax.get_ylim()[1], f'{tl_id}.{tl_index}', rotation=90, va='top', ha='right')
            
            ax.set_xlabel('Distance Traveled (m)')
            ax.set_ylabel('Elapsed Time (s)')
            ax.grid(True)
            
            # Departure time
            departure_time = bicycle_start_times[vehicle_id]

            # Trajectory detection rates
            # Time-based detection rate
            total_time = data[-1][1] - data[0][1]
            detected_segments_time = sum(
                segment[-1][1] - segment[0][1]
                for segment in segments['detected']
            )
            time_detection_rate = (detected_segments_time / total_time) * 100 if total_time > 0 else 0

            # Distance-based detection rate
            total_distance = data[-1][0] - data[0][0]
            detected_segments_distance = sum(
                segment[-1][0] - segment[0][0]
                for segment in segments['detected']
            )
            distance_detection_rate = (detected_segments_distance / total_distance) * 100 if total_distance > 0 else 0

            # Trajectory statistics
            try:
                # Count only the unique conflicts that we're plotting
                plotted_conflicts = 0
                if vehicle_id in bicycle_conflicts and bicycle_conflicts[vehicle_id]:
                    # Group conflicts by foe vehicle ID
                    conflicts_by_foe = {}
                    for conflict in bicycle_conflicts[vehicle_id]:
                        if not conflict:  # Skip if conflict is empty
                            continue
                        
                        foe_id = conflict.get('foe_id')
                        if foe_id and 'distance' in conflict:  # Only count if we have both foe_id and distance
                            if foe_id not in conflicts_by_foe:
                                conflicts_by_foe[foe_id] = []
                            conflicts_by_foe[foe_id].append(conflict)
                    
                    # Store conflicts in individual dictionary
                    bicycle_conflicts_ind[vehicle_id] = conflicts_by_foe
                    
                    # Count number of unique foe vehicles that had conflicts
                    plotted_conflicts = len(conflicts_by_foe)

                info_text = (
                    f"Bicycle: {vehicle_id}\n"
                    f"Departure time: {bicycle_start_times[vehicle_id]:.1f} s\n"
                    f"Time-based detection rate: {time_detection_rate:.1f}%\n"
                    f"Distance-based detection rate: {distance_detection_rate:.1f}%\n"
                    f"Number of potential conflicts: {plotted_conflicts}"
                )
            except Exception as e:
                continue
            
             # Add info text box with same style as legend
            ax.text(0.01, 0.99, info_text,
                   transform=ax.transAxes,
                   verticalalignment='top',
                   horizontalalignment='left',
                   fontsize=plt.rcParams['legend.fontsize'],
                   bbox=dict(
                       facecolor='white',
                       edgecolor='black',
                       alpha=0.8,
                       boxstyle='round'
                   ))
            
            # Add legend with all elements (keep existing legend code)
            handles = [
                plt.Line2D([0], [0], color='black', lw=2, label='bicycle undetected'),
                plt.Line2D([0], [0], color='darkturquoise', lw=2, label='bicycle detected'),
                plt.Line2D([0], [0], marker='o', color='firebrick', linestyle='None', 
                          markerfacecolor='none', markersize=10, label='potential conflict detected'),
                plt.Line2D([0], [0], color='red', lw=2, label='Red TL'),
                plt.Line2D([0], [0], color='yellow', lw=2, label='Yellow TL'),
                plt.Line2D([0], [0], color='green', lw=2, label='Green TL')
            ]
            ax.legend(handles=handles, loc='lower right', bbox_to_anchor=(0.99, 0.01))
            
            # Save the plot
            plt.savefig(f'out_2d_individual_trajectories/{vehicle_id}_space_time_diagram_{file_tag}_FCO{FCO_share*100:.0f}%_FBO{FBO_share*100:.0f}%.png', bbox_inches='tight')
            print(f"\nIndividual space-time diagram for bicycle {vehicle_id} saved as out_2d_individual_trajectories/{vehicle_id}_space_time_diagram_{file_tag}_FCO{FCO_share*100:.0f}%_FBO{FBO_share*100:.0f}%.png.")
            plt.close(fig)

            # Remove this bicycle from the data dictionaries
            del bicycle_data[vehicle_id]
            del bicycle_start_times[vehicle_id]
            del traffic_light_positions[vehicle_id]
            del bicycle_tls[vehicle_id]
            # if vehicle_id in bicycle_detection_data:
            #    del bicycle_detection_data[vehicle_id]
    
    # Collect data for current bicycles
    for vehicle_id in current_vehicles:
        vehicle_type = traci.vehicle.getTypeID(vehicle_id)
        if vehicle_type in ["bicycle", "DEFAULT_BIKETYPE", "floating_bike_observer"]:
            distance = traci.vehicle.getDistance(vehicle_id)
            current_time = traci.simulation.getTime()
            
            # Initialize if first time seeing this bicycle
            if vehicle_id not in bicycle_start_times:
                bicycle_start_times[vehicle_id] = current_time
                bicycle_data[vehicle_id] = []
                traffic_light_positions[vehicle_id] = {}
                bicycle_tls[vehicle_id] = {}
            
            start_time = bicycle_start_times[vehicle_id]
            elapsed_time = current_time - start_time
            bicycle_data[vehicle_id].append((distance, elapsed_time))
            
            # Check for conflicts
            try:
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
                    
                    # Check for conflict
                    if (ttc < TTC_THRESHOLD or pet < PET_THRESHOLD or drac > DRAC_THRESHOLD):
                        if vehicle_id not in bicycle_conflicts_ind:
                            bicycle_conflicts_ind[vehicle_id] = []
                        
                        # Calculate severity
                        ttc_severity = 1 - (ttc / TTC_THRESHOLD) if ttc < TTC_THRESHOLD else 0
                        pet_severity = 1 - (pet / PET_THRESHOLD) if pet < PET_THRESHOLD else 0
                        drac_severity = min(drac / DRAC_THRESHOLD, 1.0) if drac > 0 else 0
                        
                        conflict_severity = max(ttc_severity, pet_severity, drac_severity)
                        
                        bicycle_conflicts_ind[vehicle_id].append({
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
                print(f"Error in conflict detection for {vehicle_id}: {str(e)}")

            # Check for the next traffic light
            next_tls = traci.vehicle.getNextTLS(vehicle_id)
            if next_tls:
                tl_id, tl_index, tl_distance, tl_state = next_tls[0]
                if tl_id not in traffic_light_ids:
                    traffic_light_ids[tl_id] = len(traffic_light_ids) + 1
                short_tl_id = f"TL{traffic_light_ids[tl_id]}"
                tl_pos = distance + tl_distance  # Position of the traffic light relative to the start
                if short_tl_id not in traffic_light_positions[vehicle_id]:
                    traffic_light_positions[vehicle_id][short_tl_id] = [tl_pos, []]
                bicycle_tls[vehicle_id][short_tl_id] = tl_index

            # Update states for all known traffic lights
            for short_tl_id, tl_index in bicycle_tls[vehicle_id].items():
                full_tl_id = next((id for id, num in traffic_light_ids.items() if f"TL{num}" == short_tl_id), None)
                if full_tl_id:
                    full_state = traci.trafficlight.getRedYellowGreenState(full_tl_id)
                    relevant_state = full_state[tl_index]
                    traffic_light_positions[vehicle_id][short_tl_id][1].append(relevant_state)

def important_trajectory_parts(frame):
    """
    Creates space-time diagrams for each bicycle when they leave the simulation.
    Simplified version that shows trajectories with different colors for test areas.
    """
    global bicycle_trajectories, transformer, flow_ids

    # Initialize at frame 0
    if frame == 0:
        transformer = pyproj.Transformer.from_crs('EPSG:4326', 'EPSG:32632', always_xy=True)
        bicycle_trajectories.clear()
        flow_ids.clear()

    # Ensure transformer is initialized
    if transformer is None:
        transformer = pyproj.Transformer.from_crs('EPSG:4326', 'EPSG:32632', always_xy=True)

    # Get list of test polygons and their shapes
    test_polygons = []
    for poly_id in traci.polygon.getIDList():
        if traci.polygon.getType(poly_id) == "test":
            shape = traci.polygon.getShape(poly_id)
            # Convert shape to shapely polygon
            poly = Polygon(shape)
            test_polygons.append(poly)

    # Get current vehicles and collect trajectory data
    current_vehicles = set(traci.vehicle.getIDList())
    current_time = frame * step_length

    for vehicle_id in current_vehicles:
        vehicle_type = traci.vehicle.getTypeID(vehicle_id)
        
        # Only process bicycles
        if vehicle_type in ["bicycle", "DEFAULT_BIKETYPE", "floating_bike_observer"]:
            # Get position
            x_sumo, y_sumo = traci.vehicle.getPosition(vehicle_id)
            point = Point(x_sumo, y_sumo)
            
            # Check if position is in any test polygon
            in_test_area = any(poly.contains(point) for poly in test_polygons)
            
            lon, lat = traci.simulation.convertGeo(x_sumo, y_sumo)
            x_utm, y_utm = transformer.transform(lon, lat)
            
            # Store trajectory data with test area flag
            flow_id = vehicle_id.rsplit('.', 1)[0]
            flow_ids.add(flow_id)
            if vehicle_id not in bicycle_trajectories:
                bicycle_trajectories[vehicle_id] = []
            bicycle_trajectories[vehicle_id].append((x_utm, y_utm, current_time, in_test_area))

    # Check for bicycles that have finished their trajectory
    finished_bicycles = set(bicycle_trajectories.keys()) - current_vehicles
    
    # Generate plots for finished bicycles
    for vehicle_id in finished_bicycles:
        if len(bicycle_trajectories[vehicle_id]) > 0:  # Only plot if we have trajectory data
            # Create figure
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Extract trajectory data
            trajectory = bicycle_trajectories[vehicle_id]
            distances = []
            times = []
            in_test_area = []
            
            # Calculate cumulative distance
            total_distance = 0
            prev_x, prev_y = None, None
            
            for x, y, t, test_area in trajectory:
                if prev_x is not None:
                    # Calculate distance between consecutive points
                    dx = x - prev_x
                    dy = y - prev_y
                    distance = np.sqrt(dx**2 + dy**2)
                    total_distance += distance
                
                distances.append(total_distance)
                times.append(t)
                in_test_area.append(test_area)
                prev_x, prev_y = x, y
            
            # Split trajectory into segments based on test area status
            segments = []
            current_segment = {'distances': [], 'times': [], 'in_test_area': None}
            
            for d, t, test_area in zip(distances, times, in_test_area):
                if current_segment['in_test_area'] is None:
                    current_segment['in_test_area'] = test_area
                
                if current_segment['in_test_area'] == test_area:
                    current_segment['distances'].append(d)
                    current_segment['times'].append(t)
                else:
                    segments.append(current_segment)
                    current_segment = {
                        'distances': [d],
                        'times': [t],
                        'in_test_area': test_area
                    }
            
            segments.append(current_segment)
            
            # Plot segments with different colors
            for segment in segments:
                color = 'red' if segment['in_test_area'] else 'darkslateblue'
                ax.plot(segment['distances'], segment['times'], 
                       color=color, linewidth=2)
            
            # Add legend
            ax.plot([], [], color='darkslateblue', linewidth=2, label='Normal trajectory')
            ax.plot([], [], color='red', linewidth=2, label='Test area trajectory')
            
            # Customize plot
            ax.set_xlabel('Distance traveled (m)')
            ax.set_ylabel('Time (s)')
            ax.set_title(f'Space-Time Diagram for Bicycle {vehicle_id}')
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend()

            # Save plot
            os.makedirs('out_test', exist_ok=True)
            plt.savefig(f'out_test/trajectory_{vehicle_id}_{file_tag}_FCO{FCO_share*100:.0f}%_FBO{FBO_share*100:.0f}%.png',
                       bbox_inches='tight', dpi=300)
            print(f'\nTrajectory plot saved for bicycle {vehicle_id}')
            plt.close(fig)

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
    os.makedirs('out_2d_flow_based_trajectories', exist_ok=True)
    
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
                if len(detection_buffer) > max_gap_bridge:
                    detection_buffer.pop(0)

                # Apply smoothing logic
                recent_detection = any(detection_buffer[-3:]) if len(detection_buffer) >= 3 else is_detected
                if not recent_detection and len(detection_buffer) >= max_gap_bridge:
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
            if len(detection_buffer) > max_gap_bridge:
                detection_buffer.pop(0)

            # Apply smoothing logic
            recent_detection = any(detection_buffer[-3:]) if len(detection_buffer) >= 3 else is_detected
            if not recent_detection and len(detection_buffer) >= max_gap_bridge:
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
                    if len(detection_buffer) > max_gap_bridge:
                        detection_buffer.pop(0)
                    
                    # If there's any detection in the last 3 frames, consider it detected
                    recent_detection = any(detection_buffer[-3:]) if len(detection_buffer) >= 3 else is_detected
                    # For longer gaps, only bridge if there are detections on both sides
                    if not recent_detection and len(detection_buffer) >= max_gap_bridge:
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
                        if len(current_points) >= min_segment_length:
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
                plt.Line2D([0], [0], color='black', lw=2, label='bicycle undetected'),
                plt.Line2D([0], [0], color='darkturquoise', lw=2, label='bicycle detected'),
                plt.Line2D([0], [0], marker='o', color='firebrick', linestyle='None', 
                          markerfacecolor='none', markersize=10, label='potential conflict detected'),
                plt.Line2D([0], [0], color='red', lw=2, label='Red TL'),
                plt.Line2D([0], [0], color='yellow', lw=2, label='Yellow TL'),
                plt.Line2D([0], [0], color='green', lw=2, label='Green TL')
            ]
            ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.01, 0.99))

            plt.savefig(f'out_2d_flow_based_trajectories/{flow_id}_space_time_diagram_{file_tag}_FCO{FCO_share*100:.0f}%_FBO{FBO_share*100:.0f}%.png', 
                       bbox_inches='tight')
            plt.close(fig)
            
            print(f"\nFlow-based space-time diagram for bicycle flow {flow_id} saved as out_2d_flow_based_trajectories/{flow_id}_space_time_diagram_{file_tag}_FCO{FCO_share*100:.0f}%_FBO{FBO_share*100:.0f}%.png.")

def three_dimensional_conflict_plots(frame):
    """
    Creates a 3D visualization of bicycle trajectories where the z=0 plane shows the static scene.
    Automatically generates plots for each bicycle when their trajectory ends.
    """
    global fig_3d, ax_3d, total_steps, bicycle_trajectories, transformer, flow_ids, bicycle_conflicts, foe_trajectories, bicycle_detection_data, conflict_bicycle_trajectories

    # Initialize transformer at frame 0
    if frame == 0:
        transformer = pyproj.Transformer.from_crs('EPSG:4326', 'EPSG:32632', always_xy=True)
        conflict_bicycle_trajectories.clear()
        flow_ids.clear()
        bicycle_detection_data = {}  # Add this line

    # Ensure transformer is initialized
    if transformer is None:
        transformer = pyproj.Transformer.from_crs('EPSG:4326', 'EPSG:32632', always_xy=True)

    # Create bounding box for clipping
    bbox = box(west, south, east, north)  # Create box from original coordinates
    bbox_transformed = shapely.ops.transform(
        lambda x, y: transformer.transform(x, y), 
        bbox
    )  # Transform to UTM coordinates

    # Get bounds of transformed bounding box for relative coordinates
    minx, miny, maxx, maxy = bbox_transformed.bounds
            
    # Define functions for relative coordinates
    rel_x = lambda x: x - minx
    rel_y = lambda y: y - miny

    # Collect positions for this frame
    current_vehicles = set(traci.vehicle.getIDList())
    departed_foes = set(foe_trajectories.keys()) - current_vehicles
    
    # Track foes that have complete trajectories but haven't been processed
    if not hasattr(three_dimensional_conflict_plots, 'completed_foes'):
        three_dimensional_conflict_plots.completed_foes = {}
    
    # Store completed foe trajectories before removing them
    for foe_id in departed_foes:
        if foe_id not in three_dimensional_conflict_plots.completed_foes:
            three_dimensional_conflict_plots.completed_foes[foe_id] = foe_trajectories[foe_id]

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
                is_detected = False
                # Store detection status
                if vehicle_id not in bicycle_detection_data:
                    bicycle_detection_data[vehicle_id] = []
                bicycle_detection_data[vehicle_id].append((current_time, is_detected))

                # Store trajectory data
                flow_id = vehicle_id.rsplit('.', 1)[0]
                flow_ids.add(flow_id)
                if vehicle_id not in conflict_bicycle_trajectories:
                    conflict_bicycle_trajectories[vehicle_id] = []
                conflict_bicycle_trajectories[vehicle_id].append((x_utm, y_utm, current_time))
                
                # Check for conflicts
                try:
                    # Check both leader and follower vehicles
                    leader = traci.vehicle.getLeader(vehicle_id)
                    follower = traci.vehicle.getFollower(vehicle_id)
                    
                    potential_foes = []
                    if leader and leader[0] != '':
                        potential_foes.append(('leader', *leader))
                    if follower and follower[0] != '':
                        potential_foes.append(('follower', *follower))
                    
                    # Get current distance for the bicycle
                    distance = traci.vehicle.getDistance(vehicle_id)

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
                        
                        # Check for conflict
                        if (ttc < TTC_THRESHOLD or pet < PET_THRESHOLD or drac > DRAC_THRESHOLD):
                            if vehicle_id not in bicycle_conflicts:
                                bicycle_conflicts[vehicle_id] = []
                            
                            # Calculate severity
                            ttc_severity = 1 - (ttc / TTC_THRESHOLD) if ttc < TTC_THRESHOLD else 0
                            pet_severity = 1 - (pet / PET_THRESHOLD) if pet < PET_THRESHOLD else 0
                            drac_severity = min(drac / DRAC_THRESHOLD, 1.0) if drac > 0 else 0
                            
                            conflict_severity = max(ttc_severity, pet_severity, drac_severity)
                            
                            # Get vehicle position for 3D plotting
                            x_sumo, y_sumo = traci.vehicle.getPosition(vehicle_id)
                            lon, lat = traci.simulation.convertGeo(x_sumo, y_sumo)
                            x_utm, y_utm = transformer.transform(lon, lat)
                            
                            bicycle_conflicts[vehicle_id].append({
                                'distance': distance,
                                'time': current_time,
                                'ttc': ttc,
                                'pet': pet,
                                'drac': drac,
                                'severity': conflict_severity,
                                'foe_type': foe_type,
                                'foe_id': foe_id,
                                'x': x_utm,  # Add x coordinate
                                'y': y_utm   # Add y coordinate
                            })
                
                except Exception as e:
                    if frame % 100 == 0:
                        print(f"Error in conflict detection for {vehicle_id}: {str(e)}")
            
            # Store positions for all other vehicles (potential foes)
            else:
                if vehicle_id not in foe_trajectories:
                    foe_trajectories[vehicle_id] = []
                foe_trajectories[vehicle_id].append((x_utm, y_utm, current_time))

    # Check for bicycles that have finished their trajectory
    # finished_bicycles = set(conflict_bicycle_trajectories.keys()) - current_vehicles
    
    # Generate plots for finished bicycles
    for vehicle_id in list(conflict_bicycle_trajectories.keys()):
        if vehicle_id not in current_vehicles:
        # if len(conflict_bicycle_trajectories[vehicle_id]) > 0:  # Only plot if we have trajectory data
            if vehicle_id in bicycle_conflicts and bicycle_conflicts[vehicle_id]:
                # Check if all foe trajectories are complete
                all_foes_complete = True
                for conflict in bicycle_conflicts[vehicle_id]:
                    foe_id = conflict['foe_id']
                    if foe_id in current_vehicles:  # If foe still in simulation
                        all_foes_complete = False
                        break
                
                if not all_foes_complete:
                    continue  # Skip plotting until all foes are complete

            # Now proceed with plotting
            trajectory = conflict_bicycle_trajectories[vehicle_id]
            
            # Calculate z range with padding
            z_min = min(t for _, _, t in trajectory)
            z_max = max(t for _, _, t in trajectory)
            z_padding = (z_max - z_min) * 0.05
            base_z = z_min - z_padding

            if vehicle_id in bicycle_conflicts and bicycle_conflicts[vehicle_id]:
                # Create conflict overview plot
                fig_3d = plt.figure(figsize=(15, 12))
                ax_3d = fig_3d.add_subplot(111, projection='3d')
                
                # Set axis labels and limits with relative coordinates
                ax_3d.set_xlabel('Distance (m)')
                ax_3d.set_ylabel('Distance (m)')
                ax_3d.set_zlabel('Time (s)')
                ax_3d.set_xlim(0, maxx - minx)
                ax_3d.set_ylim(0, maxy - miny)
                ax_3d.set_zlim(base_z, z_max + z_padding)

                # Set background color of the planes and grid style
                ax_3d.xaxis.pane.fill = False
                ax_3d.yaxis.pane.fill = False
                ax_3d.zaxis.pane.fill = False
                ax_3d.xaxis._axinfo['grid'].update(linestyle="--")
                ax_3d.yaxis._axinfo['grid'].update(linestyle="--")
                ax_3d.zaxis._axinfo['grid'].update(linestyle="--")
                
                # Calculate aspect ratios using relative coordinates
                dx = maxx - minx
                dy = maxy - miny
                dz = (z_max + z_padding) - base_z
                max_xy = max(dx, dy)
                aspect_ratios = [dx/max_xy, dy/max_xy, dz/max_xy * 2.0]
                ax_3d.set_box_aspect(aspect_ratios)
                
                # Set view angle
                ax_3d.view_init(elev=35, azim=285)
                ax_3d.set_axisbelow(True)

                # Create base plane with relative coordinates
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

                # Plot roads with relative coordinates
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

                # Plot buildings and parks with relative coordinates
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

                # Plot bicycle trajectory with detection-based segments
                segments = {'detected': [], 'undetected': []}
                current_points = []
                current_detected = None
                detection_buffer = []  # Buffer to store recent detection states

                for x, y, t in trajectory:
                    # Get detection status for this time
                    is_detected = False
                    if vehicle_id in bicycle_detection_data:
                        # for det_time, det_status in bicycle_detection_data[vehicle_id]:
                        for detection in bicycle_detection_data[vehicle_id]:
                            det_time = detection[0]
                            det_status = detection[1]
                            if abs(det_time - t) < step_length:
                                is_detected = det_status
                                break
                    
                    # Update detection buffer
                    detection_buffer.append(is_detected)
                    if len(detection_buffer) > max_gap_bridge:
                        detection_buffer.pop(0)
                    
                    # If there's any detection in the last 3 frames, consider it detected
                    recent_detection = any(detection_buffer[-3:]) if len(detection_buffer) >= 3 else is_detected
                    if not recent_detection and len(detection_buffer) >= max_gap_bridge:
                        if any(detection_buffer[:3]) and any(detection_buffer[-3:]):
                            smoothed_detection = True
                        else:
                            smoothed_detection = False
                    else:
                        smoothed_detection = recent_detection
                    
                    if current_detected is None:
                        current_detected = smoothed_detection
                        current_points = [(rel_x(x), rel_y(y), t)]
                    elif smoothed_detection != current_detected:
                        if len(current_points) >= min_segment_length:
                            segments['detected' if current_detected else 'undetected'].append(current_points)
                            current_points = [(rel_x(x), rel_y(y), t)]
                            current_detected = smoothed_detection
                        else:
                            current_points.append((rel_x(x), rel_y(y), t))
                    else:
                        current_points.append((rel_x(x), rel_y(y), t))

                if current_points:
                    segments['detected' if current_detected else 'undetected'].append(current_points)

                # Plot segments with appropriate colors
                all_segments = []  # Store all segments in order
                for state in ['undetected', 'detected']:
                    for segment in segments[state]:
                        if len(segment) > 1:
                            all_segments.append((state, segment))
                
                # Sort segments by time to ensure proper order
                all_segments.sort(key=lambda x: x[1][0][2])  # Sort by first time point
                
                for i, (state, segment) in enumerate(all_segments):
                    if len(segment) > 1:
                        x_coords, y_coords, times = zip(*segment)
                        color = 'cornflowerblue' if state == 'detected' else 'darkslateblue'
                        
                        # Plot 3D trajectory
                        ax_3d.plot(x_coords, y_coords, times, 
                                color=color, linewidth=2, alpha=1.0,
                                zorder=1000)
                        # Plot ground projection
                        ax_3d.plot(x_coords, y_coords, [base_z]*len(x_coords),
                                color=color, linestyle='--', linewidth=2, alpha=1.0,
                                zorder=1000)
                        
                        # Add projection plane
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
                                next_state, next_segment = all_segments[i+1]
                                next_x, next_y, next_t = next_segment[0]
                                transition_quad = [
                                    (x_coords[j+1], y_coords[j+1], times[j+1]),
                                    (next_x, next_y, next_t),
                                    (next_x, next_y, base_z),
                                    (x_coords[j+1], y_coords[j+1], base_z)
                                ]
                                plane_vertices.append(transition_quad)
                        
                        proj_plane = Poly3DCollection(plane_vertices, alpha=0.2)
                        proj_plane.set_facecolor(color)
                        proj_plane.set_edgecolor('none')
                        proj_plane.set_sort_zpos(900)
                        ax_3d.add_collection3d(proj_plane)

                # Group conflicts by foe and plot most severe ones
                conflicts_by_foe = {}
                for conflict in bicycle_conflicts[vehicle_id]:
                    foe_id = conflict.get('foe_id')
                    if foe_id:
                        if foe_id not in conflicts_by_foe:
                            conflicts_by_foe[foe_id] = []
                        conflicts_by_foe[foe_id].append(conflict)
                
                # Plot conflict points in conflict overview plot
                for foe_conflicts in conflicts_by_foe.values():
                    most_severe = max(foe_conflicts, key=lambda x: x['severity'])
                    size = 50 + (most_severe['severity'] * 100)

                    # Plot conflict point with relative coordinates
                    ax_3d.scatter(rel_x(most_severe['x']), rel_y(most_severe['y']), most_severe['time'],
                                color='firebrick', s=size, marker='o',
                                facecolors='none', edgecolors='firebrick',
                                linewidth=0.75, zorder=1001)
                    
                    # Add vertical line from base with relative coordinates
                    ax_3d.plot([rel_x(most_severe['x']), rel_x(most_severe['x'])],
                             [rel_y(most_severe['y']), rel_y(most_severe['y'])],
                             [base_z, most_severe['time']],
                             color='firebrick', linestyle=':', alpha=0.3,
                             zorder=1001)

                # Add bicycle label with relative coordinates
                x_coords_rel = [rel_x(x) for x, _, _ in trajectory]
                y_coords_rel = [rel_y(y) for _, y, _ in trajectory]
                ax_3d.text(x_coords_rel[-1], y_coords_rel[-1], base_z,
                          f'bicycle {vehicle_id}',
                          color='darkslateblue',
                          horizontalalignment='right',
                          verticalalignment='bottom',
                          rotation=90,
                          bbox=dict(facecolor='white', alpha=1.0, edgecolor='none'),
                          zorder=1000)

                # Create legend for conflict overview plot
                handles = [
                    plt.Line2D([0], [0], color='darkslateblue', linewidth=2, label='Bicycle Undetected'),
                    plt.Line2D([0], [0], color='cornflowerblue', linewidth=2, label='Bicycle Detected'),
                    plt.Line2D([0], [0], color='black', linestyle='--', label='Ground Projections'),
                    plt.Line2D([0], [0], marker='o', color='firebrick', linestyle='None', 
                              markerfacecolor='none', markersize=10, label='Potential Conflict')
                ]
                ax_3d.legend(handles=handles, loc='upper left')
                
                # Save conflict overview plot
                os.makedirs('out_3d_conflicts', exist_ok=True)
                plt.savefig(f'out_3d_conflicts/3d_bicycle_trajectory_{vehicle_id}_conflict-overview_{file_tag}_FCO{FCO_share*100:.0f}%_FBO{FBO_share*100:.0f}%.png', 
                           bbox_inches='tight', dpi=300)
                print(f'\nConflict overview plot for bicycle {vehicle_id} saved as out_3d_conflicts/3d_bicycle_trajectory_{vehicle_id}_conflict-overview_{file_tag}_FCO{FCO_share*100:.0f}%_FBO{FBO_share*100:.0f}%.png.')
                plt.close(fig_3d)

                # Create individual conflict plots for each conflict
                for foe_id, foe_conflicts in conflicts_by_foe.items():
                    most_severe = max(foe_conflicts, key=lambda x: x['severity'])
                    conflict_time = most_severe['time']
                    
                    # Create new figure for this conflict
                    fig_3d = plt.figure(figsize=(15, 12))
                    ax_3d = fig_3d.add_subplot(111, projection='3d')
                    
                    # Set up static scene with relative coordinates
                    ax_3d.set_xlabel('Distance (m)')
                    ax_3d.set_ylabel('Distance (m)')
                    ax_3d.set_zlabel('Time (s)')
                    ax_3d.set_xlim(0, maxx - minx)
                    ax_3d.set_ylim(0, maxy - miny)
                    ax_3d.set_zlim(base_z, z_max + z_padding)

                    # Set background color of the planes and grid style
                    ax_3d.xaxis.pane.fill = False
                    ax_3d.yaxis.pane.fill = False
                    ax_3d.zaxis.pane.fill = False
                    ax_3d.xaxis._axinfo['grid'].update(linestyle="--")
                    ax_3d.yaxis._axinfo['grid'].update(linestyle="--")
                    ax_3d.zaxis._axinfo['grid'].update(linestyle="--")
                    
                    # Calculate aspect ratios using relative coordinates
                    dx = maxx - minx
                    dy = maxy - miny
                    dz = (z_max + z_padding) - base_z
                    max_xy = max(dx, dy)
                    aspect_ratios = [dx/max_xy, dy/max_xy, dz/max_xy * 2.0]
                    ax_3d.set_box_aspect(aspect_ratios)
                    
                    # Set view angle
                    ax_3d.view_init(elev=35, azim=285)
                    ax_3d.set_axisbelow(True)

                    # Create base plane with relative coordinates
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

                    # Plot roads with relative coordinates
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

                    # Plot buildings and parks with relative coordinates
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
                    
                    # Plot bicycle trajectory with detection-based segments
                    segments = {'detected': [], 'undetected': []}
                    current_points = []
                    current_detected = None
                    detection_buffer = []  # Buffer to store recent detection states

                    for x, y, t in trajectory:
                        # Get detection status for this time
                        is_detected = False
                        if vehicle_id in bicycle_detection_data:
                            # for det_time, det_status in bicycle_detection_data[vehicle_id]:
                            for detection in bicycle_detection_data[vehicle_id]:
                                det_time = detection[0]
                                det_status = detection[1]
                                if abs(det_time - t) < step_length:
                                    is_detected = det_status
                                    break
                        
                        # Update detection buffer
                        detection_buffer.append(is_detected)
                        if len(detection_buffer) > max_gap_bridge:
                            detection_buffer.pop(0)
                        
                        # If there's any detection in the last                         
                        recent_detection = any(detection_buffer[-3:]) if len(detection_buffer) >= 3 else is_detected
                        if not recent_detection and len(detection_buffer) >= max_gap_bridge:
                            if any(detection_buffer[:3]) and any(detection_buffer[-3:]):
                                smoothed_detection = True
                            else:
                                smoothed_detection = False
                        else:
                            smoothed_detection = recent_detection
                        
                        if current_detected is None:
                            current_detected = smoothed_detection
                            current_points = [(rel_x(x), rel_y(y), t)]
                        elif smoothed_detection != current_detected:
                            if len(current_points) >= min_segment_length:
                                segments['detected' if current_detected else 'undetected'].append(current_points)
                                current_points = [(rel_x(x), rel_y(y), t)]
                                current_detected = smoothed_detection
                            else:
                                current_points.append((rel_x(x), rel_y(y), t))
                        else:
                            current_points.append((rel_x(x), rel_y(y), t))

                    if current_points:
                        segments['detected' if current_detected else 'undetected'].append(current_points)

                    # Plot segments with appropriate colors
                    all_segments = []  # Store all segments in order
                    for state in ['undetected', 'detected']:
                        for segment in segments[state]:
                            if len(segment) > 1:
                                all_segments.append((state, segment))
                    
                    # Sort segments by time to ensure proper order
                    all_segments.sort(key=lambda x: x[1][0][2])  # Sort by first time point
                    
                    for i, (state, segment) in enumerate(all_segments):
                        if len(segment) > 1:
                            x_coords, y_coords, times = zip(*segment)
                            color = 'cornflowerblue' if state == 'detected' else 'darkslateblue'
                            
                            # Plot 3D trajectory
                            ax_3d.plot(x_coords, y_coords, times, 
                                    color=color, linewidth=2, alpha=1.0,
                                    zorder=1000)
                            # Plot ground projection
                            ax_3d.plot(x_coords, y_coords, [base_z]*len(x_coords),
                                    color=color, linestyle='--', linewidth=2, alpha=1.0,
                                    zorder=1000)
                            
                            # Add projection plane
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
                                    next_state, next_segment = all_segments[i+1]
                                    next_x, next_y, next_t = next_segment[0]
                                    transition_quad = [
                                        (x_coords[j+1], y_coords[j+1], times[j+1]),
                                        (next_x, next_y, next_t),
                                        (next_x, next_y, base_z),
                                        (x_coords[j+1], y_coords[j+1], base_z)
                                    ]
                                    plane_vertices.append(transition_quad)
                            
                            proj_plane = Poly3DCollection(plane_vertices, alpha=0.2)
                            proj_plane.set_facecolor(color)
                            proj_plane.set_edgecolor('none')
                            proj_plane.set_sort_zpos(900)
                            ax_3d.add_collection3d(proj_plane)
                    
                    # Plot conflict point with relative coordinates
                    size = 50 + (most_severe['severity'] * 100)
                    ax_3d.scatter(rel_x(most_severe['x']), rel_y(most_severe['y']), most_severe['time'],
                                color='firebrick', s=size, marker='o',
                                facecolors='none', edgecolors='firebrick',
                                linewidth=0.75, zorder=1001)
                    
                    # Add vertical line from base with relative coordinates
                    ax_3d.plot([rel_x(most_severe['x']), rel_x(most_severe['x'])],
                             [rel_y(most_severe['y']), rel_y(most_severe['y'])],
                             [base_z, most_severe['time']],
                             color='firebrick', linestyle=':', alpha=0.3,
                             zorder=1001)
                    
                    # Plot foe trajectory if available with relative coordinates
                    foe_traj = None
                    if foe_id in foe_trajectories:
                        foe_traj = foe_trajectories[foe_id]
                    elif foe_id in three_dimensional_conflict_plots.completed_foes:
                        foe_traj = three_dimensional_conflict_plots.completed_foes[foe_id]

                    if foe_traj:  # Only proceed if we have a trajectory
                        # Filter trajectory points to only those within the bounding box
                        filtered_foe_traj = []
                        for x, y, t in foe_traj:
                            point = Point(x, y)
                            if bbox_transformed.contains(point):
                                # Ensure coordinates are within bounds
                                x_rel = np.clip(rel_x(x), 0, maxx - minx)
                                y_rel = np.clip(rel_y(y), 0, maxy - miny)
                                filtered_foe_traj.append((x_rel, y_rel, t))
                        
                        if filtered_foe_traj:  # Only plot if we have valid points
                            foe_x_rel, foe_y_rel, foe_times = zip(*filtered_foe_traj)
                        
                        # 1. Plot ground projection
                        ax_3d.plot(foe_x_rel, foe_y_rel, [base_z]*len(foe_x_rel),
                                 color='black', linestyle='--', 
                                 linewidth=2, alpha=0.7, zorder=1000)

                        # 2. Create projection plane
                        foe_plane_vertices = []
                        for i in range(len(foe_x_rel)-1):
                            quad = [
                                (foe_x_rel[i], foe_y_rel[i], foe_times[i]),
                                (foe_x_rel[i+1], foe_y_rel[i+1], foe_times[i+1]),
                                (foe_x_rel[i+1], foe_y_rel[i+1], base_z),
                                (foe_x_rel[i], foe_y_rel[i], base_z)
                            ]
                            foe_plane_vertices.append(quad)
                        
                        foe_proj_plane = Poly3DCollection(foe_plane_vertices, alpha=0.1)
                        foe_proj_plane.set_facecolor('black')
                        foe_proj_plane.set_edgecolor('none')
                        foe_proj_plane.set_sort_zpos(900)
                        ax_3d.add_collection3d(foe_proj_plane)

                        # 3. Plot 3D trajectory
                        ax_3d.plot(foe_x_rel, foe_y_rel, foe_times,
                                 color='black', linewidth=2, alpha=0.7,
                                 zorder=1000)
                        
                        # 4. Add foe label with relative coordinates
                        ax_3d.text(foe_x_rel[-1], foe_y_rel[-1], base_z,
                                 f'foe {foe_id}',
                                 color='black',
                                 horizontalalignment='right',
                                 verticalalignment='bottom',
                                 rotation=90,
                                 bbox=dict(facecolor='white', alpha=1.0, edgecolor='none'),
                                 zorder=999)
                    
                    # Add bicycle label with relative coordinates
                    x_coords_rel = [rel_x(x) for x, _, _ in trajectory]
                    y_coords_rel = [rel_y(y) for _, y, _ in trajectory]
                    ax_3d.text(x_coords_rel[-1], y_coords_rel[-1], base_z,
                             f'bicycle {vehicle_id}',
                             color='darkslateblue',
                             horizontalalignment='right',
                             verticalalignment='bottom',
                             rotation=90,
                             bbox=dict(facecolor='white', alpha=1.0, edgecolor='none'),
                             zorder=1000)

                    # Create legend for individual conflict plot
                    handles = [
                        plt.Line2D([0], [0], color='darkslateblue', linewidth=2, label='Bicycle Undetected'),
                        plt.Line2D([0], [0], color='cornflowerblue', linewidth=2, label='Bicycle Detected'),
                        plt.Line2D([0], [0], color='black', linewidth=2, label='Foe Trajectory'),
                        plt.Line2D([0], [0], color='black', linestyle='--', label='Ground Projections'),
                        plt.Line2D([0], [0], marker='o', color='firebrick', linestyle='None', 
                                  markerfacecolor='none', markersize=10, label='Potential Conflict')
                    ]
                    ax_3d.legend(handles=handles, loc='upper left')
                    
                    # Save individual conflict plot
                    plt.savefig(f'out_3d_conflicts/3d_bicycle_trajectory_{vehicle_id}_conflict_{foe_id}_{file_tag}_FCO{FCO_share*100:.0f}%_FBO{FBO_share*100:.0f}%.png', 
                               bbox_inches='tight', dpi=300)
                    print(f'\nIndividual conflict plot for bicycle {vehicle_id} and foe {foe_id} saved as out_3d_conflicts/3d_bicycle_trajectory_{vehicle_id}_conflict_{foe_id}_{file_tag}_FCO{FCO_share*100:.0f}%_FBO{FBO_share*100:.0f}%.png.')
                    plt.close(fig_3d)
            
            else:
                # If no conflicts, create bicycle trajectory plot (without conflict points)
                fig_3d = plt.figure(figsize=(15, 12))
                ax_3d = fig_3d.add_subplot(111, projection='3d')
                
                # Set up static scene with relative coordinates
                ax_3d.set_xlabel('Distance (m)')
                ax_3d.set_ylabel('Distance (m)')
                ax_3d.set_zlabel('Time (s)')
                ax_3d.set_xlim(0, maxx - minx)
                ax_3d.set_ylim(0, maxy - miny)
                ax_3d.set_zlim(base_z, z_max + z_padding)

                # Set background color of the planes and grid style
                ax_3d.xaxis.pane.fill = False
                ax_3d.yaxis.pane.fill = False
                ax_3d.zaxis.pane.fill = False
                ax_3d.xaxis._axinfo['grid'].update(linestyle="--")
                ax_3d.yaxis._axinfo['grid'].update(linestyle="--")
                ax_3d.zaxis._axinfo['grid'].update(linestyle="--")
                
                # Calculate aspect ratios using relative coordinates
                dx = maxx - minx
                dy = maxy - miny
                dz = (z_max + z_padding) - base_z
                max_xy = max(dx, dy)
                aspect_ratios = [dx/max_xy, dy/max_xy, dz/max_xy * 2.0]
                ax_3d.set_box_aspect(aspect_ratios)
                
                # Set view angle
                ax_3d.view_init(elev=35, azim=285)
                ax_3d.set_axisbelow(True)

                # Create base plane with relative coordinates
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

                # Plot roads with relative coordinates
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

                # Plot buildings and parks with relative coordinates
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

                # Plot bicycle trajectory with detection-based segments
                segments = {'detected': [], 'undetected': []}
                current_points = []
                current_detected = None
                detection_buffer = []  # Buffer to store recent detection states

                for x, y, t in trajectory:
                    # Get detection status for this time
                    is_detected = False
                    if vehicle_id in bicycle_detection_data:
                        # for det_time, det_status in bicycle_detection_data[vehicle_id]:
                        for detection in bicycle_detection_data[vehicle_id]:
                            det_time = detection[0]
                            det_status = detection[1]
                            if abs(det_time - t) < step_length:
                                is_detected = det_status
                                break
                    
                    # Update detection buffer
                    detection_buffer.append(is_detected)
                    if len(detection_buffer) > max_gap_bridge:
                        detection_buffer.pop(0)
                    
                    # If there's any detection in the last 3 frames, consider it detected
                    recent_detection = any(detection_buffer[-3:]) if len(detection_buffer) >= 3 else is_detected
                    if not recent_detection and len(detection_buffer) >= max_gap_bridge:
                        if any(detection_buffer[:3]) and any(detection_buffer[-3:]):
                            smoothed_detection = True
                        else:
                            smoothed_detection = False
                    else:
                        smoothed_detection = recent_detection
                    
                    if current_detected is None:
                        current_detected = smoothed_detection
                        current_points = [(rel_x(x), rel_y(y), t)]
                    elif smoothed_detection != current_detected:
                        if len(current_points) >= min_segment_length:
                            segments['detected' if current_detected else 'undetected'].append(current_points)
                            current_points = [(rel_x(x), rel_y(y), t)]
                            current_detected = smoothed_detection
                        else:
                            current_points.append((rel_x(x), rel_y(y), t))
                    else:
                        current_points.append((rel_x(x), rel_y(y), t))

                if current_points:
                    segments['detected' if current_detected else 'undetected'].append(current_points)

                # Plot segments with appropriate colors
                all_segments = []  # Store all segments in order
                for state in ['undetected', 'detected']:
                    for segment in segments[state]:
                        if len(segment) > 1:
                            all_segments.append((state, segment))
                
                # Sort segments by time to ensure proper order
                all_segments.sort(key=lambda x: x[1][0][2])  # Sort by first time point
                
                for i, (state, segment) in enumerate(all_segments):
                    if len(segment) > 1:
                        x_coords, y_coords, times = zip(*segment)
                        color = 'cornflowerblue' if state == 'detected' else 'darkslateblue'
                        
                        # Plot 3D trajectory
                        ax_3d.plot(x_coords, y_coords, times, 
                                color=color, linewidth=2, alpha=1.0,
                                zorder=1000)
                        # Plot ground projection
                        ax_3d.plot(x_coords, y_coords, [base_z]*len(x_coords),
                                color=color, linestyle='--', linewidth=2, alpha=1.0,
                                zorder=1000)
                        
                        # Add projection plane
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
                                next_state, next_segment = all_segments[i+1]
                                next_x, next_y, next_t = next_segment[0]
                                transition_quad = [
                                    (x_coords[j+1], y_coords[j+1], times[j+1]),
                                    (next_x, next_y, next_t),
                                    (next_x, next_y, base_z),
                                    (x_coords[j+1], y_coords[j+1], base_z)
                                ]
                                plane_vertices.append(transition_quad)
                        
                        proj_plane = Poly3DCollection(plane_vertices, alpha=0.2)
                        proj_plane.set_facecolor(color)
                        proj_plane.set_edgecolor('none')
                        proj_plane.set_sort_zpos(900)
                        ax_3d.add_collection3d(proj_plane)
                
                # Add bicycle label with relative coordinates
                x_coords_rel = [rel_x(x) for x, _, _ in trajectory]
                y_coords_rel = [rel_y(y) for _, y, _ in trajectory]
                ax_3d.text(x_coords_rel[-1], y_coords_rel[-1], base_z,
                          f'bicycle {vehicle_id}',
                          color='darkslateblue',
                          horizontalalignment='right',
                          verticalalignment='bottom',
                          rotation=90,
                          bbox=dict(facecolor='white', alpha=1.0, edgecolor='none'),
                          zorder=1000)

                # Create legend for bicycle trajectory plot
                handles = [
                    plt.Line2D([0], [0], color='darkslateblue', linewidth=2, label='Bicycle Undetected'),
                    plt.Line2D([0], [0], color='cornflowerblue', linewidth=2, label='Bicycle Detected'),
                    plt.Line2D([0], [0], color='black', linestyle='--', label='Ground Projection')
                ]
                ax_3d.legend(handles=handles, loc='upper left')
                
                # Save bicycle trajectory plot
                os.makedirs('out_3d_conflicts', exist_ok=True)
                plt.savefig(f'out_3d_conflicts/3d_bicycle_trajectory_{vehicle_id}_{file_tag}_FCO{FCO_share*100:.0f}%_FBO{FBO_share*100:.0f}%.png', 
                           bbox_inches='tight', dpi=300)
                print(f'\n3D bicycle trajectory plot saved as out_3d_conflicts/3d_bicycle_trajectory_{vehicle_id}_{file_tag}_FCO{FCO_share*100:.0f}%_FBO{FBO_share*100:.0f}%.png.')
                plt.close(fig_3d)
            
            # Clean up trajectories
            del conflict_bicycle_trajectories[vehicle_id]
            # if vehicle_id in bicycle_conflicts:
            #     del bicycle_conflicts[vehicle_id]

def three_dimensional_detection_plots(frame):
    """
    Creates a 3D visualization of bicycle trajectories where the z=0 plane shows the static scene.
    The time dimension is represented on the z-axis. Observer vehicles (FCOs/FBOs) are shown
    with their detection ranges, and bicycles are colored based on their detection status.
    """
    global fig_3d, ax_3d, total_steps, bicycle_trajectories, transformer, flow_ids, bicycle_detection_data, detection_bicycle_trajectories

    # Initialize transformer at frame 0
    if frame == 0:
        transformer = pyproj.Transformer.from_crs('EPSG:4326', 'EPSG:32632', always_xy=True)
        detection_bicycle_trajectories.clear()
        flow_ids.clear()
        bicycle_detection_data = {}  # Store detection status over time
        
        # Add new dictionary for observer trajectories
        if not hasattr(three_dimensional_detection_plots, 'observer_trajectories'):
            three_dimensional_detection_plots.observer_trajectories = {}
        three_dimensional_detection_plots.observer_trajectories.clear()

    # Ensure transformer is initialized
    if transformer is None:
        transformer = pyproj.Transformer.from_crs('EPSG:4326', 'EPSG:32632', always_xy=True)

    # Create bounding box for clipping
    bbox = box(west, south, east, north)
    bbox_transformed = shapely.ops.transform(
        lambda x, y: transformer.transform(x, y), 
        bbox
    )

    # Get bounds of transformed bounding box for relative coordinates
    minx, miny, maxx, maxy = bbox_transformed.bounds
            
    # Define functions for relative coordinates
    rel_x = lambda x: x - minx
    rel_y = lambda y: y - miny

    # Get current vehicles and collect data
    current_vehicles = set(traci.vehicle.getIDList())
    current_time = frame * step_length

    # Dictionary to store current frame's detections
    current_frame_detections = {}

    # Collect observer positions and their detections
    for vehicle_id in current_vehicles:
        vehicle_type = traci.vehicle.getTypeID(vehicle_id)
        if vehicle_type in ["floating_car_observer", "floating_bike_observer"]:
            x_sumo, y_sumo = traci.vehicle.getPosition(vehicle_id)
            lon, lat = traci.simulation.convertGeo(x_sumo, y_sumo)
            x_utm, y_utm = transformer.transform(lon, lat)
            
            # Store observer trajectory
            if not hasattr(three_dimensional_detection_plots, 'observer_trajectories'):
                three_dimensional_detection_plots.observer_trajectories = {}
            if vehicle_id not in three_dimensional_detection_plots.observer_trajectories:
                three_dimensional_detection_plots.observer_trajectories[vehicle_id] = {
                    'type': vehicle_type,
                    'trajectory': []
                }
            three_dimensional_detection_plots.observer_trajectories[vehicle_id]['trajectory'].append(
                (x_utm, y_utm, current_time)
            )

            # Check which bicycles this observer can detect
            for bicycle_id in bicycle_detection_data:
                # Get the latest detection status for this bicycle
                if bicycle_detection_data[bicycle_id]:  # If there are any detections
                    latest_detection = bicycle_detection_data[bicycle_id][-1]
                    if latest_detection[0] == current_time and latest_detection[1]:  # If detected at current time
                        if bicycle_id not in current_frame_detections:
                            current_frame_detections[bicycle_id] = []
                        current_frame_detections[bicycle_id].append(vehicle_id)

    for vehicle_id in current_vehicles:
        vehicle_type = traci.vehicle.getTypeID(vehicle_id)
        x_sumo, y_sumo = traci.vehicle.getPosition(vehicle_id)
        lon, lat = traci.simulation.convertGeo(x_sumo, y_sumo)
        x_utm, y_utm = transformer.transform(lon, lat)
        point = Point(x_utm, y_utm)

        if bbox_transformed.contains(point):
            # Store positions for bicycles
            if vehicle_type in ["bicycle", "DEFAULT_BIKETYPE", "floating_bike_observer"]:
                # Get current observers that detect this bicycle
                current_observers = []
                if vehicle_id in current_frame_detections:
                    current_observers = [{'id': obs_id} for obs_id in current_frame_detections[vehicle_id]]
                
                # Determine if bicycle is detected
                is_detected = len(current_observers) > 0

                # Store detection status with observers
                if vehicle_id not in bicycle_detection_data:
                    bicycle_detection_data[vehicle_id] = []
                bicycle_detection_data[vehicle_id].append((current_time, is_detected, current_observers))

                # Store trajectory data
                flow_id = vehicle_id.rsplit('.', 1)[0]
                flow_ids.add(flow_id)
                if vehicle_id not in detection_bicycle_trajectories:
                    detection_bicycle_trajectories[vehicle_id] = []
                detection_bicycle_trajectories[vehicle_id].append((x_utm, y_utm, current_time))

    # Create plots for bicycles that have finished their trajectory
    for vehicle_id in list(detection_bicycle_trajectories.keys()):
        if vehicle_id not in current_vehicles:
            trajectory = detection_bicycle_trajectories[vehicle_id]
            
            # Create new figure for this bicycle
            fig_3d = plt.figure(figsize=(15, 12))
            ax_3d = fig_3d.add_subplot(111, projection='3d')
            
            # Calculate z range with padding
            z_min = min(t for _, _, t in trajectory)
            z_max = max(t for _, _, t in trajectory)
            z_padding = (z_max - z_min) * 0.05
            base_z = z_min - z_padding

            # Set up the plot (axis labels, limits, background, etc.)
            ax_3d.set_xlabel('Distance (m)')
            ax_3d.set_ylabel('Distance (m)')
            ax_3d.set_zlabel('Time (s)')
            ax_3d.set_xlim(0, maxx - minx)
            ax_3d.set_ylim(0, maxy - miny)
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
            ax_3d.set_box_aspect(aspect_ratios)
            
            # Set view angle
            ax_3d.view_init(elev=35, azim=285)
            ax_3d.set_axisbelow(True)

            # Create base plane with relative coordinates
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

            # Plot roads with relative coordinates
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

            # Plot buildings and parks with relative coordinates
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
                if len(detection_info) == 3:  # New format
                    time, is_detected, observers = detection_info
                else:  # Old format (2-tuple)
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
            current_observers = set()

            # Process trajectory points with smoothing
            for x, y, t in trajectory:
                # Get detection status for this time
                detection_info = next((info for info in bicycle_detection_data[vehicle_id] 
                                    if abs(info[0] - t) < step_length), None)
                
                if detection_info:
                    if len(detection_info) == 3:  # New format
                        is_detected = detection_info[1]
                        time_observers = set(obs['id'] for obs in detection_info[2])
                    else:  # Old format
                        is_detected = detection_info[1]
                        time_observers = set()
                else:
                    is_detected = False
                    time_observers = set()

                # Update detection buffer for smoothing
                detection_buffer.append(is_detected)
                if len(detection_buffer) > max_gap_bridge:
                    detection_buffer.pop(0)

                # Apply smoothing logic
                recent_detection = any(detection_buffer[-3:]) if len(detection_buffer) >= 3 else is_detected
                if not recent_detection and len(detection_buffer) >= max_gap_bridge:
                    smoothed_detection = any(detection_buffer[:3]) and any(detection_buffer[-3:])
                else:
                    smoothed_detection = recent_detection

                # Handle segment creation
                if current_detected is None:
                    current_detected = smoothed_detection
                    current_points = [(rel_x(x), rel_y(y), t)]
                    current_observers = time_observers
                elif smoothed_detection != current_detected:
                    if len(current_points) >= min_segment_length:
                        segments['detected' if current_detected else 'undetected'].append(
                            (current_points, list({'id': obs_id} for obs_id in current_observers))
                        )
                        current_points = [(rel_x(x), rel_y(y), t)]
                        current_detected = smoothed_detection
                        current_observers = time_observers
                    else:
                        current_points.append((rel_x(x), rel_y(y), t))
                        current_observers.update(time_observers)
                else:
                    current_points.append((rel_x(x), rel_y(y), t))
                    if smoothed_detection:
                        current_observers.update(time_observers)

            # Add final segment
            if current_points:
                segments['detected' if current_detected else 'undetected'].append(
                    (current_points, list({'id': obs_id} for obs_id in current_observers))
                )

            # Plot bicycle trajectory segments
            all_segments = [(state, segment) for state in ['undetected', 'detected'] 
                          for segment in segments[state]]
            all_segments.sort(key=lambda x: x[1][0][0][2])  # Sort by first time point

            # Plot bicycle segments
            for state, (points, observers) in all_segments:
                if len(points) > 1:
                    x_coords, y_coords, times = zip(*points)
                    color = 'cornflowerblue' if state == 'detected' else 'darkslateblue'
                    
                    # Plot 3D trajectory
                    ax_3d.plot(x_coords, y_coords, times,
                             color=color, linewidth=2, alpha=1.0, zorder=1000)
                    
                    # Plot ground projection
                    ax_3d.plot(x_coords, y_coords, [base_z]*len(x_coords),
                             color=color, linestyle='--', linewidth=2, alpha=1.0, zorder=1000)
                    
                    # Add projection plane
                    plane_vertices = []
                    for i in range(len(points)-1):
                        quad = [
                            (x_coords[i], y_coords[i], times[i]),
                            (x_coords[i+1], y_coords[i+1], times[i+1]),
                            (x_coords[i+1], y_coords[i+1], base_z),
                            (x_coords[i], y_coords[i], base_z)
                        ]
                        plane_vertices.append(quad)
                    
                    proj_plane = Poly3DCollection(plane_vertices, alpha=0.2)
                    proj_plane.set_facecolor(color)
                    proj_plane.set_edgecolor('none')
                    proj_plane.set_sort_zpos(900)
                    ax_3d.add_collection3d(proj_plane)

            # Plot observer trajectories
            detecting_observers = set()
            for detection_info in bicycle_detection_data[vehicle_id]:
                if len(detection_info) == 3:  # New format with observers
                    _, is_detected, observers = detection_info
                    if is_detected and observers:
                        detecting_observers.update(obs['id'] for obs in observers)

            for observer_id in detecting_observers:
                if observer_id in three_dimensional_detection_plots.observer_trajectories:
                    obs_traj = three_dimensional_detection_plots.observer_trajectories[observer_id]['trajectory']
                    
                    # Filter trajectory points within time range and bounding box
                    rel_obs_traj = []
                    for x, y, t in obs_traj:
                        if base_z <= t <= z_max + z_padding:
                            point = Point(x, y)
                            if bbox_transformed.contains(point):
                                rel_obs_traj.append((rel_x(x), rel_y(y), t))
                    
                    if rel_obs_traj:
                        obs_x, obs_y, obs_t = zip(*rel_obs_traj)
                        
                        # First, create segments based on detection status
                        observer_segments = []
                        current_segment = {'points': [], 'is_detecting': None}
                        
                        for i in range(len(obs_x)):
                            is_detecting = any(
                                info[1] and any(obs['id'] == observer_id for obs in info[2])
                                for info in bicycle_detection_data[vehicle_id]
                                if abs(info[0] - obs_t[i]) < step_length
                            )
                            
                            if current_segment['is_detecting'] is None:
                                current_segment['is_detecting'] = is_detecting
                            
                            if is_detecting != current_segment['is_detecting']:
                                if current_segment['points']:
                                    observer_segments.append(current_segment)
                                current_segment = {'points': [], 'is_detecting': is_detecting}
                            
                            current_segment['points'].append((obs_x[i], obs_y[i], obs_t[i]))
                        
                        if current_segment['points']:
                            observer_segments.append(current_segment)
                        
                        # Now plot each segment
                        for segment in observer_segments:
                            seg_x, seg_y, seg_t = zip(*segment['points'])
                            color = 'darkred' if segment['is_detecting'] else 'indianred'
                            
                            # Plot 3D trajectory
                            ax_3d.plot(seg_x, seg_y, seg_t,
                                     color=color, linewidth=2, alpha=1.0, zorder=1000)
                            
                            # Plot ground projection (as one segment)
                            ax_3d.plot(seg_x, seg_y, [base_z]*len(seg_x),
                                     color=color, linestyle='--', linewidth=2, alpha=0.7, zorder=1000)
                        
                        # Add projection plane for observer
                        obs_plane_vertices = []
                        for i in range(len(obs_x)-1):
                            quad = [
                                (obs_x[i], obs_y[i], obs_t[i]),
                                (obs_x[i+1], obs_y[i+1], obs_t[i+1]),
                                (obs_x[i+1], obs_y[i+1], base_z),
                                (obs_x[i], obs_y[i], base_z)
                            ]
                            obs_plane_vertices.append(quad)
                        
                        obs_proj_plane = Poly3DCollection(obs_plane_vertices, alpha=0.1)
                        obs_proj_plane.set_facecolor('indianred')
                        obs_proj_plane.set_edgecolor('none')
                        obs_proj_plane.set_sort_zpos(900)
                        ax_3d.add_collection3d(obs_proj_plane)

            # Add bicycle label with relative coordinates
            x_coords_rel = [rel_x(x) for x, y, _ in trajectory]
            y_coords_rel = [rel_y(y) for _, y, _ in trajectory]
            ax_3d.text(x_coords_rel[-1], y_coords_rel[-1], base_z,
                      f'bicycle {vehicle_id}',
                      color='darkslateblue',
                      horizontalalignment='right',
                      verticalalignment='bottom',
                      rotation=90,
                      bbox=dict(facecolor='white', alpha=1.0, edgecolor='none'),
                      zorder=1000)

            # Create legend
            handles = [
                plt.Line2D([0], [0], color='darkslateblue', linewidth=2, label='Bicycle Undetected'),
                plt.Line2D([0], [0], color='cornflowerblue', linewidth=2, label='Bicycle Detected'),
                plt.Line2D([0], [0], color='indianred', linewidth=2, label='Observer Vehicle'),
                plt.Line2D([0], [0], color='darkred', linewidth=2, label='Observer Vehicle (Detecting)'),
                plt.Line2D([0], [0], color='black', linestyle='--', label='Ground Projections')
            ]
            ax_3d.legend(handles=handles, loc='upper left')

            # Save individual bicycle plot
            os.makedirs('out_3d_detections', exist_ok=True)
            plt.savefig(
                f'out_3d_detections/3d_bicycle_trajectory_{vehicle_id}_{file_tag}_FCO{FCO_share*100:.0f}%_FBO{FBO_share*100:.0f}%.png',
                bbox_inches='tight', dpi=300
            )
            print(f'\n3D detection plot saved for bicycle {vehicle_id}')
            plt.close(fig_3d)
        
            # Clean up trajectories
            del detection_bicycle_trajectories[vehicle_id]

def three_dimensional_conflict_plots_gif(frame):
    """
    Creates a 3D visualization of bicycle trajectories where the z=0 plane shows the static scene.
    Automatically generates plots for each bicycle when their trajectory ends.
    """
    global fig_3d, ax_3d, total_steps, bicycle_trajectories, transformer, flow_ids, bicycle_conflicts, foe_trajectories, bicycle_detection_data, conflict_gif_trajectories

    # Initialize transformer at frame 0
    if frame == 0:
        transformer = pyproj.Transformer.from_crs('EPSG:4326', 'EPSG:32632', always_xy=True)
        conflict_gif_trajectories.clear()
        flow_ids.clear()
        bicycle_detection_data = {}  # Add this line

    # Ensure transformer is initialized
    if transformer is None:
        transformer = pyproj.Transformer.from_crs('EPSG:4326', 'EPSG:32632', always_xy=True)

    # Create bounding box for clipping
    bbox = box(west, south, east, north)  # Create box from original coordinates
    bbox_transformed = shapely.ops.transform(
        lambda x, y: transformer.transform(x, y), 
        bbox
    )  # Transform to UTM coordinates

    # Collect positions for this frame
    current_vehicles = set(traci.vehicle.getIDList())
    departed_foes = set(foe_trajectories.keys()) - current_vehicles
    
    # Track foes that have complete trajectories but haven't been processed
    if not hasattr(three_dimensional_conflict_plots_gif, 'completed_foes'):
        three_dimensional_conflict_plots_gif.completed_foes = {}
    
    # Store completed foe trajectories before removing them
    for foe_id in departed_foes:
        if foe_id not in three_dimensional_conflict_plots_gif.completed_foes:
            three_dimensional_conflict_plots_gif.completed_foes[foe_id] = foe_trajectories[foe_id]

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
                is_detected = False
                # Store detection status
                if vehicle_id not in bicycle_detection_data:
                    bicycle_detection_data[vehicle_id] = []
                bicycle_detection_data[vehicle_id].append((current_time, is_detected))
                
                # Store detection status
                if vehicle_id not in bicycle_detection_data:
                    bicycle_detection_data[vehicle_id] = []
                bicycle_detection_data[vehicle_id].append((current_time, is_detected))

                # Store trajectory data
                flow_id = vehicle_id.rsplit('.', 1)[0]
                flow_ids.add(flow_id)
                if vehicle_id not in conflict_gif_trajectories:
                    conflict_gif_trajectories[vehicle_id] = []
                conflict_gif_trajectories[vehicle_id].append((x_utm, y_utm, current_time))
                
                # Check for conflicts
                try:
                    # Check both leader and follower vehicles
                    leader = traci.vehicle.getLeader(vehicle_id)
                    follower = traci.vehicle.getFollower(vehicle_id)
                    
                    potential_foes = []
                    if leader and leader[0] != '':
                        potential_foes.append(('leader', *leader))
                    if follower and follower[0] != '':
                        potential_foes.append(('follower', *follower))
                    
                    # Get current distance for the bicycle
                    distance = traci.vehicle.getDistance(vehicle_id)

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
                        
                        # Check for conflict
                        if (ttc < TTC_THRESHOLD or pet < PET_THRESHOLD or drac > DRAC_THRESHOLD):
                            if vehicle_id not in bicycle_conflicts:
                                bicycle_conflicts[vehicle_id] = []
                            
                            # Calculate severity
                            ttc_severity = 1 - (ttc / TTC_THRESHOLD) if ttc < TTC_THRESHOLD else 0
                            pet_severity = 1 - (pet / PET_THRESHOLD) if pet < PET_THRESHOLD else 0
                            drac_severity = min(drac / DRAC_THRESHOLD, 1.0) if drac > 0 else 0
                            
                            conflict_severity = max(ttc_severity, pet_severity, drac_severity)
                            
                            # Get vehicle position for 3D plotting
                            x_sumo, y_sumo = traci.vehicle.getPosition(vehicle_id)
                            lon, lat = traci.simulation.convertGeo(x_sumo, y_sumo)
                            x_utm, y_utm = transformer.transform(lon, lat)
                            
                            bicycle_conflicts[vehicle_id].append({
                                'distance': distance,
                                'time': current_time,
                                'ttc': ttc,
                                'pet': pet,
                                'drac': drac,
                                'severity': conflict_severity,
                                'foe_type': foe_type,
                                'foe_id': foe_id,
                                'x': x_utm,  # Add x coordinate
                                'y': y_utm   # Add y coordinate
                            })
                
                except Exception as e:
                    if frame % 100 == 0:
                        print(f"Error in conflict detection for {vehicle_id}: {str(e)}")
            
            # Store positions for all other vehicles (potential foes)
            else:
                if vehicle_id not in foe_trajectories:
                    foe_trajectories[vehicle_id] = []
                foe_trajectories[vehicle_id].append((x_utm, y_utm, current_time))

    # Check for bicycles that have finished their trajectory
    # finished_bicycles = set(conflict_gif_trajectories.keys()) - current_vehicles
    
    # Generate plots for finished bicycles
    for vehicle_id in list(conflict_gif_trajectories.keys()):
        # if len(conflict_gif_trajectories[vehicle_id]) > 0:  # Only plot if we have trajectory data
        if vehicle_id not in current_vehicles:
            if vehicle_id in bicycle_conflicts and bicycle_conflicts[vehicle_id]:
                # Check if all foe trajectories are complete
                all_foes_complete = True
                for conflict in bicycle_conflicts[vehicle_id]:
                    foe_id = conflict['foe_id']
                    if foe_id in current_vehicles:  # If foe still in simulation
                        all_foes_complete = False
                        break
                
                if not all_foes_complete:
                    continue  # Skip plotting until all foes are complete

            # Now proceed with plotting
            trajectory = conflict_gif_trajectories[vehicle_id]
            x_coords, y_coords, times = zip(*trajectory)
            
            # Calculate z range with padding
            z_min = min(times)
            z_max = max(times)
            z_padding = (z_max - z_min) * 0.05
            base_z = z_min - z_padding

            if vehicle_id in bicycle_conflicts and bicycle_conflicts[vehicle_id]:
                # Create conflict overview plot
                fig_3d = plt.figure(figsize=(15, 12))
                ax_3d = fig_3d.add_subplot(111, projection='3d')
                
                # Get bounds of transformed bounding box
                minx, miny, maxx, maxy = bbox_transformed.bounds
                
                # Set axis labels and limits
                ax_3d.set_xlabel('X (m)')
                ax_3d.set_ylabel('Y (m)')
                ax_3d.set_zlabel('Time (s)')
                ax_3d.set_xlim(minx, maxx)
                ax_3d.set_ylim(miny, maxy)
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
                
                # Normalize the dimensions to make z-axis more prominent
                max_xy = max(dx, dy)
                aspect_ratios = [dx/max_xy, dy/max_xy, dz/max_xy * 2.0]
                
                # Set box aspect with normalized ratios
                ax_3d.set_box_aspect(aspect_ratios)
                
                # Set view angle
                ax_3d.view_init(elev=35, azim=285)
                ax_3d.set_axisbelow(True)

                # Create base plane
                base_vertices = [
                    [minx, miny, base_z],
                    [maxx, miny, base_z],
                    [maxx, maxy, base_z],
                    [minx, maxy, base_z]
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
                                xs = np.clip(xs, minx, maxx)
                                ys = np.clip(ys, miny, maxy)
                                verts = [(x, y, base_z) for x, y in zip(xs, ys)]
                                poly = Poly3DCollection([verts], alpha=0.5)
                                poly.set_facecolor('lightgray')
                                poly.set_edgecolor('darkgray')
                                poly.set_linewidth(1.0)
                                poly.set_sort_zpos(-1)
                                ax_3d.add_collection3d(poly)
                        
                        elif isinstance(clipped_geom, LineString):
                            xs, ys = clipped_geom.xy
                            xs = np.clip(xs, minx, maxx)
                            ys = np.clip(ys, miny, maxy)
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
                                        xs = np.clip(xs, minx, maxx)
                                        ys = np.clip(ys, miny, maxy)
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

                # Plot bicycle trajectory with detection-based segments
                segments = {'detected': [], 'undetected': []}
                current_points = []
                current_detected = None
                detection_buffer = []  # Buffer to store recent detection states

                for x, y, t in trajectory:
                    # Get detection status for this time
                    is_detected = False
                    if vehicle_id in bicycle_detection_data:
                        # Update this loop to handle both 2-tuple and 3-tuple formats
                        for detection_info in bicycle_detection_data[vehicle_id]:
                            if len(detection_info) == 3:  # New format
                                det_time, det_status, _ = detection_info
                            else:  # Old format
                                det_time, det_status = detection_info
                            
                            if abs(det_time - t) < step_length:
                                is_detected = det_status
                                break
                    
                    # Update detection buffer
                    detection_buffer.append(is_detected)
                    if len(detection_buffer) > max_gap_bridge:
                        detection_buffer.pop(0)
                    
                    # If there's any detection in the last 3 frames, consider it detected
                    recent_detection = any(detection_buffer[-3:]) if len(detection_buffer) >= 3 else is_detected
                    if not recent_detection and len(detection_buffer) >= max_gap_bridge:
                        if any(detection_buffer[:3]) and any(detection_buffer[-3:]):
                            smoothed_detection = True
                        else:
                            smoothed_detection = False
                    else:
                        smoothed_detection = recent_detection
                    
                    if current_detected is None:
                        current_detected = smoothed_detection
                        current_points = [(x, y, t)]
                    elif smoothed_detection != current_detected:
                        if len(current_points) >= min_segment_length:
                            segments['detected' if current_detected else 'undetected'].append(current_points)
                            current_points = [(x, y, t)]
                            current_detected = smoothed_detection
                        else:
                            current_points.append((x, y, t))
                    else:
                        current_points.append((x, y, t))

                if current_points:
                    segments['detected' if current_detected else 'undetected'].append(current_points)

                # Plot segments with appropriate colors
                all_segments = []  # Store all segments in order
                for state in ['undetected', 'detected']:
                    for segment in segments[state]:
                        if len(segment) > 1:
                            all_segments.append((state, segment))
                
                # Sort segments by time to ensure proper order
                all_segments.sort(key=lambda x: x[1][0][2])  # Sort by first time point
                
                for i, (state, segment) in enumerate(all_segments):
                    if len(segment) > 1:
                        x_coords, y_coords, times = zip(*segment)
                        color = 'cornflowerblue' if state == 'detected' else 'darkslateblue'
                        
                        # Plot 3D trajectory
                        ax_3d.plot(x_coords, y_coords, times, 
                                color=color, linewidth=2, alpha=1.0,
                                zorder=1000)
                        # Plot ground projection
                        ax_3d.plot(x_coords, y_coords, [base_z]*len(x_coords),
                                color=color, linestyle='--', linewidth=2, alpha=0.7,
                                zorder=1000)
                        
                        # Add projection plane
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
                                next_state, next_segment = all_segments[i+1]
                                next_x, next_y, next_t = next_segment[0]
                                transition_quad = [
                                    (x_coords[j+1], y_coords[j+1], times[j+1]),
                                    (next_x, next_y, next_t),
                                    (next_x, next_y, base_z),
                                    (x_coords[j+1], y_coords[j+1], base_z)
                                ]
                                plane_vertices.append(transition_quad)
                        
                        proj_plane = Poly3DCollection(plane_vertices, alpha=0.2)
                        proj_plane.set_facecolor(color)
                        proj_plane.set_edgecolor('none')
                        proj_plane.set_sort_zpos(900)
                        ax_3d.add_collection3d(proj_plane)

                # Group conflicts by foe and plot most severe ones
                conflicts_by_foe = {}
                for conflict in bicycle_conflicts[vehicle_id]:
                    foe_id = conflict.get('foe_id')
                    if foe_id:
                        if foe_id not in conflicts_by_foe:
                            conflicts_by_foe[foe_id] = []
                        conflicts_by_foe[foe_id].append(conflict)
                
                # Plot conflict points in conflict overview plot
                for foe_conflicts in conflicts_by_foe.values():
                    most_severe = max(foe_conflicts, key=lambda x: x['severity'])
                    size = 50 + (most_severe['severity'] * 100)
                    
                    # Plot conflict point
                    ax_3d.scatter(most_severe['x'], most_severe['y'], most_severe['time'],
                                color='firebrick', s=size, marker='o',
                                facecolors='none', edgecolors='firebrick',
                                linewidth=0.75, zorder=1001)
                    
                    # Add vertical line from base
                    ax_3d.plot([most_severe['x'], most_severe['x']],
                             [most_severe['y'], most_severe['y']],
                             [base_z, most_severe['time']],
                             color='firebrick', linestyle=':', alpha=0.3,
                             zorder=1001)

                # Add bicycle label
                ax_3d.text(x_coords[-1], y_coords[-1], base_z,
                          f'bicycle {vehicle_id}',
                          color='darkslateblue',
                          horizontalalignment='right',
                          verticalalignment='bottom',
                          rotation=90,
                          bbox=dict(facecolor='white', alpha=1.0, edgecolor='none'),
                          zorder=1000)

                # Create legend for conflict overview plot
                handles = [
                    plt.Line2D([0], [0], color='darkslateblue', linewidth=2, label='Bicycle Undetected'),
                    plt.Line2D([0], [0], color='cornflowerblue', linewidth=2, label='Bicycle Detected'),
                    plt.Line2D([0], [0], color='black', linestyle='--', label='Ground Projection'),
                    plt.Line2D([0], [0], marker='o', color='firebrick', linestyle='None', 
                              markerfacecolor='none', markersize=10, label='Potential Conflict')
                ]
                ax_3d.legend(handles=handles, loc='upper left')
                
                # Save conflict overview plot
                base_filename = f'bicycle_{vehicle_id}_conflict_overview_FCO{FCO_share*100:.0f}_FBO{FBO_share*100:.0f}'
                save_rotating_view_frames(ax_3d, base_filename)
                plt.close(fig_3d)
                create_rotating_view_gif(base_filename)

                # Create individual conflict plots for each conflict
                for foe_id, foe_conflicts in conflicts_by_foe.items():
                    most_severe = max(foe_conflicts, key=lambda x: x['severity'])
                    conflict_time = most_severe['time']
                    
                    # Create new figure for this conflict
                    fig_3d = plt.figure(figsize=(15, 12))
                    ax_3d = fig_3d.add_subplot(111, projection='3d')
                    
                    # Set up static scene
                    minx, miny, maxx, maxy = bbox_transformed.bounds
                    
                    # Set axis labels and limits
                    ax_3d.set_xlabel('X (m)')
                    ax_3d.set_ylabel('Y (m)')
                    ax_3d.set_zlabel('Time (s)')
                    ax_3d.set_xlim(minx, maxx)
                    ax_3d.set_ylim(miny, maxy)
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
                    ax_3d.set_box_aspect(aspect_ratios)
                    
                    # Set view angle
                    ax_3d.view_init(elev=35, azim=285)
                    ax_3d.set_axisbelow(True)

                    # Create base plane
                    base_vertices = [
                        [minx, miny, base_z],
                        [maxx, miny, base_z],
                        [maxx, maxy, base_z],
                        [minx, maxy, base_z]
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
                                    xs = np.clip(xs, minx, maxx)
                                    ys = np.clip(ys, miny, maxy)
                                    verts = [(x, y, base_z) for x, y in zip(xs, ys)]
                                    poly = Poly3DCollection([verts], alpha=0.5)
                                    poly.set_facecolor('lightgray')
                                    poly.set_edgecolor('darkgray')
                                    poly.set_linewidth(1.0)
                                    poly.set_sort_zpos(-1)
                                    ax_3d.add_collection3d(poly)
                            
                            elif isinstance(clipped_geom, LineString):
                                xs, ys = clipped_geom.xy
                                xs = np.clip(xs, minx, maxx)
                                ys = np.clip(ys, miny, maxy)
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
                                            xs = np.clip(xs, minx, maxx)
                                            ys = np.clip(ys, miny, maxy)
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

                    # Plot bicycle trajectory with detection-based segments
                    segments = {'detected': [], 'undetected': []}
                    current_points = []
                    current_detected = None
                    detection_buffer = []  # Buffer to store recent detection states

                    for x, y, t in trajectory:
                        # Get detection status for this time
                        is_detected = False
                        if vehicle_id in bicycle_detection_data:
                            for detection_info in bicycle_detection_data[vehicle_id]:
                                if len(detection_info) == 3:  # New format
                                    det_time, det_status, _ = detection_info
                                else:  # Old format
                                    det_time, det_status = detection_info
                                
                                if abs(det_time - t) < step_length:
                                    is_detected = det_status
                                    break
                        
                        # Update detection buffer
                        detection_buffer.append(is_detected)
                        if len(detection_buffer) > max_gap_bridge:
                            detection_buffer.pop(0)
                        
                        # If there's any detection in the last 3 frames, consider it detected
                        recent_detection = any(detection_buffer[-3:]) if len(detection_buffer) >= 3 else is_detected
                        if not recent_detection and len(detection_buffer) >= max_gap_bridge:
                            if any(detection_buffer[:3]) and any(detection_buffer[-3:]):
                                smoothed_detection = True
                            else:
                                smoothed_detection = False
                        else:
                            smoothed_detection = recent_detection
                        
                        if current_detected is None:
                            current_detected = smoothed_detection
                            current_points = [(x, y, t)]
                        elif smoothed_detection != current_detected:
                            if len(current_points) >= min_segment_length:
                                segments['detected' if current_detected else 'undetected'].append(current_points)
                                current_points = [(x, y, t)]
                                current_detected = smoothed_detection
                            else:
                                current_points.append((x, y, t))
                        else:
                            current_points.append((x, y, t))

                    if current_points:
                        segments['detected' if current_detected else 'undetected'].append(current_points)

                    # Plot segments with appropriate colors
                    all_segments = []  # Store all segments in order
                    for state in ['undetected', 'detected']:
                        for segment in segments[state]:
                            if len(segment) > 1:
                                all_segments.append((state, segment))
                    
                    # Sort segments by time to ensure proper order
                    all_segments.sort(key=lambda x: x[1][0][2])  # Sort by first time point
                    
                    for i, (state, segment) in enumerate(all_segments):
                        if len(segment) > 1:
                            x_coords, y_coords, times = zip(*segment)
                            color = 'cornflowerblue' if state == 'detected' else 'darkslateblue'
                            
                            # Plot 3D trajectory
                            ax_3d.plot(x_coords, y_coords, times, 
                                    color=color, linewidth=2, alpha=1.0,
                                    zorder=1000)
                            # Plot ground projection
                            ax_3d.plot(x_coords, y_coords, [base_z]*len(x_coords),
                                    color=color, linestyle='--', linewidth=2, alpha=0.7,
                                    zorder=1000)
                            
                            # Add projection plane
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
                                    next_state, next_segment = all_segments[i+1]
                                    next_x, next_y, next_t = next_segment[0]
                                    transition_quad = [
                                        (x_coords[j+1], y_coords[j+1], times[j+1]),
                                        (next_x, next_y, next_t),
                                        (next_x, next_y, base_z),
                                        (x_coords[j+1], y_coords[j+1], base_z)
                                    ]
                                    plane_vertices.append(transition_quad)
                            
                            proj_plane = Poly3DCollection(plane_vertices, alpha=0.2)
                            proj_plane.set_facecolor(color)
                            proj_plane.set_edgecolor('none')
                            proj_plane.set_sort_zpos(999)
                            ax_3d.add_collection3d(proj_plane)
                    
                    # Plot conflict point
                    size = 50 + (most_severe['severity'] * 100)
                    ax_3d.scatter(most_severe['x'], most_severe['y'], most_severe['time'],
                                color='firebrick', s=size, marker='o',
                                facecolors='none', edgecolors='firebrick',
                                linewidth=0.75, zorder=1001)
                    
                    # Add vertical line from base to conflict point                    
                    ax_3d.plot([most_severe['x'], most_severe['x']],
                             [most_severe['y'], most_severe['y']],
                             [base_z, most_severe['time']],
                             color='firebrick', linestyle=':', alpha=0.3,
                             zorder=1001)
                    
                    # Plot foe trajectory if available with relative coordinates
                    foe_traj = None
                    if foe_id in foe_trajectories:
                        foe_traj = foe_trajectories[foe_id]
                    elif foe_id in three_dimensional_conflict_plots.completed_foes:
                        foe_traj = three_dimensional_conflict_plots.completed_foes[foe_id]

                    if foe_traj:  # Only proceed if we have a trajectory
                        # Filter trajectory points to only those within the bounding box
                        filtered_foe_traj = []
                        for x, y, t in foe_traj:
                            point = Point(x, y)
                            if bbox_transformed.contains(point):
                                # Ensure coordinates are within bounds
                                x_rel = np.clip(rel_x(x), 0, maxx - minx)
                                y_rel = np.clip(rel_y(y), 0, maxy - miny)
                                filtered_foe_traj.append((x_rel, y_rel, t))
                        
                        if filtered_foe_traj:  # Only plot if we have valid points
                            foe_x_rel, foe_y_rel, foe_times = zip(*filtered_foe_traj)
                        
                        # 1. Plot ground projection
                        ax_3d.plot(foe_x, foe_y, [base_z]*len(foe_x),
                                 color='black', linestyle='--', 
                                 linewidth=2, alpha=0.7, zorder=1000)

                        # 2. Create projection plane
                        foe_plane_vertices = []
                        for i in range(len(foe_x)-1):
                            quad = [
                                (foe_x[i], foe_y[i], foe_times[i]),
                                (foe_x[i+1], foe_y[i+1], foe_times[i+1]),
                                (foe_x[i+1], foe_y[i+1], base_z),
                                (foe_x[i], foe_y[i], base_z)
                            ]
                            foe_plane_vertices.append(quad)
                        
                        foe_proj_plane = Poly3DCollection(foe_plane_vertices, alpha=0.1)
                        foe_proj_plane.set_facecolor('black')
                        foe_proj_plane.set_edgecolor('none')
                        foe_proj_plane.set_sort_zpos(900)
                        ax_3d.add_collection3d(foe_proj_plane)

                        # 3. Plot 3D trajectory
                        ax_3d.plot(foe_x, foe_y, foe_times,
                                 color='black', linewidth=2, alpha=0.7,
                                 zorder=1000)
                        
                        # 4. Add foe label
                        ax_3d.text(foe_x[-1], foe_y[-1], base_z,
                                 f'foe {foe_id}',
                                 color='black',
                                 horizontalalignment='right',
                                 verticalalignment='bottom',
                                 rotation=90,
                                 bbox=dict(facecolor='white', alpha=1.0, edgecolor='none'),
                                 zorder=999)
                    
                    # Add bicycle label
                    ax_3d.text(x_coords[-1], y_coords[-1], base_z,
                             f'bicycle {vehicle_id}',
                             color='darkslateblue',
                             horizontalalignment='right',
                             verticalalignment='bottom',
                             rotation=90,
                             bbox=dict(facecolor='white', alpha=1.0, edgecolor='none'),
                             zorder=1000)

                    # Create legend for individual conflict plot
                    handles = [
                        plt.Line2D([0], [0], color='darkslateblue', linewidth=2, label='Bicycle Undetected'),
                        plt.Line2D([0], [0], color='cornflowerblue', linewidth=2, label='Bicycle Detected'),
                        plt.Line2D([0], [0], color='black', linewidth=2, label='Foe Trajectory'),
                        plt.Line2D([0], [0], color='black', linestyle='--', label='Ground Projections'),
                        plt.Line2D([0], [0], marker='o', color='firebrick', linestyle='None', 
                                  markerfacecolor='none', markersize=10, label='Potential Conflict')
                    ]
                    ax_3d.legend(handles=handles, loc='upper left')
                    
                    # Save individual conflict plot
                    base_filename = f'bicycle_{vehicle_id}_conflict_{foe_id}_FCO{FCO_share*100:.0f}_FBO{FBO_share*100:.0f}'
                    save_rotating_view_frames(ax_3d, base_filename)
                    plt.close(fig_3d)
                    create_rotating_view_gif(base_filename)
            
            else:
                # If no conflicts, create bicycle trajectory plot (without conflict points)
                fig_3d = plt.figure(figsize=(15, 12))
                ax_3d = fig_3d.add_subplot(111, projection='3d')
                
                # Set up static scene
                minx, miny, maxx, maxy = bbox_transformed.bounds
                
                # Set axis labels and limits
                ax_3d.set_xlabel('X (m)')
                ax_3d.set_ylabel('Y (m)')
                ax_3d.set_zlabel('Time (s)')
                ax_3d.set_xlim(minx, maxx)
                ax_3d.set_ylim(miny, maxy)
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
                ax_3d.set_box_aspect(aspect_ratios)
                
                # Set view angle
                ax_3d.view_init(elev=35, azim=285)
                ax_3d.set_axisbelow(True)

                # Create base plane
                base_vertices = [
                    [minx, miny, base_z],
                    [maxx, miny, base_z],
                    [maxx, maxy, base_z],
                    [minx, maxy, base_z]
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
                                xs = np.clip(xs, minx, maxx)
                                ys = np.clip(ys, miny, maxy)
                                verts = [(x, y, base_z) for x, y in zip(xs, ys)]
                                poly = Poly3DCollection([verts], alpha=0.5)
                                poly.set_facecolor('lightgray')
                                poly.set_edgecolor('darkgray')
                                poly.set_linewidth(1.0)
                                poly.set_sort_zpos(-1)
                                ax_3d.add_collection3d(poly)
                        
                        elif isinstance(clipped_geom, LineString):
                            xs, ys = clipped_geom.xy
                            xs = np.clip(xs, minx, maxx)
                            ys = np.clip(ys, miny, maxy)
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
                                        xs = np.clip(xs, minx, maxx)
                                        ys = np.clip(ys, miny, maxy)
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

                # Plot bicycle trajectory with detection-based segments
                segments = {'detected': [], 'undetected': []}
                current_points = []
                current_detected = None
                detection_buffer = []  # Buffer to store recent detection states

                for x, y, t in trajectory:
                    # Get detection status for this time
                    is_detected = False
                    if vehicle_id in bicycle_detection_data:
                        for detection_info in bicycle_detection_data[vehicle_id]:
                            if len(detection_info) == 3:  # New format
                                det_time, det_status, _ = detection_info
                            else:  # Old format
                                det_time, det_status = detection_info
                            
                            if abs(det_time - t) < step_length:
                                is_detected = det_status
                                break
                    
                    # Update detection buffer
                    detection_buffer.append(is_detected)
                    if len(detection_buffer) > max_gap_bridge:
                        detection_buffer.pop(0)
                    
                    # If there's any detection in the last 3 frames, consider it detected
                    recent_detection = any(detection_buffer[-3:]) if len(detection_buffer) >= 3 else is_detected
                    if not recent_detection and len(detection_buffer) >= max_gap_bridge:
                        if any(detection_buffer[:3]) and any(detection_buffer[-3:]):
                            smoothed_detection = True
                        else:
                            smoothed_detection = False
                    else:
                        smoothed_detection = recent_detection
                    
                    if current_detected is None:
                        current_detected = smoothed_detection
                        current_points = [(x, y, t)]
                    elif smoothed_detection != current_detected:
                        if len(current_points) >= min_segment_length:
                            segments['detected' if current_detected else 'undetected'].append(current_points)
                            current_points = [(x, y, t)]
                            current_detected = smoothed_detection
                        else:
                            current_points.append((x, y, t))
                    else:
                        current_points.append((x, y, t))

                if current_points:
                    segments['detected' if current_detected else 'undetected'].append(current_points)

                # Plot segments with appropriate colors
                all_segments = []  # Store all segments in order
                for state in ['undetected', 'detected']:
                    for segment in segments[state]:
                        if len(segment) > 1:
                            all_segments.append((state, segment))
                
                # Sort segments by time to ensure proper order
                all_segments.sort(key=lambda x: x[1][0][2])  # Sort by first time point
                
                for i, (state, segment) in enumerate(all_segments):
                    if len(segment) > 1:
                        x_coords, y_coords, times = zip(*segment)
                        color = 'cornflowerblue' if state == 'detected' else 'darkslateblue'
                        
                        # Plot 3D trajectory
                        ax_3d.plot(x_coords, y_coords, times, 
                                color=color, linewidth=2, alpha=1.0,
                                zorder=1000)
                        # Plot ground projection
                        ax_3d.plot(x_coords, y_coords, [base_z]*len(x_coords),
                                color=color, linestyle='--', linewidth=2, alpha=0.7,
                                zorder=1000)
                        
                        # Add projection plane
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
                                next_state, next_segment = all_segments[i+1]
                                next_x, next_y, next_t = next_segment[0]
                                transition_quad = [
                                    (x_coords[j+1], y_coords[j+1], times[j+1]),
                                    (next_x, next_y, next_t),
                                    (next_x, next_y, base_z),
                                    (x_coords[j+1], y_coords[j+1], base_z)
                                ]
                                plane_vertices.append(transition_quad)
                        
                        proj_plane = Poly3DCollection(plane_vertices, alpha=0.2)
                        proj_plane.set_facecolor(color)
                        proj_plane.set_edgecolor('none')
                        proj_plane.set_sort_zpos(999)
                        ax_3d.add_collection3d(proj_plane)
                
                # Add bicycle label
                ax_3d.text(x_coords[-1], y_coords[-1], base_z,
                          f'bicycle {vehicle_id}',
                          color='darkslateblue',
                          horizontalalignment='right',
                          verticalalignment='bottom',
                          rotation=90,
                          bbox=dict(facecolor='white', alpha=1.0, edgecolor='none'),
                          zorder=1000)

                # Create legend for bicycle trajectory plot
                handles = [
                    plt.Line2D([0], [0], color='darkslateblue', linewidth=2, label='Bicycle Undetected'),
                    plt.Line2D([0], [0], color='cornflowerblue', linewidth=2, label='Bicycle Detected'),
                    plt.Line2D([0], [0], color='black', linestyle='--', label='Ground Projection')
                ]
                ax_3d.legend(handles=handles, loc='upper left')
                
                # Save bicycle trajectory plot
                base_filename = f'bicycle_{vehicle_id}_FCO{FCO_share*100:.0f}_FBO{FBO_share*100:.0f}'
                save_rotating_view_frames(ax_3d, base_filename)
                plt.close(fig_3d)
                create_rotating_view_gif(base_filename)
            
            # Clean up trajectories
            del conflict_gif_trajectories[vehicle_id]
            # if vehicle_id in bicycle_conflicts:
            #     del bicycle_conflicts[vehicle_id]

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

    # Ensure transformer is initialized
    if transformer is None:
        transformer = pyproj.Transformer.from_crs('EPSG:4326', 'EPSG:32632', always_xy=True)
    # Create bounding box for clipping
    bbox = box(west, south, east, north)
    bbox_transformed = shapely.ops.transform(
        lambda x, y: transformer.transform(x, y), 
        bbox
    )

    # Get bounds of transformed bounding box for relative coordinates
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
            ax_3d.set_box_aspect(aspect_ratios)
            
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
                if len(detection_buffer) > max_gap_bridge:
                    detection_buffer.pop(0)
                # Apply smoothing logic
                recent_detection = any(detection_buffer[-3:]) if len(detection_buffer) >= 3 else is_detected
                if not recent_detection and len(detection_buffer) >= max_gap_bridge:
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
                    if len(current_points) >= min_segment_length:
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
            # Add bicycle label
            ax_3d.text(rel_x(x_coords[-1]), rel_y(y_coords[-1]), base_z,
                      f'bicycle {vehicle_id}',
                      color='darkslateblue',
                      horizontalalignment='right',
                      verticalalignment='bottom',
                      rotation=90,
                      bbox=dict(facecolor='white', alpha=1.0, edgecolor='none'),
                      zorder=1000)
            # Create legend
            handles = [
                plt.Line2D([0], [0], color='darkslateblue', linewidth=2, label='Bicycle Undetected'),
                plt.Line2D([0], [0], color='cornflowerblue', linewidth=2, label='Bicycle Detected'),
                plt.Line2D([0], [0], color='indianred', linewidth=2, label='Observer Vehicle'),
                plt.Line2D([0], [0], color='darkred', linewidth=2, label='Observer Vehicle (Detecting)'),
                plt.Line2D([0], [0], color='black', linestyle='--', label='Ground Projections')
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
    # Create necessary directories
    if AnimatedThreeDimensionalConflictPlots:
        os.makedirs('out_3d_conflicts_gif/rotation_frames', exist_ok=True)
    if AnimatedThreeDimensionalDetectionPlots:
        os.makedirs('out_3d_detections_gif/rotation_frames', exist_ok=True)
        
    # Start from bird's eye view and smoothly transition both angles
    # Start: (90° elevation, 270° azimuth)    - bird's eye view
    # End: (35° elevation, 285° azimuth)    - final view
    t = np.linspace(0, 1, n_frames)
    t_azim = t**2
    elevations = np.linspace(90, 35, n_frames)
    azimuths = 270 + (t_azim * 15)
    
    # Save frames based on which type is active and has a filename
    if AnimatedThreeDimensionalConflictPlots and base_filename_conflict:
        for i, (elev, azim) in enumerate(zip(elevations, azimuths)):
            ax_3d.view_init(elev=elev, azim=azim)
            plt.savefig(f'out_3d_conflicts_gif/rotation_frames/{base_filename_conflict}_frame_{i:03d}.png', 
                    dpi=300)
            
    if AnimatedThreeDimensionalDetectionPlots and base_filename_detection:
        for i, (elev, azim) in enumerate(zip(elevations, azimuths)):
            ax_3d.view_init(elev=elev, azim=azim)
            plt.savefig(f'out_3d_detections_gif/rotation_frames/{base_filename_detection}_frame_{i:03d}.png', 
                    dpi=300)
            
def create_rotating_view_gif(base_filename_conflict=None, base_filename_detection=None, duration=0.1):
    """Helper function to create GIF from saved frames"""
    # Get all frames for this plot
    frames_conflict = None
    frames_detection = None
    
    # Simplified conditions - check each type independently
    if AnimatedThreeDimensionalConflictPlots and base_filename_conflict:   
        frames_conflict = sorted(glob.glob(f'out_3d_conflicts_gif/rotation_frames/{base_filename_conflict}_frame_*.png'))
    if AnimatedThreeDimensionalDetectionPlots and base_filename_detection:
        frames_detection = sorted(glob.glob(f'out_3d_detections_gif/rotation_frames/{base_filename_detection}_frame_*.png'))
    
    # Create GIFs for available frames
    if frames_conflict:
        images_conflict = [imageio.imread(frame) for frame in frames_conflict]
        output_file_conflict = f'out_3d_conflicts_gif/{base_filename_conflict}_rotation.gif'
        imageio.mimsave(output_file_conflict, images_conflict, format='GIF', duration=duration)
        print(f'\nCreated rotating view animation: {output_file_conflict}')
        # Clean up conflict frames
        for frame in frames_conflict:
            os.remove(frame)
            
    if frames_detection:
        images_detection = [imageio.imread(frame) for frame in frames_detection]
        output_file_detection = f'out_3d_detections_gif/{base_filename_detection}_rotation.gif'
        imageio.mimsave(output_file_detection, images_detection, format='GIF', duration=duration)
        print(f'\nCreated rotating view animation: {output_file_detection}')
        # Clean up detection frames
        for frame in frames_detection:
            os.remove(frame)
# ------

# ---------------------
# APPLICATIONS - URBAN TRAFFIC CONTROL
# ---------------------

# yet to come ...

# ---------------------
# EVALUATION
# ---------------------

def bicycle_safety_evaluation():
    """
    Evaluates bicycle safety metrics using data from the simulation logs.
    Generates a comprehensive evaluation file with safety metrics and analysis.
    """
    
    def calculate_segment_distance(segment):
        """Helper function to calculate total distance of a segment using coordinates"""
        if len(segment) < 2:
            return 0.0
        
        # Calculate Euclidean distance between consecutive points
        dx = np.diff(segment['x_coord'].values)
        dy = np.diff(segment['y_coord'].values)
        distances = np.sqrt(dx**2 + dy**2)
        return np.sum(distances)
    
    # Load logging data
    trajectory_file = f'out_logging/log_bicycle_trajectories_{file_tag}_FCO{FCO_share*100:.0f}%_FBO{FBO_share*100:.0f}%.csv'
    df = pd.read_csv(trajectory_file,
                     skiprows=23,  # Adjust this number to match the number of header lines
                     skip_blank_lines=True,
                     skipinitialspace=True,
                     na_values=['-'])
    conflict_file = f'out_logging/log_conflicts_{file_tag}_FCO{FCO_share*100:.0f}%_FBO{FBO_share*100:.0f}%.csv'
    conflict_df = pd.read_csv(conflict_file,
                            skiprows=20,  # Adjust this number to match the number of header lines
                            skip_blank_lines=True,
                            skipinitialspace=True,
                            na_values=['-'])

    # Initialization of dictionaries
    bicycle_conflict_metrics = {}
    flow_conflict_metrics = {}
    bicycle_metrics = {}
    flow_metrics = {}
    
    # Bicycle detection rates
    for bicycle_id in df['vehicle_id'].unique():
        bike_data = df[df['vehicle_id'] == bicycle_id]
        flow_id = bicycle_id.split('.')[0]  # Extract flow ID
        # Calculate overall temporal detection rates
        total_steps = len(bike_data)
        detected_steps = len(bike_data[bike_data['is_detected'] == 1])
        temporal_rate = (detected_steps / total_steps) * 100 if total_steps > 0 else 0
        # Calculate overall spatial detection rates using coordinates
        total_distance = calculate_segment_distance(bike_data)
        detected_segments = bike_data[bike_data['is_detected'] == 1]
        total_detected_distance = calculate_segment_distance(detected_segments)
        spatial_rate = (total_detected_distance / total_distance) * 100 if total_distance > 0 else 0
        # Calculate overall spatio-temporal detection rate
        spatiotemporal_rate = (temporal_rate + spatial_rate) / 2
        # Calculate important area metrics
        important_area_data = bike_data[bike_data['in_test_area'] == 1].copy()
        important_area_data.loc[:, 'time_diff'] = important_area_data['time_step'].diff()
        important_area_data.loc[:, 'segment_id'] = (important_area_data['time_diff'] > 1).cumsum()
        # Calculate temporal detection rate in important areas
        important_area_steps = len(important_area_data)
        important_area_detected_steps = len(important_area_data[important_area_data['is_detected'] == 1])
        important_temporal_rate = (important_area_detected_steps / important_area_steps * 100 
                                if important_area_steps > 0 else 0)
        # Create segments based on discontinuities in time_step
        important_area_data['time_diff'] = important_area_data['time_step'].diff()
        important_area_data['segment_id'] = (important_area_data['time_diff'] > 1).cumsum()
        # Group by continuous segments
        important_segments = important_area_data.groupby('segment_id')
        total_important_distance = 0
        total_important_detected_distance = 0
        for segment_id, segment in important_segments:
            if len(segment) > 1:
                # Calculate total distance for this segment
                segment_distance = calculate_segment_distance(segment)
                total_important_distance += segment_distance
                # Calculate detected distance within this segment
                detected_subsegments = segment[segment['is_detected'] == 1]
                if not detected_subsegments.empty and len(detected_subsegments) > 1:
                    detected_distance = calculate_segment_distance(detected_subsegments)
                    total_important_detected_distance += detected_distance
        # Calculate important spatial rate
        important_spatial_rate = (total_important_detected_distance / total_important_distance * 100 
                                if total_important_distance > 0 else 0)
        # Calculate important spatio-temporal rate
        important_spatiotemporal_rate = (important_temporal_rate + important_spatial_rate) / 2
        # Store individual bicycle metrics
        bicycle_metrics[bicycle_id] = {
            'temporal_rate': temporal_rate,
            'spatial_rate': spatial_rate,
            'spatiotemporal_rate': spatiotemporal_rate,
            'total_distance': total_distance,
            'total_time_steps': total_steps,
            'detected_distance': total_detected_distance,
            'detected_steps': detected_steps,
            'important_temporal_rate': important_temporal_rate,
            'important_spatial_rate': important_spatial_rate,
            'important_spatiotemporal_rate': important_spatiotemporal_rate,
            'important_total_distance': total_important_distance,
            'important_total_steps': important_area_steps,
            'important_detected_distance': total_important_detected_distance,
            'important_detected_steps': important_area_detected_steps,
            'flow_id': flow_id
        }
        # Initialize or update flow metrics
        if flow_id not in flow_metrics:
            flow_metrics[flow_id] = {
                'bicycles': [],
                'total_steps': 0,
                'detected_steps': 0,
                'total_distance': 0,
                'detected_distance': 0,
                'important_total_steps': 0,
                'important_detected_steps': 0,
                'important_total_distance': 0,
                'important_detected_distance': 0
            }
        # Aggregate metrics per flow
        flow_metrics[flow_id]['bicycles'].append(bicycle_id)
        flow_metrics[flow_id]['total_steps'] += total_steps
        flow_metrics[flow_id]['detected_steps'] += detected_steps
        flow_metrics[flow_id]['total_distance'] += total_distance
        flow_metrics[flow_id]['detected_distance'] += total_detected_distance
        flow_metrics[flow_id]['important_total_steps'] += important_area_steps
        flow_metrics[flow_id]['important_detected_steps'] += important_area_detected_steps
        flow_metrics[flow_id]['important_total_distance'] += total_important_distance
        flow_metrics[flow_id]['important_detected_distance'] += total_important_detected_distance
    # Calculate individual detection rates
    for flow_id in flow_metrics:
        metrics = flow_metrics[flow_id]
        metrics['temporal_rate'] = (metrics['detected_steps'] / metrics['total_steps'] * 100 
                                  if metrics['total_steps'] > 0 else 0)
        metrics['spatial_rate'] = (metrics['detected_distance'] / metrics['total_distance'] * 100 
                                 if metrics['total_distance'] > 0 else 0)
        metrics['spatiotemporal_rate'] = (metrics['temporal_rate'] + metrics['spatial_rate']) / 2
        metrics['important_temporal_rate'] = (metrics['important_detected_steps'] / metrics['important_total_steps'] * 100 
                                       if metrics['important_total_steps'] > 0 else 0)
        metrics['important_spatial_rate'] = (metrics['important_detected_distance'] / metrics['important_total_distance'] * 100 
                                      if metrics['important_total_distance'] > 0 else 0)
        metrics['important_spatiotemporal_rate'] = (metrics['important_temporal_rate'] + metrics['important_spatial_rate']) / 2
    avg_temporal_rate = np.mean([m['temporal_rate'] for m in bicycle_metrics.values()])
    avg_spatial_rate = np.mean([m['spatial_rate'] for m in bicycle_metrics.values()])
    avg_spatiotemporal_rate = np.mean([m['spatiotemporal_rate'] for m in bicycle_metrics.values()])
    avg_important_temporal_rate = np.mean([m['important_temporal_rate'] for m in bicycle_metrics.values()])
    avg_important_spatial_rate = np.mean([m['important_spatial_rate'] for m in bicycle_metrics.values()])
    avg_important_spatiotemporal_rate = np.mean([m['important_spatiotemporal_rate'] for m in bicycle_metrics.values()])
    # Calculate flow-based detection rates
    avg_flow_temporal_rate = np.mean([m['temporal_rate'] for m in flow_metrics.values()])
    avg_flow_spatial_rate = np.mean([m['spatial_rate'] for m in flow_metrics.values()])
    avg_flow_spatiotemporal_rate = np.mean([m['spatiotemporal_rate'] for m in flow_metrics.values()])
    avg_flow_important_temporal_rate = np.mean([m['important_temporal_rate'] for m in flow_metrics.values()])
    avg_flow_important_spatial_rate = np.mean([m['important_spatial_rate'] for m in flow_metrics.values()])
    avg_flow_important_spatiotemporal_rate = np.mean([m['important_spatiotemporal_rate'] for m in flow_metrics.values()])
    # Calculate system-wide detection rates
    total_system_steps = sum(metrics['total_time_steps'] for metrics in bicycle_metrics.values())
    total_system_detected_steps = sum(metrics['detected_steps'] for metrics in bicycle_metrics.values())
    total_system_distance = sum(metrics['total_distance'] for metrics in bicycle_metrics.values())
    total_system_detected_distance = sum(metrics['detected_distance'] for metrics in bicycle_metrics.values())
    total_system_important_steps = sum(metrics['important_total_steps'] for metrics in bicycle_metrics.values())
    total_system_important_detected_steps = sum(metrics['important_detected_steps'] for metrics in bicycle_metrics.values())
    total_system_important_distance = sum(metrics['important_total_distance'] for metrics in bicycle_metrics.values())
    total_system_important_detected_distance = sum(metrics['important_detected_distance'] for metrics in bicycle_metrics.values())
    overall_temporal_rate = (total_system_detected_steps / total_system_steps * 100 
                           if total_system_steps > 0 else 0)
    overall_spatial_rate = (total_system_detected_distance / total_system_distance * 100 
                          if total_system_distance > 0 else 0)
    overall_spatiotemporal_rate = (overall_temporal_rate + overall_spatial_rate) / 2
    overall_important_temporal_rate = (total_system_important_detected_steps / total_system_important_steps * 100 
                                if total_system_important_steps > 0 else 0)
    overall_important_spatial_rate = (total_system_important_detected_distance / total_system_important_distance * 100 
                               if total_system_important_distance > 0 else 0)
    overall_important_spatiotemporal_rate = (overall_important_temporal_rate + overall_important_spatial_rate) / 2

    # Conflict detection rates
    for bicycle_id in conflict_df['bicycle_id'].unique():
        bike_conflicts = conflict_df[conflict_df['bicycle_id'] == bicycle_id].copy()
        flow_id = bicycle_id.split('.')[0]
        # Calculate temporal conflict detection rate
        total_conflict_steps = len(bike_conflicts)
        detected_conflict_steps = len(bike_conflicts[bike_conflicts['is_detected'] == 1])
        conflict_temporal_rate = (detected_conflict_steps / total_conflict_steps * 100 
                                if total_conflict_steps > 0 else 0)
        # Calculate spatial conflict detection rate
        total_conflict_distance = bike_conflicts['distance'].max() - bike_conflicts['distance'].min()
        detected_conflicts = bike_conflicts[bike_conflicts['is_detected'] == 1]
        detected_conflict_distance = (detected_conflicts['distance'].max() - detected_conflicts['distance'].min() 
                                    if not detected_conflicts.empty else 0)
        conflict_spatial_rate = (detected_conflict_distance / total_conflict_distance * 100 
                               if total_conflict_distance > 0 else 0)
        # Calculate spatio-temporal conflict detection rate
        conflict_spatiotemporal_rate = (conflict_temporal_rate + conflict_spatial_rate) / 2
        # Store individual conflict metrics
        bicycle_conflict_metrics[bicycle_id] = {
            'conflict_temporal_rate': conflict_temporal_rate,
            'conflict_spatial_rate': conflict_spatial_rate,
            'conflict_spatiotemporal_rate': conflict_spatiotemporal_rate,
            'total_conflict_steps': total_conflict_steps,
            'detected_conflict_steps': detected_conflict_steps,
            'total_conflict_distance': total_conflict_distance,
            'detected_conflict_distance': detected_conflict_distance,
            'flow_id': flow_id
        }
        # Initialize or update flow conflict metrics
        if flow_id not in flow_conflict_metrics:
            flow_conflict_metrics[flow_id] = {
                'bicycles': [],
                'total_conflict_steps': 0,
                'detected_conflict_steps': 0,
                'total_conflict_distance': 0,
                'detected_conflict_distance': 0
            }
        # Aggregate conflict metrics per flow
        flow_conf_metrics = flow_conflict_metrics[flow_id]
        flow_conf_metrics['bicycles'].append(bicycle_id)
        flow_conf_metrics['total_conflict_steps'] += total_conflict_steps
        flow_conf_metrics['detected_conflict_steps'] += detected_conflict_steps
        flow_conf_metrics['total_conflict_distance'] += total_conflict_distance
        flow_conf_metrics['detected_conflict_distance'] += detected_conflict_distance
    # Calculate average conflict rates per flow
    for flow_id in flow_conflict_metrics:
        metrics = flow_conflict_metrics[flow_id]
        metrics['conflict_temporal_rate'] = (metrics['detected_conflict_steps'] / metrics['total_conflict_steps'] * 100 
                                           if metrics['total_conflict_steps'] > 0 else 0)
        metrics['conflict_spatial_rate'] = (metrics['detected_conflict_distance'] / metrics['total_conflict_distance'] * 100 
                                          if metrics['total_conflict_distance'] > 0 else 0)
        metrics['conflict_spatiotemporal_rate'] = (metrics['conflict_temporal_rate'] + metrics['conflict_spatial_rate']) / 2
    # Calculate system-wide conflict detection rates
    total_system_conflict_steps = sum(m['total_conflict_steps'] for m in bicycle_conflict_metrics.values())
    total_system_detected_conflict_steps = sum(m['detected_conflict_steps'] for m in bicycle_conflict_metrics.values())
    total_system_conflict_distance = sum(m['total_conflict_distance'] for m in bicycle_conflict_metrics.values())
    total_system_detected_conflict_distance = sum(m['detected_conflict_distance'] for m in bicycle_conflict_metrics.values())
    system_conflict_temporal_rate = (total_system_detected_conflict_steps / total_system_conflict_steps * 100 
                                   if total_system_conflict_steps > 0 else 0)
    system_conflict_spatial_rate = (total_system_detected_conflict_distance / total_system_conflict_distance * 100 
                                  if total_system_conflict_distance > 0 else 0)
    system_conflict_spatiotemporal_rate = (system_conflict_temporal_rate + system_conflict_spatial_rate) / 2
    
    # Create evaluation file
    with open(f'out_evaluation/bicycle_safety_evaluation_{file_tag}_FCO{FCO_share*100:.0f}%_FBO{FBO_share*100:.0f}%.txt', 'w') as f:
        # Title and general information
        f.write('=======================================================\n')
        f.write('BICYCLE SAFETY EVALUATION\n')
        f.write('=======================================================\n')
        f.write(f'Generated on: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write(f'Simulation Configuration:\n')
        f.write(f'- FCO Share: {FCO_share*100:.0f}%\n')
        f.write(f'- FBO Share: {FBO_share*100:.0f}%\n')
        f.write(f'- Step Length: {step_length} seconds\n')
        f.write('=======================================================\n\n')
        
        # Individual Analysis
        f.write('-------------------------------------------------------\n')
        f.write('1. INDIVIDUAL BICYCLE SAFETY EVALUATION\n')
        f.write('-------------------------------------------------------\n')
        # Overall Statistics
        f.write('1.1 Overall Detection Rate Statistics\n')
        f.write(f'    Total number of bicycles: {len(bicycle_metrics)}\n')
        f.write(f'    Average temporal detection rate: {avg_temporal_rate:.2f}%\n')
        f.write(f'    Average spatial detection rate: {avg_spatial_rate:.2f}%\n')
        f.write(f'    Average spatio-temporal detection rate: {avg_spatiotemporal_rate:.2f}%\n')
        f.write(f'    Average important area temporal detection rate: {avg_important_temporal_rate:.2f}%\n')
        f.write(f'    Average important area spatial detection rate: {avg_important_spatial_rate:.2f}%\n')
        f.write(f'    Average important area spatio-temporal detection rate: {avg_important_spatiotemporal_rate:.2f}%\n\n')
        # Individual Bicycle Analysis
        f.write('1.2 Detection Rates by Bicycle\n')
        for bicycle_id, metrics in sorted(bicycle_metrics.items()):
            f.write(f'    - Bicycle {bicycle_id}:\n')
            f.write(f'      * Trajectory Detection Rates:\n')
            f.write(f'        - Temporal detection rate: {metrics["temporal_rate"]:.2f}%\n')
            f.write(f'        - Spatial detection rate: {metrics["spatial_rate"]:.2f}%\n')
            f.write(f'        - Spatio-temporal detection rate: {metrics["spatiotemporal_rate"]:.2f}%\n')
            f.write(f'        - Total time steps: {metrics["total_time_steps"]}\n')
            f.write(f'        - Total distance: {metrics["total_distance"]:.2f} meters\n')
            f.write(f'      * "Important Area" Trajectory Detection Rates:\n')
            f.write(f'        - Temporal detection rate: {metrics["important_temporal_rate"]:.2f}%\n')
            f.write(f'        - Spatial detection rate: {metrics["important_spatial_rate"]:.2f}%\n')
            f.write(f'        - Spatio-temporal detection rate: {metrics["important_spatiotemporal_rate"]:.2f}%\n')
            f.write(f'        - Total time steps: {metrics["important_total_steps"]}\n')
            f.write(f'        - Total distance: {metrics["important_total_distance"]:.2f} meters\n')
            conflict_metrics = bicycle_conflict_metrics.get(bicycle_id, {})
            f.write(f'      * Conflict Detection Rates:\n')
            f.write(f'        - Temporal conflict detection rate: {conflict_metrics.get("conflict_temporal_rate", 0):.2f}%\n')
            f.write(f'        - Spatial conflict detection rate: {conflict_metrics.get("conflict_spatial_rate", 0):.2f}%\n')
            f.write(f'        - Spatio-temporal conflict detection rate: {conflict_metrics.get("conflict_spatiotemporal_rate", 0):.2f}%\n')
            f.write(f'        - Total conflict steps: {conflict_metrics.get("total_conflict_steps", 0)}\n')
            f.write(f'        - Total conflict distance: {conflict_metrics.get("total_conflict_distance", 0):.2f} meters\n')
        f.write('\n')
        
        # Flow-based Analysis
        f.write('-------------------------------------------------------\n')
        f.write('2. FLOW-BASED BICYCLE SAFETY EVALUATION\n')
        f.write('-------------------------------------------------------\n')
        # Overall Statistics
        f.write('2.1 Overall Flow-Based Detection Rate Statistics\n')
        f.write(f'    Total number of flows: {len(flow_metrics)}\n')
        f.write(f'    Average bicycles per flow: {np.mean([len(m["bicycles"]) for m in flow_metrics.values()]):.2f}\n')
        f.write(f'    Average flow-based temporal detection rate: {avg_flow_temporal_rate:.2f}%\n')
        f.write(f'    Average flow-based spatial detection rate: {avg_flow_spatial_rate:.2f}%\n')
        f.write(f'    Average flow-based spatio-temporal detection rate: {avg_flow_spatiotemporal_rate:.2f}%\n')
        f.write(f'    Average flow-based important area temporal detection rate: {avg_flow_important_temporal_rate:.2f}%\n')
        f.write(f'    Average flow-based important area spatial detection rate: {avg_flow_important_spatial_rate:.2f}%\n')
        f.write(f'    Average flow-based important area spatio-temporal detection rate: {avg_flow_important_spatiotemporal_rate:.2f}%\n\n')
        # Flow-based Bicycle Analysis
        f.write('2.2 Detection Rates by Flow\n')
        for flow_id, metrics in sorted(flow_metrics.items()):
            f.write(f'    - {flow_id}:\n')
            f.write(f'      * Number of bicycles: {len(metrics["bicycles"])}\n')
            f.write(f'      * Flow-based Detection Rates:\n')
            f.write(f'        - Temporal detection rate: {metrics["temporal_rate"]:.2f}%\n')
            f.write(f'        - Spatial detection rate: {metrics["spatial_rate"]:.2f}%\n')
            f.write(f'        - Spatio-temporal detection rate: {metrics["spatiotemporal_rate"]:.2f}%\n')
            f.write(f'        - Cumulative time steps: {metrics["total_steps"]}\n')
            f.write(f'        - Cumulative distance: {metrics["total_distance"]:.2f} meters\n')
            f.write(f'      * "Important Area" Flow-based Detection Rates:\n')
            f.write(f'        - Temporal detection rate: {metrics["important_temporal_rate"]:.2f}%\n')
            f.write(f'        - Spatial detection rate: {metrics["important_spatial_rate"]:.2f}%\n')
            f.write(f'        - Spatio-temporal detection rate: {metrics["important_spatiotemporal_rate"]:.2f}%\n')
            f.write(f'        - Cumulative time steps: {metrics["important_total_steps"]}\n')
            f.write(f'        - Cumulative distance: {metrics["important_total_distance"]:.2f} meters\n')
            conflict_metrics = flow_conflict_metrics.get(flow_id, {})
            f.write(f'      * Flow-based Conflict Detection Rates:\n')
            f.write(f'        - Temporal conflict detection rate: {conflict_metrics.get("conflict_temporal_rate", 0):.2f}%\n')
            f.write(f'        - Spatial conflict detection rate: {conflict_metrics.get("conflict_spatial_rate", 0):.2f}%\n')
            f.write(f'        - Spatio-temporal conflict detection rate: {conflict_metrics.get("conflict_spatiotemporal_rate", 0):.2f}%\n')
            f.write(f'        - Cumulative conflict steps: {conflict_metrics.get("total_conflict_steps", 0)}\n')
            f.write(f'        - Cumulative conflict distance: {conflict_metrics.get("total_conflict_distance", 0):.2f} meters\n')
        f.write('\n')
        
        # Overall Analysis
        f.write('-------------------------------------------------------\n')
        f.write('3. SYSTEM-WIDE BICYCLE SAFETY EVALUATION\n')
        f.write('-------------------------------------------------------\n')
        f.write('3.1 Overall System-Wide Detection Rate Statistics:\n')
        f.write(f'    Cumulative time steps in simulation: {total_system_steps}\n')
        f.write(f'    Cumulative detected time steps: {total_system_detected_steps}\n')
        f.write(f'    Cumulative distance traveled: {total_system_distance:.2f} meters\n')
        f.write(f'    Cumulative detected distance: {total_system_detected_distance:.2f} meters\n')
        f.write(f'    System-wide temporal detection rate: {overall_temporal_rate:.2f}%\n')
        f.write(f'    System-wide spatial detection rate: {overall_spatial_rate:.2f}%\n')
        f.write(f'    System-wide spatio-temporal detection rate: {overall_spatiotemporal_rate:.2f}%\n\n')
        f.write('3.2 "Important Area" System-Wide Detection Rate Statistics:\n')
        f.write(f'    Cumulative time steps in important areas: {total_system_important_steps}\n')
        f.write(f'    Cumulative detected time steps in important areas: {total_system_important_detected_steps}\n')
        f.write(f'    Cumulative distance traveled in important areas: {total_system_important_distance:.2f} meters\n')
        f.write(f'    Cumulative detected distance in important areas: {total_system_important_detected_distance:.2f} meters\n')
        f.write(f'    System-wide important area temporal detection rate: {overall_important_temporal_rate:.2f}%\n')
        f.write(f'    System-wide important area spatial detection rate: {overall_important_spatial_rate:.2f}%\n')
        f.write(f'    System-wide important area spatio-temporal detection rate: {overall_important_spatiotemporal_rate:.2f}%\n\n')
        f.write('3.3 System-Wide Conflict Detection Rate Statistics:\n')
        f.write(f'    Temporal conflict detection rate: {system_conflict_temporal_rate:.2f}%\n')
        f.write(f'    Spatial conflict detection rate: {system_conflict_spatial_rate:.2f}%\n')
        f.write(f'    Spatio-temporal conflict detection rate: {system_conflict_spatiotemporal_rate:.2f}%\n')
        f.write(f'    Total conflict steps: {total_system_conflict_steps}\n')
        f.write(f'    Total conflict distance: {total_system_conflict_distance:.2f} meters\n')
        f.write('\n')
    
    print('Bicycle safety evaluation completed.')

# ... further evaluation functions to be implemented

# ---------------------
# MAIN EXECUTION
# ---------------------

if __name__ == "__main__":  
    with TimingContext("simulation_setup"):
        load_sumo_simulation()
        gdf1, G, buildings, parks, trees, leaves, barriers, PT_shelters = load_geospatial_data()
        print('Geospatial data loaded.')
        gdf1_proj, G_proj, buildings_proj, parks_proj, trees_proj, leaves_proj, barriers_proj, PT_shelters_proj = project_geospatial_data(gdf1, G, buildings, parks, trees, leaves, barriers, PT_shelters)
        print('Geospatial data projected.')
        setup_plot()
        plot_geospatial_data(gdf1_proj, G_proj, buildings_proj, parks_proj, trees_proj, leaves_proj, barriers_proj, PT_shelters_proj)
        if RelativeVisibility or LoVheatmap:
            x_coords, y_coords, grid_cells, visibility_counts = initialize_grid(buildings_proj)
        else:
            visibility_counts = {}
        total_steps = get_total_simulation_steps(sumo_config_path)
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
        if BicycleSafetyEvaluation:
            bicycle_safety_evaluation()
        traci.close()
        print("SUMO simulation closed and TraCi disconnected.")
    if RelativeVisibility:
        with TimingContext("visibility_heatmap"):
            create_relative_visibility_heatmap(x_coords, y_coords, visibility_counts, buildings_proj, parks_proj, trees_proj, leaves_proj, barriers_proj, PT_shelters_proj)
    if LoVheatmap:
        with TimingContext("LoV_heatmap"):
            output_prefix = f'{file_tag}_FCO{FCO_share*100:.0f}%_FBO{FBO_share*100:.0f}%'
            visibility_counts_path = os.path.join(parent_dir, 'out_visibility', 'visibility_counts', f'visibility_counts_{output_prefix}.csv')
            create_lov_heatmap(visibility_counts_path, output_prefix, buildings_proj, parks_proj, trees_proj, leaves_proj, barriers_proj, PT_shelters_proj)