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

# Setup logging (showing only errors in the terminal, no "irrelevant" messages or warnings)
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------------
# CONFIGURATION
# ---------------------

# General Settings:

useLiveVisualization = False # Live Visualization of Ray Tracing
visualizeRays = False # Visualize rays additionaly to the visibility polygon
useManualFrameForwarding = False # Visualization of each frame, manual input necessary to forward the visualization
saveAnimation = False # Save the animation (currently not compatible with live visualization)

# Bounding Box Settings:

north, south, east, west = 48.150600, 48.149000, 11.570800, 11.567600
bbox = (north, south, east, west)

# Path Settings:

base_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(base_dir)
sumo_config_path = os.path.join(parent_dir, 'SUMO_example', 'SUMO_example.sumocfg') # Path to SUMO config-file
geojson_path = os.path.join(parent_dir, 'SUMO_example', 'SUMO_example.geojson') # Path to GEOjson file

# FCO / FBO Settings:

FCO_share = 0 # Penetration rate of FCOs
FBO_share = 0 # Penetration rate of FBOs
numberOfRays = 360 # Number of rays emerging from the observer vehicle's (FCO/FBO) center point
radius = 30 # Radius of the rays emerging from the observer vehicle's (FCO/FBO) center point
min_segment_length = 3  # Base minimum segment length (for bicycle trajectory analysis)
max_gap_bridge = 10  # Maximum number of undetected frames to bridge between detected segments (for bicycle trajectory analysis)

# Warm Up Settings:

delay = 60 #warm-up time in seconds (during this time in the beginning of the simulation, no ray tracing is performed)

# Grid Map Settings:

grid_size =  0.5 # Grid Size for Heat Map Visualization (the smaller the grid size, the higher the resolution)

# Application Settings:

relativeVisibility = False # Generate relative visibility heatmaps
IndividualBicycleTrajectories = True # Generate 2D space-time diagrams of bicycle trajectories (individual trajectory plots)
FlowBasedBicycleTrajectories = False # Generate 2D space-time diagrams of bicycle trajectories (flow-based trajectory plots)
ThreeDimensionalBicycleTrajectories = False # Generate 3D space-time diagrams of bicycle trajectories (3D trajectory plots)
AnimatedThreeDimensionalBicycleTrajectories = False # Generate animated 3D space-time diagrams of bicycle trajectories (3D trajectory plots)

# ---------------------

# General Visualization Settings

fig, ax = plt.subplots(figsize=(12, 8))

# 3D Visualization Settings
bicycle_trajectories = {}
flow_ids = set()
transformer = None
fig_3d = None
ax_3d = None

# Loading of Geospatial Data

buildings = ox.features_from_bbox(bbox=bbox, tags={'building': True})
buildings_proj = buildings.to_crs("EPSG:32632")

# Projection Settings

proj_from = pyproj.Proj('epsg:4326')   # Source projection: WGS 84
proj_to = pyproj.Proj('epsg:32632')    # Target projection: UTM zone 32N
project = pyproj.Transformer.from_proj(proj_from, proj_to, always_xy=True).transform

# Initialization of empty lists

vehicle_patches = []
ray_lines = []
visibility_polygons = []

# Initialization of empty disctionaries
bicycle_trajectory_data = {}
bicycle_flow_data = {}

# Initialization of Grid Parameters

x_min, y_min, x_max, y_max = buildings_proj.total_bounds
x_coords = np.arange(x_min, x_max, grid_size)
y_coords = np.arange(y_min, y_max, grid_size)
grid_points = [(x, y) for x in x_coords for y in y_coords]
grid_cells = [box(x, y, x + grid_size, y + grid_size) for x, y in grid_points]
visibility_counts = {cell: 0 for cell in grid_cells}

# Logging Settings
unique_vehicles = set()
vehicle_type_set = set()
log_columns = ['time_step']
simulation_log = pd.DataFrame(columns=log_columns)

# Global variables to store bicycle data (for Bicycle Trajectory Analysis)
bicycle_data = defaultdict(list)
bicycle_start_times = {}
traffic_light_ids = {}
traffic_light_positions = {}
bicycle_tls = {}
bicycle_detection_data = {}
bicycle_conflicts = defaultdict(list)
foe_trajectories = {}  # To store trajectories of foe vehicles

# ---------------------

# ---------------------
# INITIALIZATION
# ---------------------

def load_sumo_simulation():
    """
    Initializes and starts SUMO traffic simulation with error logging and warnings disabled.
    """
    sumoCmd = ["sumo", "-c", sumo_config_path, "--message-log", "error", "--no-warnings", "true"] # showing only errors in the terminal, no "irrelevant" messages or warnings
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
    
    return gdf1, G, buildings, parks

def load_geospatial_data_new():
    """
    Loads road space distribution from SUMO .net file using SumoNetVis.
    Returns data in format compatible with existing code.
    """
    import SumoNetVis
    print(f"Target bbox (north, south, east, west): {bbox}")
    
    # Load SUMO network using SumoNetVis
    net = SumoNetVis.Net('SUMO_example/network.net.xml')
    
    # Convert the network geometries to a GeoDataFrame
    network_elements = []
    
    # Process edges and their lanes
    for edge_id, edge in net.edges.items():
        if not hasattr(edge, 'function') or edge.function != "internal":
            for lane in edge.lanes:
                if lane.shape is not None:
                    network_elements.append({
                        'geometry': lane.shape,
                        'Type': 'LaneBoundary',
                        'id': lane.id
                    })
    
    # Process junctions
    for junction_id, junction in net.junctions.items():
        if junction.shape is not None:
            network_elements.append({
                'geometry': junction.shape,
                'Type': 'Junction',
                'id': junction_id
            })
    
    # Create GeoDataFrame
    gdf1 = gpd.GeoDataFrame(network_elements, crs="EPSG:32632")  # SUMO uses UTM32N by default
    
    # Debug prints
    print("\nNetwork Data Summary:")
    print(f"Total elements: {len(network_elements)}")
    print(f"Types: {gdf1['Type'].value_counts().to_dict()}")
    print(f"CRS: {gdf1.crs}")
    print(f"Bounds: {gdf1.total_bounds}")
    print("\nSample geometries:")
    print(gdf1.head())
    
    # Load other data (keeping the original return structure)
    G = None  # We don't need the NetworkX graph anymore
    buildings = ox.features_from_bbox(bbox=bbox, tags={'building': True})
    parks = ox.features_from_bbox(bbox=bbox, tags={'leisure': 'park'})
    
    return gdf1, G, buildings, parks

def project_geospatial_data(gdf1, G, buildings, parks):
    """
    Projects all geospatial data (NetworkX graph, road space distribution, buildings, parks) to UTM zone 32N for consistent spatial analysis.
    """
    gdf1_proj = gdf1.to_crs("EPSG:32632")  # road space distribution
    G_proj = ox.project_graph(G, to_crs="EPSG:32632") # NetworkX graph (bounding box)
    # Project buildings if they exist
    buildings_proj = buildings.to_crs("EPSG:32632") if buildings is not None else None
    # Project parks if they exist
    parks_proj = parks.to_crs("EPSG:32632") if parks is not None else None
    
    # Debug prints
    print("\nProjection Summary:")
    print(f"Road network CRS: {gdf1_proj.crs}")
    print(f"Road network bounds: {gdf1_proj.total_bounds}")
    if buildings_proj is not None:
        print(f"Buildings bounds: {buildings_proj.total_bounds}")
    if parks_proj is not None:
        print(f"Parks bounds: {parks_proj.total_bounds}")
    
    return gdf1_proj, G_proj, buildings_proj, parks_proj

def project_geospatial_data_new(gdf1, buildings, parks):
    """
    Projects all geospatial data (NetworkX graph, road space distribution, buildings, parks) to UTM zone 32N for consistent spatial analysis.
    """
    gdf1_proj = gdf1.to_crs("EPSG:32632")  # road space distribution
    # Project buildings if they exist
    buildings_proj = buildings.to_crs("EPSG:32632") if buildings is not None else None
    # Project parks if they exist
    parks_proj = parks.to_crs("EPSG:32632") if parks is not None else None
    
    # Debug prints
    print("\nProjection Summary:")
    print(f"Road network CRS: {gdf1_proj.crs}")
    print(f"Road network bounds: {gdf1_proj.total_bounds}")
    if buildings_proj is not None:
        print(f"Buildings bounds: {buildings_proj.total_bounds}")
    if parks_proj is not None:
        print(f"Parks bounds: {parks_proj.total_bounds}")
    
    return gdf1_proj, buildings_proj, parks_proj

def initialize_grid(buildings_proj, grid_size=1.0):
    """
    Creates a grid of cells over the simulation area for tracking visibility.
    Each cell is a square of size grid_size and is initiated with a visibility count of 0.
    """
    if relativeVisibility:
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
    Configures the ray tracing visualization plot with title and legend showing buildings, parks, and vehicle types.
    """
    global fig, ax
    ax.set_aspect('equal')

    ax.set_title(f'Ray Tracing Visualization for penetration rates FCO {FCO_share*100:.0f}% and FBO {FBO_share*100:.0f}%')
    
    # Create legend handles depending on FCO and FBO penetration rates
    if FCO_share > 0 and FBO_share > 0:
        legend_handles = [
            # Static elements
            Rectangle((0, 0), 1, 1, facecolor='darkgray', edgecolor='black', linewidth=0.5, alpha=0.7, label='Buildings'),
            Rectangle((0, 0), 1, 1, facecolor='forestgreen', edgecolor='black', linewidth=0.5, alpha=0.7, label='Parks'),
            
            # Vehicle types
            Rectangle((0, 0), 0.36, 1, facecolor='none', edgecolor='black', label='Passenger Car'),
            Rectangle((0, 0), 0.13, 0.32, facecolor='none', edgecolor='blue', label='Bicycle'),
            Rectangle((0, 0), 0.36, 1, facecolor='none', edgecolor='red', label='FCO'),
            Rectangle((0, 0), 0.13, 0.32, facecolor='none', edgecolor='orange', label='FBO')
        ]
    elif FCO_share > 0 and FBO_share == 0:
        legend_handles = [
            # Static elements
            Rectangle((0, 0), 1, 1, facecolor='darkgray', edgecolor='black', linewidth=0.5, alpha=0.7, label='Buildings'),
            Rectangle((0, 0), 1, 1, facecolor='forestgreen', edgecolor='black', linewidth=0.5, alpha=0.7, label='Parks'),
            
            # Vehicle types
            Rectangle((0, 0), 0.36, 1, facecolor='none', edgecolor='black', label='Passenger Car'),
            Rectangle((0, 0), 0.13, 0.32, facecolor='none', edgecolor='blue', label='Bicycle'),
            Rectangle((0, 0), 0.36, 1, facecolor='none', edgecolor='red', label='FCO')
        ]
    elif FCO_share == 0 and FBO_share > 0:
        legend_handles = [
            # Static elements
            Rectangle((0, 0), 1, 1, facecolor='darkgray', edgecolor='black', linewidth=0.5, alpha=0.7, label='Buildings'),
            Rectangle((0, 0), 1, 1, facecolor='forestgreen', edgecolor='black', linewidth=0.5, alpha=0.7, label='Parks'),
            
            # Vehicle types
            Rectangle((0, 0), 0.36, 1, facecolor='none', edgecolor='black', label='Passenger Car'),
            Rectangle((0, 0), 0.13, 0.32, facecolor='none', edgecolor='blue', label='Bicycle'),
            Rectangle((0, 0), 0.13, 0.32, facecolor='none', edgecolor='orange', label='FBO')
        ]
    elif FCO_share == 0 and FBO_share == 0:
        legend_handles = [
            # Static elements
            Rectangle((0, 0), 1, 1, facecolor='darkgray', edgecolor='black', linewidth=0.5, alpha=0.7, label='Buildings'),
            Rectangle((0, 0), 1, 1, facecolor='forestgreen', edgecolor='black', linewidth=0.5, alpha=0.7, label='Parks'),
            
            # Vehicle types
            Rectangle((0, 0), 0.36, 1, facecolor='none', edgecolor='black', label='Passenger Car'),
            Rectangle((0, 0), 0.13, 0.32, facecolor='none', edgecolor='blue', label='Bicycle')
        ]
    
    ax.legend(handles=legend_handles, loc='upper right', fontsize=7)

    # Add initial warm-up text box (only for the first frame, further text boxes are updated in the update_with_ray_tracing function)
    ax.warm_up_text = ax.text(0.02, 0.98, f'Warm-up phase\nRemaining: {delay}s', 
                             transform=ax.transAxes,
                             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
                             verticalalignment='top',
                             fontsize=10)
    ax.warm_up_text.set_visible(True)

def plot_geospatial_data(gdf1_proj, G_proj, buildings_proj, parks_proj):
    """
    Plots the geospatial data (base map, road network, buildings and parks) onto the current axes.
    """
    gdf1_proj.plot(ax=ax, color='lightgray', alpha=0.5, edgecolor='lightgray', zorder=1)  # Plot the road space distribution
    ox.plot_graph(G_proj, ax=ax, bgcolor='none', edge_color='none', node_size=0, show=False, close=False)  # Plot the NetworkX graph
    # Plot parks if they exist
    if parks_proj is not None:
        parks_proj.plot(ax=ax, facecolor='forestgreen', edgecolor='black', linewidth=0.5, zorder=2)  # Plot parks
    # Plot buildings if they exist
    if buildings_proj is not None:
        buildings_proj.plot(ax=ax, facecolor='darkgray', edgecolor='black', linewidth=0.5, zorder=3)  # Plot buildings

def plot_geospatial_data_new(gdf1_proj, buildings_proj, parks_proj):
    """
    Plots the geospatial data with debug information.
    """
    print("\nPlotting Data Summary:")
    print(f"Projected CRS: {gdf1_proj.crs}")
    print(f"Projected bounds: {gdf1_proj.total_bounds}")
    
    # Create figure and axis if they don't exist
    if 'ax' not in globals():
        global fig, ax
        fig, ax = plt.subplots(figsize=(10, 10))
    
    # Clear the axis
    ax.clear()
    
    # Plot junctions
    junctions = gdf1_proj[gdf1_proj['Type'] == 'Junction']
    if not junctions.empty:
        junctions.plot(ax=ax, 
                      color='lightgray', 
                      alpha=0.7, 
                      edgecolor='dimgray',
                      linewidth=0.5,
                      zorder=1)
    
    # Plot lane boundaries
    lanes = gdf1_proj[gdf1_proj['Type'] == 'LaneBoundary']
    if not lanes.empty:
        lanes.plot(ax=ax, 
                  color='white', 
                  alpha=0.8, 
                  edgecolor='dimgray',
                  linewidth=1,
                  zorder=2)
    
    # Plot parks if they exist
    if parks_proj is not None and not parks_proj.empty:
        parks_proj.plot(ax=ax, 
                       facecolor='forestgreen', 
                       edgecolor='black', 
                       linewidth=0.5, 
                       alpha=0.5,
                       zorder=3)
    
    # Plot buildings if they exist
    if buildings_proj is not None and not buildings_proj.empty:
        buildings_proj.plot(ax=ax, 
                          facecolor='darkgray', 
                          edgecolor='black', 
                          linewidth=0.5, 
                          alpha=0.7,
                          zorder=4)
    
    # Set plot limits based on the data bounds
    bounds = gdf1_proj.total_bounds
    ax.set_xlim([bounds[0], bounds[2]])
    ax.set_ylim([bounds[1], bounds[3]])
    
    # Force aspect ratio to be equal
    ax.set_aspect('equal')
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.5)
    
    # Debug print
    print("\nPlot Configuration:")
    print(f"X limits: {ax.get_xlim()}")
    print(f"Y limits: {ax.get_ylim()}")
    print(f"Aspect ratio: {ax.get_aspect()}")

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
    global vehicle_patches, ray_lines, visibility_polygons, FCO_share, FBO_share, visibility_counts, numberOfRays, useRTREEmethod, visualizeRays, useManualFrameForwarding, delay, bicycle_detection_data
    detected_color = (1.0, 0.27, 0, 0.5)
    undetected_color = (0.53, 0.81, 0.98, 0.5)

    if frame == 1:
        print('Ray Tracing initiated:')
    traci.simulationStep()  # Advance the simulation by one step

    if useManualFrameForwarding:
        input("Press Enter to continue...")  # Wait for user input if manual forwarding is enabled

    if IndividualBicycleTrajectories:
        if frame == 1:
            print('Individual bicycle trajectory tracking initiated:')
        individual_bicycle_trajectories(frame)
    if FlowBasedBicycleTrajectories:
        if frame == 1:
            print('Flow-based bicycle trajectory tracking initiated:')
        flow_based_bicycle_trajectories(frame, total_steps)
    if ThreeDimensionalBicycleTrajectories:
        if frame == 1:
            print('3D bicycle trajectory tracking initiated:')
        three_dimensional_bicycle_trajectories(frame)
    if AnimatedThreeDimensionalBicycleTrajectories:
        if frame == 1:
            print('3D bicycle trajectory tracking and animation initiated:')
        three_dimensional_bicycle_trajectories_gif(frame)

    print(f"Frame: {frame + 1}")

    # Set vehicle types for FCOs and FBOs based on probability
    FCO_type = "floating_car_observer"
    FBO_type = "floating_bike_observer"
    for vehicle_id in traci.simulation.getDepartedIDList():
        if traci.vehicle.getTypeID(vehicle_id) == "DEFAULT_VEHTYPE" and np.random.uniform() < FCO_share:
            traci.vehicle.setType(vehicle_id, FCO_type)
        if traci.vehicle.getTypeID(vehicle_id) == "DEFAULT_BIKETYPE" and np.random.uniform() < FBO_share:
            traci.vehicle.setType(vehicle_id, FBO_type)

    stepLength = get_step_length(sumo_config_path)

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

        # Create static objects (buildings and parked vehicles)
        static_objects = [building.geometry for building in buildings_proj.itertuples()]
        for vehicle_id in traci.vehicle.getIDList():
            vehicle_type = traci.vehicle.getTypeID(vehicle_id)
            if vehicle_type == "parked_vehicle":
                # Add parked vehicles to static objects
                x, y = traci.vehicle.getPosition(vehicle_id)
                x_32632, y_32632 = convert_simulation_coordinates(x, y)
                width, length = vehicle_attributes(vehicle_type)[2]
                angle = traci.vehicle.getAngle(vehicle_id)
                parked_vehicle_geom = create_vehicle_polygon(x_32632, y_32632, width, length, angle)
                static_objects.append(parked_vehicle_geom)

        # Clear previous ray lines and visibility polygons
        for line in ray_lines:
            if useLiveVisualization and visualizeRays:
                line.remove()
        ray_lines.clear()

        if useLiveVisualization:
            for polygon in visibility_polygons:
                polygon.remove()
        visibility_polygons.clear()

        # Process each vehicle
        for vehicle_id in traci.vehicle.getIDList():
            vehicle_type = traci.vehicle.getTypeID(vehicle_id)
            Shape, edgecolor, (width, length) = vehicle_attributes(vehicle_type)
            # Update vehicle patches
            x, y = traci.vehicle.getPosition(vehicle_id)
            x_32632, y_32632 = convert_simulation_coordinates(x, y)
            angle = traci.vehicle.getAngle(vehicle_id)
            adjusted_angle = (-angle) % 360
            lower_left_corner = (x_32632 - width / 2, y_32632 - length / 2)
            if vehicle_type == "parked_vehicle":
                patch = Rectangle(lower_left_corner, width, length, facecolor='lightgray', edgecolor='gray')
            else:
                patch = Rectangle(lower_left_corner, width, length, facecolor='white', edgecolor=edgecolor)
            
            # Create dynamic objects (other vehicles)
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

            # Perform ray tracing for FCOs and FBOs
            if vehicle_type in ["floating_car_observer", "floating_bike_observer"]:
                center = (x_32632, y_32632)
                rays = generate_rays(center)
                all_objects = static_objects + dynamic_objects_geom
                ray_endpoints = []

                # Use multithreading for ray tracing
                with ThreadPoolExecutor() as executor:
                    futures = {executor.submit(detect_intersections, ray, all_objects): ray for ray in rays}

                    # Process ray tracing results
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
                        ray_line = Line2D([ray[0][0], end_point[0]], [ray[0][1], end_point[1]], color=ray_color, linewidth=1)
                        if useLiveVisualization and visualizeRays:
                            ax.add_line(ray_line)
                        new_ray_lines.append(ray_line)

                # Create visibility polygon
                ray_endpoints.sort(key=lambda point: np.arctan2(point[1] - center[1], point[0] - center[0]))

                if len(ray_endpoints) > 2:
                    visibility_polygon = MatPolygon(ray_endpoints, color='green', alpha=0.5, fill=None)
                    if useLiveVisualization:
                        ax.add_patch(visibility_polygon)
                    visibility_polygons.append(visibility_polygon)

                # Update visibility counts
                visibility_polygon_shape = Polygon(ray_endpoints)
                for cell in visibility_counts.keys():
                    if visibility_polygon_shape.contains(cell):
                        if cell not in updated_cells:
                            visibility_counts[cell] += 1
                            updated_cells.add(cell)

        # Checking if bicycles have been detected by FCOs / FBOs
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
                # Check if this bicycle is detected by any FCO/FBO visibility polygon
                for vis_polygon in visibility_polygons:
                    if vis_polygon and vehicle_polygon.intersects(Polygon(vis_polygon.get_xy())):
                        is_detected = True
                        break
                bicycle_detection_data[vehicle_id].append((traci.simulation.getTime(), is_detected))
        
        # Update visualization
        if useLiveVisualization:
            for patch in vehicle_patches:
                patch.remove()
        
        vehicle_patches = new_vehicle_patches
        ray_lines = new_ray_lines

        if useLiveVisualization:
            for patch in vehicle_patches:
                ax.add_patch(patch)
    
    detailled_logging(frame)  # Log detailed information for this frame
    if frame == total_steps - 1:
        print('Ray tracing completed.')

def run_animation(total_steps):
    """
    Runs and displays a matplotlib animation of the ray tracing simulation.
    """
    global fig, ax

    if useLiveVisualization:
        # Close existing figure and switch backend
        plt.close(fig)
        matplotlib.use('TkAgg', force=True)
        print('Ray tracing animation initiated:')
        
        # Create new figure with interactive backend
        fig, ax = plt.subplots(figsize=(12, 8))
        plot_geospatial_data(gdf1_proj, G_proj, buildings_proj, parks_proj)
        # plot_geospatial_data_new(gdf1_proj, buildings_proj, parks_proj)
        setup_plot()
    
    # Create animation
    anim = FuncAnimation(fig, update_with_ray_tracing, frames=range(1, total_steps), 
                        interval=33, repeat=False)
    
    if saveAnimation:
        writer = FFMpegWriter(fps=1, metadata=dict(artist='Me'), bitrate=1800)
        filename = f'out_raytracing/ray_tracing_animation_FCO{FCO_share*100:.0f}%_FBO{FBO_share*100:.0f}%.mp4'
        print('Saving ray tracing animation.')
        anim.save(filename, writer=writer)
        print(f"Ray tracing animation saved.")

    if useLiveVisualization:
        plt.show()
    
    return anim

# ---------------------
# LOGGING
# ---------------------

def detailled_logging(time_step):
    """
    Logs detailed information about vehicles in the simulation at each time step.
    Tracks new and present vehicles by type, updates global variables for unique vehicles and vehicle types, and logs the information.
    """
    global unique_vehicles, vehicle_type_set, simulation_log

    # Initialize counters for new and present vehicles
    new_vehicle_counts = {}
    present_vehicle_counts = {}
    
    # Count new vehicles that have entered the simulation
    for vehicle_id in traci.simulation.getDepartedIDList():
        if vehicle_id not in unique_vehicles:
            unique_vehicles.add(vehicle_id)
            vehicle_type = traci.vehicle.getTypeID(vehicle_id)
            vehicle_type_set.add(vehicle_type)

            # Increment count for new vehicles of this type
            if vehicle_type not in new_vehicle_counts:
                new_vehicle_counts[vehicle_type] = 0
            new_vehicle_counts[vehicle_type] += 1

    # Count vehicles currently present in the simulation
    for vehicle_id in traci.vehicle.getIDList():
        vehicle_type = traci.vehicle.getTypeID(vehicle_id)
        if vehicle_type not in present_vehicle_counts:
            present_vehicle_counts[vehicle_type] = 0
        present_vehicle_counts[vehicle_type] += 1

    # Prepare log entry for this time step
    log_entry = {'time_step': time_step}
    for vehicle_type in vehicle_type_set:
        log_entry[f'new_{vehicle_type}_count'] = new_vehicle_counts.get(vehicle_type, 0)
        log_entry[f'present_{vehicle_type}_count'] = present_vehicle_counts.get(vehicle_type, 0)

    # Add log entry to the simulation log
    log_entry_df = pd.DataFrame([log_entry])
    simulation_log = pd.concat([simulation_log, log_entry_df], ignore_index=True)

def summary_logging():
    """
    Generates and saves summary logs of the simulation.
    Calculates total vehicle counts, penetration rates for FCOs and FBOs, and logs the information.
    """
    global simulation_log

    # Initialize counters for each vehicle type
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
        total_vehicle_counts[vehicle_type] = simulation_log[f'new_{vehicle_type}_count'].sum()
        # Sum up relevant car and bike counts
        if vehicle_type in relevant_car_types:
            total_relevant_cars += total_vehicle_counts[vehicle_type]
            if vehicle_type == "floating_car_observer":
                total_floating_car_observers += total_vehicle_counts[vehicle_type]
        if vehicle_type in relevant_bike_types:
            total_relevant_bikes += total_vehicle_counts[vehicle_type]
            if vehicle_type == "floating_bike_observer":
                total_floating_bike_observers += total_vehicle_counts[vehicle_type]

    # Calculate penetration rates
    fco_penetration_rate = total_floating_car_observers / total_relevant_cars if total_relevant_cars > 0 else 0
    fbo_penetration_rate = total_floating_bike_observers / total_relevant_bikes if total_relevant_bikes > 0 else 0

    # Save detailed log to CSV
    simulation_log.to_csv(f'out_logging/detailed_log_FCO{str(FCO_share*100)}%_FBO{str(FBO_share*100)}%.csv', index=False)
    print('Detailed logging completed.')

    # Write summary log to CSV
    with open(f'out_logging/summary_log_FCO{str(FCO_share*100)}%_FBO{str(FBO_share*100)}%.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write simulation input parameters
        writer.writerow(["Simulation Input:"])
        writer.writerow([])
        writer.writerow(["Description", "Value"])
        writer.writerow(["FCO share (input)", f"{(FCO_share):.2%}"])
        writer.writerow(["FBO share (input)", f"{(FBO_share):.2%}"])
        writer.writerow(["Number of rays (for Ray Tracing)", numberOfRays])
        writer.writerow(["Grid size (for Heat Map Visualizations)", grid_size])
        writer.writerow([])
        writer.writerow([])
        # Write simulation output results
        writer.writerow(["Simulation Output:"])
        writer.writerow([])
        writer.writerow(["Description", "Value"])
        for vehicle_type in vehicle_type_set:
            writer.writerow([f"Total {vehicle_type} vehicles", total_vehicle_counts[vehicle_type]])
        writer.writerow(["Total relevant cars", total_relevant_cars])
        writer.writerow(["Total relevant bikes", total_relevant_bikes])
        writer.writerow(["FCO penetration rate", f"{fco_penetration_rate:.2%}"])
        writer.writerow(["FBO penetration rate", f"{fbo_penetration_rate:.2%}"])
    
    print('Summary logging completed.')

# ---------------------
# APPLICATIONS
# ---------------------

def create_visibility_heatmap(x_coords, y_coords, visibility_counts):
    """
    Generates a CSV file with raw visibility data (visibility counts) and plots a normalized heatmap.
    Saves the heatmap as a PNG file if 'relative visibility' is enabled in the Application Settings.
    """
    if relativeVisibility: # Only create heatmap if relative visibility is enabled
        print('Relative visibility heatmap generation initiated.')

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
        with open(f'out_visibility/visibility_counts_FCO{str(FCO_share*100)}%_FBO{str(FBO_share*100)}%.csv', 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['x_coord', 'y_coord', 'visibility_count'])
            for i, x in enumerate(x_coords):
                for j, y in enumerate(y_coords):
                    if not np.isnan(heatmap_data[i, j]):
                        csvwriter.writerow([x, y, heatmap_data[i, j]])

        # Normalize heatmap data
        heatmap_data = heatmap_data / np.nanmax(heatmap_data)

        # Plot and save heatmap
        fig, ax = plt.subplots(figsize=(12, 8), facecolor='lightgray')
        ax.set_facecolor('lightgray')
        buildings_proj.plot(ax=ax, facecolor='darkgray', edgecolor='black', linewidth=0.5)
        parks_proj.plot(ax=ax, facecolor='forestgreen', edgecolor='black', linewidth=0.5)
        cax = ax.imshow(heatmap_data.T, origin='lower', cmap='hot', extent=[x_min, x_max, y_min, y_max], alpha=0.6)
        ax.set_title('Relative Visibility Heatmap')
        fig.colorbar(cax, ax=ax, label='Relative Visibility')
        plt.savefig(f'out_raytracing/relative_visibility_heatmap_FCO{str(FCO_share*100)}%_FBO{str(FBO_share*100)}%.png')
        print(f'Relative visibility heatmap generated and saved as out_raytracing/relative_visibility_heatmap_FCO{str(FCO_share*100)}%_FBO{str(FBO_share*100)}%.png.')

def individual_bicycle_trajectories(frame):
    """
    Creates space-time diagrams for individual bicycles, including detection status, traffic lights,
    and conflicts detected by SUMO's SSM device.
    """
    global bicycle_data, bicycle_start_times, traffic_light_positions, bicycle_tls, bicycle_waiting_times

    # Create output directory if it doesn't exist
    os.makedirs('out_bicycle_trajectories', exist_ok=True)
    
    bicycle_waiting_times = {}

    current_vehicles = set(traci.vehicle.getIDList())
    
    # Check for bicycles that have left the simulation
    for vehicle_id in list(bicycle_data.keys()):
        if vehicle_id not in current_vehicles:
            # This bicycle has left the simulation, plot its trajectory
            data = bicycle_data[vehicle_id]
            detection_data = bicycle_detection_data.get(vehicle_id, [])
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Split trajectory into detected and undetected segments
            current_points = []
            current_detected = None
            segments = {'detected': [], 'undetected': []}
            detection_buffer = []  # Buffer to store recent detection states
            
            for i, (distance, time) in enumerate(data):
                # Find corresponding detection status
                detection_time = time + bicycle_start_times[vehicle_id]
                is_detected = False
                for det_time, det_status in detection_data:
                    if abs(det_time - detection_time) < step_length:
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
            if vehicle_id in bicycle_conflicts and bicycle_conflicts[vehicle_id]:
                # Group conflicts by foe vehicle ID
                conflicts_by_foe = {}
                for conflict in bicycle_conflicts[vehicle_id]:
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
                        most_severe = max(foe_conflicts, key=lambda x: x['severity'])
                        size = 50 + (most_severe['severity'] * 100)
                        ax.scatter(most_severe['distance'], most_severe['time'], 
                                color='firebrick', marker='o', s=size, zorder=5,
                                facecolors='none', edgecolors='firebrick', linewidth=0.75)

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
            plt.savefig(f'out_bicycle_trajectories/{vehicle_id}_space_time_diagram_FCO{FCO_share*100:.0f}%_FBO{FBO_share*100:.0f}%.png', bbox_inches='tight')
            print(f"Individual space-time diagram for bicycle {vehicle_id} saved as out_bicycle_trajectories/{vehicle_id}_space_time_diagram_FCO{FCO_share*100:.0f}%_FBO{FBO_share*100:.0f}%.png.")
            plt.close(fig)
            
            # Remove this bicycle from the data dictionaries
            del bicycle_data[vehicle_id]
            del bicycle_start_times[vehicle_id]
            del traffic_light_positions[vehicle_id]
            del bicycle_tls[vehicle_id]
            if vehicle_id in bicycle_detection_data:
                del bicycle_detection_data[vehicle_id]
    
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
                        if vehicle_id not in bicycle_conflicts:
                            bicycle_conflicts[vehicle_id] = []
                        
                        # Calculate severity
                        ttc_severity = 1 - (ttc / TTC_THRESHOLD) if ttc < TTC_THRESHOLD else 0
                        pet_severity = 1 - (pet / PET_THRESHOLD) if pet < PET_THRESHOLD else 0
                        drac_severity = min(drac / DRAC_THRESHOLD, 1.0) if drac > 0 else 0
                        
                        conflict_severity = max(ttc_severity, pet_severity, drac_severity)
                        
                        # Remove coordinate transformation, only store what we need for the space-time diagram
                        bicycle_conflicts[vehicle_id].append({
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

def flow_based_bicycle_trajectories(frame, total_steps):
    """
    Creates space-time diagrams for bicycle flows, including detection status, traffic lights,
    and conflicts detected by SUMO's SSM device.
    """
    global bicycle_flow_data, traffic_light_positions, bicycle_tls, step_length, bicycle_conflicts, traffic_light_programs, flow_detection_data, foe_trajectories

    # Initialize traffic light programs at frame 0
    if frame == 0:
        traffic_light_programs = {}
        for tl_id in traci.trafficlight.getIDList():
            if tl_id not in traffic_light_ids:
                traffic_light_ids[tl_id] = len(traffic_light_ids) + 1
            traffic_light_programs[tl_id] = {
                'program': []
            }
    
    # Record traffic light states every frame
    current_time = traci.simulation.getTime()
    for tl_id in traffic_light_programs:
        full_state = traci.trafficlight.getRedYellowGreenState(tl_id)
        traffic_light_programs[tl_id]['program'].append((current_time, full_state))

    # Create output directory if it doesn't exist
    os.makedirs('out_flow_trajectories', exist_ok=True)

    current_time = traci.simulation.getTime()

    # detection status collection
    if frame == 0:
        flow_detection_data = {}  # Store detection data per flow/bicycle
    
    # During simulation, collect detection data
    current_vehicles = set(traci.vehicle.getIDList())
    
    for vehicle_id in current_vehicles:
        vehicle_type = traci.vehicle.getTypeID(vehicle_id)
        if vehicle_type in ["bicycle", "DEFAULT_BIKETYPE", "floating_bike_observer"]:
            flow_id = vehicle_id.rsplit('.', 1)[0]
            if flow_id not in flow_detection_data:
                flow_detection_data[flow_id] = {}
            if vehicle_id not in flow_detection_data[flow_id]:
                flow_detection_data[flow_id][vehicle_id] = []
            
            # Check if bicycle is currently detected
            is_detected = False
            for observer_id in traci.vehicle.getIDList():
                observer_type = traci.vehicle.getTypeID(observer_id)
                # Only check FCD parameters for observer vehicles
                if observer_type in ["floating_car_observer", "floating_bike_observer"]:
                    try:
                        observed_vehicles = traci.vehicle.getParameter(observer_id, "device.fcd.observedVehicles").split()
                        if vehicle_id in observed_vehicles:
                            is_detected = True
                            break
                    except:
                        # Skip if the vehicle doesn't have the FCD device
                        continue
            
            current_time = traci.simulation.getTime()
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
        
        # Check detection status from bicycle_detection_data
        is_detected = False
        if bicycle_id in bicycle_detection_data:
            for det_time, det_status in bicycle_detection_data[bicycle_id]:
                if abs(det_time - current_time) < step_length:
                    is_detected = det_status
                    break
        
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
                    if bicycle_id not in bicycle_conflicts:
                        bicycle_conflicts[bicycle_id] = []
                    
                    # Calculate severity
                    ttc_severity = 1 - (ttc / TTC_THRESHOLD) if ttc < TTC_THRESHOLD else 0
                    pet_severity = 1 - (pet / PET_THRESHOLD) if pet < PET_THRESHOLD else 0
                    drac_severity = min(drac / DRAC_THRESHOLD, 1.0) if drac > 0 else 0
                    
                    conflict_severity = max(ttc_severity, pet_severity, drac_severity)
                    
                    bicycle_conflicts[bicycle_id].append({
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
        detection_data = bicycle_detection_data.get(bicycle_id, [])
        for detection_time, detection_status in detection_data:
            if abs(detection_time - current_time) < step_length:
                is_detected = detection_status
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
                if vehicle_id in bicycle_conflicts:
                    conflicts_by_foe = {}
                    for conflict in bicycle_conflicts[vehicle_id]:
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
                        most_severe = max(foe_conflicts, key=lambda x: x['severity'])
                        size = 50 + (most_severe['severity'] * 100)
                        ax.scatter(most_severe['distance'], most_severe['time'], 
                                  color='firebrick', marker='o', s=size, zorder=1000,
                                  facecolors='none', edgecolors='firebrick', linewidth=0.75)

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

            plt.savefig(f'out_flow_trajectories/{flow_id}_space_time_diagram_FCO{FCO_share*100:.0f}%_FBO{FBO_share*100:.0f}%.png', 
                       bbox_inches='tight')
            plt.close(fig)
            
            print(f"Flow-based space-time diagram for bicycle flow {flow_id} saved as out_flow_trajectories/{flow_id}_space_time_diagram_FCO{FCO_share*100:.0f}%_FBO{FBO_share*100:.0f}%.png.")

def three_dimensional_bicycle_trajectories(frame):
    """
    Creates a 3D visualization of bicycle trajectories where the z=0 plane shows the static scene.
    Automatically generates plots for each bicycle when their trajectory ends.
    """
    global fig_3d, ax_3d, total_steps, bicycle_trajectories, transformer, flow_ids, bicycle_conflicts, foe_trajectories, bicycle_detection_data
    
    # Initialize transformer at frame 0
    if frame == 0:
        transformer = pyproj.Transformer.from_crs('EPSG:4326', 'EPSG:32632', always_xy=True)
        bicycle_trajectories.clear()
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
    if not hasattr(three_dimensional_bicycle_trajectories, 'completed_foes'):
        three_dimensional_bicycle_trajectories.completed_foes = {}
    
    # Store completed foe trajectories before removing them
    for foe_id in departed_foes:
        if foe_id not in three_dimensional_bicycle_trajectories.completed_foes:
            three_dimensional_bicycle_trajectories.completed_foes[foe_id] = foe_trajectories[foe_id]

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
                # Check if bicycle is currently detected
                is_detected = False
                for observer_id in traci.vehicle.getIDList():
                    observer_type = traci.vehicle.getTypeID(observer_id)
                    # Only check FCD parameters for observer vehicles
                    if observer_type in ["floating_car_observer", "floating_bike_observer"]:
                        try:
                            observed_vehicles = traci.vehicle.getParameter(observer_id, "device.fcd.observedVehicles").split()
                            if vehicle_id in observed_vehicles:
                                is_detected = True
                                break
                        except:
                            # Skip if the vehicle doesn't have the FCD device
                            continue
                
                # Store detection status
                if vehicle_id not in bicycle_detection_data:
                    bicycle_detection_data[vehicle_id] = []
                bicycle_detection_data[vehicle_id].append((current_time, is_detected))

                # Store trajectory data
                flow_id = vehicle_id.rsplit('.', 1)[0]
                flow_ids.add(flow_id)
                if vehicle_id not in bicycle_trajectories:
                    bicycle_trajectories[vehicle_id] = []
                bicycle_trajectories[vehicle_id].append((x_utm, y_utm, current_time))
                
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
    finished_bicycles = set(bicycle_trajectories.keys()) - current_vehicles
    
    # Generate plots for finished bicycles
    for vehicle_id in finished_bicycles:
        if len(bicycle_trajectories[vehicle_id]) > 0:  # Only plot if we have trajectory data
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
            trajectory = bicycle_trajectories[vehicle_id]
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
                        for det_time, det_status in bicycle_detection_data[vehicle_id]:
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
                    plt.Line2D([0], [0], marker='o', color='firebrick', linestyle='None', 
                              markerfacecolor='none', markersize=10, label='Potential Conflict')
                ]
                ax_3d.legend(handles=handles, loc='upper left')
                
                # Save conflict overview plot
                os.makedirs('out_3d_trajectories', exist_ok=True)
                plt.savefig(f'out_3d_trajectories/3d_bicycle_trajectory_{vehicle_id}_conflict-overview_FCO{FCO_share*100:.0f}%_FBO{FBO_share*100:.0f}%.png', 
                           bbox_inches='tight', dpi=300)
                print(f'Conflict overview plot for bicycle {vehicle_id} saved as out_3d_trajectories/3d_bicycle_trajectory_{vehicle_id}_conflict-overview_FCO{FCO_share*100:.0f}%_FBO{FBO_share*100:.0f}%.png.')
                plt.close(fig_3d)

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
                            for det_time, det_status in bicycle_detection_data[vehicle_id]:
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
                    
                    # Plot foe trajectory if available
                    foe_traj = None
                    if foe_id in foe_trajectories:
                        foe_traj = foe_trajectories[foe_id]
                    elif foe_id in three_dimensional_bicycle_trajectories.completed_foes:
                        foe_traj = three_dimensional_bicycle_trajectories.completed_foes[foe_id]
                    
                    if foe_traj:
                        foe_x, foe_y, foe_times = zip(*foe_traj)
                        
                        # 1. Plot ground projection
                        ax_3d.plot(foe_x, foe_y, [base_z]*len(foe_x),
                                 color='black', linestyle='--', 
                                 linewidth=2, alpha=0.7, zorder=999)

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
                        
                        foe_proj_plane = Poly3DCollection(foe_plane_vertices, alpha=0.2)
                        foe_proj_plane.set_facecolor('black')
                        foe_proj_plane.set_edgecolor('none')
                        foe_proj_plane.set_sort_zpos(997)
                        ax_3d.add_collection3d(foe_proj_plane)

                        # 3. Plot 3D trajectory
                        ax_3d.plot(foe_x, foe_y, foe_times,
                                 color='black', linewidth=2, alpha=1.0,
                                 zorder=999)
                        
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
                        plt.Line2D([0], [0], marker='o', color='firebrick', linestyle='None', 
                                  markerfacecolor='none', markersize=10, label='Potential Conflict')
                    ]
                    ax_3d.legend(handles=handles, loc='upper left')
                    
                    # Save individual conflict plot
                    plt.savefig(f'out_3d_trajectories/3d_bicycle_trajectory_{vehicle_id}_conflict_{foe_id}_FCO{FCO_share*100:.0f}%_FBO{FBO_share*100:.0f}%.png', 
                               bbox_inches='tight', dpi=300)
                    print(f'Individual conflict plot for bicycle {vehicle_id} and foe {foe_id} saved as out_3d_trajectories/3d_bicycle_trajectory_{vehicle_id}_conflict_{foe_id}_FCO{FCO_share*100:.0f}%_FBO{FBO_share*100:.0f}%.png.')
                    plt.close(fig_3d)
            
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
                        for det_time, det_status in bicycle_detection_data[vehicle_id]:
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
                    plt.Line2D([0], [0], color='darkslateblue', linestyle='--', label='Ground Projection')
                ]
                ax_3d.legend(handles=handles, loc='upper left')
                
                # Save bicycle trajectory plot
                os.makedirs('out_3d_trajectories', exist_ok=True)
                plt.savefig(f'out_3d_trajectories/3d_bicycle_trajectory_{vehicle_id}_FCO{FCO_share*100:.0f}%_FBO{FBO_share*100:.0f}%.png', 
                           bbox_inches='tight', dpi=300)
                print(f'3D bicycle trajectory plot saved as out_3d_trajectories/3d_bicycle_trajectory_{vehicle_id}_FCO{FCO_share*100:.0f}%_FBO{FBO_share*100:.0f}%.png.')
                plt.close(fig_3d)
            
            # Clean up trajectories
            del bicycle_trajectories[vehicle_id]
            if vehicle_id in bicycle_conflicts:
                del bicycle_conflicts[vehicle_id]

def three_dimensional_bicycle_trajectories_gif(frame):
    """
    Creates a 3D visualization of bicycle trajectories where the z=0 plane shows the static scene.
    Automatically generates plots for each bicycle when their trajectory ends.
    """
    global fig_3d, ax_3d, total_steps, bicycle_trajectories, transformer, flow_ids, bicycle_conflicts, foe_trajectories, bicycle_detection_data
    
    def save_rotating_view_frames(ax_3d, base_filename, n_frames=30):
        """Helper function to save frames for rotating view animation"""
        os.makedirs('out_3d_trajectories_gifs/rotation_frames', exist_ok=True)
        
        # Start from bird's eye view and smoothly transition both angles
        # Start: (90° elevation, 270° azimuth)    - bird's eye view
        # End: (35° elevation, 285° azimuth)    - final view
        
        # Create non-linear transitions to make the azimuth change more gradual
        # Use first half of frames mainly for elevation change, second half for azimuth
        t = np.linspace(0, 1, n_frames)
        t_azim = t**2  # Square the parameter to make azimuth change more gradual at start
        
        elevations = np.linspace(90, 35, n_frames)
        azimuths = 270 + (t_azim * 15)  # Smooth transition from 270° to 285°
        
        # Save a frame for each view angle
        for i, (elev, azim) in enumerate(zip(elevations, azimuths)):
            ax_3d.view_init(elev=elev, azim=azim)
            plt.savefig(f'out_3d_trajectories_gifs/rotation_frames/{base_filename}_frame_{i:03d}.png', 
                       dpi=300)

    def create_rotating_view_gif(base_filename, duration=0.1):
        """Helper function to create GIF from saved frames"""
        # Get all frames for this plot
        frames = sorted(glob.glob(f'out_3d_trajectories_gifs/rotation_frames/{base_filename}_frame_*.png'))
        
        # Read frames and create GIF
        images = [imageio.imread(frame) for frame in frames]
        output_file = f'out_3d_trajectories_gifs/{base_filename}_rotation.gif'
        imageio.mimsave(output_file, images, format='GIF', duration=duration)
        
        # Clean up frame files
        for frame in frames:
            os.remove(frame)
        
        print(f'Created rotating view animation: {output_file}')

    # Initialize transformer at frame 0
    if frame == 0:
        transformer = pyproj.Transformer.from_crs('EPSG:4326', 'EPSG:32632', always_xy=True)
        bicycle_trajectories.clear()
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
    if not hasattr(three_dimensional_bicycle_trajectories, 'completed_foes'):
        three_dimensional_bicycle_trajectories.completed_foes = {}
    
    # Store completed foe trajectories before removing them
    for foe_id in departed_foes:
        if foe_id not in three_dimensional_bicycle_trajectories.completed_foes:
            three_dimensional_bicycle_trajectories.completed_foes[foe_id] = foe_trajectories[foe_id]

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
                # Check if bicycle is currently detected
                is_detected = False
                for observer_id in traci.vehicle.getIDList():
                    observer_type = traci.vehicle.getTypeID(observer_id)
                    # Only check FCD parameters for observer vehicles
                    if observer_type in ["floating_car_observer", "floating_bike_observer"]:
                        try:
                            observed_vehicles = traci.vehicle.getParameter(observer_id, "device.fcd.observedVehicles").split()
                            if vehicle_id in observed_vehicles:
                                is_detected = True
                                break
                        except:
                            # Skip if the vehicle doesn't have the FCD device
                            continue
                
                # Store detection status
                if vehicle_id not in bicycle_detection_data:
                    bicycle_detection_data[vehicle_id] = []
                bicycle_detection_data[vehicle_id].append((current_time, is_detected))

                # Store trajectory data
                flow_id = vehicle_id.rsplit('.', 1)[0]
                flow_ids.add(flow_id)
                if vehicle_id not in bicycle_trajectories:
                    bicycle_trajectories[vehicle_id] = []
                bicycle_trajectories[vehicle_id].append((x_utm, y_utm, current_time))
                
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
    finished_bicycles = set(bicycle_trajectories.keys()) - current_vehicles
    
    # Generate plots for finished bicycles
    for vehicle_id in finished_bicycles:
        if len(bicycle_trajectories[vehicle_id]) > 0:  # Only plot if we have trajectory data
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
            trajectory = bicycle_trajectories[vehicle_id]
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
                        for det_time, det_status in bicycle_detection_data[vehicle_id]:
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
                            for det_time, det_status in bicycle_detection_data[vehicle_id]:
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
                    
                    # Plot foe trajectory if available
                    foe_traj = None
                    if foe_id in foe_trajectories:
                        foe_traj = foe_trajectories[foe_id]
                    elif foe_id in three_dimensional_bicycle_trajectories.completed_foes:
                        foe_traj = three_dimensional_bicycle_trajectories.completed_foes[foe_id]
                    
                    if foe_traj:
                        foe_x, foe_y, foe_times = zip(*foe_traj)
                        
                        # 1. Plot ground projection
                        ax_3d.plot(foe_x, foe_y, [base_z]*len(foe_x),
                                 color='black', linestyle='--', 
                                 linewidth=2, alpha=0.7, zorder=999)

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
                        
                        foe_proj_plane = Poly3DCollection(foe_plane_vertices, alpha=0.2)
                        foe_proj_plane.set_facecolor('black')
                        foe_proj_plane.set_edgecolor('none')
                        foe_proj_plane.set_sort_zpos(997)
                        ax_3d.add_collection3d(foe_proj_plane)

                        # 3. Plot 3D trajectory
                        ax_3d.plot(foe_x, foe_y, foe_times,
                                 color='black', linewidth=2, alpha=1.0,
                                 zorder=999)
                        
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
                        for det_time, det_status in bicycle_detection_data[vehicle_id]:
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
                    plt.Line2D([0], [0], color='darkslateblue', linestyle='--', label='Ground Projection')
                ]
                ax_3d.legend(handles=handles, loc='upper left')
                
                # Save bicycle trajectory plot
                base_filename = f'bicycle_{vehicle_id}_FCO{FCO_share*100:.0f}_FBO{FBO_share*100:.0f}'
                save_rotating_view_frames(ax_3d, base_filename)
                plt.close(fig_3d)
                create_rotating_view_gif(base_filename)
            
            # Clean up trajectories
            del bicycle_trajectories[vehicle_id]
            if vehicle_id in bicycle_conflicts:
                del bicycle_conflicts[vehicle_id]

def three_dimensional_bicycle_trajectories_gif_old(frame):
    """
    Creates a 3D visualization of bicycle trajectories where the z=0 plane shows the static scene.
    Saves a gif of the rotating view animation.
    """
    global fig_3d, ax_3d, total_steps, bicycle_trajectories, transformer, flow_ids, bicycle_conflicts, foe_trajectories
    
    def save_rotating_view_frames(ax_3d, base_filename, n_frames=30):
        """Helper function to save frames for rotating view animation"""
        os.makedirs('out_3d_trajectories_gifs/rotation_frames', exist_ok=True)
        
        # Start from bird's eye view and smoothly transition both angles
        # Start: (90° elevation, 270° azimuth)    - bird's eye view
        # End: (35° elevation, 285° azimuth)    - final view
        
        # Create non-linear transitions to make the azimuth change more gradual
        # Use first half of frames mainly for elevation change, second half for azimuth
        t = np.linspace(0, 1, n_frames)
        t_azim = t**2  # Square the parameter to make azimuth change more gradual at start
        
        elevations = np.linspace(90, 35, n_frames)
        azimuths = 270 + (t_azim * 15)  # Smooth transition from 270° to 285°
        
        # Save a frame for each view angle
        for i, (elev, azim) in enumerate(zip(elevations, azimuths)):
            ax_3d.view_init(elev=elev, azim=azim)
            plt.savefig(f'out_3d_trajectories_gifs/rotation_frames/{base_filename}_frame_{i:03d}.png', 
                       dpi=300)

    def create_rotating_view_gif(base_filename, duration=0.1):
        """Helper function to create GIF from saved frames"""
        # Get all frames for this plot
        frames = sorted(glob.glob(f'out_3d_trajectories_gifs/rotation_frames/{base_filename}_frame_*.png'))
        
        # Read frames and create GIF
        images = [imageio.imread(frame) for frame in frames]
        output_file = f'out_3d_trajectories_gifs/{base_filename}_rotation.gif'
        imageio.mimsave(output_file, images, format='GIF', duration=duration)
        
        # Clean up frame files
        for frame in frames:
            os.remove(frame)
        
        print(f'Created rotating view animation: {output_file}')

    # Initialize transformer at frame 0
    if frame == 0:
        transformer = pyproj.Transformer.from_crs('EPSG:4326', 'EPSG:32632', always_xy=True)
        bicycle_trajectories.clear()
        flow_ids.clear()

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
    if not hasattr(three_dimensional_bicycle_trajectories, 'completed_foes'):
        three_dimensional_bicycle_trajectories.completed_foes = {}
    
    # Store completed foe trajectories before removing them
    for foe_id in departed_foes:
        if foe_id not in three_dimensional_bicycle_trajectories.completed_foes:
            three_dimensional_bicycle_trajectories.completed_foes[foe_id] = foe_trajectories[foe_id]

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
                flow_id = vehicle_id.rsplit('.', 1)[0]
                flow_ids.add(flow_id)
                if vehicle_id not in bicycle_trajectories:
                    bicycle_trajectories[vehicle_id] = []
                bicycle_trajectories[vehicle_id].append((x_utm, y_utm, current_time))
                
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
    finished_bicycles = set(bicycle_trajectories.keys()) - current_vehicles
    
    # Generate plots for finished bicycles
    for vehicle_id in finished_bicycles:
        if len(bicycle_trajectories[vehicle_id]) > 0:  # Only plot if we have trajectory data
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
            trajectory = bicycle_trajectories[vehicle_id]
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

                # Plot bicycle trajectory
                x_coords = np.array(x_coords)
                y_coords = np.array(y_coords)
                times = np.array(times)

                # Create a mask for points within the bounding box
                within_bounds = (x_coords >= minx) & (x_coords <= maxx) & (y_coords >= miny) & (y_coords <= maxy)

                # Filter points to only those within bounds
                x_coords_clipped = x_coords[within_bounds]
                y_coords_clipped = y_coords[within_bounds]
                times_clipped = times[within_bounds]

                # Plot bicycle ground projection with clipped coordinates
                ax_3d.plot(x_coords_clipped, y_coords_clipped, [base_z]*len(x_coords_clipped),
                          color='darkslateblue', linestyle='--', linewidth=2, alpha=0.7,
                          zorder=1000)

                # Create projection plane vertices with clipped coordinates
                plane_vertices = []
                for i in range(len(x_coords_clipped)-1):
                    quad = [
                        (x_coords_clipped[i], y_coords_clipped[i], times_clipped[i]),
                        (x_coords_clipped[i+1], y_coords_clipped[i+1], times_clipped[i+1]),
                        (x_coords_clipped[i+1], y_coords_clipped[i+1], base_z),
                        (x_coords_clipped[i], y_coords_clipped[i], base_z)
                    ]
                    plane_vertices.append(quad)
                
                proj_plane = Poly3DCollection(plane_vertices, alpha=0.2)
                proj_plane.set_facecolor('darkslateblue')
                proj_plane.set_edgecolor('none')
                proj_plane.set_sort_zpos(999)
                ax_3d.add_collection3d(proj_plane)

                # Plot 3D trajectory
                ax_3d.plot(x_coords_clipped, y_coords_clipped, times_clipped, 
                          color='darkslateblue', linewidth=2, alpha=1.0,
                          zorder=1000)

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
                ax_3d.text(x_coords_clipped[-1], y_coords_clipped[-1], base_z,
                          f'bicycle {vehicle_id}',
                          color='darkslateblue',
                          horizontalalignment='right',
                          verticalalignment='bottom',
                          rotation=90,
                          bbox=dict(facecolor='white', alpha=1.0, edgecolor='none'),
                          zorder=1000)

                # Create legend for conflict overview plot
                handles = [
                    plt.Line2D([0], [0], color='darkslateblue', linewidth=2, label='Bicycle Trajectory'),
                    plt.Line2D([0], [0], color='darkslateblue', linestyle='--', label='Ground Projection'),
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

                    # Plot bicycle trajectory
                    x_coords = np.array(x_coords)
                    y_coords = np.array(y_coords)
                    times = np.array(times)

                    # Create a mask for points within the bounding box
                    within_bounds = (x_coords >= minx) & (x_coords <= maxx) & (y_coords >= miny) & (y_coords <= maxy)

                    # Filter points to only those within bounds
                    x_coords_clipped = x_coords[within_bounds]
                    y_coords_clipped = y_coords[within_bounds]
                    times_clipped = times[within_bounds]

                    # Plot bicycle ground projection with clipped coordinates
                    ax_3d.plot(x_coords_clipped, y_coords_clipped, [base_z]*len(x_coords_clipped),
                             color='darkslateblue', linestyle='--', linewidth=2, alpha=0.7,
                             zorder=1000)

                    # Create projection plane vertices with clipped coordinates
                    plane_vertices = []
                    for i in range(len(x_coords_clipped)-1):
                        quad = [
                            (x_coords_clipped[i], y_coords_clipped[i], times_clipped[i]),
                            (x_coords_clipped[i+1], y_coords_clipped[i+1], times_clipped[i+1]),
                            (x_coords_clipped[i+1], y_coords_clipped[i+1], base_z),
                            (x_coords_clipped[i], y_coords_clipped[i], base_z)
                        ]
                        plane_vertices.append(quad)
                    
                    proj_plane = Poly3DCollection(plane_vertices, alpha=0.2)
                    proj_plane.set_facecolor('darkslateblue')
                    proj_plane.set_edgecolor('none')
                    proj_plane.set_sort_zpos(999)
                    ax_3d.add_collection3d(proj_plane)
                    
                    # Plot 3D bicycle trajectory
                    ax_3d.plot(x_coords_clipped, y_coords_clipped, times_clipped,
                             color='darkslateblue', linewidth=2, alpha=1.0,
                             zorder=1000)
                    
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
                    
                    # Plot foe trajectory if available
                    foe_traj = None
                    if foe_id in foe_trajectories:
                        foe_traj = foe_trajectories[foe_id]
                    elif foe_id in three_dimensional_bicycle_trajectories.completed_foes:
                        foe_traj = three_dimensional_bicycle_trajectories.completed_foes[foe_id]
                    
                    if foe_traj:
                        foe_x, foe_y, foe_times = zip(*foe_traj)
                        
                        # 1. Plot ground projection
                        ax_3d.plot(foe_x, foe_y, [base_z]*len(foe_x),
                                 color='black', linestyle='--', 
                                 linewidth=2, alpha=0.7, zorder=999)

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
                        
                        foe_proj_plane = Poly3DCollection(foe_plane_vertices, alpha=0.2)
                        foe_proj_plane.set_facecolor('black')
                        foe_proj_plane.set_edgecolor('none')
                        foe_proj_plane.set_sort_zpos(997)
                        ax_3d.add_collection3d(foe_proj_plane)

                        # 3. Plot 3D trajectory
                        ax_3d.plot(foe_x, foe_y, foe_times,
                                 color='black', linewidth=2, alpha=1.0,
                                 zorder=999)
                        
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
                    ax_3d.text(x_coords_clipped[-1], y_coords_clipped[-1], base_z,
                             f'bicycle {vehicle_id}',
                             color='darkslateblue',
                             horizontalalignment='right',
                             verticalalignment='bottom',
                             rotation=90,
                             bbox=dict(facecolor='white', alpha=1.0, edgecolor='none'),
                             zorder=1000)

                    # Create legend for individual conflict plot
                    handles = [
                        plt.Line2D([0], [0], color='darkslateblue', linewidth=2, label='Bicycle Trajectory'),
                        plt.Line2D([0], [0], color='black', linewidth=2, label='Foe Trajectory'),
                        plt.Line2D([0], [0], color='darkslateblue', linestyle='--', label='Ground Projection'),
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

                # Plot bicycle trajectory
                x_coords = np.array(x_coords)
                y_coords = np.array(y_coords)
                times = np.array(times)

                # Create a mask for points within the bounding box
                within_bounds = (x_coords >= minx) & (x_coords <= maxx) & (y_coords >= miny) & (y_coords <= maxy)

                # Filter points to only those within bounds
                x_coords_clipped = x_coords[within_bounds]
                y_coords_clipped = y_coords[within_bounds]
                times_clipped = times[within_bounds]

                # Plot bicycle ground projection with clipped coordinates
                ax_3d.plot(x_coords_clipped, y_coords_clipped, [base_z]*len(x_coords_clipped),
                          color='darkslateblue', linestyle='--', linewidth=2, alpha=0.7,
                          zorder=1000)

                # Create projection plane vertices with clipped coordinates
                plane_vertices = []
                for i in range(len(x_coords_clipped)-1):
                    quad = [
                        (x_coords_clipped[i], y_coords_clipped[i], times_clipped[i]),
                        (x_coords_clipped[i+1], y_coords_clipped[i+1], times_clipped[i+1]),
                        (x_coords_clipped[i+1], y_coords_clipped[i+1], base_z),
                        (x_coords_clipped[i], y_coords_clipped[i], base_z)
                    ]
                    plane_vertices.append(quad)
                
                proj_plane = Poly3DCollection(plane_vertices, alpha=0.2)
                proj_plane.set_facecolor('darkslateblue')
                proj_plane.set_edgecolor('none')
                proj_plane.set_sort_zpos(999)
                ax_3d.add_collection3d(proj_plane)

                # Plot 3D trajectory
                ax_3d.plot(x_coords_clipped, y_coords_clipped, times_clipped, 
                          color='darkslateblue', linewidth=2, alpha=1.0,
                          zorder=1000)
                
                # Add bicycle label
                ax_3d.text(x_coords_clipped[-1], y_coords_clipped[-1], base_z,
                          f'bicycle {vehicle_id}',
                          color='darkslateblue',
                          horizontalalignment='right',
                          verticalalignment='bottom',
                          rotation=90,
                          bbox=dict(facecolor='white', alpha=1.0, edgecolor='none'),
                          zorder=1000)

                # Create legend for bicycle trajectory plot
                handles = [
                    plt.Line2D([0], [0], color='darkslateblue', linewidth=2, label='Bicycle Trajectory'),
                    plt.Line2D([0], [0], color='darkslateblue', linestyle='--', label='Ground Projection')
                ]
                ax_3d.legend(handles=handles, loc='upper left')
                
                # Save bicycle trajectory plot
                base_filename = f'bicycle_{vehicle_id}_FCO{FCO_share*100:.0f}_FBO{FBO_share*100:.0f}'
                save_rotating_view_frames(ax_3d, base_filename)
                plt.close(fig_3d)
                create_rotating_view_gif(base_filename)
            
            # Clean up trajectories
            del bicycle_trajectories[vehicle_id]
            if vehicle_id in bicycle_conflicts:
                del bicycle_conflicts[vehicle_id]

# ---------------------
# MAIN EXECUTION
# ---------------------

if __name__ == "__main__":  
    load_sumo_simulation()
    gdf1, G, buildings, parks = load_geospatial_data()
    print('Geospatial data loaded.')
    gdf1_proj, G_proj, buildings_proj, parks_proj = project_geospatial_data(gdf1, G, buildings, parks)
    # gdf1_proj, buildings_proj, parks_proj = project_geospatial_data_new(gdf1, buildings, parks)
    print('Geospatial data projected.')
    setup_plot()
    plot_geospatial_data(gdf1_proj, G_proj, buildings_proj, parks_proj)
    # plot_geospatial_data_new(gdf1_proj, buildings_proj, parks_proj)
    if relativeVisibility:
        x_coords, y_coords, grid_cells, visibility_counts = initialize_grid(buildings_proj)
        print('Binning Map (Grid Map) initiated.')
    total_steps = get_total_simulation_steps(sumo_config_path)
    if useLiveVisualization:
        anim = run_animation(total_steps)
    else:
        for frame in range(total_steps):
            update_with_ray_tracing(frame)
    summary_logging()
    traci.close()
    print("SUMO simulation closed and TraCi disconnected.")
    if relativeVisibility:
        create_visibility_heatmap(x_coords, y_coords, visibility_counts)