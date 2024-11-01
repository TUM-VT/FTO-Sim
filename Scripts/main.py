import os
import osmnx as ox
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import libsumo as traci
from shapely.geometry import Point, box, Polygon
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from rtree import index
import csv
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from collections import Counter
from collections import defaultdict
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from io import BytesIO
from PIL import Image
from shapely.geometry import Polygon, MultiPolygon, box
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import shapely

# Setup logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------------
# CONFIGURATION
# ---------------------

# General Settings:

useLiveVisualization = False # Live Visualization of Ray Tracing
visualizeRays = False # Visualize rays additionaly to the visibility polygon
useManualFrameForwarding = False # Visualization of each frame, manual input necessary to forward the visualization
saveAnimation = False # Save the animation (currently not compatible with live visualization)

useRTREEmethod = False

# Bounding Box Settings:

north, south, east, west = 48.150600, 48.148800, 11.570800, 11.567600
bbox = (north, south, east, west)

# Path Settings:

base_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(base_dir)
sumo_config_path = os.path.join(parent_dir, 'SUMO_example', 'SUMO_example.sumocfg') # Path to SUMO config-file
geojson_path = os.path.join(parent_dir, 'SUMO_example', 'SUMO_example.geojson') # Path to GEOjson file

# FCO / FBO Settings:

FCO_share = 0.1
FBO_share = 0
numberOfRays = 360

# Warm Up Settings:

delay = 60 #warm-up time in seconds (during this time in the beginning of the simulation, no ray tracing is performed)

# Grid Map Settings:

grid_size =  0.5 # Grid Size for Heat Map Visualization (the smaller the grid size, the higher the resolution)

# Application Settings:

relativeVisibility = False # Generate relative visibility heatmaps
IndividualBicycleTrajectories = True # Generate 2D space-time diagrams of bicycle trajectories (individual trajectory plots)
FlowBasedBicycleTrajectories = True # Generate 2D space-time diagrams of bicycle trajectories (flow-based trajectory plots)
ThreeDimensionalBicycleTrajectories = False # Generate 3D space-time diagrams of bicycle trajectories (3D trajectory plots)

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

# ---------------------

# ---------------------
# INITIALIZATION
# ---------------------

def load_sumo_simulation():
    """
    Loads and starts the SUMO simulation using the specified configuration file.
    """
    sumoCmd = ["sumo", "-c", sumo_config_path]
    traci.start(sumoCmd)

def load_geospatial_data():
    """
    Loads geospatial data for the simulation area.
    Returns GeoDataFrames (buildings, parks, road space) and a NetworkX graph.
    """
    gdf1 = gpd.read_file(geojson_path)  # road space distribution
    G = ox.graph_from_bbox(bbox=bbox, network_type='all')  # NetworkX graph (bounding box)
    buildings = ox.features_from_bbox(bbox=bbox, tags={'building': True})  # buildings
    parks = ox.features_from_bbox(bbox=bbox, tags={'leisure': 'park'})  # parks
    return gdf1, G, buildings, parks  # returning loaded geospatial data

def project_geospatial_data(gdf1, G, buildings, parks):
    """
    Projects geospatial data to a UTM zone 32N (EPSG:32632) coordinate system.
    Takes GeoDataFrames and NetworkX graph as input and returns their projected versions.
    """
    gdf1_proj = gdf1.to_crs("EPSG:32632")  # road space distribution
    G_proj = ox.project_graph(G, to_crs="EPSG:32632")  # NetworkX graph (bounding box)
    buildings_proj = buildings.to_crs("EPSG:32632")  # buildings
    parks_proj = parks.to_crs("EPSG:32632")  # parks
    return gdf1_proj, G_proj, buildings_proj, parks_proj  # returning projected data

def initialize_grid(buildings_proj, grid_size=1.0):
    """
    Initializes a grid for visibility analysis.
    Creates a grid of cells covering the simulated scene.
    Calculates the coordinates for each cell and initializes visibility counts.
    """
    x_min, y_min, x_max, y_max = buildings_proj.total_bounds  # bounding box
    x_coords = np.arange(x_min, x_max, grid_size)  # array of x-coordinates with specified grid size
    y_coords = np.arange(y_min, y_max, grid_size)  # array of y-coordinates with specified grid size
    grid_points = [(x, y) for x in x_coords for y in y_coords]  # grid points as (x, y) tuples
    grid_cells = [box(x, y, x + grid_size, y + grid_size) for x, y in grid_points]  # box geometries for each grid cell
    visibility_counts = {cell: 0 for cell in grid_cells}  # initialization of visibility count for each cell to 0
    return x_coords, y_coords, grid_cells, visibility_counts  # returning grid information and visibility counts

def get_total_simulation_steps(sumo_config_file):
    """
    Calculates the total number of simulation steps based on the SUMO configuration file.
    Parses the XML file to extract begin time, end time, and step length.
    Returns the total number of steps as an integer.
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
    Retrieves the step length from the SUMO configuration file.
    Parses the XML file to extract the step length value.
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
    Sets up the plot by adding a title and legend for buildings and parks.
    """
    ax.set_title('Ray Tracing Visualization')  # Set the title of the plot
    legend_handles = [
        Rectangle((0, 0), 1, 1, facecolor='gray', edgecolor='black', linewidth=0.5, alpha=0.7, label='Buildings'),  # Create a rectangle for buildings legend
        Rectangle((0, 0), 1, 1, facecolor='green', edgecolor='black', linewidth=0.5, alpha=0.7, label='Parks')  # Create a rectangle for parks legend
    ]
    ax.legend(handles=legend_handles)  # Add the legend to the plot

def plot_geospatial_data(gdf1_proj, G_proj, buildings_proj, parks_proj):
    """
    Plots the geospatial data including the road network, buildings, and parks on the current axes.
    """
    ox.plot_graph(G_proj, ax=ax, bgcolor='none', edge_color='none', node_size=0, show=False, close=False)  # Plot the road network without visible nodes or edges
    gdf1_proj.plot(ax=ax, color='lightgray', alpha=0.5, edgecolor='lightgray')  # Plot the first GeoDataFrame (likely the base map) in light gray
    buildings_proj.plot(ax=ax, facecolor='gray', edgecolor='black', linewidth=0.5, alpha=0.7)  # Plot buildings in gray with black outlines
    parks_proj.plot(ax=ax, facecolor='green', edgecolor='black', linewidth=0.5, alpha=0.7)  # Plot parks in green with black outlines

def convert_simulation_coordinates(x, y):
    """
    Converts SUMO simulation coordinates to UTM zone 32N (EPSG:32632) coordinates.
    """
    lon, lat = traci.simulation.convertGeo(x, y)  # Convert SUMO coordinates to longitude and latitude
    x_32632, y_32632 = project(lon, lat)  # Project longitude and latitude to UTM zone 32N
    return x_32632, y_32632  # Return the converted coordinates

def vehicle_attributes(vehicle_type):
    """
    Determines the attributes of a vehicle based on its type.
    Returns a tuple containing the shape (Rectangle), color, and dimensions (width, length) for the given vehicle type.
    If the vehicle type is not recognized, default attributes are returned.
    """
    # Define a dictionary to store vehicle attributes
    vehicle_types = {
        # Observer vehicles
        "floating_car_observer": (Rectangle, 'red', (1.8, 5)),
        "floating_bike_observer": (Rectangle, 'red', (0.65, 1.6)),
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
    Creates a polygon representation of a vehicle given its position, dimensions, and angle.
    Returns a Polygon object representing the vehicle shape.
    """
    adjusted_angle = (-angle) % 360  # Adjust angle for correct rotation
    rect = Polygon([(-width / 2, -length / 2), (-width / 2, length / 2), (width / 2, length / 2), (width / 2, -length / 2)])  # Create initial rectangle
    rotated_rect = rotate(rect, adjusted_angle, use_radians=False, origin=(0, 0))  # Rotate rectangle
    translated_rect = translate(rotated_rect, xoff=x, yoff=y)  # Move rectangle to correct position
    return translated_rect  # Return final polygon

# ---------------------
# RAY TRACING
# ---------------------

def generate_rays(center, num_rays=360, radius=30):
    """
    Generates a set ('num_rays') of evenly spaced rays emerging from the center point of an object with the length 'radius'.
    Returns a list of ray segments, where each segment is defined by its start point (center) and end point.
    """
    angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)  # Create evenly spaced angles
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

def detect_intersections_rtree(ray, objects, rtree_idx):
    closest_intersection = None
    min_distance = float('inf')
    ray_line = LineString(ray)
    possible_matches = list(rtree_idx.intersection(ray_line.bounds))
    for i in possible_matches:
        obj = objects[i]
        if ray_line.intersects(obj):
            intersection_point = ray_line.intersection(obj)
            if intersection_point.is_empty:
                continue
            if intersection_point.geom_type.startswith('Multi'):
                for part in intersection_point.geoms:
                    if part.is_empty or not hasattr(part, 'coords'):
                        continue
                    for coord in part.coords:
                        distance = Point(ray[0]).distance(Point(coord))  # Calculate distance
                        if distance < min_distance:
                            min_distance = distance
                            closest_intersection = coord
            else:
                if not hasattr(intersection_point, 'coords'):
                    continue
                for coord in intersection_point.coords:
                    distance = Point(ray[0]).distance(Point(coord))  # Calculate distance
                    if distance < min_distance:
                        min_distance = distance
                        closest_intersection = coord
    return closest_intersection

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

    traci.simulationStep()  # Advance the simulation by one step

    if useManualFrameForwarding:
        input("Press Enter to continue...")  # Wait for user input if manual forwarding is enabled

    if IndividualBicycleTrajectories:
        individual_bicycle_trajectories(frame)
    if FlowBasedBicycleTrajectories:
        flow_based_bicycle_trajectories(frame, total_steps)
    if ThreeDimensionalBicycleTrajectories:
        three_dimensional_bicycle_trajectories(frame)

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
            
            # Create dynamic objects (other vehicles)
            dynamic_objects_geom = [
                create_vehicle_polygon(
                    *convert_simulation_coordinates(*traci.vehicle.getPosition(vid)),
                    *vehicle_attributes(traci.vehicle.getTypeID(vid))[2],
                    traci.vehicle.getAngle(vid)
                ) for vid in traci.vehicle.getIDList() if vid != vehicle_id
            ]

            # Update vehicle patches
            x, y = traci.vehicle.getPosition(vehicle_id)
            x_32632, y_32632 = convert_simulation_coordinates(x, y)
            angle = traci.vehicle.getAngle(vehicle_id)
            adjusted_angle = (-angle) % 360
            
            lower_left_corner = (x_32632 - width / 2, y_32632 - length / 2)
            patch = Rectangle(lower_left_corner, width, length, edgecolor=edgecolor, fill=None)

            t = transforms.Affine2D().rotate_deg_around(x_32632, y_32632, adjusted_angle) + ax.transData
            patch.set_transform(t)
            new_vehicle_patches.append(patch)

            # Perform ray tracing for FCOs and FBOs
            if vehicle_type in ["floating_car_observer", "floating_bike_observer"]:
                center = (x_32632, y_32632)
                rays = generate_rays(center, num_rays=numberOfRays, radius=30)
                all_objects = static_objects + dynamic_objects_geom
                ray_endpoints = []

                # Use multithreading for ray tracing
                with ThreadPoolExecutor() as executor:
                    if useRTREEmethod:
                        # Use R-tree for spatial indexing
                        static_index = index.Index()
                        for i, obj in enumerate(static_objects):
                            static_index.insert(i, obj.bounds)
                        futures = {executor.submit(detect_intersections_rtree, ray, all_objects, static_index): ray for ray in rays}
                    else:
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
                            end_point = (ray[0][0] + np.cos(angle) * 30, ray[0][1] + np.sin(angle) * 30)
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

def run_animation(total_steps):
    """
    Runs and displays a matplotlib animation of the ray tracing simulation.
    """
    global fig, ax

    if useLiveVisualization:
        # Close existing figure and switch backend
        plt.close(fig)
        matplotlib.use('TkAgg', force=True)
        
        # Create new figure with interactive backend
        fig, ax = plt.subplots(figsize=(12, 8))
        plot_geospatial_data(gdf1_proj, G_proj, buildings_proj, parks_proj)
        setup_plot()
    
    # Create animation
    anim = FuncAnimation(fig, update_with_ray_tracing, frames=range(1, total_steps), 
                        interval=33, repeat=False)
    
    if saveAnimation:
        writer = FFMpegWriter(fps=1, metadata=dict(artist='Me'), bitrate=1800)
        filename = f'out_raytracing/ray_tracing_animation_FCO{FCO_share*100:.0f}%_FBO{FBO_share*100:.0f}%.mp4'
        anim.save(filename, writer=writer)
        print(f"Animation saved as {filename}")

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

# ---------------------
# APPLICATIONS
# ---------------------

def create_visibility_heatmap(x_coords, y_coords, visibility_counts):
    """
    Generates a CSV file with raw visibility data (visibility counts) and plots a normalized heatmap.
    Saves the heatmap as a PNG file if 'relative visibility' is enabled in the Application Settings.
    """
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

    # Plot and save heatmap if relative visibility is enabled
    if relativeVisibility:
        fig, ax = plt.subplots(figsize=(12, 8), facecolor='lightgray')
        ax.set_facecolor('lightgray')
        buildings_proj.plot(ax=ax, facecolor='gray', edgecolor='black', linewidth=0.5, alpha=0.7)
        parks_proj.plot(ax=ax, facecolor='green', edgecolor='black', linewidth=0.5, alpha=0.7)
        cax = ax.imshow(heatmap_data.T, origin='lower', cmap='hot', extent=[x_min, x_max, y_min, y_max], alpha=0.6)
        ax.set_title('Relative Visibility Heatmap')
        fig.colorbar(cax, ax=ax, label='Relative Visibility')
        plt.savefig(f'out_raytracing/relative_visibility_heatmap_FCO{str(FCO_share*100)}%_FBO{str(FBO_share*100)}%.png')

def individual_bicycle_trajectories(frame):
    """
    Creates space-time diagrams for individual bicycles, including detection status, traffic lights,
    and conflicts detected by SUMO's SSM device.
    """
    global bicycle_data, bicycle_start_times, traffic_light_positions, bicycle_tls

    # Create output directory if it doesn't exist
    os.makedirs('out_bicycle_trajectories', exist_ok=True)
    
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
            
            for i, (distance, time) in enumerate(data):
                # Find corresponding detection status
                detection_time = time + bicycle_start_times[vehicle_id]
                detection_status = False
                for det_time, det_status in detection_data:
                    if abs(det_time - detection_time) < step_length:
                        detection_status = det_status
                        break
                
                if current_detected is None:
                    current_detected = detection_status
                    current_points = [(distance, time)]
                elif detection_status != current_detected:
                    if current_points:
                        segments['detected' if current_detected else 'undetected'].append(current_points)
                    current_points = [(distance, time)]
                    current_detected = detection_status
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
            if vehicle_id in bicycle_conflicts:
                # Group conflicts by foe vehicle ID
                conflicts_by_foe = {}
                for conflict in bicycle_conflicts[vehicle_id]:
                    # Convert absolute simulation time to elapsed time for this bicycle
                    conflict_elapsed_time = conflict['time'] - bicycle_start_times[vehicle_id]
                    
                    foe_id = conflict.get('foe_id')
                    if foe_id:
                        if foe_id not in conflicts_by_foe:
                            conflicts_by_foe[foe_id] = []
                        # Store conflict with elapsed time instead of simulation time
                        conflicts_by_foe[foe_id].append({
                            'distance': conflict['distance'],
                            'time': conflict_elapsed_time,  # This is now relative to bicycle start
                            'ttc': conflict['ttc'],
                            'pet': conflict['pet'],
                            'drac': conflict['drac'],
                            'severity': conflict['severity'],
                            'foe_type': conflict['foe_type']
                        })
                
                # Plot only the most severe conflict for each foe
                for foe_conflicts in conflicts_by_foe.values():
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
            
            # Add bicycle ID and departure time information
            departure_time = bicycle_start_times[vehicle_id]
            info_text = f"Bicycle: {vehicle_id}\nDeparture Time: {departure_time:.2f}s"
            ax.text(0.02, 0.98, info_text, 
                    transform=ax.transAxes, verticalalignment='top', 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Add legend with all elements
            handles = [
                plt.Line2D([0], [0], color='black', lw=2, label='bicycle undetected'),
                plt.Line2D([0], [0], color='darkturquoise', lw=2, label='bicycle detected'),
                plt.Line2D([0], [0], marker='o', color='firebrick', linestyle='None', 
                          markerfacecolor='none', markersize=10, label='potential conflict detected'),
                plt.Line2D([0], [0], color='red', lw=2, label='Red TL'),
                plt.Line2D([0], [0], color='yellow', lw=2, label='Yellow TL'),
                plt.Line2D([0], [0], color='green', lw=2, label='Green TL')
            ]
            ax.legend(handles=handles, loc='lower right', bbox_to_anchor=(0.95, 0.05))
            
            # Save the plot
            plt.savefig(f'out_bicycle_trajectories/space_time_diagram_{vehicle_id}.png', bbox_inches='tight')
            plt.close(fig)
            
            print(f"Space-time diagram for bicycle {vehicle_id} has been saved.")
            
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
            
            if vehicle_id not in bicycle_start_times:
                # First time we see this bicycle
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
                        
                        bicycle_conflicts[vehicle_id].append({
                            'distance': distance,
                            'time': current_time,  # Store absolute simulation time
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
    global bicycle_flow_data, traffic_light_positions, bicycle_tls, step_length, bicycle_conflicts, traffic_light_programs, flow_detection_data

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

                for distance, time, is_detected in trajectory_data:
                    if current_detected is None:
                        current_detected = is_detected
                        current_points = [(distance, time)]
                    elif is_detected != current_detected:
                        if current_points:
                            segments['detected' if current_detected else 'undetected'].append(current_points)
                        current_points = [(distance, time)]
                        current_detected = is_detected
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
                        if foe_id:
                            if foe_id not in conflicts_by_foe:
                                conflicts_by_foe[foe_id] = []
                            conflicts_by_foe[foe_id].append(conflict)
                    
                    for foe_conflicts in conflicts_by_foe.values():
                        most_severe = max(foe_conflicts, key=lambda x: x['severity'])
                        size = 50 + (most_severe['severity'] * 100)
                        ax.scatter(most_severe['distance'], most_severe['time'], 
                                 color='firebrick', marker='o', s=size, zorder=5,
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
            ax.legend(handles=handles, loc='upper left')

            plt.savefig(f'out_flow_trajectories/flow_{flow_id}_space_time_diagram.png', 
                       bbox_inches='tight')
            plt.close(fig)
            
            print(f"Space-time diagram for flow {flow_id} has been saved.")

def three_dimensional_bicycle_trajectories(frame):
    """
    Creates a 3D visualization of bicycle trajectories where the z=0 plane shows the static scene.
    The plot is only generated once the simulation is complete.
    """
    global fig_3d, ax_3d, total_steps, bicycle_trajectories, transformer, flow_ids

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

    # Collect bicycle positions for this frame
    current_time = frame * step_length
    for vehicle_id in traci.vehicle.getIDList():
        vehicle_type = traci.vehicle.getTypeID(vehicle_id)
        if vehicle_type in ["bicycle", "DEFAULT_BIKETYPE", "floating_bike_observer"]:
            flow_id = vehicle_id.rsplit('.', 1)[0]
            flow_ids.add(flow_id)
            
            x_sumo, y_sumo = traci.vehicle.getPosition(vehicle_id)
            lon, lat = traci.simulation.convertGeo(x_sumo, y_sumo)
            x_utm, y_utm = transformer.transform(lon, lat)
            
            # Only store positions within the bounding box
            point = Point(x_utm, y_utm)
            if bbox_transformed.contains(point):
                if vehicle_id not in bicycle_trajectories:
                    bicycle_trajectories[vehicle_id] = []
                bicycle_trajectories[vehicle_id].append((x_utm, y_utm, current_time))

    # Plot the static scene at frame 0
    if frame == 0:
        fig_3d = plt.figure(figsize=(15, 8))
        ax_3d = fig_3d.add_subplot(111, projection='3d')
        
        # Get bounds of transformed bounding box
        minx, miny, maxx, maxy = bbox_transformed.bounds
        z = 0  # All elements will be placed at z=0
        
        # Plot static scene with reduced alpha for better visibility of trajectories
        # Plot road network first (bottom layer)
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
                        verts = [(x, y, z) for x, y in zip(xs, ys)]
                        poly = Poly3DCollection([verts], alpha=0.3)
                        poly.set_facecolor('lightgray')
                        ax_3d.add_collection3d(poly)

        # Plot parks (middle layer)
        for _, park in parks_proj.iterrows():
            if park.geometry.intersects(bbox_transformed):
                clipped_geom = park.geometry.intersection(bbox_transformed)
                if isinstance(clipped_geom, (MultiPolygon, Polygon)):
                    if isinstance(clipped_geom, MultiPolygon):
                        polygons = clipped_geom.geoms
                    else:
                        polygons = [clipped_geom]
                    
                    for polygon in polygons:
                        xs, ys = polygon.exterior.xy
                        verts = [(x, y, z) for x, y in zip(xs, ys)]
                        poly = Poly3DCollection([verts], alpha=0.4)
                        poly.set_facecolor('green')
                        poly.set_edgecolor('black')
                        ax_3d.add_collection3d(poly)

        # Plot buildings (top layer of static scene)
        for _, building in buildings_proj.iterrows():
            if building.geometry.intersects(bbox_transformed):
                clipped_geom = building.geometry.intersection(bbox_transformed)
                if isinstance(clipped_geom, (MultiPolygon, Polygon)):
                    if isinstance(clipped_geom, MultiPolygon):
                        polygons = clipped_geom.geoms
                    else:
                        polygons = [clipped_geom]
                    
                    for polygon in polygons:
                        xs, ys = polygon.exterior.xy
                        verts = [(x, y, z) for x, y in zip(xs, ys)]
                        poly = Poly3DCollection([verts], alpha=0.5)
                        poly.set_facecolor('gray')
                        poly.set_edgecolor('black')
                        ax_3d.add_collection3d(poly)
        
        # Set axis limits using transformed bounding box
        ax_3d.set_xlim(minx, maxx)
        ax_3d.set_ylim(miny, maxy)
        ax_3d.set_zlim(0, total_steps * step_length)
        
        # Set correct aspect ratio
        dx = maxx - minx
        dy = maxy - miny
        dz = total_steps * step_length
        
        # Set aspect ratio to be equal for x and y, but different for z
        ax_3d.set_box_aspect([dx, dy, dz])
        
        # Set labels and title
        ax_3d.set_xlabel('X (m)')
        ax_3d.set_ylabel('Y (m)')
        ax_3d.set_zlabel('Time (s)')
        ax_3d.set_title('3D Bicycle Trajectories')

        # Adjust the view angle for better visualization
        ax_3d.view_init(elev=35, azim=285)

    # At the end of simulation, get user input and plot trajectories
    if frame == total_steps - 1:
        print("\nAvailable bicycle flows:")
        sorted_flows = sorted(flow_ids)
        for flow_id in sorted_flows:
            print(f"- {flow_id}")
        
        print("\nEnter the flow numbers you want to plot (separated by spaces)")
        print("Example: '10 11 12' or 'all' for all flows")
        selected_flows_input = input("Select flows to plot: ").strip().lower()
        
        if selected_flows_input == 'all':
            selected_flows = sorted_flows
        else:
            try:
                # Convert input to flow IDs
                flow_numbers = [int(num) for num in selected_flows_input.split()]
                selected_flows = [f"flow_{num}" for num in flow_numbers]
                # Validate that selected flows exist
                selected_flows = [flow for flow in selected_flows if flow in flow_ids]
                if not selected_flows:
                    print("No valid flows selected. Plotting all flows.")
                    selected_flows = sorted_flows
                else:
                    print(f"Valid flows selected: {', '.join(selected_flows)}")
            except ValueError:
                print("Invalid input. Plotting all flows.")
                selected_flows = sorted_flows

        print("\nPlotting bicycle trajectories...")
        print(f"Selected flows: {', '.join(selected_flows)}")
        
        # Group trajectories by flow
        flow_trajectories = {}
        for vehicle_id, trajectory in bicycle_trajectories.items():
            flow_id = vehicle_id.rsplit('.', 1)[0]
            if flow_id in selected_flows:
                if flow_id not in flow_trajectories:
                    flow_trajectories[flow_id] = []
                flow_trajectories[flow_id].append(trajectory)

        trajectories_plotted = 0
        for i, (flow_id, trajectories) in enumerate(flow_trajectories.items()):
            print(f"\nProcessing flow {flow_id}:")
            
            # Plot each trajectory in the flow
            for trajectory in trajectories:
                x_coords, y_coords, times = zip(*trajectory)
                print(f"Number of positions: {len(trajectory)}")
                print(f"X range: {min(x_coords):.2f} to {max(x_coords):.2f}")
                print(f"Y range: {min(y_coords):.2f} to {max(y_coords):.2f}")
                print(f"Time range: {min(times):.2f} to {max(times):.2f}")
                
                # Plot 3D trajectory
                line = ax_3d.plot(x_coords, y_coords, times, 
                                color='black', linewidth=1, alpha=1.0,
                                zorder=10)
                trajectories_plotted += 1
            
            # Plot dashed line on z=0 plane using the first trajectory of this flow
            first_trajectory = trajectories[0]
            x_coords, y_coords, _ = zip(*first_trajectory)
            ax_3d.plot(x_coords, y_coords, [0]*len(x_coords),
                      color='black', linestyle='--', linewidth=1, alpha=0.7,
                      zorder=5)
            
            # Add flow label at the end of the dashed line
            # Alternate label positions above and below the line to avoid overlap
            vertical_align = 'bottom' if i % 2 == 0 else 'top'
            y_offset = 10 if i % 2 == 0 else -10  # Adjust this value to control spacing
            
            ax_3d.text(x_coords[-1], y_coords[-1], y_offset,
                      f'Flow {flow_id.split("_")[1]}',
                      color='black',
                      horizontalalignment='right',
                      verticalalignment=vertical_align,
                      rotation=90)  # Rotate text for better readability

        print(f"\nTrajectories plotted: {trajectories_plotted}")
        
        # Save the plot
        os.makedirs('out_3d_trajectories', exist_ok=True)
        plt.savefig('out_3d_trajectories/3d_bicycle_trajectories.png', bbox_inches='tight', dpi=300)
        print(f"3D trajectory plot saved.")
        plt.close(fig_3d)

# ---------------------
# MAIN EXECUTION
# ---------------------

if __name__ == "__main__":  
    load_sumo_simulation()
    print('SUMO simulation loaded.')
    gdf1, G, buildings, parks = load_geospatial_data()
    print('Geospatial data loaded.')
    gdf1_proj, G_proj, buildings_proj, parks_proj = project_geospatial_data(gdf1, G, buildings, parks)
    print('Geospatial data projected.')
    setup_plot()
    plot_geospatial_data(gdf1_proj, G_proj, buildings_proj, parks_proj)
    x_coords, y_coords, grid_cells, visibility_counts = initialize_grid(buildings_proj)
    print('Binning Map (Grid Map) initiated.')
    total_steps = get_total_simulation_steps(sumo_config_path)
    print('Ray Tracing initiated:')
    if useLiveVisualization:
        anim = run_animation(total_steps)
    else:
        for frame in range(total_steps):
            update_with_ray_tracing(frame)
    print('Ray tracing completed.')
    if saveAnimation:
        print(f'Ray tracing animation saved in out_raytracing as ray_tracing_animation_FCO{str(FCO_share*100)}%_FBO{str(FBO_share*100)}%.mp4.')
    summary_logging()
    print('Logging completed and saved in out_logging.')
    traci.close()
    print('TraCI closed.')
    create_visibility_heatmap(x_coords, y_coords, visibility_counts)
    if relativeVisibility:
        print(f'Relative Visibility Heat Map Generation completed - file saved in out_raytracing as relative_visibility_heatmap_FCO{str(FCO_share*100)}%_FBO{str(FBO_share*100)}%.png.')

