import os
import osmnx as ox
import matplotlib
matplotlib.use('Agg')  # Agg backend for non-interactive plotting
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

# Setup logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------------
# CONFIGURATION
# ---------------------

# General Settings:

useLiveVisualization = False # Live Visualization of Ray Tracing
visualizeRays = False # Visualize rays additionaly to the visibility polygon
useManualFrameForwarding = False # Visualization of each frame, manual input necessary to forward the visualization
saveAnimation = False # Save the animation

useRTREEmethod = False
fig, ax = plt.subplots(figsize=(12, 8)) # General visualization settings

# Bounding Box Settings:

north, south, east, west = 48.1505, 48.14905, 11.5720, 11.5669
bbox = (north, south, east, west)

# Path Settings:

base_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(base_dir)
sumo_config_path = os.path.join(parent_dir, 'SUMO_example', 'SUMO_example.sumocfg') # Path to SUMO config-file
geojson_path = os.path.join(parent_dir, 'SUMO_example', 'SUMO_example.geojson') # Path to GEOjson file

# FCO / FBO Settings:

FCO_share = 0
FBO_share = 0
numberOfRays = 10

# Warm Up Settings:

delay = 90 #warm-up time in seconds (during this time in the beginning of the simulation, no ray tracing is performed)

# Grid Map Settings:

grid_size =  0.5 # Grid Size for Heat Map Visualization (the smaller the grid size, the higher the resolution)

# Application Settings:

relativeVisibility = False # Generate relative visibility heatmaps
IndividualBicycleTrajectoryTracing = True # Generate 2D space-time diagrams of bicycle trajectories (individual trajectory plots)
FlowBasedBicycleTrajectoryTracing = False # Generate 2D space-time diagrams of bicycle trajectories (plots for each flow of bicycle traffic)

# ---------------------

# Loading of Geospatial Data

buildings = ox.features_from_bbox(bbox=bbox, tags={'building': True})
buildings_proj = buildings.to_crs("EPSG:32632")

# Projection Settings:

proj_from = pyproj.Proj('epsg:4326')   # Source projection: WGS 84
proj_to = pyproj.Proj('epsg:32632')    # Target projection: UTM zone 32N
project = pyproj.Transformer.from_proj(proj_from, proj_to, always_xy=True).transform

# Initialization of empty lists:

vehicle_patches = []
ray_lines = []
visibility_polygons = []

# Initialization of empty disctionaries
bicycle_trajectory_data = {}
bicycle_flow_data = {}

# Initialization of Grid Parameters:

x_min, y_min, x_max, y_max = buildings_proj.total_bounds
x_coords = np.arange(x_min, x_max, grid_size)
y_coords = np.arange(y_min, y_max, grid_size)
grid_points = [(x, y) for x in x_coords for y in y_coords]
grid_cells = [box(x, y, x + grid_size, y + grid_size) for x, y in grid_points]

# Initialization of Visibility Counts (for Heat Map Visualization)

visibility_counts = {cell: 0 for cell in grid_cells}

# Logging Settings:

# Initialization of sets to track unique vehicles
unique_vehicles = set()
vehicle_type_set = set()
# Initialize a DataFrame to log information at each time step
log_columns = ['time_step']
simulation_log = pd.DataFrame(columns=log_columns)

# Global variables to track if bicycle flow data has been processed
global flows_processed
flows_processed = False

# ---------------------

# ---------------------
# INITIALIZATION
# ---------------------

def load_sumo_simulation():
    """
    Loads and starts the SUMO simulation using the specified configuration file.
    """
    sumoCmd = ["sumo", "-c", sumo_config_path]  # Define the SUMO command with config file
    traci.start(sumoCmd)  # Start the TraCI connection to SUMO

def load_geospatial_data():
    """
    Loads geospatial data for the simulation area.
    Returns GeoDataFrames (buildings, parks, road space) and a NetworkX graph.
    """
    north, south, east, west = 48.1505, 48.14905, 11.5720, 11.5669  # Define the bounding box coordinates
    bbox = (north, south, east, west)  # Create a bounding box tuple
    gdf1 = gpd.read_file(geojson_path)  # Load GeoJSON file into a GeoDataFrame
    G = ox.graph_from_bbox(bbox=bbox, network_type='all')  # Create a NetworkX graph from the bounding box
    buildings = ox.features_from_bbox(bbox=bbox, tags={'building': True})  # Extract building features from OSM
    parks = ox.features_from_bbox(bbox=bbox, tags={'leisure': 'park'})  # Extract park features from OSM
    return gdf1, G, buildings, parks  # Return all loaded geospatial data

def project_geospatial_data(gdf1, G, buildings, parks):
    """
    Projects geospatial data to a UTM zone 32N (EPSG:32632) coordinate system.
    Takes GeoDataFrames and NetworkX graph as input and returns their projected versions.
    """
    gdf1_proj = gdf1.to_crs("EPSG:32632")  # Project the first GeoDataFrame to UTM zone 32N
    G_proj = ox.project_graph(G, to_crs="EPSG:32632")  # Project the NetworkX graph to UTM zone 32N
    buildings_proj = buildings.to_crs("EPSG:32632")  # Project the buildings GeoDataFrame to UTM zone 32N
    parks_proj = parks.to_crs("EPSG:32632")  # Project the parks GeoDataFrame to UTM zone 32N
    return gdf1_proj, G_proj, buildings_proj, parks_proj  # Return all projected data

def initialize_grid(buildings_proj, grid_size=1.0):
    """
    Initializes a grid for visibility analysis.
    Creates a grid of cells covering the area of the buildings.
    Calculates the coordinates for each cell and initializes visibility counts.
    """
    x_min, y_min, x_max, y_max = buildings_proj.total_bounds  # Get the bounding box of the buildings
    x_coords = np.arange(x_min, x_max, grid_size)  # Create an array of x-coordinates with specified grid size
    y_coords = np.arange(y_min, y_max, grid_size)  # Create an array of y-coordinates with specified grid size
    grid_points = [(x, y) for x in x_coords for y in y_coords]  # Generate all grid points as (x, y) tuples
    grid_cells = [box(x, y, x + grid_size, y + grid_size) for x, y in grid_points]  # Create box geometries for each grid cell
    visibility_counts = {cell: 0 for cell in grid_cells}  # Initialize visibility count for each cell to 0
    return x_coords, y_coords, grid_cells, visibility_counts  # Return grid information and visibility counts

def get_total_simulation_steps(sumo_config_file):
    """
    Calculates the total number of simulation steps based on the SUMO configuration file.
    Parses the XML file to extract begin time, end time, and step length.
    Returns the total number of steps as an integer.
    """
    global step_length  # Declare step_length as a global variable
    tree = ET.parse(sumo_config_file)  # Parse the XML file
    root = tree.getroot()  # Get the root element of the XML tree
    begin = 0  # Default start time (can be adjusted)
    end = 3600  # Default end time (1 hour, can be adjusted)
    step_length = 0.1  # Default step length (can be adjusted)
    for time in root.findall('time'):  # Find all 'time' elements in the XML
        begin = float(time.find('begin').get('value', begin))  # Get 'begin' value, use default if not found
        end = float(time.find('end').get('value', end))  # Get 'end' value, use default if not found
        step_length = float(time.find('step-length').get('value', step_length))  # Get 'step-length', use default if not found
    total_steps = int((end - begin) / step_length)  # Calculate total steps
    return total_steps  # Return the total number of simulation steps

def get_step_length(sumo_config_file):
    """
    Retrieves the step length from the SUMO configuration file.
    Parses the XML file to extract the step length value.
    Returns the step length as a float.
    """
    tree = ET.parse(sumo_config_file)  # Parse the XML file
    root = tree.getroot()  # Get the root element of the XML tree
    step_length = -1  # Initialize step_length with a default value
    for time in root.findall('time'):  # Iterate through all 'time' elements
        step_length = float(time.find('step-length').get('value', step_length))  # Extract step-length value
    return step_length  # Return the extracted step length

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
                        distance = Point(ray[0]).distance(Point(coord))
                        if distance < min_distance:
                            min_distance = distance
                            closest_intersection = coord
            else:
                if not hasattr(intersection_point, 'coords'):
                    continue
                for coord in intersection_point.coords:
                    distance = Point(ray[0]).distance(Point(coord))
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
    global vehicle_patches, ray_lines, visibility_polygons, FCO_share, FBO_share, visibility_counts, numberOfRays, useRTREEmethod, visualizeRays, useManualFrameForwarding, delay
    detected_color = (1.0, 0.27, 0, 0.5)
    undetected_color = (0.53, 0.81, 0.98, 0.5)

    traci.simulationStep()  # Advance the simulation by one step

    if useManualFrameForwarding:
        input("Press Enter to continue...")  # Wait for user input if manual forwarding is enabled

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

            # Update bicycle trajectory tracing if enabled
            if IndividualBicycleTrajectoryTracing:
                update_bicycle_2d_diagram(frame)
            if FlowBasedBicycleTrajectoryTracing:
                update_bicycle_flow_diagrams(frame)

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
    Runs and displays a matplotlib animation of the ray tracing simulation for the specified number of steps, and optionally saves it as an MP4 file.
    Uses the update_with_ray_tracing function for each frame of the animation.
    """
    # Set up matplotlib backend
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, FFMpegWriter

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create animation
    ani = FuncAnimation(fig, update_with_ray_tracing, frames=range(1, total_steps), interval=33, repeat=False)
    
    # Display the animation
    plt.show()

    # Save animation if enabled
    if saveAnimation:
        writer = FFMpegWriter(fps=1, metadata=dict(artist='Me'), bitrate=1800)
        filename = f'out_raytracing/ray_tracing_animation_FCO{FCO_share*100:.0f}%_FBO{FBO_share*100:.0f}%.mp4'
        ani.save(filename, writer=writer)
        print(f"Animation saved as {filename}")

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

def get_tl_color(state):
    if state in ['G', 'g']:
        return 'green'
    elif state == 'y':
        return 'yellow'
    elif state == 'r':
        return 'red'
    else:
        print(f"Unexpected traffic light state: {state}")  # Debug print
        return 'gray'  # for other states like 'o' (blinking) or 'O' (off)
    
def get_relevant_tl_state(vehicle_id, tl_id):
    try:
        # Get the vehicle's current lane ID
        lane_id = traci.vehicle.getLaneID(vehicle_id)
        
        # Get all controlled links for this traffic light
        controlled_links = traci.trafficlight.getControlledLinks(tl_id)
        
        # Get the current program of the traffic light
        current_program = traci.trafficlight.getRedYellowGreenState(tl_id)
        
        for i, links in enumerate(controlled_links):
            for link in links:
                from_lane, to_lane, _ = link
                if from_lane == lane_id:
                    # We found the relevant link, return its state
                    return current_program[i]
        
        # If we didn't find a matching link, return None
        get_relevant_tl_state.error_count = getattr(get_relevant_tl_state, 'error_count', 0) + 1
        if get_relevant_tl_state.error_count <= 5:
            print(f"No matching link found for vehicle {vehicle_id} on lane {lane_id} at traffic light {tl_id}")
        return None
    except traci.exceptions.TraCIException as e:
        print(f"TraCI error for vehicle {vehicle_id} at traffic light {tl_id}: {str(e)}")
        return None
    except Exception as e:
        print(f"Unexpected error for vehicle {vehicle_id} at traffic light {tl_id}: {str(e)}")
        return None

def update_bicycle_2d_diagram(time_step):
    """
    Updates the space-time diagram for all individual bicycles if Individual Bicycle Trajectory Tracing is enabled in the Application Settings.
    Tracks bicycle trajectories, calculates driven distances, checks for the location of traffic lights and FCO/FBO detection.
    Saves plots for bicycles that have completed their routes.
    """
    global bicycle_trajectory_data
    step_length = get_step_length(sumo_config_path)

    # Get current bicycles if Individual Bicycle Trajectory Tracing is enabled
    if IndividualBicycleTrajectoryTracing:
        current_bicycles = [vid for vid in traci.vehicle.getIDList() if traci.vehicle.getTypeID(vid) in ["DEFAULT_BIKETYPE", "bicycle", "floating_bike_observer"]]
    
    for vehicle_id in current_bicycles:
        # Get bicycle position and convert coordinates
        x, y = traci.vehicle.getPosition(vehicle_id)
        x_32632, y_32632 = convert_simulation_coordinates(x, y)
        
        # Initialize data structure for new bicycles
        if vehicle_id not in bicycle_trajectory_data:
            bicycle_trajectory_data[vehicle_id] = {
                'times': [], 'positions': [], 'colors': [],
                'traffic_lights': {}, 'departure_time': time_step * step_length,
                'route': traci.vehicle.getRoute(vehicle_id)
            }
            
            # Initialize traffic light data for the entire route
            route = bicycle_trajectory_data[vehicle_id]['route']
            next_tls = traci.vehicle.getNextTLS(vehicle_id)
            for tls in next_tls:
                tl_id, tl_link_index, distance, _ = tls
                if tl_id not in bicycle_trajectory_data[vehicle_id]['traffic_lights']:
                    tl_pos = traci.junction.getPosition(tl_id)
                    tl_pos_32632 = convert_simulation_coordinates(*tl_pos)
                    bicycle_trajectory_data[vehicle_id]['traffic_lights'][tl_id] = {
                        'distance': distance,
                        'link_index': tl_link_index,
                        'states': []
                    }
        
        # Calculate distance traveled
        total_distance = traci.vehicle.getDistance(vehicle_id)

        # Check if bicycle is detected by FCO/FBO
        bicycle_hit = False
        bicycle_polygon = create_vehicle_polygon(x_32632, y_32632, 0.65, 1.6, traci.vehicle.getAngle(vehicle_id))
        
        for observer_id in traci.vehicle.getIDList():
            observer_type = traci.vehicle.getTypeID(observer_id)
            if observer_type in ["floating_car_observer", "floating_bike_observer"]:
                observer_x, observer_y = convert_simulation_coordinates(*traci.vehicle.getPosition(observer_id))
                rays = generate_rays((observer_x, observer_y), num_rays=numberOfRays, radius=30)
                
                for ray in rays:
                    ray_line = LineString(ray)
                    if ray_line.intersects(bicycle_polygon):
                        bicycle_hit = True
                        break
            
            if bicycle_hit:
                break

        # Update trajectory data
        elapsed_time = (time_step * step_length) - bicycle_trajectory_data[vehicle_id]['departure_time']
        bicycle_trajectory_data[vehicle_id]['times'].append(elapsed_time)
        bicycle_trajectory_data[vehicle_id]['positions'].append(total_distance)
        bicycle_trajectory_data[vehicle_id]['colors'].append('limegreen' if bicycle_hit else 'mediumblue')

        # Update traffic light states
        for tl_id in bicycle_trajectory_data[vehicle_id]['traffic_lights']:
            tl_state = get_relevant_tl_state(vehicle_id, tl_id)
            if tl_state is not None:
                bicycle_trajectory_data[vehicle_id]['traffic_lights'][tl_id]['states'].append((elapsed_time, tl_state))

    # Process bicycles that have completed their routes
    all_bicycles = set(bicycle_trajectory_data.keys())
    departed_bicycles = all_bicycles - set(current_bicycles)

    for vehicle_id in departed_bicycles:
        # Calculate the actual tracked distance
        if len(bicycle_trajectory_data[vehicle_id]['positions']) >= 2:
            tracked_distance = bicycle_trajectory_data[vehicle_id]['positions'][-1] - bicycle_trajectory_data[vehicle_id]['positions'][0]
        else:
            tracked_distance = 0

        print(f"Debug: Calculated tracked distance for {vehicle_id}: {tracked_distance:.2f}")

        if tracked_distance >= 150:  # Only save trajectories longer than 150 meters
            print(f"Saving plot for bicycle with ID: {vehicle_id}")

            # Create a new figure for this bicycle
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Ensure all data lists have the same length
            min_length = min(len(bicycle_trajectory_data[vehicle_id]['positions']),
                             len(bicycle_trajectory_data[vehicle_id]['times']),
                             len(bicycle_trajectory_data[vehicle_id]['colors']))
            
            distances = bicycle_trajectory_data[vehicle_id]['positions'][:min_length]
            times = bicycle_trajectory_data[vehicle_id]['times'][:min_length]
            colors = bicycle_trajectory_data[vehicle_id]['colors'][:min_length]

            # Plot trajectory
            ax.scatter(distances, times, c=colors, s=5)

            # Plot traffic light locations
            for tl_id, tl_data in bicycle_trajectory_data[vehicle_id]['traffic_lights'].items():
                tl_distance = tl_data['distance']
                tl_states = tl_data['states']
                
                if not tl_states:
                    continue

                # Sort states by time and remove duplicates
                tl_states.sort(key=lambda x: x[0])
                unique_states = []
                for time, state in tl_states:
                    if not unique_states or state != unique_states[-1][1]:
                        unique_states.append((time, state))

                # Plot colored segments for each unique state
                for i in range(len(unique_states)):
                    start_time, state = unique_states[i]
                    end_time = unique_states[i+1][0] if i+1 < len(unique_states) else max(times)
                    color = get_tl_color(state)
                    ax.plot([tl_distance, tl_distance], [start_time, end_time], 
                            color=color, linewidth=1.0, linestyle='--', dashes=(2, 2))

            # Add legend
            ax.plot([], [], color='mediumblue', label='Bicycle not detected', marker='o', linestyle='None')
            ax.plot([], [], color='limegreen', label='Bicycle detected by FCO/FBO', marker='o', linestyle='None')
            ax.plot([], [], color='red', label='Traffic Light (Red)', linewidth=1.0, linestyle='--', dashes=(2, 2))
            ax.plot([], [], color='yellow', label='Traffic Light (Yellow)', linewidth=1.0, linestyle='--', dashes=(2, 2))
            ax.plot([], [], color='green', label='Traffic Light (Green)', linewidth=1.0, linestyle='--', dashes=(2, 2))
            ax.plot([], [], color='gray', label='Traffic Light (Other)', linewidth=1.0, linestyle='--', dashes=(2, 2))
            
            ax.legend(fontsize=8)

            # Set labels and title
            ax.set_xlabel('Distance (m)')
            ax.set_ylabel('Time (s)')
            ax.set_title(f'Bicycle Trajectory for Vehicle {vehicle_id}')

            # Save plot
            filename = f'out_bicycle_trajectories/bicycle_trajectory_{vehicle_id}.png'
            fig.savefig(filename)
            print(f"Plot saved as {filename}")
            
            # Close the figure to free up memory
            plt.close(fig)
        else:
            print(f"Skipping plot for bicycle with ID: {vehicle_id} (tracked trajectory shorter than 150 meters, actual distance: {tracked_distance:.2f})")

    # Clean up
    for vehicle_id in departed_bicycles:
        del bicycle_trajectory_data[vehicle_id]

def update_bicycle_flow_diagrams(time_step):
    global bicycle_flow_data, flows_processed
    step_length = get_step_length(sumo_config_path)

    if FlowBasedBicycleTrajectoryTracing:
        current_bicycles = [vid for vid in traci.vehicle.getIDList() if traci.vehicle.getTypeID(vid) in ["DEFAULT_BIKETYPE", "bicycle", "floating_bike_observer"]]

        for vehicle_id in current_bicycles:
            flow_id = vehicle_id.split('.')[0]  # Assuming flow_id is the part before the first dot
            
            if flow_id not in bicycle_flow_data:
                bicycle_flow_data[flow_id] = {
                    'vehicles': {},
                    'traffic_lights': defaultdict(lambda: {'distance': None, 'states': []})
                }

            # Update bicycle data
            if vehicle_id not in bicycle_flow_data[flow_id]['vehicles']:
                bicycle_flow_data[flow_id]['vehicles'][vehicle_id] = {
                    'distance': [],
                    'time': [],
                    'colors': []
                }

            # Get bicycle position and convert coordinates
            x, y = traci.vehicle.getPosition(vehicle_id)
            x_32632, y_32632 = convert_simulation_coordinates(x, y)

            # Calculate distance traveled
            if bicycle_flow_data[flow_id]['vehicles'][vehicle_id]['distance']:
                last_distance = bicycle_flow_data[flow_id]['vehicles'][vehicle_id]['distance'][-1]
                total_distance = traci.vehicle.getDistance(vehicle_id)
            else:
                total_distance = 0

            # Check if bicycle is detected by FCO/FBO
            bicycle_hit = False
            bicycle_polygon = create_vehicle_polygon(x_32632, y_32632, 0.65, 1.6, traci.vehicle.getAngle(vehicle_id))
            
            for observer_id in traci.vehicle.getIDList():
                observer_type = traci.vehicle.getTypeID(observer_id)
                if observer_type in ["floating_car_observer", "floating_bike_observer"]:
                    observer_x, observer_y = convert_simulation_coordinates(*traci.vehicle.getPosition(observer_id))
                    rays = generate_rays((observer_x, observer_y), num_rays=numberOfRays, radius=30)
                    
                    for ray in rays:
                        ray_line = LineString(ray)
                        if ray_line.intersects(bicycle_polygon):
                            bicycle_hit = True
                            break
                
                if bicycle_hit:
                    break

            # Update data
            bicycle_flow_data[flow_id]['vehicles'][vehicle_id]['distance'].append(total_distance)
            bicycle_flow_data[flow_id]['vehicles'][vehicle_id]['time'].append(time_step * step_length)
            bicycle_flow_data[flow_id]['vehicles'][vehicle_id]['colors'].append('limegreen' if bicycle_hit else 'mediumblue')

            # Check for nearby traffic lights
            next_tls = traci.vehicle.getNextTLS(vehicle_id)
            for tls in next_tls:
                tl_id, _, distance, state = tls
                relevant_index, distance_to_tl = get_relevant_tl_index(vehicle_id, tl_id)
                
                if relevant_index is not None and 0 <= relevant_index < len(state):
                    relevant_state = state[relevant_index]
                    tl_distance = total_distance + distance_to_tl
                    
                    if tl_id not in bicycle_flow_data[flow_id]['traffic_lights']:
                        bicycle_flow_data[flow_id]['traffic_lights'][tl_id] = {'distance': tl_distance, 'states': []}
                    
                    bicycle_flow_data[flow_id]['traffic_lights'][tl_id]['states'].append((time_step * step_length, relevant_state))
                    
                    print(f"Debug: TL {tl_id} for vehicle {vehicle_id}: full state = {state}, relevant state = {relevant_state}, distance = {tl_distance}")
                else:
                    print(f"No relevant state found for TL {tl_id} and vehicle {vehicle_id}")

        # Generate and save flow diagrams at the end of simulation
        if time_step == get_total_simulation_steps(sumo_config_path) - 1 and not flows_processed:
            for flow_id, flow_data in bicycle_flow_data.items():
                fig, ax = plt.subplots(figsize=(12, 8))
                ax.set_title(f"Bicycle Flow Space-Time Diagram - {flow_id}")
                ax.set_xlabel("Distance [m]")
                ax.set_ylabel("Time [s]")

                # Plot trajectories for all vehicles in this flow that exceed 150 meters
                trajectories_plotted = False
                max_time = 0
                for vehicle_id, vehicle_data in flow_data['vehicles'].items():
                    if vehicle_data['distance'] and vehicle_data['distance'][-1] >= 150:
                        trajectories_plotted = True
                        min_length = min(len(vehicle_data['distance']), len(vehicle_data['time']), len(vehicle_data['colors']))
                        for i in range(1, min_length):
                            ax.plot(vehicle_data['distance'][i-1:i+1], 
                                    vehicle_data['time'][i-1:i+1], 
                                    color=vehicle_data['colors'][i],
                                    linewidth=2)
                        max_time = max(max_time, max(vehicle_data['time'][:min_length]))

                # Only save the plot if at least one trajectory was plotted
                if trajectories_plotted:
                    # Plot traffic light locations
                    for tl_id, tl_data in flow_data['traffic_lights'].items():
                        tl_distance = tl_data['distance']
                        tl_states = tl_data['states']
                        
                        if not tl_states:
                            continue
                        
                        print(f"Traffic light {tl_id} states for flow {flow_id}: {tl_states}")  # Debug print
                        
                        # Plot the states
                        for i in range(len(tl_states) - 1):
                            start_time, start_state = tl_states[i]
                            end_time, _ = tl_states[i+1]
                            color = get_tl_color(start_state)
                            print(f"Traffic light {tl_id} for flow {flow_id} from {start_time} to {end_time}: state = {start_state}, color = {color}")  # Debug print
                            ax.plot([tl_distance, tl_distance], [start_time, end_time], 
                                    color=color, linestyle='--', linewidth=0.5)
                        
                        # Plot the last state to the end of the time range
                        if tl_states:
                            start_time, start_state = tl_states[-1]
                            color = get_tl_color(start_state)
                            ax.plot([tl_distance, tl_distance], [start_time, max_time], 
                                    color=color, linestyle='--', linewidth=0.5)

                    # Add legend
                    ax.plot([], [], color='mediumblue', label='Bicycle not detected')
                    ax.plot([], [], color='limegreen', label='Bicycle detected by FCO/FBO')
                    ax.plot([], [], color='red', linestyle='--', label='Traffic Light (Red)', linewidth=0.5)
                    ax.plot([], [], color='yellow', linestyle='--', label='Traffic Light (Yellow)', linewidth=0.5)
                    ax.plot([], [], color='green', linestyle='--', label='Traffic Light (Green)', linewidth=0.5)
                    ax.plot([], [], color='gray', linestyle='--', label='Traffic Light (Other)', linewidth=0.5)
                    
                    ax.legend(fontsize=8)

                    filename = f'out_bicycle_trajectories/bicycle_flow_diagram_{flow_id}.png'
                    fig.savefig(filename)
                    print(f"Bicycle flow diagram for {flow_id} saved as {filename}")
                else:
                    print(f"No trajectories exceeding 150 meters for flow {flow_id}. Skipping plot generation.")
                
                plt.close(fig)

            # Set the flag to indicate that flows have been processed
            flows_processed = True

            # Clear the bicycle_flow_data after generating all plots
            bicycle_flow_data.clear()

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
        run_animation(total_steps)
    else:
        for frame in range(total_steps):
            update_with_ray_tracing(frame)
    print('Ray tracing completed.')
    if IndividualBicycleTrajectoryTracing:
        print('Individual Bicycle Trajectory Tracing completed and files saved in out_bicycle_trajectories')
    if FlowBasedBicycleTrajectoryTracing:
        print('Flow-Based Bicycle Trajectory Tracing completed and files saved in out_bicycle_trajectories')
    if saveAnimation:
        print(f'Ray tracing animation saved in out_raytracing as ray_tracing_animation_FCO{str(FCO_share*100)}%_FBO{str(FBO_share*100)}%.mp4.')
    summary_logging()
    print('Logging completed and saved in out_logging.')
    traci.close()
    print('TraCI closed.')
    create_visibility_heatmap(x_coords, y_coords, visibility_counts)
    if relativeVisibility:
        print(f'Relative Visibility Heat Map Generation completed - file saved in out_raytracing as relative_visibility_heatmap_FCO{str(FCO_share*100)}%_FBO{str(FBO_share*100)}%.png.')