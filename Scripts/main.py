import os
import osmnx as ox
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

# Setup logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------------------
# CONFIGURATION
# ---------------------

# General Settings:

useLiveVisualization = True # Live Visualization of Ray Tracing
visualizeRays = True # Visualize rays additionaly to the visibility polygon
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

FCO_share = 0.2
FBO_share = 0
numberOfRays = 360

# Warm Up Settings:

delay = 90 #warm-up time in seconds (during this time in the beginning of the simulation, no ray tracing is performed)

# Grid Map Settings:

grid_size =  0.5 # Grid Size for Heat Map Visualization (the smaller the grid size, the higher the resolution)

# ---------------------

# Loading of Geospatial Data (for Heatmap Data)

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

# Initialization of Grid Parameters:

x_min, y_min, x_max, y_max = buildings_proj.total_bounds
x_coords = np.arange(x_min, x_max, grid_size)
y_coords = np.arange(y_min, y_max, grid_size)
grid_points = [(x, y) for x in x_coords for y in y_coords]
grid_cells = [box(x, y, x + grid_size, y + grid_size) for x, y in grid_points]

# Initialization of Visibility Counts (for Heat Map Visualization)

visibility_counts = {cell: 0 for cell in grid_cells}

# Logging Settings:

# Initialize sets to track unique vehicles
unique_vehicles = set()
vehicle_type_set = set()
# Initialize a DataFrame to log information at each time step
log_columns = ['time_step']
simulation_log = pd.DataFrame(columns=log_columns)

# ---------------------

def convert_simulation_coordinates(x, y):
    lon, lat = traci.simulation.convertGeo(x, y)
    x_32632, y_32632 = project(lon, lat)
    return x_32632, y_32632

def load_sumo_simulation():
    sumoCmd = ["sumo", "-c", sumo_config_path]
    traci.start(sumoCmd)

def load_geospatial_data():
    north, south, east, west = 48.1505, 48.14905, 11.5720, 11.5669
    bbox = (north, south, east, west)
    gdf1 = gpd.read_file(geojson_path)
    G = ox.graph_from_bbox(bbox=bbox, network_type='all')
    buildings = ox.features_from_bbox(bbox=bbox, tags={'building': True})
    parks = ox.features_from_bbox(bbox=bbox, tags={'leisure': 'park'})
    return gdf1, G, buildings, parks

def project_geospatial_data(gdf1, G, buildings, parks):
    gdf1_proj = gdf1.to_crs("EPSG:32632")
    G_proj = ox.project_graph(G, to_crs="EPSG:32632")
    buildings_proj = buildings.to_crs("EPSG:32632")
    parks_proj = parks.to_crs("EPSG:32632")
    return gdf1_proj, G_proj, buildings_proj, parks_proj

def setup_plot():
    ax.set_title('Ray Tracing Visualization')
    legend_handles = [
        Rectangle((0, 0), 1, 1, facecolor='gray', edgecolor='black', linewidth=0.5, alpha=0.7, label='Buildings'),
        Rectangle((0, 0), 1, 1, facecolor='green', edgecolor='black', linewidth=0.5, alpha=0.7, label='Parks')
    ]
    ax.legend(handles=legend_handles)

def plot_geospatial_data(gdf1_proj, G_proj, buildings_proj, parks_proj):
    ox.plot_graph(G_proj, ax=ax, bgcolor='none', edge_color='none', node_size=0, show=False, close=False)
    gdf1_proj.plot(ax=ax, color='lightgray', alpha=0.5, edgecolor='lightgray')
    buildings_proj.plot(ax=ax, facecolor='gray', edgecolor='black', linewidth=0.5, alpha=0.7)
    parks_proj.plot(ax=ax, facecolor='green', edgecolor='black', linewidth=0.5, alpha=0.7)

def initialize_grid(buildings_proj, grid_size=1.0):
    x_min, y_min, x_max, y_max = buildings_proj.total_bounds
    x_coords = np.arange(x_min, x_max, grid_size)
    y_coords = np.arange(y_min, y_max, grid_size)
    grid_points = [(x, y) for x in x_coords for y in y_coords]
    grid_cells = [box(x, y, x + grid_size, y + grid_size) for x, y in grid_points]
    visibility_counts = {cell: 0 for cell in grid_cells}
    return x_coords, y_coords, grid_cells, visibility_counts

def get_total_simulation_steps(sumo_config_file):
    global step_length
    tree = ET.parse(sumo_config_file)
    root = tree.getroot()
    begin = 0
    end = 3600
    step_length = 0.1
    for time in root.findall('time'):
        begin = float(time.find('begin').get('value', begin))
        end = float(time.find('end').get('value', end))
        step_length = float(time.find('step-length').get('value', step_length))
    total_steps = int((end - begin) / step_length)
    return total_steps

def get_step_length(sumo_config_file):
    tree = ET.parse(sumo_config_file)
    root = tree.getroot()
    step_length = -1
    for time in root.findall('time'):
        step_length = float(time.find('step-length').get('value', step_length))
    return step_length

def vehicle_attributes(vehicle_type):
    if vehicle_type == "floating_car_observer":
        return Rectangle, 'red', (1.8, 5)
    if vehicle_type == "floating_bike_observer":
        return Rectangle, 'red', (0.65, 1.6)
    if vehicle_type == "veh_passenger":
        return Rectangle, 'gray', (1.8, 5)
    if vehicle_type == "parked_vehicle":
        return Rectangle, 'gray', (1.8, 5)
    if vehicle_type == "DEFAULT_VEHTYPE":
        return Rectangle, 'gray', (1.8, 5)
    elif vehicle_type == "pt_bus":
        return Rectangle, 'gray', (2.5, 12)
    elif vehicle_type == "bus_bus":
        return Rectangle, 'gray', (2.5, 12)
    elif vehicle_type == "pt_tram":
        return Rectangle, 'gray', (2.5, 12)
    elif vehicle_type == "truck_truck":
        return Rectangle, 'gray', (2.4, 7.1)
    elif vehicle_type == "bike_bicycle":
        return Rectangle, 'blue', (0.65, 1.6)
    elif vehicle_type == "DEFAULT_BIKETYPE":
        return Rectangle, 'blue', (0.65, 1.6)
    elif vehicle_type == "ped_pedestrian":
        return Rectangle, 'green', (0.5, 0.5)
    else:
        return Rectangle, 'gray', (1.8, 5)

def generate_rays(center, num_rays=360, radius=30):
    angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
    rays = [(center, (center[0] + np.cos(angle) * radius, center[1] + np.sin(angle) * radius)) for angle in angles]
    return rays

def detect_intersections(ray, objects):
    logging.info(f"Thread started for ray: {ray}")
    closest_intersection = None
    min_distance = float('inf')
    ray_line = LineString(ray)
    for obj in objects:
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
    logging.info(f"Thread completed for ray: {ray}")
    return closest_intersection

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

def create_vehicle_polygon(x, y, width, length, angle):
    adjusted_angle = (-angle) % 360
    rect = Polygon([(-width / 2, -length / 2), (-width / 2, length / 2), (width / 2, length / 2), (width / 2, -length / 2)])
    rotated_rect = rotate(rect, adjusted_angle, use_radians=False, origin=(0, 0))
    translated_rect = translate(rotated_rect, xoff=x, yoff=y)
    return translated_rect

def update_with_ray_tracing(frame):
    global vehicle_patches, ray_lines, visibility_polygons, FCO_share, FBO_share, visibility_counts, numberOfRays, useRTREEmethod, visualizeRays, useManualFrameForwarding, delay
    detected_color = (1.0, 0.27, 0, 0.5)
    undetected_color = (0.53, 0.81, 0.98, 0.5)

    traci.simulationStep()

    if useManualFrameForwarding:
        input("Press Enter to continue...")

    print(f"Frame: {frame + 1}")

    FCO_type = "floating_car_observer"
    FBO_type = "floating_bike_observer"
    for vehicle_id in traci.simulation.getDepartedIDList():
        if traci.vehicle.getTypeID(vehicle_id) == "DEFAULT_VEHTYPE" and np.random.uniform() < FCO_share:
            traci.vehicle.setType(vehicle_id, FCO_type)
        if traci.vehicle.getTypeID(vehicle_id) == "DEFAULT_BIKETYPE" and np.random.uniform() < FBO_share:
            traci.vehicle.setType(vehicle_id, FBO_type)

    stepLength = get_step_length(sumo_config_path)


    if frame > delay / stepLength:
        new_vehicle_patches = []
        new_ray_lines = []
        updated_cells = set()

        static_objects = [building.geometry for building in buildings_proj.itertuples()]
        for vehicle_id in traci.vehicle.getIDList():
            vehicle_type = traci.vehicle.getTypeID(vehicle_id)
            if vehicle_type == "parked_vehicle":
                x, y = traci.vehicle.getPosition(vehicle_id)
                x_32632, y_32632 = convert_simulation_coordinates(x, y)
                width, length = vehicle_attributes(vehicle_type)[2]
                angle = traci.vehicle.getAngle(vehicle_id)
                parked_vehicle_geom = create_vehicle_polygon(x_32632, y_32632, width, length, angle)
                static_objects.append(parked_vehicle_geom)

        for line in ray_lines:
            if useLiveVisualization:
                if visualizeRays:
                    line.remove()
                else:
                    pass
        ray_lines.clear()

        if useLiveVisualization:
            for polygon in visibility_polygons:
                polygon.remove()
        visibility_polygons.clear()

        for vehicle_id in traci.vehicle.getIDList():
            vehicle_type = traci.vehicle.getTypeID(vehicle_id)
            Shape, edgecolor, (width, length) = vehicle_attributes(vehicle_type)
            dynamic_objects_geom = [
                create_vehicle_polygon(
                    *convert_simulation_coordinates(*traci.vehicle.getPosition(vid)),
                    *vehicle_attributes(traci.vehicle.getTypeID(vid))[2],
                    traci.vehicle.getAngle(vid)
                ) for vid in traci.vehicle.getIDList() if vid != vehicle_id
            ]

            x, y = traci.vehicle.getPosition(vehicle_id)
            x_32632, y_32632 = convert_simulation_coordinates(x, y)
            angle = traci.vehicle.getAngle(vehicle_id)
            adjusted_angle = (-angle) % 360
            
            lower_left_corner = (x_32632 - width / 2, y_32632 - length / 2)
            patch = Rectangle(lower_left_corner, width, length, edgecolor=edgecolor, fill=None)

            t = transforms.Affine2D().rotate_deg_around(x_32632, y_32632, adjusted_angle) + ax.transData
            patch.set_transform(t)
            new_vehicle_patches.append(patch)

            if vehicle_type == "floating_car_observer" or vehicle_type == "floating_bike_observer":
                center = (x_32632, y_32632)
                rays = generate_rays(center, num_rays=numberOfRays, radius=30)
                all_objects = static_objects + dynamic_objects_geom
                ray_endpoints = []

                with ThreadPoolExecutor() as executor:
                    if useRTREEmethod:
                        static_index = index.Index()
                        for i, obj in enumerate(static_objects):
                            static_index.insert(i, obj.bounds)
                        futures = {executor.submit(detect_intersections_rtree, ray, all_objects, static_index): ray for ray in rays}
                    else:
                        futures = {executor.submit(detect_intersections, ray, all_objects): ray for ray in rays}

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
                        if useLiveVisualization:
                            if visualizeRays:
                                ax.add_line(ray_line)
                        new_ray_lines.append(ray_line)

                ray_endpoints.sort(key=lambda point: np.arctan2(point[1] - center[1], point[0] - center[0]))

                if len(ray_endpoints) > 2:
                    visibility_polygon = MatPolygon(ray_endpoints, color='green', alpha=0.5, fill=None)
                    if useLiveVisualization:
                        ax.add_patch(visibility_polygon)
                    visibility_polygons.append(visibility_polygon)

                visibility_polygon_shape = Polygon(ray_endpoints)
                for cell in visibility_counts.keys():
                    if visibility_polygon_shape.contains(cell):
                        if cell not in updated_cells:  # Check if the cell is already in updated_cells
                            visibility_counts[cell] += 1
                            updated_cells.add(cell)
        
        if useLiveVisualization:
            for patch in vehicle_patches:
                patch.remove()
        
        vehicle_patches = new_vehicle_patches
        ray_lines = new_ray_lines

        if useLiveVisualization:
            for patch in vehicle_patches:
                ax.add_patch(patch)
    
    # Call log_simulation_step function to log data at each step
    detailled_logging(frame)

def generate_animation(total_steps):
    if useLiveVisualization:
        ani = FuncAnimation(fig, update_with_ray_tracing, frames=range(1, total_steps), interval=33, repeat=False)
        plt.show()
    else:
        max_frames = total_steps
        for frames in range(max_frames):
            update_with_ray_tracing(frames)

    if useLiveVisualization and saveAnimation:
        writer = FFMpegWriter(fps=1, metadata=dict(artist='Me'), bitrate=1800)
        ani.save('out_raytracing/ray_tracing_animation_' + str(FCO_share*100) + '%.mp4', writer=writer)

def create_visibility_heatmap(x_coords, y_coords, visibility_counts):
    # Generate heatmap data (obtain visbility counts)
    heatmap_data = np.zeros((len(x_coords), len(y_coords)))
    for cell, count in visibility_counts.items():
        x_idx = np.searchsorted(x_coords, cell.bounds[0])
        y_idx = np.searchsorted(y_coords, cell.bounds[1])
        if x_idx < len(x_coords) and y_idx < len(y_coords):
            heatmap_data[x_idx, y_idx] = count
    heatmap_data[heatmap_data == 0] = np.nan

    # Save heatmap data to CSV before normalizing the visibility counts!
    with open(f'out_visibility/visibility_counts_FCO{str(FCO_share*100)}%_FBO{str(FBO_share*100)}%.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['x_coord', 'y_coord', 'visibility_count'])
        for i, x in enumerate(x_coords):
            for j, y in enumerate(y_coords):
                if not np.isnan(heatmap_data[i, j]):
                    csvwriter.writerow([i, j, heatmap_data[i, j]])

    # Normalizing visibility counts by the max. value --> resulting in values [0, 1]
    heatmap_data = heatmap_data / np.nanmax(heatmap_data)

    # Plotting the heatmap
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='lightgray')
    ax.set_facecolor('lightgray')
    buildings_proj.plot(ax=ax, facecolor='gray', edgecolor='black', linewidth=0.5, alpha=0.7)
    parks_proj.plot(ax=ax, facecolor='green', edgecolor='black', linewidth=0.5, alpha=0.7)

    cax = ax.imshow(heatmap_data.T, origin='lower', cmap='hot', extent=[0, len(x_coords), 0, len(y_coords)], alpha=0.6)
    ax.set_title('Visibility Heatmap')
    fig.colorbar(cax, ax=ax, label='Visibility (normalized)')
    plt.savefig(f'out_raytracing/ray_tracing_heatmap_FCO{str(FCO_share*100)}%_FBO{str(FBO_share*100)}%.png')
    #plt.show()

def detailled_logging(time_step):
    global unique_vehicles, vehicle_type_set, simulation_log

    new_vehicle_counts = {}
    present_vehicle_counts = {}
    
    for vehicle_id in traci.simulation.getDepartedIDList():
        if vehicle_id not in unique_vehicles:
            unique_vehicles.add(vehicle_id)
            vehicle_type = traci.vehicle.getTypeID(vehicle_id)
            vehicle_type_set.add(vehicle_type)

            if vehicle_type not in new_vehicle_counts:
                new_vehicle_counts[vehicle_type] = 0
            new_vehicle_counts[vehicle_type] += 1

    for vehicle_id in traci.vehicle.getIDList():
        vehicle_type = traci.vehicle.getTypeID(vehicle_id)
        if vehicle_type not in present_vehicle_counts:
            present_vehicle_counts[vehicle_type] = 0
        present_vehicle_counts[vehicle_type] += 1

    log_entry = {'time_step': time_step}
    for vehicle_type in vehicle_type_set:
        log_entry[f'new_{vehicle_type}_count'] = new_vehicle_counts.get(vehicle_type, 0)
        log_entry[f'present_{vehicle_type}_count'] = present_vehicle_counts.get(vehicle_type, 0)

    log_entry_df = pd.DataFrame([log_entry])
    simulation_log = pd.concat([simulation_log, log_entry_df], ignore_index=True)

def summary_logging():
    global simulation_log

    # Compute totals
    total_vehicle_counts = {vehicle_type: 0 for vehicle_type in vehicle_type_set}
    
    # Define relevant vehicle types for calculation of penetration rates
    relevant_car_types = {"floating_car_observer", "DEFAULT_VEHTYPE", "pt_bus", "passenger"} # for FCO penetration rate
    relevant_bike_types = {"floating_bike_observer", "DEFAULT_BIKETYPE"} # for FBO penetration rate
    
    total_relevant_cars = 0
    total_relevant_bikes = 0
    total_floating_car_observers = 0
    total_floating_bike_observers = 0

    for vehicle_type in vehicle_type_set:
        total_vehicle_counts[vehicle_type] = simulation_log[f'new_{vehicle_type}_count'].sum()
        if vehicle_type in relevant_car_types:
            total_relevant_cars += total_vehicle_counts[vehicle_type]
            if vehicle_type == "floating_car_observer":
                total_floating_car_observers += total_vehicle_counts[vehicle_type]
        if vehicle_type in relevant_bike_types:
            total_relevant_bikes += total_vehicle_counts[vehicle_type]
            if vehicle_type == "floating_bike_observer":
                total_floating_bike_observers += total_vehicle_counts[vehicle_type]

    fco_penetration_rate = total_floating_car_observers / total_relevant_cars if total_relevant_cars > 0 else 0
    fbo_penetration_rate = total_floating_bike_observers / total_relevant_bikes if total_relevant_bikes > 0 else 0

    # Save detailed log
    simulation_log.to_csv(f'out_logging/detailed_log_FCO{str(FCO_share*100)}%_FBO{str(FBO_share*100)}%.csv', index=False)

    # Save summary log
    with open(f'out_logging/summary_log_FCO{str(FCO_share*100)}%_FBO{str(FBO_share*100)}%.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        # Header for Simulation Input
        writer.writerow(["Simulation Input:"])
        writer.writerow([])
        writer.writerow(["Description", "Value"])
        # Simulation Inputs
        writer.writerow(["FCO share (input)", f"{(FCO_share):.2%}"])
        writer.writerow(["FBO share (input)", f"{(FBO_share):.2%}"])
        writer.writerow(["Number of rays (for Ray Tracing)", numberOfRays])
        writer.writerow(["Grid size (for Heat Map Visualizations)", grid_size])
        writer.writerow([])
        writer.writerow([])
        # Header for Simulation Output
        writer.writerow(["Simulation Output:"])
        writer.writerow([])
        writer.writerow(["Description", "Value"])
        # Simulation Outputs
        for vehicle_type in vehicle_type_set:
            writer.writerow([f"Total {vehicle_type} vehicles", total_vehicle_counts[vehicle_type]])
        writer.writerow(["Total relevant cars", total_relevant_cars])
        writer.writerow(["Total relevant bikes", total_relevant_bikes])
        writer.writerow(["FCO penetration rate", f"{fco_penetration_rate:.2%}"])
        writer.writerow(["FBO penetration rate", f"{fbo_penetration_rate:.2%}"])

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
    generate_animation(total_steps) # Ray Tracing is actually performed in this function
    print('Ray tracing completed.')
    if saveAnimation:
        print(f'Ray tracing animation saved in out_raytracing as ray_tracing_animation_FCO{str(FCO_share*100)}%_FBO{str(FBO_share*100)}%.mp4.')
    summary_logging()
    print('Logging completed and saved in out_logging.')
    traci.close()
    print('TraCI closed.')
    create_visibility_heatmap(x_coords, y_coords, visibility_counts)
    print(f'Visibility Heat Map Generation completed - file saved in out_raytracing as ray_tracing_heatmap_FCO{str(FCO_share*100)}%_FBO{str(FBO_share*100)}%.png.')