import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, to_rgba
import geopandas as gpd
import osmnx as ox
import xml.etree.ElementTree as ET
import pyproj
import os
from matplotlib.patches import Patch

# ---------------------
# CONFIGURATION
# ---------------------

# Warm Up Settings (make sure, they are the same as for the Ray Tracing):

delay = 90 #warm-up time in seconds (during this time in the beginning of the simulation, no ray tracing is performed)

# Bounding Box Settings:

north, south, east, west = 48.1505, 48.14905, 11.5720, 11.5669
bbox = (north, south, east, west)

# Path Settings:

base_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(base_dir)
heatmap_csv_path = os.path.join(parent_dir, 'out_visibility/visibility_counts/visibility_counts_FCO50.0%_FBO0%.csv')
sumo_config_path = os.path.join(parent_dir, 'SUMO_example', 'SUMO_example.sumocfg')

# Extract just the filename from the full path
filename = os.path.basename(heatmap_csv_path)
# Split the filename on 'visibility_counts_' to get just the parameter part
trimmed_path = filename.split('visibility_counts_')[1]
# Remove the .csv extension
trimmed_path = os.path.splitext(trimmed_path)[0]
logging_csv_path = os.path.join(parent_dir, 'out_visibility', 'LoV_logging', 'log_LoV_' + trimmed_path + '.csv')
print(trimmed_path)

# ---------------------

def load_heatmap_data_from_csv(filename):
    x_coords = []
    y_coords = []
    visibility_counts = []

    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # Skip header
        for row in csvreader:
            x_coords.append(float(row[0]))
            y_coords.append(float(row[1]))
            visibility_counts.append(float(row[2]))

    return np.array(x_coords), np.array(y_coords), np.array(visibility_counts)

def get_simulation_params(sumo_config_file):
    global delay
    tree = ET.parse(sumo_config_file)
    root = tree.getroot()
    begin = 0
    end = 3600
    step_length = 0.1
    for time in root.findall('time'):
        begin = float(time.find('begin').get('value', begin))
        end = float(time.find('end').get('value', end))
        step_length = float(time.find('step-length').get('value', step_length))
    total_steps = int((end - begin - delay) / step_length)
    step_size = step_length
    return total_steps, step_size

def load_geospatial_data(bbox):
    buildings = ox.features_from_bbox(bbox=bbox, tags={'building': True})
    parks = ox.features_from_bbox(bbox=bbox, tags={'leisure': 'park'})
    
    # Project to target coordinate system
    proj_to = pyproj.CRS("EPSG:32632")
    buildings_proj = buildings.to_crs(proj_to)
    parks_proj = parks.to_crs(proj_to)
    
    return buildings_proj, parks_proj

def hex_to_rgba(hex_color, alpha=0.5):
    """Convert hex color to RGBA with the specified transparency (alpha)."""
    rgba_color = to_rgba(hex_color, alpha)
    return rgba_color

def create_lov_heatmap(x_coords, y_coords, visibility_counts, total_steps, step_size, buildings_proj, parks_proj, x_min, y_min, debug_csv_path):
    
    # Collect debug information
    logging_info = []
    
    logging_info.append(['Max. visibility count', np.max(visibility_counts)])
    logging_info.append(['Total simulation steps', total_steps])
    
    lov_data = visibility_counts / total_steps
    max_lov = 1 / step_size

    logging_info.append(['Step Size', step_size])
    logging_info.append(['LoV scale', f'0 - {max_lov}'])

    # Calculate the maximum and mean values of lov_data
    max_lov_data = np.max(lov_data)
    mean_lov_data = np.mean(lov_data)
    logging_info.append(['Max. LoV value', max_lov_data])
    logging_info.append(['Mean LoV value', mean_lov_data])

    # Write debug information to CSV
    with open(logging_csv_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['Description', 'Value'])
        csvwriter.writerows(logging_info)

    # Translate coordinates
    x_coords_translated = x_coords - x_min
    y_coords_translated = y_coords - y_min

    # Check for any invalid (NaN or inf) values in coordinates
    if np.any(np.isnan(x_coords_translated)) or np.any(np.isnan(y_coords_translated)):
        raise ValueError("Translated coordinates contain NaN values.")
    if np.any(np.isinf(x_coords_translated)) or np.any(np.isinf(y_coords_translated)):
        raise ValueError("Translated coordinates contain infinity values.")

    # Define color map with hex codes and 50% transparency
    alpha = 0.5
    colors = [
        hex_to_rgba("#B22222", alpha),  # Firebrick
        hex_to_rgba("#FF4500", alpha),  # Orange-Red
        hex_to_rgba("#FFA500", alpha),  # Orange
        hex_to_rgba("#FFFF00", alpha),  # Yellow
        hex_to_rgba("#ADFF2F", alpha)   # Green-Yellow
    ]
    cmap = ListedColormap(colors)
    # changeFactor = 0.2
    # bounds = [0, max_lov * 0.2*changeFactor, max_lov * 0.4*changeFactor, max_lov * 0.6*changeFactor, max_lov * 0.8*changeFactor, max_lov*changeFactor]  # uniform distribution of LoV classes (can be customized)
    bounds = [0, max_lov * 0.2, max_lov * 0.4, max_lov * 0.6, max_lov * 0.8, max_lov]  # uniform distribution of LoV classes (can be customized)
    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(12, 8), facecolor='lightgray')
    ax.set_facecolor('lightgray')
    
    # Translate and plot geospatial data
    buildings_proj_translated = buildings_proj.translate(-x_min, -y_min)
    parks_proj_translated = parks_proj.translate(-x_min, -y_min)

    buildings_proj_translated.plot(ax=ax, facecolor='gray', edgecolor='black', linewidth=0.5, alpha=0.7)
    parks_proj_translated.plot(ax=ax, facecolor='green', edgecolor='black', linewidth=0.5, alpha=0.7)

    for i, x in enumerate(x_coords_translated):
        y = y_coords_translated[i]
        value = lov_data[i]
        color = cmap(norm([value])[0])
        ax.add_patch(plt.Rectangle((x, y), 1.0, 1.0, facecolor=color, edgecolor='none'))  # transparency (alpha) already set above the customized colormap

    # Create legend
    legend_patches = [
        Patch(color=hex_to_rgba("#B22222", alpha), label='LoV E'), # Firebrick
        Patch(color=hex_to_rgba("#FF4500", alpha), label='LoV D'), # Orange-Red
        Patch(color=hex_to_rgba("#FFA500", alpha), label='LoV C'), # Orange
        Patch(color=hex_to_rgba("#FFFF00", alpha), label='LoV B'), # Yellow
        Patch(color=hex_to_rgba("#ADFF2F", alpha), label='LoV A') # Green-Yellow
    ]
    legend = ax.legend(handles=legend_patches, loc='upper right', title='Level of Visibility (LoV)')
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(1.0)
    legend.get_frame().set_edgecolor('black')

    ax.set_title('Level of Visibility (LoV) Heatmap')
    ax.set_xlabel('Distance [m]')
    ax.set_ylabel('Distance [m]')

    # Ensure the entire bounding box is visible
    translated_x_max = x_max - x_min
    translated_y_max = y_max - y_min
    ax.set_xlim(0, translated_x_max)
    ax.set_ylim(0, translated_y_max)
    
    plt.savefig(os.path.join(parent_dir, 'out_visibility', 'LoV_heatmap_' + trimmed_path + '.png'))

if __name__ == "__main__":
    # Load heatmap data from CSV
    x_coords, y_coords, visibility_counts = load_heatmap_data_from_csv(heatmap_csv_path)
    print('Visibility counts loaded.')

    # Get total_steps and step_size from SUMO config file
    total_steps, step_size = get_simulation_params(sumo_config_path)
    
    # Load geospatial data
    buildings_proj, parks_proj = load_geospatial_data(bbox)
    print('Geospatial data loaded.')
    
    # Determine the minimum x and y coordinates for translation
    x_min, y_min, x_max, y_max = buildings_proj.total_bounds

    create_lov_heatmap(x_coords, y_coords, visibility_counts, total_steps, step_size, buildings_proj, parks_proj, x_min, y_min, logging_csv_path)
    print('LoV Heat Map created.')
