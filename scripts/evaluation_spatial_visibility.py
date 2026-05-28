#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# type: ignore
# See copilot-instructions.md for agent guidance
"""
Spatial Visibility Analysis

This unified script generates both Relative Visibility and Level of Visibility (LoV) heatmaps 
from existing visibility count CSV files without needing to run the full ray tracing simulation.
"""

# =============================================================================
# USER CONFIGURATION
# =============================================================================

# 1. PROJECT PATH - Set the path to your scenario output folder
SCENARIO_OUTPUT_PATH = r"C:\FTO-Sim\outputs\TR-A_final_LoV_averaged\TR-A_high-demand_no-parking_FCO50%_FBO0%"  # Path to scenario output folder (set to None to use manual configuration)

# 2. ANALYSIS SELECTION - Choose which metrics to generate
RELATIVE_VISIBILITY = True   # Generate relative visibility heatmaps
DISCRETE_LOV = True           # Generate discrete Level of Visibility (LoV) heatmaps (binary frame-based)
CONTINUOUS_LOV = False         # Generate continuous Level of Visibility (LoV) heatmaps (sensor accuracy weighted)

INFRASTRUCTURE_CLASSIFICATION = False                                                                    # Classify infrastructure types from SUMO network (vehicles_only, vru, mixed, none)
INFRASTRUCTURE_CLASSIFICATION_MAP = False                                                                # Generate visualization map of infrastructure classification
SUMO_NETWORK_FILE = r"simulation_examples\Spatial-Visibility_Ilic-TRB-2025\Ilic-2025_network.net.xml"    # Path to SUMO network file (.net.xml) - REQUIRED if INFRASTRUCTURE_CLASSIFICATION is enabled

# 3. GRID AND DISPLAY SETTINGS
VISUALIZATION_GRID_SIZE = 1.0  # Grid resolution for heatmap visualization in meters (can be different than grid size of visibility counts)
COLORMAP = 'hot'              # Color scheme for relative visibility - perceptually uniform and colorblind-friendly
ALPHA = 0.6                   # Heatmap transparency (0.0-1.0)

# 4. VISUALIZATION OPTIONS - Configure what to include in the maps
INCLUDE_ROADS = True         # Display road network from GeoJSON
INCLUDE_BUILDINGS = True     # Display buildings from OpenStreetMap
INCLUDE_PARKS = False        # Display parks from OpenStreetMap
INCLUDE_TREES = True         # Display trees from OpenStreetMap
INCLUDE_BARRIERS = False     # Display barriers from OpenStreetMap
INCLUDE_PT_SHELTERS = True   # Display public transport shelters

# If set, overrides threshold-based focus area
FOCUS_AREA_BBOX_OVERRIDE = [690131, 5333677, 690231, 5333777] # 100m x 100m focus area

# Focus area overlay (optional)
ENABLE_FOCUS_AREA_OVERLAY = False
FOCUS_AREA_THRESHOLD = 25000  # visibility count threshold
FOCUS_AREA_BUFFER = 10        # meters
FOCUS_AREA_ROUND_TO_10M = True

# Generate separate LoV heatmap for focus area only (cropped view)
PLOT_FOCUS_AREA_HEATMAP = False

# =============================================================================
# OPTIONAL MANUAL CONFIGURATION (only needed if SCENARIO_OUTPUT_PATH = None)
# =============================================================================

# REQUIRED PARAMETERS (for manual configuration)
FILE_TAG = "TRB_new_figures_LD90%"       # File tag used in the simulation
FCO_SHARE = 0                 # FCO penetration rate (0-100)
FBO_SHARE = 0                   # FBO penetration rate (0-100)
BOUNDING_BOX = [48.146200, 48.144400, 11.580650, 11.577150]  # Geographic bounds [north, south, east, west]

# SIMULATION PARAMETERS (for LoV calculation - will be auto-detected if available)
TOTAL_SIMULATION_STEPS = 2700   # Total simulation steps (fallback if not found in logs)
STEP_LENGTH = 0.1               # Simulation step length in seconds (fallback if not found in logs)

# OPTIONAL PATHS (fallbacks for manual configuration)
GEOJSON_PATH = "simulation_examples/Spatial-Visibility_Ilic-TRB-2025/Ilic-2025.geojson"  # Path to GeoJSON file (set to None if not available)
OUTPUT_DIR = "outputs/ex_singleFCO_FCO100%_FBO0%/out_visibility"  # Output directory for heatmaps

# =============================================================================
# END OF USER CONFIGURATION
# =============================================================================

# Suppress GDAL/OGR warnings from stderr (must be set before importing geopandas/fiona)
import os
os.environ['CPL_LOG'] = 'OFF'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import geopandas as gpd
import osmnx as ox
import json
import argparse
import csv
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from pyproj import Transformer
from tqdm import tqdm
from shapely.geometry import LineString, box
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch, Rectangle
import math

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*unsupported OGR type.*')

def auto_detect_parameters_from_scenario(scenario_path):
    """
    Auto-detect simulation parameters from scenario output folder structure and log files.
    
    Args:
        scenario_path: Path to the scenario output folder
        
    Returns:
        dict: Dictionary with auto-detected parameters
    """
    scenario_path = Path(scenario_path)
    
    if not scenario_path.exists():
        raise FileNotFoundError(f"Scenario path does not exist: {scenario_path}")
    
    # Extract scenario name from folder path
    scenario_name = scenario_path.name
    
    # Parse FCO and FBO shares from folder name
    import re
    match = re.search(r'FCO(\d+)%_FBO(\d+)%', scenario_name)
    if not match:
        raise ValueError(f"Cannot parse FCO/FBO shares from folder name: {scenario_name}")
    
    fco_share = int(match.group(1))
    fbo_share = int(match.group(2))
    
    # Extract file tag (everything before _FCO)
    file_tag = scenario_name.split('_FCO')[0]
    
    # Initialize parameters with defaults
    bbox = None
    total_steps = None
    step_length = None
    grid_size = None
    geojson_path = None
    sumo_config_path = None
    
    # 1. PRIMARY SOURCE: Extract from summary log file (most reliable)
    log_path = scenario_path / 'out_logging'
    if log_path.exists():
        # Look for summary_log_*.csv file
        summary_logs = list(log_path.glob('summary_log_*.csv'))
        if summary_logs:
            summary_log = summary_logs[0]
            
            try:
                with open(summary_log, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # Extract bounding box
                    bbox_pattern = r'Bounding box \(north\),([0-9.]+)\s*Bounding box \(south\),([0-9.]+)\s*Bounding box \(east\),([0-9.]+)\s*Bounding box \(west\),([0-9.]+)'
                    bbox_match = re.search(bbox_pattern, content)
                    if bbox_match:
                        bbox = [float(bbox_match.group(1)), float(bbox_match.group(2)),
                               float(bbox_match.group(3)), float(bbox_match.group(4))]
                    
                    # Extract simulation steps
                    steps_match = re.search(r'Total simulation steps,(\d+)', content)
                    if steps_match:
                        total_steps = int(steps_match.group(1))
                    
                    # Extract step length
                    step_match = re.search(r'Step length,([0-9.]+)', content)
                    if step_match:
                        step_length = float(step_match.group(1))
                    
                    # Extract grid size
                    grid_match = re.search(r'Grid size \(Heat Map\),([0-9.]+)', content)
                    if grid_match:
                        grid_size = float(grid_match.group(1))
                    
                    # Extract SUMO config path
                    sumo_match = re.search(r'SUMO configuration file,(.+?)(?:\n|$)', content)
                    if sumo_match:
                        sumo_config_path = sumo_match.group(1).strip()
                    
                    # Extract GeoJSON path
                    geojson_match = re.search(r'GeoJSON path,(.+?)(?:\n|$)', content)
                    if geojson_match:
                        geojson_path_str = geojson_match.group(1).strip()
                        if geojson_path_str and geojson_path_str != 'None':
                            # Handle relative paths from parent_dir
                            if not os.path.isabs(geojson_path_str):
                                # Reconstruct absolute path from parent_dir
                                parent_dir = Path(__file__).parent.parent
                                geojson_path = parent_dir / geojson_path_str
                            else:
                                geojson_path = Path(geojson_path_str)
                            
                            if not geojson_path.exists():
                                geojson_path = None
                    
                    # If GeoJSON not found but SUMO config is, try to infer from SUMO dir
                    if geojson_path is None and sumo_config_path:
                        sumo_dir = Path(sumo_config_path).parent
                        potential_geojson_files = list(sumo_dir.glob('*.geojson'))
                        if potential_geojson_files:
                            geojson_path = potential_geojson_files[0]
                        
            except Exception as e:
                pass
    
    # 2. FALLBACK: Try to extract from JSON log files in out_logging
    log_path = scenario_path / 'out_logging'
    if log_path.exists():
        for log_file in log_path.glob('*.json'):
            try:
                with open(log_file, 'r') as f:
                    config = json.load(f)
                    
                # Extract various parameter formats
                if 'bounding_box' in config:
                    bbox = config['bounding_box']
                elif 'bbox' in config:
                    bbox = config['bbox']
                    print(f"    ✓ Found bbox in {log_file.name}")
                    
                if 'grid_size' in config:
                    grid_size = config['grid_size']
                    print(f"    ✓ Found grid size: {grid_size}m in {log_file.name}")
                    
                if 'total_simulation_steps' in config:
                    total_steps = config['total_simulation_steps']
                    print(f"    ✓ Found simulation steps: {total_steps} in {log_file.name}")
                elif 'simulation_steps' in config:
                    total_steps = config['simulation_steps']
                    print(f"    ✓ Found simulation steps: {total_steps} in {log_file.name}")
                    
                if 'step_length' in config:
                    step_length = config['step_length']
                    print(f"    ✓ Found step length: {step_length}s in {log_file.name}")
                elif 'time_step' in config:
                    step_length = config['time_step']
                    print(f"    ✓ Found step length: {step_length}s in {log_file.name}")
                    
            except Exception as e:
                print(f"    ⚠ Could not parse {log_file.name}: {e}")
                continue
    

    
    # Check for visibility CSV to validate scenario (updated for new directory structure)
    discrete_csv_path = scenario_path / 'out_raytracing' / f'discrete_visibility_counts_{scenario_name}.csv'
    continuous_csv_pattern = scenario_path / 'out_raytracing' / f'continuous_visibility_counts_{scenario_name}_SSA*.csv'
    
    # Find continuous CSV with any SSA value
    import glob
    continuous_csv_matches = list(scenario_path.glob(f'out_raytracing/continuous_visibility_counts_{scenario_name}_SSA*.csv'))
    continuous_csv_path = continuous_csv_matches[0] if continuous_csv_matches else None
    
    # Extract SSA value from continuous CSV filename if it exists
    single_sensor_accuracy = None
    if continuous_csv_path:
        import re
        ssa_match = re.search(r'_SSA(\d+)%\.csv$', continuous_csv_path.name)
        if ssa_match:
            single_sensor_accuracy = int(ssa_match.group(1))
    
    if not discrete_csv_path.exists() and not continuous_csv_path:
        raise FileNotFoundError(f"No visibility CSV files found in: {scenario_path / 'out_raytracing'}")
    
    # Find GeoJSON path - check multiple potential locations (only if not already found from log)
    if geojson_path is None:
        potential_geojson_paths = [
            Path('simulation_examples/Ilic_TRB2025/SUMO_example.geojson'),  # New location
            Path('SUMO_example/SUMO_example.geojson'),  # Old location (fallback)
            Path(GEOJSON_PATH) if GEOJSON_PATH else None  # User-configured fallback path
        ]
        
        for potential_path in potential_geojson_paths:
            if potential_path and potential_path.exists():
                geojson_path = potential_path
                print(f"  ✓ Found GeoJSON file: {geojson_path}")
                break
        
        if geojson_path is None:
            print(f"  - No GeoJSON file found in expected locations")
    
    # Use fallbacks for missing parameters
    if bbox is None:
        bbox = BOUNDING_BOX
    if total_steps is None:
        total_steps = TOTAL_SIMULATION_STEPS
    if step_length is None:
        step_length = STEP_LENGTH  
    if grid_size is None:
        grid_size = VISUALIZATION_GRID_SIZE
    
    # Extract project name from scenario_path (last folder name)
    project_name = scenario_path.name
    
    # Create the spatial visibility output directory directly in scenario path
    spatial_output_dir = scenario_path / 'out_spatial_visibility'
    
    return {
        'file_tag': file_tag,
        'project_name': project_name,
        'fco_share': fco_share,
        'fbo_share': fbo_share,
        'bbox': bbox,
        'discrete_visibility_csv_path': str(discrete_csv_path) if discrete_csv_path.exists() else None,
        'continuous_visibility_csv_path': str(continuous_csv_path) if continuous_csv_path else None,
        'single_sensor_accuracy': single_sensor_accuracy,
        'output_dir': str(spatial_output_dir),
        'geojson_path': str(geojson_path) if geojson_path and geojson_path.exists() else None,
        'grid_size': grid_size,
        'total_simulation_steps': total_steps,
        'step_length': step_length,
        'scenario_output_path': str(scenario_path)
    }


class SpatialVisibilityAnalyzer:
    def __init__(self, config_file=None, **kwargs):
        """
        Initialize the spatial visibility analyzer.
        
        Args:
            config_file: Path to JSON configuration file (optional)
            **kwargs: Override configuration parameters
        """
        # Try automatic detection first if SCENARIO_OUTPUT_PATH is set
        if SCENARIO_OUTPUT_PATH is not None:
            try:
                auto_config = auto_detect_parameters_from_scenario(SCENARIO_OUTPUT_PATH)
                # Print will be done in analyze_spatial_visibility
                # Store for later use
            except Exception as e:
                print(f"⚠ Auto-detection failed: {e}")
                print("Falling back to manual configuration...")
                auto_config = None
        else:
            auto_config = None
        
        # Use auto-detected config or fall back to manual configuration
        if auto_config:
            self.config = {
                # Required parameters (from auto-detection)
                'bbox': auto_config['bbox'],
                'FCO_share': auto_config['fco_share'],
                'FBO_share': auto_config['fbo_share'],
                'discrete_visibility_csv_path': auto_config['discrete_visibility_csv_path'],
                'continuous_visibility_csv_path': auto_config['continuous_visibility_csv_path'],
                'single_sensor_accuracy': auto_config['single_sensor_accuracy'],
                'total_simulation_steps': auto_config['total_simulation_steps'],
                'step_length': auto_config['step_length'],
                'file_tag': auto_config['file_tag'],  # Add file_tag for naming
                'scenario_output_path': SCENARIO_OUTPUT_PATH,  # Store for infrastructure classification
                
                # Optional parameters (from auto-detection with defaults)
                'geojson_path': auto_config['geojson_path'],
                'output_dir': auto_config['output_dir'],
                'data_grid_size': auto_config['grid_size'],  # Grid size from data collection
                'visualization_grid_size': VISUALIZATION_GRID_SIZE,  # Grid size for visualization (user controlled)
                
                # Visualization options (use script defaults)
                'include_roads': INCLUDE_ROADS,
                'include_buildings': INCLUDE_BUILDINGS,
                'include_parks': INCLUDE_PARKS,
                'include_trees': INCLUDE_TREES,
                'include_barriers': INCLUDE_BARRIERS,
                'include_pt_shelters': INCLUDE_PT_SHELTERS,
                'colormap': COLORMAP,
                'alpha': ALPHA,
                'figure_size': (12, 10),
                'dpi': 300
            }
        else:
            # Use manual configuration variables defined at the top of the script
            scenario_name = f"{FILE_TAG}_FCO{FCO_SHARE}%_FBO{FBO_SHARE}%"
            scenario_dir = f"outputs/{scenario_name}"
            
            # Extract project name from the scenario directory structure
            project_name = scenario_name
            
            # Create the spatial visibility output directory directly in scenario path
            output_dir = f"{scenario_dir}/out_spatial_visibility"
            
            # Updated to use new directory structure
            discrete_visibility_csv_path = f"{scenario_dir}/out_raytracing/discrete_visibility_counts_{scenario_name}.csv"
            continuous_visibility_csv_path = f"{scenario_dir}/out_raytracing/continuous_visibility_counts_{scenario_name}_SSA70%.csv"  # Default to 70%
            
            self.config = {
                # Required parameters
                'bbox': BOUNDING_BOX,
                'project_name': project_name,
                'FCO_share': FCO_SHARE,
                'FBO_share': FBO_SHARE,
                'discrete_visibility_csv_path': discrete_visibility_csv_path,
                'continuous_visibility_csv_path': continuous_visibility_csv_path,
                'single_sensor_accuracy': 70,  # Default
                'total_simulation_steps': TOTAL_SIMULATION_STEPS,
                'step_length': STEP_LENGTH,
                'file_tag': FILE_TAG,  # Add file_tag for naming
                
                # Optional parameters  
                'geojson_path': GEOJSON_PATH,
                'output_dir': output_dir,
                'visualization_grid_size': VISUALIZATION_GRID_SIZE,
                
                # Visualization options
                'include_roads': INCLUDE_ROADS,
                'include_buildings': INCLUDE_BUILDINGS,
                'include_parks': INCLUDE_PARKS,
                'include_trees': INCLUDE_TREES,
                'include_barriers': INCLUDE_BARRIERS,
                'include_pt_shelters': INCLUDE_PT_SHELTERS,
                'colormap': COLORMAP,
                'alpha': ALPHA,
                'figure_size': (12, 10),
                'dpi': 300
            }
        
        # Override with any provided kwargs
        self.config.update(kwargs)
        
        # Ensure output directory exists
        os.makedirs(self.config['output_dir'], exist_ok=True)
    
    def project(self, lon, lat):
        """Project WGS84 coordinates to UTM Zone 32N (EPSG:32632)."""
        # Use modern Transformer API (pyproj 2+)
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)
        x, y = transformer.transform(lon, lat)
        return x, y
    
    def load_visibility_data(self, csv_type='discrete'):
        """
        Load visibility count data from CSV file.
        
        Args:
            csv_type: 'discrete' or 'continuous' to specify which CSV to load
        
        Returns:
            pandas.DataFrame: Visibility data with columns [x_coord, y_coord, {discrete|continuous}_visibility_count]
        """
        csv_path_key = f'{csv_type}_visibility_csv_path'
        csv_path = self.config.get(csv_path_key)
        
        if not csv_path or not os.path.exists(csv_path):
            raise FileNotFoundError(f"{csv_type.capitalize()} visibility CSV not found: {csv_path}")
        
        df = pd.read_csv(csv_path, comment='#')
        
        # Validate required columns based on type
        count_col = f'{csv_type}_visibility_count'
        required_cols = ['x_coord', 'y_coord', count_col]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in CSV")
        
        # Rename count column to generic 'visibility_count' for consistent processing
        df = df.rename(columns={count_col: 'visibility_count'})
        
        return df
    
    def load_geospatial_data(self):
        """
        Load and project geospatial data (roads, buildings, parks, etc.).
        
        Returns:
            dict: Dictionary containing projected geospatial data layers
        """
        north, south, east, west = self.config['bbox']
        bbox = (north, south, east, west)
        
        data = {}
        
        # Load road space distribution from GeoJSON if available
        if self.config['geojson_path']:
            geojson_path = self.config['geojson_path']
            
            if os.path.exists(geojson_path):
                # Temporarily redirect stderr to suppress GDAL/OGR warnings
                import sys
                import io
                old_stderr = sys.stderr
                sys.stderr = io.StringIO()
                
                try:
                    gdf1 = gpd.read_file(geojson_path)
                finally:
                    sys.stderr = old_stderr
                
                # Filter for line elements only (curbs) - exclude Junction polygons (intersection areas)
                if 'Type' in gdf1.columns:
                    gdf1 = gdf1[
                        gdf1['Type'].isin(['LaneBoundary', 'Gate', 'Signal']) &
                        gdf1.geometry.type.isin(['LineString', 'MultiLineString'])
                    ]
                
                if len(gdf1) > 0:
                    data['roads'] = gdf1.to_crs("EPSG:32632")
                else:
                    data['roads'] = None
            else:
                data['roads'] = None
        else:
            data['roads'] = None
        
        # Load OpenStreetMap data
        osm_messages = []
        
        # Get buildings
        if self.config['include_buildings']:
            try:
                buildings = ox.features_from_bbox(bbox=bbox, tags={'building': True})
                data['buildings'] = buildings.to_crs("EPSG:32632")
                osm_messages.append(f"  {len(buildings)} buildings")
            except Exception as e:
                osm_messages.append("  Warning: Could not load buildings")
                data['buildings'] = None
        else:
            data['buildings'] = None
        
        # Get parks
        if self.config['include_parks']:
            try:
                parks = ox.features_from_bbox(bbox=bbox, tags={'leisure': ['park', 'garden']})
                data['parks'] = parks.to_crs("EPSG:32632")
                osm_messages.append(f"  {len(parks)} parks")
            except Exception as e:
                osm_messages.append("  Warning: Could not load parks")
                data['parks'] = None
        else:
            data['parks'] = None
        
        # Get trees
        if self.config['include_trees']:
            try:
                trees = ox.features_from_bbox(bbox=bbox, tags={'natural': 'tree'})
                data['trees'] = trees.to_crs("EPSG:32632")
                osm_messages.append(f"  {len(trees)} trees")
            except Exception as e:
                osm_messages.append("  Warning: Could not load trees")
                data['trees'] = None
        else:
            data['trees'] = None
        
        # Get barriers
        if self.config['include_barriers']:
            try:
                barriers = ox.features_from_bbox(bbox=bbox, tags={'barrier': True})
                data['barriers'] = barriers.to_crs("EPSG:32632")
                osm_messages.append(f"  {len(barriers)} barriers")
            except Exception as e:
                osm_messages.append("  Warning: Could not load barriers")
                data['barriers'] = None
        else:
            data['barriers'] = None
        
        # Get public transport shelters
        if self.config['include_pt_shelters']:
            try:
                pt_shelters = ox.features_from_bbox(bbox=bbox, tags={'shelter_type': 'public_transport'})
                data['pt_shelters'] = pt_shelters.to_crs("EPSG:32632")
                osm_messages.append(f"  {len(pt_shelters)} PT shelters")
            except Exception as e:
                osm_messages.append("  Warning: Could not load PT shelters")
                data['pt_shelters'] = None
        else:
            data['pt_shelters'] = None
        
        # Print consolidated OSM loading message
        if osm_messages:
            print(f"✓ Loaded OpenStreetMap data")
            for msg in osm_messages:
                print(msg)
        
        return data
    
    def _round_bbox_to_tens(self, bbox):
        """Round bbox dimensions up to the nearest 10m, keeping center."""
        min_x, min_y, max_x, max_y = bbox
        width = max_x - min_x
        height = max_y - min_y

        rounded_w = math.ceil(width / 10) * 10
        rounded_h = math.ceil(height / 10) * 10

        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2

        new_min_x = center_x - rounded_w / 2
        new_max_x = center_x + rounded_w / 2
        new_min_y = center_y - rounded_h / 2
        new_max_y = center_y + rounded_h / 2

        return [new_min_x, new_min_y, new_max_x, new_max_y]
    
    def _compute_focus_area_bbox(self, df):
        """Compute focus area bounding box from visibility data."""
        if not ENABLE_FOCUS_AREA_OVERLAY:
            return None
        
        if FOCUS_AREA_BBOX_OVERRIDE:
            return FOCUS_AREA_BBOX_OVERRIDE

        if 'visibility_count' not in df.columns:
            return None

        mask = df['visibility_count'] >= FOCUS_AREA_THRESHOLD
        if not mask.any():
            return None

        x_min = df.loc[mask, 'x_coord'].min() - FOCUS_AREA_BUFFER
        x_max = df.loc[mask, 'x_coord'].max() + FOCUS_AREA_BUFFER
        y_min = df.loc[mask, 'y_coord'].min() - FOCUS_AREA_BUFFER
        y_max = df.loc[mask, 'y_coord'].max() + FOCUS_AREA_BUFFER

        bbox = [x_min, y_min, x_max, y_max]
        if FOCUS_AREA_ROUND_TO_10M:
            bbox = self._round_bbox_to_tens(bbox)

        return bbox
    
    def _overlay_focus_area(self, ax, focus_bbox, x_min, y_min):
        """Overlay focus area rectangle on axes with proper coordinate translation."""
        if not focus_bbox:
            return
        
        min_x, min_y, max_x, max_y = focus_bbox
        
        # Translate to plot coordinates
        plot_min_x = min_x - x_min
        plot_min_y = min_y - y_min
        plot_width = max_x - min_x
        plot_height = max_y - min_y
        
        # Semi-transparent fill
        #focus_fill = Rectangle(
        #    (plot_min_x, plot_min_y),
        #    plot_width,
        #    plot_height,
        #    linewidth=0,
        #    edgecolor='none',
        #    facecolor='darkcyan',
        #    alpha=0.12,
        #    zorder=10
        #)
        #ax.add_patch(focus_fill)

        # Solid darkcyan border
        focus_border = Rectangle(
            (plot_min_x, plot_min_y),
            plot_width,
            plot_height,
            linewidth=4.5,
            edgecolor='darkcyan',
            facecolor='none',
            linestyle='-',
            zorder=11
        )
        ax.add_patch(focus_border)
        
        # Combined label with dimensions (white background, black border and text)
        center_x = plot_min_x + plot_width / 2
        
        label_bbox_style = dict(boxstyle='round,pad=0.6', facecolor='white', 
                         alpha=1.0, edgecolor='black', linewidth=2)
        ax.text(center_x, plot_min_y + plot_height + 12,
               f'FOCUS AREA ({plot_width:.0f}m × {plot_height:.0f}m)',
               ha='center', va='bottom',
               fontsize=17, fontweight='bold',
               color='black',
               bbox=label_bbox_style,
               zorder=12)
    
    def classify_infrastructure_from_network(self, scenario_path):
        """
        Classify grid cells based on lane-level infrastructure types from SUMO network file.
        
        Returns:
            dict: Mapping of grid cells to infrastructure types:
                - 'vehicles_only': Only motorized vehicles allowed
                - 'vru': Pedestrians and/or bicycles only  
                - 'mixed': Both vehicles and VRUs allowed
                - 'none': No infrastructure present
        """
        # Check if user provided network file path
        if SUMO_NETWORK_FILE is None:
            print(f"  ⚠ SUMO_NETWORK_FILE not configured")
            print(f"    Please set SUMO_NETWORK_FILE in configuration to enable infrastructure classification")
            return None
        
        net_file = SUMO_NETWORK_FILE
        
        if not os.path.exists(net_file):
            print(f"  ⚠ Network file not found: {net_file}")
            print(f"    Please check the path in SUMO_NETWORK_FILE configuration")
            return None
        
        print(f"  ✓ Using network file: {os.path.basename(net_file)}")
        
        # Parse network file
        try:
            net_tree = ET.parse(net_file)
            net_root = net_tree.getroot()
        except Exception as e:
            print(f"  ⚠ Could not parse network file: {e}")
            return None
        
        # Create grid cells from configuration
        north, south, east, west = self.config['bbox']
        x_min, y_min = self.project(west, south)
        x_max, y_max = self.project(east, north)
        
        grid_size = self.config['visualization_grid_size']
        x_coords = np.arange(x_min, x_max, grid_size)
        y_coords = np.arange(y_min, y_max, grid_size)
        grid_cells = [box(x, y, x + grid_size, y + grid_size) 
                      for x in x_coords for y in y_coords]
        
        # Initialize cell classifications
        cell_classifications = {i: 'none' for i in range(len(grid_cells))}
        
        # Parse type definitions from network file
        type_defs = {}
        for type_elem in net_root.findall('.//type'):
            type_id = type_elem.get('id')
            allow = type_elem.get('allow', '')
            disallow = type_elem.get('disallow', '')
            type_defs[type_id] = {'allow': allow, 'disallow': disallow}
        
        print(f"  ✓ Found {len(type_defs)} type definitions")
        
        # Helper function to classify lane permissions
        def classify_lane_permissions(allow, disallow, type_id=None):
            """Classify based on allow/disallow attributes"""
            # Parse space-separated values
            allow_set = set(allow.split()) if allow else None
            disallow_set = set(disallow.split()) if disallow else set()
            
            # VRU types (anything else is considered a vehicle)
            vru_types = {'pedestrian', 'bicycle'}
            
            # For composite types (e.g., "cycleway.lane|highway.residential"), check component types
            if type_id and '|' in type_id and (not allow and not disallow):
                # No explicit allow/disallow on lane - need to check component types
                component_types = type_id.split('|')
                component_has_vru = False
                component_has_vehicles = False
                
                for comp_type in component_types:
                    if comp_type in type_defs:
                        comp_allow = type_defs[comp_type]['allow']
                        comp_disallow = type_defs[comp_type]['disallow']
                        
                        # Check what this component allows
                        if comp_allow:
                            comp_allow_set = set(comp_allow.split())
                            if comp_allow_set & vru_types:
                                component_has_vru = True
                            # Any non-VRU type is a vehicle
                            if comp_allow_set - vru_types:
                                component_has_vehicles = True
                        else:
                            # No explicit allow - check disallow
                            comp_disallow_set = set(comp_disallow.split()) if comp_disallow else set()
                            if not (comp_disallow_set & vru_types):
                                component_has_vru = True
                            # If not all non-VRU types are disallowed, vehicles are allowed
                            if not comp_disallow_set or (comp_disallow_set == vru_types):
                                component_has_vehicles = True
                
                # Use component analysis if we found component types
                if component_has_vru or component_has_vehicles:
                    if component_has_vru and component_has_vehicles:
                        return 'mixed'
                    elif component_has_vru:
                        return 'vru'
                    elif component_has_vehicles:
                        return 'vehicles_only'
            
            # Standard classification using lane's own allow/disallow
            if allow_set is not None:
                # Explicit allow list
                has_vru = bool(allow_set & vru_types)
                # Any non-VRU type in the allow list means vehicles are allowed
                has_vehicles = bool(allow_set - vru_types)
            else:
                # No explicit allow = allow all except disallow
                has_vru = not bool(disallow_set & vru_types)
                # Vehicles allowed unless all non-VRU types would be disallowed (impossible to list all)
                # In practice: if only VRUs are disallowed, vehicles are allowed
                # If VRUs are not disallowed but we have a disallow list, check if it blocks vehicles
                has_vehicles = not disallow_set or bool(vru_types - disallow_set) or not has_vru
            
            if has_vru and has_vehicles:
                return 'mixed'
            elif has_vru:
                return 'vru'
            elif has_vehicles:
                return 'vehicles_only'
            else:
                return 'none'
        
        # Get location information from SUMO network
        location = net_root.find('.//location')
        if location is not None:
            net_offset_str = location.get('netOffset', '0.0,0.0')
            net_offset_x = float(net_offset_str.split(',')[0])
            net_offset_y = float(net_offset_str.split(',')[1])
            
            proj_parameter = location.get('projParameter', '')
            conv_boundary = location.get('convBoundary', '')
            orig_boundary = location.get('origBoundary', '')
            
            print(f"  ✓ Net offset: {net_offset_str}")
            print(f"  ✓ Projection: {proj_parameter}")
            print(f"  ✓ SUMO conv boundary: {conv_boundary}")
            print(f"  ✓ Geographic orig boundary: {orig_boundary}")
            
            # SUMO coordinates transformation:
            # SUMO stores coordinates in a local system. The netOffset was SUBTRACTED from absolute
            # UTM coordinates to create the local SUMO coordinates, so we need to ADD it back (subtract the negative)
            # to convert from SUMO local coordinates to absolute UTM coordinates.
            # Formula: UTM = SUMO_local - netOffset (because netOffset is stored as negative)
            def sumo_to_utm(x_sumo, y_sumo):
                """Convert SUMO coordinates to absolute UTM by reversing the netOffset operation"""
                x_utm = x_sumo - net_offset_x  # Subtract the (negative) offset = add absolute value
                y_utm = y_sumo - net_offset_y  # Subtract the (negative) offset = add absolute value
                return x_utm, y_utm
        else:
            print(f"  ⚠ No location element found in network file")
            return None
        
        # Print grid extent for comparison
        print(f"  ✓ Grid extent: x=[{x_min:.2f}, {x_max:.2f}], y=[{y_min:.2f}, {y_max:.2f}]")
        
        # Process each lane in the network
        lane_count = 0
        classified_count = 0
        lanes_with_coords = 0
        
        # Debug: count classifications
        classification_counts = {'vehicles_only': 0, 'vru': 0, 'mixed': 0, 'none': 0}
        
        # Get all edges
        all_edges = net_root.findall('.//edge')
        
        # Pre-pass: Identify all cells that will be affected by lanes (for accurate progress tracking)
        print(f"  Identifying number of relevant grid cells (cells that intersect with lanes)...")
        relevant_cells = set()
        for edge in all_edges:
            function = edge.get('function')
            # Skip crossings and walking areas
            if function in ['crossing', 'walkingarea']:
                continue
            
            for lane in edge.findall('lane'):
                shape = lane.get('shape')
                if not shape:
                    continue
                
                # Parse shape coordinates
                coords = []
                for coord_pair in shape.split():
                    try:
                        x_sumo, y_sumo = map(float, coord_pair.split(','))
                        x_utm, y_utm = sumo_to_utm(x_sumo, y_sumo)
                        coords.append((x_utm, y_utm))
                    except:
                        continue
                
                if len(coords) < 2:
                    continue
                
                # Create approximate buffer to find relevant cells
                lane_line = LineString(coords)
                width = float(lane.get('width', '3.2'))
                lane_buffer = lane_line.buffer(width / 2)
                
                # Find all cells that intersect with this lane
                for i, cell in enumerate(grid_cells):
                    if lane_buffer.intersects(cell):
                        relevant_cells.add(i)
        
        print(f"  ✓ Found {len(relevant_cells)} relevant grid cells (cells that intersect with lanes) out of {len(grid_cells)} total cells")
        
        # Create progress bar for relevant cells only
        pbar = tqdm(total=len(relevant_cells), desc="  Classifying infrastructure", ncols=80, leave=False, unit='cells', bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}')
        cells_processed = set()  # Track which cells have been processed
        
        for edge in all_edges:
            function = edge.get('function')
            
            # Process internal edges (intersection connection lanes)
            if function == 'internal':
                # Internal lanes connect different edges within an intersection
                # Classify them based on their actual permissions, not automatically as 'mixed'
                for lane in edge.findall('lane'):
                    lane_count += 1
                    
                    # Get lane permissions for internal lanes
                    allow = lane.get('allow', '')
                    disallow = lane.get('disallow', '')
                    
                    # Classify this internal lane based on its permissions
                    # Internal lanes typically don't have type attributes
                    classification = classify_lane_permissions(allow, disallow, type_id=None)
                    classification_counts[classification] += 1
                    
                    shape = lane.get('shape')
                    if not shape:
                        continue
                    
                    # Parse shape coordinates
                    coords = []
                    for coord_pair in shape.split():
                        try:
                            x_sumo, y_sumo = map(float, coord_pair.split(','))
                            x_utm, y_utm = sumo_to_utm(x_sumo, y_sumo)
                            coords.append((x_utm, y_utm))
                        except:
                            continue
                    
                    if len(coords) < 2:
                        continue
                    
                    lanes_with_coords += 1
                    
                    # Create LineString for lane
                    lane_line = LineString(coords)
                    width = float(lane.get('width', '3.2'))
                    if classification == 'vru':
                        width = min(width, 2.5)  # VRU infrastructure typically narrower
                    lane_buffer = lane_line.buffer(width / 2)
                    
                    # Classify intersecting cells based on lane classification
                    for i, cell in enumerate(grid_cells):
                        if lane_buffer.intersects(cell):
                            if i not in cells_processed:
                                cells_processed.add(i)
                                pbar.update(1)
                            
                            current = cell_classifications[i]
                            
                            # Priority: mixed > vehicles_only > vru > none
                            if current == 'none':
                                cell_classifications[i] = classification
                                classified_count += 1
                            elif current == 'mixed':
                                # Already mixed, keep it
                                pass
                            elif current != classification:
                                # Different types intersect -> mixed
                                cell_classifications[i] = 'mixed'
                continue
            
            # Skip crossings and walking areas (let them be classified by adjacent lanes)
            if function in ['crossing', 'walkingarea']:
                continue
            
            for lane in edge.findall('lane'):
                lane_count += 1
                
                # Get lane permissions (priority: lane > edge > type definition)
                allow = lane.get('allow', '')
                disallow = lane.get('disallow', '')
                
                # Get type for composite type handling
                edge_type = edge.get('type', '')
                
                # If lane doesn't specify, inherit from edge
                if not allow and not disallow:
                    allow = edge.get('allow', '')
                    disallow = edge.get('disallow', '')
                
                # If edge doesn't specify, look up type definition
                if not allow and not disallow:
                    if edge_type and edge_type in type_defs:
                        allow = type_defs[edge_type]['allow']
                        disallow = type_defs[edge_type]['disallow']
                
                # Classify this lane (pass edge_type for composite type handling)
                classification = classify_lane_permissions(allow, disallow, edge_type)
                classification_counts[classification] += 1
                
                # Get lane geometry (shape attribute)
                shape = lane.get('shape')
                if not shape:
                    continue
                
                # Parse shape coordinates (SUMO coordinate system)
                coords = []
                for coord_pair in shape.split():
                    try:
                        x_sumo, y_sumo = map(float, coord_pair.split(','))
                        # Convert SUMO coordinates directly to UTM (no intermediate geographic conversion needed)
                        x_utm, y_utm = sumo_to_utm(x_sumo, y_sumo)
                        coords.append((x_utm, y_utm))
                    except:
                        continue
                
                if len(coords) < 2:
                    continue
                
                lanes_with_coords += 1
                
                # Create LineString for lane
                lane_line = LineString(coords)
                
                # Get lane width (default 3.2m for normal lanes, 2.0m for bike/ped)
                width = float(lane.get('width', '3.2'))
                if classification == 'vru':
                    width = min(width, 2.5)  # VRU infrastructure typically narrower
                
                # Buffer lane by half width on each side to get coverage area
                lane_buffer = lane_line.buffer(width / 2)
                
                # Check intersection with grid cells
                for i, cell in enumerate(grid_cells):
                    if lane_buffer.intersects(cell):
                        if i not in cells_processed:
                            cells_processed.add(i)
                            pbar.update(1)
                        current = cell_classifications[i]
                        
                        # Priority: mixed > vehicles_only > vru > none
                        if current == 'none':
                            cell_classifications[i] = classification
                            classified_count += 1
                        elif current == 'mixed':
                            # Already mixed, keep it
                            pass
                        elif current != classification:
                            # Different types intersect -> mixed
                            cell_classifications[i] = 'mixed'
        
        # Close progress bar
        pbar.close()
        print(f"  ✓ Processed {lane_count} lanes ({lanes_with_coords} with valid coordinates)")
        print(f"  ✓ Lane classifications: vehicles_only={classification_counts['vehicles_only']}, vru={classification_counts['vru']}, mixed={classification_counts['mixed']}, none={classification_counts['none']}")
        print(f"  ✓ Classified {classified_count} grid cells with infrastructure")
        
        # Convert to dict with cell geometries as keys (for consistency with existing code)
        result = {}
        for i, classification in cell_classifications.items():
            result[grid_cells[i]] = classification
        
        return result
    
    def _save_infrastructure_classification(self, infrastructure_types):
        """Save infrastructure classification to CSV file."""
        output_prefix = f'FCO{self.config["FCO_share"]}%_FBO{self.config["FBO_share"]}%'
        output_filename = f'infrastructure_classification_{self.config["file_tag"]}_{output_prefix}.csv'
        output_path = os.path.join(self.config['output_dir'], output_filename)
        
        # Convert to DataFrame for easy saving
        data = []
        for cell, infra_type in infrastructure_types.items():
            # Get cell center coordinates
            cell_x = cell.bounds[0] + (cell.bounds[2] - cell.bounds[0]) / 2
            cell_y = cell.bounds[1] + (cell.bounds[3] - cell.bounds[1]) / 2
            data.append({
                'x_coord': cell_x,
                'y_coord': cell_y,
                'infrastructure_type': infra_type
            })
        
        df = pd.DataFrame(data)
        
        # Save with header comments
        with open(output_path, 'w', newline='') as f:
            f.write('# Infrastructure Type Classification\n')
            f.write('# Generated from SUMO network file (lane-level analysis)\n')
            f.write('#\n')
            f.write('# Categories:\n')
            f.write('#   vehicles_only: Only motorized vehicles allowed\n')
            f.write('#   vru: Pedestrians and/or bicycles only\n')
            f.write('#   mixed: Both vehicles and VRUs allowed\n')
            f.write('#   none: No infrastructure present\n')
            f.write('#\n')
            df.to_csv(f, index=False)
        
        print(f"  ✓ Saved infrastructure classification: {output_filename}")
    
    def plot_infrastructure_classification_map(self, infrastructure_types, geospatial_data):
        """
        Generate a visualization map showing infrastructure type classification.
        Similar style to LoV heatmaps but showing discrete categories.
        
        Args:
            infrastructure_types: Dictionary mapping grid cells to infrastructure types
            geospatial_data: Dictionary with geospatial background layers
        """
        # Get coordinate system info
        north, south, east, west = self.config['bbox']
        x_min, y_min = self.project(west, south)
        x_max, y_max = self.project(east, north)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.config['figure_size'])
        
        # Plot background layers
        self._plot_background_layers(ax, geospatial_data, x_min, y_min, 
                                     include_roads=self.config['include_roads'])
        
        # Define colors for each infrastructure type
        color_map = {
            'vehicles_only': '#4ECDC4',  # Cyan for vehicles only
            'vru': '#FF0000',            # Red for VRU (pedestrian/bicycle)
            'mixed': '#FFA500',          # Orange for mixed
            'none': '#FFFFFF'            # Transparent for none
        }
        
        # Plot infrastructure classification as colored rectangles
        grid_size = self.config['visualization_grid_size']
        
        for cell, infra_type in infrastructure_types.items():
            if infra_type == 'none':
                continue  # Skip cells with no infrastructure
            
            # Get cell bounds (already in UTM)
            minx, miny, maxx, maxy = cell.bounds
            
            # Translate to plot coordinates
            plot_x = minx - x_min
            plot_y = miny - y_min
            plot_width = maxx - minx
            plot_height = maxy - miny
            
            # Add rectangle for this cell
            rect = Rectangle((plot_x, plot_y), plot_width, plot_height,
                           facecolor=color_map[infra_type], 
                           edgecolor='none',
                           alpha=0.7,
                           zorder=2)
            ax.add_patch(rect)
        
        # Set axis properties (match LoV map formatting)
        ax.set_xlim(0, x_max - x_min)
        ax.set_ylim(0, y_max - y_min)
        ax.set_xlabel('Longitude [m]')
        ax.set_ylabel('Latitude [m]')
        ax.set_aspect('equal')
        
        # Create legend - only infrastructure classes
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=color_map['vehicles_only'], alpha=0.7, label='Vehicles Only'),
            Patch(facecolor=color_map['vru'], alpha=0.7, label='VRU (Pedestrian/Bicycle)'),
            Patch(facecolor=color_map['mixed'], alpha=0.7, label='Mixed (Vehicles + VRU)'),
        ]
        
        ax.legend(handles=legend_elements, loc='upper right', framealpha=0.9, fontsize=10)
        
        # Save figure
        output_prefix = f'FCO{self.config["FCO_share"]}%_FBO{self.config["FBO_share"]}%'
        output_filename = f'infrastructure_map_{self.config["file_tag"]}_{output_prefix}.png'
        output_path = os.path.join(self.config['output_dir'], output_filename)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.config['dpi'], bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved infrastructure map: {output_filename}")
    
    def create_grid_from_data(self, df):
        """
        Create visualization grid based on user-configured grid size.
        If visualization grid is coarser than data grid, aggregate data.
        If visualization grid is finer than data grid, show error.
        
        Args:
            df: DataFrame with visibility data
            
        Returns:
            tuple: (x_coords, y_coords, data_grid_size) for visualization
        """
        # Detect the actual grid size from the data
        x_coords_data = np.sort(df['x_coord'].unique())
        y_coords_data = np.sort(df['y_coord'].unique())
        
        if len(x_coords_data) > 1 and len(y_coords_data) > 1:
            data_grid_size_x = x_coords_data[1] - x_coords_data[0]
            data_grid_size_y = y_coords_data[1] - y_coords_data[0]
            data_grid_size = (data_grid_size_x + data_grid_size_y) / 2
        else:
            print("Warning: Cannot determine grid size from data - using single point")
            data_grid_size = self.config['visualization_grid_size']
        
        # Store detected data grid size
        self.config['data_grid_size'] = data_grid_size
        
        visualization_grid_size = self.config['visualization_grid_size']
        
        # Check if visualization grid is finer than data grid
        if visualization_grid_size < data_grid_size - 0.001:  # Small tolerance for floating point errors
            raise ValueError(f"Visualization grid size ({visualization_grid_size:.3f}m) cannot be smaller than data collection grid size ({data_grid_size:.3f}m). "
                           f"Cannot create finer resolution than what was collected.")
        
        # If visualization grid size matches data grid size, use original coordinates
        if abs(visualization_grid_size - data_grid_size) < 0.001:
            print("Using original data grid coordinates (grid sizes match)")
            x_coords = x_coords_data
            y_coords = y_coords_data
        else:
            # Create coarser grid for visualization by resampling the data extent
            print(f"Creating coarser visualization grid (aggregating {data_grid_size:.3f}m data to {visualization_grid_size:.3f}m grid)")
            
            # Get data extent
            x_min, x_max = x_coords_data[0], x_coords_data[-1]
            y_min, y_max = y_coords_data[0], y_coords_data[-1]
            
            # Create new grid coordinates aligned with data
            # Start from data minimum and create grid with visualization_grid_size spacing
            x_coords = np.arange(x_min, x_max + visualization_grid_size, visualization_grid_size)
            y_coords = np.arange(y_min, y_max + visualization_grid_size, visualization_grid_size)
            
            # Ensure we don't exceed the data extent
            x_coords = x_coords[x_coords <= x_max + 0.001]
            y_coords = y_coords[y_coords <= y_max + 0.001]
        
        return x_coords, y_coords, data_grid_size
    
    def _plot_background_layers(self, ax, data, x_min, y_min, include_roads):
        """Plot all background geospatial layers.
        
        Args:
            ax: Matplotlib axes object
            data: Dictionary with geospatial data
            x_min, y_min: Translation offset coordinates
            include_roads: Whether to include road space visualization from GeoJSON
            x_min, y_min: Translation offset coordinates
            include_roads: Whether to include road space visualization from GeoJSON
        """
        
        # Plot road space distribution (only if enabled)
        if include_roads and data['roads'] is not None:
            roads_translated = data['roads'].translate(-x_min, -y_min)
            roads_translated.plot(ax=ax, color='lightgray', alpha=0.5, edgecolor='lightgray', zorder=1)
        
        # Plot parks
        if data['parks'] is not None:
            parks_translated = data['parks'].translate(-x_min, -y_min)
            parks_translated.plot(ax=ax, facecolor='lightgreen', edgecolor='green', alpha=0.3, linewidth=0.5, zorder=2)
        
        # Plot buildings
        if data['buildings'] is not None:
            buildings_translated = data['buildings'].translate(-x_min, -y_min)
            buildings_translated.plot(ax=ax, facecolor='lightgray', edgecolor='black', alpha=0.7, linewidth=0.5, zorder=4)
        
        # Plot trees
        if data['trees'] is not None:
            trees_translated = data['trees'].translate(-x_min, -y_min)
            trees_circle = trees_translated.buffer(0.5)
            trees_circle.plot(ax=ax, facecolor='forestgreen', edgecolor='black', linewidth=0.5, zorder=5)
            
            # Plot tree canopies (leaves)
            leaves_circle = trees_translated.buffer(2.5)
            leaves_circle.plot(ax=ax, facecolor='forestgreen', alpha=0.5, edgecolor='black', linewidth=0.5, zorder=3)
        
        # Plot barriers
        if data['barriers'] is not None:
            barriers_translated = data['barriers'].translate(-x_min, -y_min)
            barriers_translated.plot(ax=ax, facecolor='brown', edgecolor='darkbrown', linewidth=0.5, zorder=5)
        
        # Plot PT shelters
        if data['pt_shelters'] is not None:
            pt_shelters_translated = data['pt_shelters'].translate(-x_min, -y_min)
            pt_shelters_translated.plot(ax=ax, facecolor='lightgray', edgecolor='black', linewidth=0.5, zorder=6)
    
    # =============================
    # RELATIVE VISIBILITY METHODS
    # =============================
    
    def create_relative_visibility_heatmap_data(self, df, x_coords, y_coords, data_grid_size):
        """
        Convert visibility data to 2D heatmap array for relative visibility using the visualization grid.
        If visualization grid is coarser than data grid, aggregate values.
        
        Args:
            df: DataFrame with visibility data
            x_coords, y_coords: Visualization grid coordinate arrays
            data_grid_size: Original data collection grid size
            
        Returns:
            numpy.ndarray: 2D heatmap data array
        """
        visualization_grid_size = self.config['visualization_grid_size']
        
        # Initialize heatmap array for visualization grid
        heatmap_data = np.zeros((len(x_coords), len(y_coords)))
        
        if abs(visualization_grid_size - data_grid_size) < 0.001:
            # Grid sizes match - use direct mapping (original approach)
            for _, row in df.iterrows():
                # Find closest grid indices for each data point
                x_idx = np.argmin(np.abs(x_coords - row['x_coord']))
                y_idx = np.argmin(np.abs(y_coords - row['y_coord']))
                
                # Ensure indices are within bounds
                if 0 <= x_idx < len(x_coords) and 0 <= y_idx < len(y_coords):
                    heatmap_data[x_idx, y_idx] = row['visibility_count']
        else:
            # Aggregate data to coarser grid
            count_data = np.zeros((len(x_coords), len(y_coords)))  # Count of data points per cell
            
            for _, row in df.iterrows():
                # Find which visualization grid cell this data point belongs to
                x_idx = np.digitize(row['x_coord'], x_coords) - 1
                y_idx = np.digitize(row['y_coord'], y_coords) - 1
                
                # Ensure indices are within bounds
                if 0 <= x_idx < len(x_coords) and 0 <= y_idx < len(y_coords):
                    heatmap_data[x_idx, y_idx] += row['visibility_count']
                    count_data[x_idx, y_idx] += 1
            
            # Average the aggregated values
            with np.errstate(divide='ignore', invalid='ignore'):
                heatmap_data = np.where(count_data > 0, heatmap_data / count_data, 0)
        
        # Set zero counts to NaN for better visualization
        heatmap_data[heatmap_data == 0] = np.nan
        
        # Normalize data
        max_val = np.nanmax(heatmap_data)
        if max_val > 0:
            heatmap_data = heatmap_data / max_val
        
        
        return heatmap_data
    
    def plot_relative_visibility_heatmap(self, heatmap_data, x_coords, y_coords, geospatial_data):
        """
        Create and save the relative visibility heatmap visualization.
        
        Args:
            heatmap_data: 2D numpy array with normalized visibility data
            x_coords, y_coords: Grid coordinate arrays
            geospatial_data: Dictionary with projected geospatial data
        """
        # Calculate bounds for translation (using full bounding box for scene extent)
        north, south, east, west = self.config['bbox']
        x_min, y_min = self.project(west, south)
        x_max, y_max = self.project(east, north)
        
        # Calculate dynamic figure size with FIXED HEIGHT and variable width
        # This ensures consistent vertical scale across different bounding boxes
        plot_width_m = x_max - x_min
        plot_height_m = y_max - y_min
        aspect_ratio = plot_width_m / plot_height_m  # Note: width/height for calculating width
        
        # Set fixed height and calculate width to maintain geographic proportions
        fixed_height = 10  # inches - consistent across all plots
        calculated_width = fixed_height * aspect_ratio
        
        # Ensure reasonable figure width limits
        min_width, max_width = 6, 20
        final_width = max(min_width, min(max_width, calculated_width))
        final_height = fixed_height
            
        dynamic_figure_size = (final_width, final_height)
        
        # Create figure with dynamic sizing (fixed height, variable width)
        fig, ax = plt.subplots(figsize=dynamic_figure_size, facecolor='white', dpi=self.config['dpi'])
        ax.set_facecolor('white')
        
        # Plot geospatial background layers covering the full bounding box
        include_roads = self.config['include_roads']
        self._plot_background_layers(ax, geospatial_data, x_min, y_min, include_roads)  # Include roads for relative visibility
        
        # Plot heatmap with proper extent (heatmap data extent, not bounding box)
        visualization_grid_size = self.config['visualization_grid_size']
        extent = [x_coords[0], x_coords[-1] + visualization_grid_size, 
                 y_coords[0], y_coords[-1] + visualization_grid_size]
        extent_translated = [x - x_min for x in extent[:2]] + [y - y_min for y in extent[2:]]
        
        # Use configured colormap for better scientific visualization
        cmap = plt.get_cmap(self.config['colormap'])  # Use configured colormap
        cmap.set_bad(color='white', alpha=0.0)  # Set NaN values to transparent white
        
        cax = ax.imshow(heatmap_data.T, origin='lower', cmap=cmap, 
                       extent=tuple(extent_translated), alpha=self.config['alpha'])  # Convert list to tuple
        
        # Set axis limits to match the full bounding box (this extends the scene)
        ax.set_xlim(0, x_max - x_min)
        ax.set_ylim(0, y_max - y_min)
        
        # Add colorbar with height matching the plot
        # Calculate aspect ratio to determine appropriate colorbar sizing
        plot_width = x_max - x_min
        plot_height = y_max - y_min
        aspect_ratio = plot_height / plot_width
        
        # Adjust colorbar parameters based on plot dimensions
        # For wider plots (low aspect ratio), use smaller fraction
        # For taller plots (high aspect ratio), use larger fraction
        if aspect_ratio < 0.7:  # Wide plot
            colorbar_fraction = 0.046 * aspect_ratio * 1.4  # Scale down for wide plots
        else:  # Taller plot
            colorbar_fraction = 0.046 * min(aspect_ratio, 1.5)  # Cap the scaling for very tall plots
        
        colorbar_pad = 0.04
        
        # Create colorbar with calculated sizing
        cbar = fig.colorbar(cax, ax=ax, fraction=colorbar_fraction, pad=colorbar_pad)
        cbar.set_label('Relative Visibility', rotation=270, labelpad=20, fontsize=20)
        cbar.ax.tick_params(labelsize=20)
        
        # Set labels (no title for cleaner appearance)
        ax.set_xlabel('Longitude [m]', fontsize=20)
        ax.set_ylabel('Latitude [m]', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=20)
        # ax.set_title('Relative Visibility Heatmap')
        
        # Save figure
        output_prefix = f'FCO{self.config["FCO_share"]}%_FBO{self.config["FBO_share"]}%'
        output_filename = f'relVis_heatmap_{self.config["file_tag"]}_{output_prefix}.png'
        output_path = os.path.join(self.config['output_dir'], output_filename)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.config['dpi'], bbox_inches='tight')
        plt.close()
    
    # ==================================
    # DISCRETE LEVEL OF VISIBILITY METHODS
    # ==================================
    
    def calculate_discrete_lov_data(self, df):
        """
        Calculate discrete Level of Visibility (LoV) from visibility counts.
        Uses binary frame counting (each frame counts as 1, regardless of observer count).
        
        Args:
            df: DataFrame with visibility data
            
        Returns:
            tuple: (lov_data, max_lov, logging_info)
        """
        # Calculate LoV for each data point
        lov_data = df['visibility_count'].values / (self.config['total_simulation_steps'] * self.config['step_length'])
        
        # Set LoV to NaN for cells with no visibility observations (count = 0)
        # This distinguishes between "never observed" (NaN) and "observed but poor visibility" (LoV E)
        lov_data = np.where(df['visibility_count'].values == 0, np.nan, lov_data)
        
        max_lov = 1 / self.config['step_length']
        
        # Calculate statistics only for cells with actual observations
        valid_lov = lov_data[~np.isnan(lov_data)]
        
        # Prepare logging information
        logging_info = [
            ['Visibility Type', 'Discrete (Binary Frame Counting)'],
            ['Max. visibility count', np.max(df['visibility_count'].values)],
            ['Total simulation steps', self.config['total_simulation_steps']],
            ['Step Size', self.config['step_length']],
            ['LoV scale', f'0 - {max_lov}'],
            ['Max. LoV value', np.max(valid_lov) if len(valid_lov) > 0 else 0],
            ['Mean LoV value', np.mean(valid_lov) if len(valid_lov) > 0 else 0],
            ['Cells with observations', len(valid_lov)],
            ['Cells without observations', np.sum(np.isnan(lov_data))]
        ]
        
        
        return lov_data, max_lov, logging_info
    
    def save_discrete_lov_logging_info(self, logging_info):
        """Save discrete LoV logging information to CSV file."""
        output_prefix = f'FCO{self.config["FCO_share"]}%_FBO{self.config["FBO_share"]}%'
        log_filename = f'discrete_LoV_log_{self.config["file_tag"]}_{output_prefix}.csv'
        log_path = os.path.join(self.config['output_dir'], log_filename)
        
        with open(log_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Description', 'Value'])
            csvwriter.writerows(logging_info)
    
    def aggregate_discrete_lov_data(self, df, lov_data, x_coords, y_coords, data_grid_size):
        """
        Aggregate discrete LoV data to coarser visualization grid if needed.
        
        Args:
            df: Original DataFrame with visibility data
            lov_data: Original LoV values array
            x_coords, y_coords: Visualization grid coordinates
            data_grid_size: Original data collection grid size
            
        Returns:
            tuple: (aggregated_lov_data, aggregated_counts) for visualization
        """
        visualization_grid_size = self.config['visualization_grid_size']
        
        if abs(visualization_grid_size - data_grid_size) < 0.001:
            # No aggregation needed - grid sizes match
            return lov_data, df['visibility_count'].values
        
        # Aggregate to coarser grid
        
        # Get original coordinates
        x_coords_data = np.sort(df['x_coord'].unique())
        y_coords_data = np.sort(df['y_coord'].unique())
        
        # Initialize aggregation arrays
        aggregated_lov = np.zeros(len(x_coords) * len(y_coords))
        aggregated_counts = np.zeros(len(x_coords) * len(y_coords))
        weighted_sums = np.zeros(len(x_coords) * len(y_coords))
        
        # Process each original data point
        for i, (x_orig, y_orig) in enumerate(zip(df['x_coord'].values, df['y_coord'].values)):
            # Find which visualization grid cell this belongs to
            x_idx = np.digitize(x_orig, x_coords) - 1
            y_idx = np.digitize(y_orig, y_coords) - 1
            
            # Ensure indices are within bounds
            if 0 <= x_idx < len(x_coords) and 0 <= y_idx < len(y_coords):
                flat_idx = x_idx * len(y_coords) + y_idx
                
                # Skip NaN values in aggregation
                if not np.isnan(lov_data[i]):
                    weighted_sums[flat_idx] += lov_data[i] * df['visibility_count'].iloc[i]
                    aggregated_counts[flat_idx] += df['visibility_count'].iloc[i]
        
        # Calculate weighted average LoV values
        with np.errstate(divide='ignore', invalid='ignore'):
            aggregated_lov = np.where(aggregated_counts > 0, weighted_sums / aggregated_counts, np.nan)
        
        
        return aggregated_lov, aggregated_counts
    
    def plot_discrete_lov_heatmap(self, df, lov_data, max_lov, x_coords, y_coords, data_grid_size, geospatial_data):
        """
        Create and save the discrete LoV heatmap visualization.
        
        Args:
            df: DataFrame with visibility data  
            lov_data: Calculated discrete LoV values (possibly aggregated)
            max_lov: Maximum LoV value
            x_coords, y_coords: Visualization grid coordinates
            data_grid_size: Original data collection grid size
            geospatial_data: Dictionary with projected geospatial data
        """
        # Create figure
        fig, ax = plt.subplots(figsize=self.config['figure_size'], facecolor='white', dpi=self.config['dpi'])
        ax.set_facecolor('white')
        
        # Calculate bounds for translation (using full bounding box for scene extent)
        north, south, east, west = self.config['bbox']
        x_min, y_min = self.project(west, south)
        x_max, y_max = self.project(east, north)
        
        # Plot geospatial background layers covering the full bounding box
        include_roads = self.config['include_roads']
        self._plot_background_layers(ax, geospatial_data, x_min, y_min, include_roads)  # Include roads for LoV heatmap
        
        # Create LoV color map (same as main.py)
        alpha = self.config['alpha']
        colors = [
            (0.698, 0.133, 0.133, alpha),  # Firebrick - LoV E
            (1.000, 0.271, 0.000, alpha),  # Orange-Red - LoV D
            (1.000, 0.647, 0.000, alpha),  # Orange - LoV C
            (1.000, 1.000, 0.000, alpha),  # Yellow - LoV B
            (0.678, 1.000, 0.184, alpha)   # Green-Yellow - LoV A
        ]
        cmap = ListedColormap(colors)
        # Set NaN values (no observations) to be transparent
        cmap.set_bad(color='white', alpha=0.0)  # Transparent white for NaN values
        bounds = [0, max_lov * 0.2, max_lov * 0.4, max_lov * 0.6, max_lov * 0.8, max_lov]
        norm = BoundaryNorm(bounds, cmap.N)
        
        # Create a grid for LoV visualization using provided coordinates
        # Create grid data array
        grid_data = np.full((len(x_coords), len(y_coords)), np.nan)
        
        # Map LoV values to visualization grid
        if len(lov_data) == len(x_coords) * len(y_coords):
            # Data is already aggregated to visualization grid
            grid_data = lov_data.reshape((len(x_coords), len(y_coords)))
        else:
            # Map original data points to visualization grid
            for i, (x, y, lov_val) in enumerate(zip(df['x_coord'].values, df['y_coord'].values, lov_data)):
                x_idx = np.argmin(np.abs(x_coords - x))
                y_idx = np.argmin(np.abs(y_coords - y))
                if 0 <= x_idx < len(x_coords) and 0 <= y_idx < len(y_coords):
                    grid_data[x_idx, y_idx] = lov_val
        
        # Plot LoV grid using imshow for smooth appearance
        visualization_grid_size = self.config['visualization_grid_size']
        extent = [x_coords[0] - x_min, x_coords[-1] - x_min + visualization_grid_size,
                 y_coords[0] - y_min, y_coords[-1] - y_min + visualization_grid_size]
        
        im = ax.imshow(grid_data.T, origin='lower', extent=tuple(extent), cmap=cmap, norm=norm,  # Convert list to tuple
                      alpha=alpha, interpolation='nearest')
        
        # Create legend
        legend_patches = [
            Patch(color=colors[4], label='LoV A'),
            Patch(color=colors[3], label='LoV B'),
            Patch(color=colors[2], label='LoV C'),
            Patch(color=colors[1], label='LoV D'),
            Patch(color=colors[0], label='LoV E')
        ]
        legend = ax.legend(handles=legend_patches, loc='upper right', title='Level of Visibility', fontsize=20, title_fontsize=20)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(1.0)
        legend.get_frame().set_edgecolor('black')
        
        # Set axis limits to match the full bounding box
        ax.set_xlim(0, x_max - x_min)
        ax.set_ylim(0, y_max - y_min)
        
        # Set labels (no title for cleaner appearance)
        ax.set_xlabel('Longitude [m]', fontsize=20)
        ax.set_ylabel('Latitude [m]', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=20)
        # ax.set_title('Level of Visibility (LoV) Heatmap')
        
        # Save figure
        output_prefix = f'FCO{self.config["FCO_share"]}%_FBO{self.config["FBO_share"]}%'
        output_filename = f'discrete_LoV_heatmap_{self.config["file_tag"]}_{output_prefix}.png'
        output_path = os.path.join(self.config['output_dir'], output_filename)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.config['dpi'], bbox_inches='tight')
        plt.close()

        # Save additional focus area map (if enabled)
        if ENABLE_FOCUS_AREA_OVERLAY and PLOT_FOCUS_AREA_HEATMAP:
            focus_bbox = self._compute_focus_area_bbox(df)

            if focus_bbox:
                print(f"  Creating focus area map...")
                fig, ax = plt.subplots(figsize=self.config['figure_size'], facecolor='white', dpi=self.config['dpi'])
                ax.set_facecolor('white')

                self._plot_background_layers(ax, geospatial_data, x_min, y_min, include_roads)

                im = ax.imshow(grid_data.T, origin='lower', extent=tuple(extent), cmap=cmap, norm=norm,
                              alpha=alpha, interpolation='nearest')

                legend = ax.legend(handles=legend_patches, loc='upper right', title='Level of Visibility', fontsize=20, title_fontsize=20)
                legend.get_frame().set_facecolor('white')
                legend.get_frame().set_alpha(1.0)
                legend.get_frame().set_edgecolor('black')

                ax.set_xlim(0, x_max - x_min)
                ax.set_ylim(0, y_max - y_min)
                ax.set_xlabel('Longitude [m]', fontsize=20)
                ax.set_ylabel('Latitude [m]', fontsize=20)
                ax.tick_params(axis='both', which='major', labelsize=20)

                self._overlay_focus_area(ax, focus_bbox, x_min, y_min)

                focus_output_filename = f'discrete_LoV_heatmap_focus_area_{self.config["file_tag"]}_{output_prefix}.png'
                focus_output_path = os.path.join(self.config['output_dir'], focus_output_filename)

                plt.tight_layout()
                plt.savefig(focus_output_path, dpi=self.config['dpi'], bbox_inches='tight')
                plt.close()
                
                print(f"✓ Saved focus area map: {focus_output_path}")
    
    # ====================================
    # CONTINUOUS LEVEL OF VISIBILITY METHODS
    # ====================================
    
    def calculate_continuous_lov_data(self, df):
        """
        Calculate continuous Level of Visibility (LoV) from weighted visibility counts.
        Uses sensor accuracy weighted counting (observer count affects values).
        
        Args:
            df: DataFrame with visibility data
            
        Returns:
            tuple: (lov_data, max_lov, logging_info)
        """
        # Calculate LoV for each data point
        lov_data = df['visibility_count'].values / (self.config['total_simulation_steps'] * self.config['step_length'])
        
        # Set LoV to NaN for cells with no visibility observations (count = 0)
        # This distinguishes between "never observed" (NaN) and "observed but poor visibility" (LoV E)
        lov_data = np.where(df['visibility_count'].values == 0, np.nan, lov_data)
        
        max_lov = 1 / self.config['step_length']
        
        # Calculate statistics only for cells with actual observations
        valid_lov = lov_data[~np.isnan(lov_data)]
        
        # Prepare logging information
        ssa_value = self.config.get('single_sensor_accuracy', 'Unknown')
        logging_info = [
            ['Visibility Type', f'Continuous (Weighted by {ssa_value}% Sensor Accuracy)'],
            ['Single Sensor Accuracy', f'{ssa_value}%'],
            ['Max. visibility count', np.max(df['visibility_count'].values)],
            ['Total simulation steps', self.config['total_simulation_steps']],
            ['Step Size', self.config['step_length']],
            ['LoV scale', f'0 - {max_lov}'],
            ['Max. LoV value', np.max(valid_lov) if len(valid_lov) > 0 else 0],
            ['Mean LoV value', np.mean(valid_lov) if len(valid_lov) > 0 else 0],
            ['Cells with observations', len(valid_lov)],
            ['Cells without observations', np.sum(np.isnan(lov_data))]
        ]
        
        return lov_data, max_lov, logging_info
    
    def save_continuous_lov_logging_info(self, logging_info):
        """Save continuous LoV logging information to CSV file."""
        output_prefix = f'FCO{self.config["FCO_share"]}%_FBO{self.config["FBO_share"]}%'
        ssa_suffix = f'_SSA{self.config.get("single_sensor_accuracy", "Unknown")}%'
        log_filename = f'continuous_LoV_log_{self.config["file_tag"]}_{output_prefix}{ssa_suffix}.csv'
        log_path = os.path.join(self.config['output_dir'], log_filename)
        
        with open(log_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Description', 'Value'])
            csvwriter.writerows(logging_info)
    
    def aggregate_continuous_lov_data(self, df, lov_data, x_coords, y_coords, data_grid_size):
        """
        Aggregate continuous LoV data to coarser visualization grid if needed.
        
        Args:
            df: Original DataFrame with visibility data
            lov_data: Original LoV values array
            x_coords, y_coords: Visualization grid coordinates
            data_grid_size: Original data collection grid size
            
        Returns:
            tuple: (aggregated_lov_data, aggregated_counts) for visualization
        """
        visualization_grid_size = self.config['visualization_grid_size']
        
        if abs(visualization_grid_size - data_grid_size) < 0.001:
            # No aggregation needed - grid sizes match
            return lov_data, df['visibility_count'].values
        
        # Aggregate to coarser grid
        
        # Get original coordinates
        x_coords_data = np.sort(df['x_coord'].unique())
        y_coords_data = np.sort(df['y_coord'].unique())
        
        # Initialize aggregation arrays
        aggregated_lov = np.zeros(len(x_coords) * len(y_coords))
        aggregated_counts = np.zeros(len(x_coords) * len(y_coords))
        weighted_sums = np.zeros(len(x_coords) * len(y_coords))
        
        # Process each original data point
        for i, (x_orig, y_orig) in enumerate(zip(df['x_coord'].values, df['y_coord'].values)):
            # Find which visualization grid cell this belongs to
            x_idx = np.digitize(x_orig, x_coords) - 1
            y_idx = np.digitize(y_orig, y_coords) - 1
            
            # Ensure indices are within bounds
            if 0 <= x_idx < len(x_coords) and 0 <= y_idx < len(y_coords):
                flat_idx = x_idx * len(y_coords) + y_idx
                
                # Skip NaN values in aggregation
                if not np.isnan(lov_data[i]):
                    weighted_sums[flat_idx] += lov_data[i] * df['visibility_count'].iloc[i]
                    aggregated_counts[flat_idx] += df['visibility_count'].iloc[i]
        
        # Calculate weighted average LoV values
        with np.errstate(divide='ignore', invalid='ignore'):
            aggregated_lov = np.where(aggregated_counts > 0, weighted_sums / aggregated_counts, np.nan)
        
        
        return aggregated_lov, aggregated_counts
    
    def plot_continuous_lov_heatmap(self, df, lov_data, max_lov, x_coords, y_coords, data_grid_size, geospatial_data):
        """
        Create and save the continuous LoV heatmap visualization.
        
        Args:
            df: DataFrame with visibility data  
            lov_data: Calculated continuous LoV values (possibly aggregated)
            max_lov: Maximum LoV value
            x_coords, y_coords: Visualization grid coordinates
            data_grid_size: Original data collection grid size
            geospatial_data: Dictionary with projected geospatial data
        """
        # Create figure
        fig, ax = plt.subplots(figsize=self.config['figure_size'], facecolor='white', dpi=self.config['dpi'])
        ax.set_facecolor('white')
        
        # Calculate bounds for translation (using full bounding box for scene extent)
        north, south, east, west = self.config['bbox']
        x_min, y_min = self.project(west, south)
        x_max, y_max = self.project(east, north)
        
        # Plot geospatial background layers covering the full bounding box
        include_roads = self.config['include_roads']
        self._plot_background_layers(ax, geospatial_data, x_min, y_min, include_roads)  # Include roads for LoV heatmap
        
        # Create LoV color map (same as main.py)
        alpha = self.config['alpha']
        colors = [
            (0.698, 0.133, 0.133, alpha),  # Firebrick - LoV E
            (1.000, 0.271, 0.000, alpha),  # Orange-Red - LoV D
            (1.000, 0.647, 0.000, alpha),  # Orange - LoV C
            (1.000, 1.000, 0.000, alpha),  # Yellow - LoV B
            (0.678, 1.000, 0.184, alpha)   # Green-Yellow - LoV A
        ]
        cmap = ListedColormap(colors)
        # Set NaN values (no observations) to be transparent
        cmap.set_bad(color='white', alpha=0.0)  # Transparent white for NaN values
        bounds = [0, max_lov * 0.2, max_lov * 0.4, max_lov * 0.6, max_lov * 0.8, max_lov]
        norm = BoundaryNorm(bounds, cmap.N)
        
        # Create a grid for LoV visualization using provided coordinates
        # Create grid data array
        grid_data = np.full((len(x_coords), len(y_coords)), np.nan)
        
        # Map LoV values to visualization grid
        if len(lov_data) == len(x_coords) * len(y_coords):
            # Data is already aggregated to visualization grid
            grid_data = lov_data.reshape((len(x_coords), len(y_coords)))
        else:
            # Map original data points to visualization grid
            for i, (x, y, lov_val) in enumerate(zip(df['x_coord'].values, df['y_coord'].values, lov_data)):
                x_idx = np.argmin(np.abs(x_coords - x))
                y_idx = np.argmin(np.abs(y_coords - y))
                if 0 <= x_idx < len(x_coords) and 0 <= y_idx < len(y_coords):
                    grid_data[x_idx, y_idx] = lov_val
        
        # Plot LoV grid using imshow for smooth appearance
        visualization_grid_size = self.config['visualization_grid_size']
        extent = [x_coords[0] - x_min, x_coords[-1] - x_min + visualization_grid_size,
                 y_coords[0] - y_min, y_coords[-1] - y_min + visualization_grid_size]
        
        im = ax.imshow(grid_data.T, origin='lower', extent=tuple(extent), cmap=cmap, norm=norm,  # Convert list to tuple
                      alpha=alpha, interpolation='nearest')
        
        # Create legend
        legend_patches = [
            Patch(color=colors[4], label='LoPP A'),
            Patch(color=colors[3], label='LoPP B'),
            Patch(color=colors[2], label='LoPP C'),
            Patch(color=colors[1], label='LoPP D'),
            Patch(color=colors[0], label='LoPP E')
        ]
        legend = ax.legend(handles=legend_patches, loc='upper right', title='Level of\nPerception Potential', fontsize=20, title_fontsize=20)
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_alpha(1.0)
        legend.get_frame().set_edgecolor('black')
        # Center the legend title
        if legend.get_title() is not None:
            legend.get_title().set_ha('center')
        
        # Set axis limits to match the full bounding box
        ax.set_xlim(0, x_max - x_min)
        ax.set_ylim(0, y_max - y_min)
        
        # Set labels (no title for cleaner appearance)
        ax.set_xlabel('Longitude [m]', fontsize=20)
        ax.set_ylabel('Latitude [m]', fontsize=20)
        ax.tick_params(axis='both', which='major', labelsize=20)
        # ax.set_title('Continuous Level of Visibility (LoV) Heatmap')
        
        # Save figure
        output_prefix = f'FCO{self.config["FCO_share"]}%_FBO{self.config["FBO_share"]}%'
        ssa_suffix = f'_SSA{self.config.get("single_sensor_accuracy", "Unknown")}%'
        output_filename = f'continuous_LoV_heatmap_{self.config["file_tag"]}_{output_prefix}{ssa_suffix}.png'
        output_path = os.path.join(self.config['output_dir'], output_filename)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.config['dpi'], bbox_inches='tight')
        plt.close()
    
    # =============================
    # MAIN ANALYSIS METHOD
    # =============================
    
    def analyze_spatial_visibility(self):
        """
        Main method to perform spatial visibility analysis and generate heatmaps.
        """
        print("=== Initialization ===")
        
        # Print auto-detected/configured parameters
        if self.config.get('file_tag'):
            fco = self.config['FCO_share']
            fbo = self.config['FBO_share']
            step = self.config.get('step_length', 'N/A')
            print(f"✓ Auto-detected parameters: {self.config['file_tag']}, FCO: {fco}%, FBO: {fbo}%, step length: {step}s")
        
        print(f"✓ Output directory: {self.config['output_dir']}")
        
        # Print data sources
        if self.config.get('discrete_visibility_csv_path'):
            print(f"✓ Loaded discrete visibility data")
        if self.config.get('continuous_visibility_csv_path'):
            ssa = self.config.get('single_sensor_accuracy', 'N/A')
            print(f"✓ Loaded continuous visibility data (SSA: {ssa}%)")
        if self.config.get('geojson_path'):
            print(f"✓ Loaded road geometry from GeoJSON")
        
        # Load geospatial data early to show OSM loading in Initialization
        geospatial_data = self.load_geospatial_data()
        
        print("\n=== Spatial Visibility Analysis ===")
        print("Analysis types:")
        print(f"  - Infrastructure Classification: {'ENABLED' if INFRASTRUCTURE_CLASSIFICATION else 'DISABLED'}")
        print(f"  - Infrastructure Classification Map: {'ENABLED' if INFRASTRUCTURE_CLASSIFICATION_MAP else 'DISABLED'}")
        print(f"  - Relative Visibility: {'ENABLED' if RELATIVE_VISIBILITY else 'DISABLED'}")
        print(f"  - Discrete LoV: {'ENABLED' if DISCRETE_LOV else 'DISABLED'}")
        print(f"  - Continuous LoV: {'ENABLED' if CONTINUOUS_LOV else 'DISABLED'}")
        
        if not INFRASTRUCTURE_CLASSIFICATION and not RELATIVE_VISIBILITY and not DISCRETE_LOV and not CONTINUOUS_LOV:
            print("No analysis enabled! Please set INFRASTRUCTURE_CLASSIFICATION, RELATIVE_VISIBILITY, DISCRETE_LOV, and/or CONTINUOUS_LOV to True.")
            return
        
        try:
            # Classify infrastructure types from SUMO network
            if INFRASTRUCTURE_CLASSIFICATION and 'scenario_output_path' in self.config and self.config['scenario_output_path']:
                print("\n=== Infrastructure Classification ===")
                infrastructure_types = self.classify_infrastructure_from_network(
                    self.config['scenario_output_path']
                )
                
                if infrastructure_types:
                    # Print statistics
                    type_counts = {}
                    for infra_type in infrastructure_types.values():
                        type_counts[infra_type] = type_counts.get(infra_type, 0) + 1
                    
                    print(f"\n  Infrastructure type distribution:")
                    total_cells = len(infrastructure_types)
                    for infra_type in sorted(type_counts.keys()):
                        count = type_counts[infra_type]
                        pct = (count / total_cells) * 100
                        print(f"    - {infra_type}: {count} cells ({pct:.1f}%)")
                    
                    # Save infrastructure classification to CSV
                    self._save_infrastructure_classification(infrastructure_types)
                    
                    # Generate visualization map if enabled
                    if INFRASTRUCTURE_CLASSIFICATION_MAP:
                        print("\n  Generating infrastructure classification map...")
                        self.plot_infrastructure_classification_map(infrastructure_types, geospatial_data)
                else:
                    print("  ⚠ Infrastructure classification skipped")
            
            # Perform Relative Visibility Analysis
            if RELATIVE_VISIBILITY:
                print("\n=== Processing Relative Visibility Heatmap ===")
                # Load discrete visibility data for relative visibility
                df_discrete = self.load_visibility_data(csv_type='discrete')
                
                # Check if we have valid data
                if self.config['FCO_share'] == 0 and self.config['FBO_share'] == 0:
                    print("Warning: No visibility data to plot: FCO and FBO penetration rates are both set to 0%.")
                    print("The heatmaps may not be meaningful.")
                
                # Create grid from data
                x_coords, y_coords, data_grid_size = self.create_grid_from_data(df_discrete)
                
                print(f"Data grid size: {data_grid_size:.3f} m")
                print(f"Visualization grid size: {self.config['visualization_grid_size']:.3f} m")
                
                heatmap_data = self.create_relative_visibility_heatmap_data(df_discrete, x_coords, y_coords, data_grid_size)
                
                self.plot_relative_visibility_heatmap(heatmap_data, x_coords, y_coords, geospatial_data)
                
                # Print save location
                output_prefix = f'FCO{self.config["FCO_share"]}%_FBO{self.config["FBO_share"]}%'
                output_filename = f'relVis_heatmap_{self.config["file_tag"]}_{output_prefix}.png'
                output_path = os.path.join(self.config['output_dir'], output_filename)
                print(f"✓ Saved heatmap: {output_path}")
            
            # Perform Discrete Level of Visibility Analysis
            if DISCRETE_LOV:
                print("\n=== Processing Discrete Level of Visibility ===")
                # Load discrete visibility data
                df_discrete = self.load_visibility_data(csv_type='discrete')
                
                # Create grid from data
                x_coords, y_coords, data_grid_size = self.create_grid_from_data(df_discrete)
                
                print(f"Data grid size: {data_grid_size:.3f} m")
                print(f"Visualization grid size: {self.config['visualization_grid_size']:.3f} m")
                
                # Calculate discrete LoV data
                lov_data, max_lov, logging_info = self.calculate_discrete_lov_data(df_discrete)
                
                # Extract LoV statistics from logging_info
                valid_lov = lov_data[~np.isnan(lov_data)]
                min_lov = np.min(valid_lov) if len(valid_lov) > 0 else 0
                max_lov_val = np.max(valid_lov) if len(valid_lov) > 0 else 0
                mean_lov = np.mean(valid_lov) if len(valid_lov) > 0 else 0
                lov_range = f"0 - {max_lov}"
                
                print(f"Discrete LoV Info:")
                print(f"  - LoV range: {lov_range}")
                print(f"  - Mean LoV: {mean_lov:.4f}")
                print(f"  - Max LoV: {max_lov_val:.4f}")
                print(f"  - Min LoV: {min_lov:.4f}")
                
                # Aggregate LoV data if using coarser visualization grid
                lov_data_viz, aggregated_counts = self.aggregate_discrete_lov_data(df_discrete, lov_data, x_coords, y_coords, data_grid_size)
                
                # Save logging information
                self.save_discrete_lov_logging_info(logging_info)
                
                # Generate visualization
                self.plot_discrete_lov_heatmap(df_discrete, lov_data_viz, max_lov, x_coords, y_coords, data_grid_size, geospatial_data)
                
                # Print save locations
                output_prefix = f'FCO{self.config["FCO_share"]}%_FBO{self.config["FBO_share"]}%'
                heatmap_filename = f'discrete_LoV_heatmap_{self.config["file_tag"]}_{output_prefix}.png'
                heatmap_path = os.path.join(self.config['output_dir'], heatmap_filename)
                log_filename = f'discrete_LoV_log_{self.config["file_tag"]}_{output_prefix}.csv'
                log_path = os.path.join(self.config['output_dir'], log_filename)
                print(f"✓ Saved heatmap: {heatmap_path}")
                print(f"✓ Saved log file: {log_path}")
            
            # Perform Continuous Level of Visibility Analysis
            if CONTINUOUS_LOV:
                print("\n=== Processing Continuous Level of Visibility ===")
                # Load continuous visibility data
                df_continuous = self.load_visibility_data(csv_type='continuous')
                
                # Create grid from data
                x_coords, y_coords, data_grid_size = self.create_grid_from_data(df_continuous)
                
                print(f"Data grid size: {data_grid_size:.3f} m")
                print(f"Visualization grid size: {self.config['visualization_grid_size']:.3f} m")
                
                # Calculate continuous LoV data
                lov_data, max_lov, logging_info = self.calculate_continuous_lov_data(df_continuous)
                
                # Extract LoV statistics from logging_info
                valid_lov = lov_data[~np.isnan(lov_data)]
                min_lov = np.min(valid_lov) if len(valid_lov) > 0 else 0
                max_lov_val = np.max(valid_lov) if len(valid_lov) > 0 else 0
                mean_lov = np.mean(valid_lov) if len(valid_lov) > 0 else 0
                lov_range = f"0 - {max_lov}"
                
                print(f"Continuous LoV Info:")
                print(f"  - LoV range: {lov_range}")
                print(f"  - Mean LoV: {mean_lov:.4f}")
                print(f"  - Max LoV: {max_lov_val:.4f}")
                print(f"  - Min LoV: {min_lov:.4f}")
                
                # Aggregate LoV data if using coarser visualization grid
                lov_data_viz, aggregated_counts = self.aggregate_continuous_lov_data(df_continuous, lov_data, x_coords, y_coords, data_grid_size)
                
                # Save logging information
                self.save_continuous_lov_logging_info(logging_info)
                
                # Generate visualization
                self.plot_continuous_lov_heatmap(df_continuous, lov_data_viz, max_lov, x_coords, y_coords, data_grid_size, geospatial_data)
                
                # Print save locations
                output_prefix = f'FCO{self.config["FCO_share"]}%_FBO{self.config["FBO_share"]}%'
                ssa_suffix = f'_SSA{self.config.get("single_sensor_accuracy", "Unknown")}%'
                heatmap_filename = f'continuous_LoV_heatmap_{self.config["file_tag"]}_{output_prefix}{ssa_suffix}.png'
                heatmap_path = os.path.join(self.config['output_dir'], heatmap_filename)
                log_filename = f'continuous_LoV_log_{self.config["file_tag"]}_{output_prefix}{ssa_suffix}.csv'
                log_path = os.path.join(self.config['output_dir'], log_filename)
                print(f"✓ Saved heatmap: {heatmap_path}")
                print(f"✓ Saved log file: {log_path}")
            
            print("\n=== Spatial Visibility Analysis completed successfully! ===")
            
        except Exception as e:
            print(f"Error during analysis: {e}")
            raise


def main():
    """Command line interface for spatial visibility analysis."""
    # Declare global variables at the start
    global RELATIVE_VISIBILITY, DISCRETE_LOV, CONTINUOUS_LOV
    
    parser = argparse.ArgumentParser(description='Generate spatial visibility heatmaps from CSV data')
    parser.add_argument('--config', help='Path to JSON configuration file')
    
    # Analysis selection
    parser.add_argument('--relative-visibility', action='store_true', default=RELATIVE_VISIBILITY, 
                       help='Enable relative visibility analysis')
    parser.add_argument('--discrete-lov', action='store_true', default=DISCRETE_LOV,
                       help='Enable discrete Level of Visibility analysis')
    parser.add_argument('--continuous-lov', action='store_true', default=CONTINUOUS_LOV,
                       help='Enable continuous Level of Visibility analysis')
    
    # Required parameters
    parser.add_argument('--csv-path', help='Path to visibility counts CSV file')
    parser.add_argument('--bbox', nargs=4, type=float, metavar=('N', 'S', 'E', 'W'),
                       help='Bounding box: north south east west')
    parser.add_argument('--fco-share', type=int, help='FCO penetration rate (0-100)')
    parser.add_argument('--fbo-share', type=int, help='FBO penetration rate (0-100)')
    
    # LoV-specific parameters
    parser.add_argument('--total-steps', type=int, help='Total simulation steps')
    parser.add_argument('--step-length', type=float, help='Simulation step length in seconds')
    
    # Optional parameters
    parser.add_argument('--geojson-path', help='Path to GeoJSON file for road geometry')
    parser.add_argument('--output-dir', default=None, help='Output directory')
    parser.add_argument('--grid-size', type=float, default=None, help='Visualization grid resolution in meters')
    parser.add_argument('--save-config', help='Save configuration to JSON file')
    
    args = parser.parse_args()
    
    # Update global analysis flags from command line arguments
    RELATIVE_VISIBILITY = args.relative_visibility
    DISCRETE_LOV = args.discrete_lov
    CONTINUOUS_LOV = args.continuous_lov
    
    # Determine if we're using built-in configuration or command line configuration
    using_builtin_config = SCENARIO_OUTPUT_PATH is not None
    
    # Validate required parameters only if not using built-in configuration
    if not using_builtin_config and not args.config and not all([args.csv_path, args.bbox, args.fco_share is not None, args.fbo_share is not None]):
        parser.error("Either --config file, built-in SCENARIO_OUTPUT_PATH, or all required parameters (--csv-path, --bbox, --fco-share, --fbo-share) must be provided")
    
    # Validate LoV-specific parameters if LoV is enabled and not using built-in config
    if (DISCRETE_LOV or CONTINUOUS_LOV) and not using_builtin_config and not args.config:
        if args.total_steps is None or args.step_length is None:
            parser.error("Level of Visibility analysis requires --total-steps and --step-length parameters")
    
    # Build configuration from arguments
    config_kwargs = {}
    if args.csv_path:
        config_kwargs['visibility_csv_path'] = args.csv_path
    if args.bbox:
        config_kwargs['bbox'] = args.bbox
    if args.fco_share is not None:
        config_kwargs['FCO_share'] = args.fco_share
    if args.fbo_share is not None:
        config_kwargs['FBO_share'] = args.fbo_share
    if args.total_steps:
        config_kwargs['total_simulation_steps'] = args.total_steps
    if args.step_length:
        config_kwargs['step_length'] = args.step_length
    if args.geojson_path:
        config_kwargs['geojson_path'] = args.geojson_path
    if args.output_dir:
        config_kwargs['output_dir'] = args.output_dir
    if args.grid_size:
        config_kwargs['visualization_grid_size'] = args.grid_size
    
    # Initialize analyzer
    try:
        analyzer = SpatialVisibilityAnalyzer(config_file=args.config, **config_kwargs)
    except Exception as e:
        print(f"Configuration error: {e}")
        return 1
    
    # Save configuration if requested
    if args.save_config:
        analyzer.save_config(args.save_config)
        print(f"Configuration saved to: {args.save_config}")
    
    # Perform analysis
    try:
        analyzer.analyze_spatial_visibility()
        return 0
    except Exception as e:
        print(f"Analysis failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
