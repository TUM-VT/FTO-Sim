"""
Spatial Visibility Analysis

This unified script generates both Relative Visibility and Level of Visibility (LoV) heatmaps 
from existing visibility count CSV files without needing to run the full ray tracing simulation.
"""

"""
Spatial Visibility Analysis

This unified script generates both Relative Visibility and Level of Visibility (LoV) heatmaps 
from existing visibility count CSV files without needing to run the full ray tracing simulation.
"""

# =============================================================================
# USER CONFIGURATION
# =============================================================================

# 1. PROJECT PATH - Set the path to your scenario output folder
SCENARIO_OUTPUT_PATH = "outputs/ex_singleFCO_FCO100%_FBO0%"  # Path to scenario output folder (set to None to use manual configuration)

# 2. ANALYSIS SELECTION - Choose which metrics to generate
RELATIVE_VISIBILITY = True   # Generate relative visibility heatmaps
LEVEL_OF_VISIBILITY = True   # Generate Level of Visibility (LoV) heatmaps

# 3. GRID AND DISPLAY SETTINGS
VISUALIZATION_GRID_SIZE = 0.2  # Grid resolution for heatmap visualization in meters (can be different than grid size of visibility counts)
COLORMAP = 'hot'              # Color scheme for relative visibility - perceptually uniform and colorblind-friendly
ALPHA = 0.6                   # Heatmap transparency (0.0-1.0)

# 4. VISUALIZATION OPTIONS - Configure what to include in the maps
INCLUDE_ROADS = True         # Display road network from GeoJSON
INCLUDE_BUILDINGS = True     # Display buildings from OpenStreetMap  
INCLUDE_PARKS = True         # Display parks from OpenStreetMap
INCLUDE_TREES = True         # Display trees from OpenStreetMap
INCLUDE_BARRIERS = False     # Display barriers from OpenStreetMap
INCLUDE_PT_SHELTERS = False  # Display public transport shelters

# =============================================================================
# OPTIONAL MANUAL CONFIGURATION (only needed if SCENARIO_OUTPUT_PATH = None)
# =============================================================================

# REQUIRED PARAMETERS (for manual configuration)
FILE_TAG = "ex_singleFCO"       # File tag used in the simulation
FCO_SHARE = 100                 # FCO penetration rate (0-100)
FBO_SHARE = 0                   # FBO penetration rate (0-100)
BOUNDING_BOX = [48.15050, 48.14905, 11.57100, 11.56790]  # Geographic bounds [north, south, east, west]

# SIMULATION PARAMETERS (for LoV calculation - will be auto-detected if available)
TOTAL_SIMULATION_STEPS = 2700   # Total simulation steps (fallback if not found in logs)
STEP_LENGTH = 0.1               # Simulation step length in seconds (fallback if not found in logs)

# OPTIONAL PATHS (fallbacks for manual configuration)
GEOJSON_PATH = "SUMO_example/SUMO_example.geojson"  # Path to GeoJSON file (set to None if not available)
OUTPUT_DIR = "outputs/ex_singleFCO_FCO100%_FBO0%/out_visibility"  # Output directory for heatmaps

# =============================================================================
# END OF USER CONFIGURATION
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import geopandas as gpd
import osmnx as ox
import os
import json
import argparse
import csv
from pathlib import Path
from pyproj import Proj, transform
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

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
    
    print(f"Auto-detecting parameters from: {scenario_path}")
    print(f"  - Parsed scenario: {file_tag}, FCO: {fco_share}%, FBO: {fbo_share}%")
    
    # 1. Try to extract from JSON log files in out_logging
    log_path = scenario_path / 'out_logging'
    if log_path.exists():
        print(f"  - Searching for parameters in log files...")
        for log_file in log_path.glob('*.json'):
            try:
                with open(log_file, 'r') as f:
                    config = json.load(f)
                    
                # Extract various parameter formats
                if 'bounding_box' in config:
                    bbox = config['bounding_box']
                    print(f"    ✓ Found bounding box in {log_file.name}")
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
    
    # 2. Try to extract simulation parameters from SUMO config files if available
    if total_steps is None or step_length is None:
        print(f"  - Searching for SUMO configuration files...")
        
        # Check common SUMO config locations
        sumo_config_paths = [
            scenario_path / 'simulation.sumocfg',
            scenario_path / f'{file_tag}.sumocfg',
            Path('Additionals') / 'small_example' / 'osm_small.sumocfg',
            Path('Additionals') / 'osm.sumocfg'
        ]
        
        for sumo_config in sumo_config_paths:
            if sumo_config.exists():
                try:
                    import xml.etree.ElementTree as ET
                    tree = ET.parse(sumo_config)
                    root = tree.getroot()
                    
                    # Extract time parameters
                    time_elem = root.find('.//time')
                    if time_elem is not None:
                        if total_steps is None and 'end' in time_elem.attrib:
                            end_time = float(time_elem.attrib['end'])
                            if step_length is not None:
                                total_steps = int(end_time / step_length)
                                print(f"    ✓ Calculated simulation steps from SUMO config: {total_steps}")
                            
                        if step_length is None and 'step-length' in time_elem.attrib:
                            step_length = float(time_elem.attrib['step-length'])
                            print(f"    ✓ Found step length in SUMO config: {step_length}s")
                    
                    break
                except Exception as e:
                    print(f"    ⚠ Could not parse SUMO config {sumo_config}: {e}")
                    continue
    
    # 3. Extract from main.py if still missing
    if total_steps is None or step_length is None or grid_size is None:
        print(f"  - Searching main.py for simulation parameters...")
        main_py_path = Path('Scripts/main.py')
        if main_py_path.exists():
            try:
                with open(main_py_path, 'r') as f:
                    content = f.read()
                    
                # Extract grid size
                if grid_size is None:
                    grid_match = re.search(r'grid_size\s*=\s*([0-9.]+)', content)
                    if grid_match:
                        grid_size = float(grid_match.group(1))
                        print(f"    ✓ Found grid size in main.py: {grid_size}m")
                
                # Extract step length (look for SUMO step-length parameter)
                if step_length is None:
                    step_match = re.search(r'step-length["\']?\s*[:\s=]\s*["\']?([0-9.]+)', content)
                    if step_match:
                        step_length = float(step_match.group(1))
                        print(f"    ✓ Found step length in main.py: {step_length}s")
            
            except Exception as e:
                print(f"    ⚠ Could not parse main.py: {e}")
    
    # Check for visibility CSV to validate scenario
    visibility_csv_path = scenario_path / 'out_visibility' / 'visibility_counts' / f'visibility_counts_{scenario_name}.csv'
    
    if not visibility_csv_path.exists():
        raise FileNotFoundError(f"Visibility CSV not found: {visibility_csv_path}")
    
    # Find GeoJSON path
    geojson_path = None
    potential_geojson = Path('SUMO_example/SUMO_example.geojson')
    if potential_geojson.exists():
        geojson_path = potential_geojson
    
    # Use fallbacks for missing parameters
    if bbox is None:
        bbox = BOUNDING_BOX
        print(f"  - Using fallback bounding box: {bbox}")
    if total_steps is None:
        total_steps = TOTAL_SIMULATION_STEPS
        print(f"  - Using fallback simulation steps: {total_steps}")
    if step_length is None:
        step_length = STEP_LENGTH  
        print(f"  - Using fallback step length: {step_length}s")
    if grid_size is None:
        grid_size = VISUALIZATION_GRID_SIZE
        print(f"  - Using fallback grid size: {grid_size}m")
    
    print(f"  ✓ Auto-detection completed successfully!")
    
    return {
        'file_tag': file_tag,
        'fco_share': fco_share,
        'fbo_share': fbo_share,
        'bbox': bbox,
        'visibility_csv_path': str(visibility_csv_path),
        'output_dir': str(scenario_path / 'out_visibility'),
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
                print(f"✓ Auto-detected parameters from: {SCENARIO_OUTPUT_PATH}")
                print(f"  - File tag: {auto_config['file_tag']}")
                print(f"  - FCO/FBO shares: {auto_config['fco_share']}%/{auto_config['fbo_share']}%")
                print(f"  - Bounding box: {auto_config['bbox']}")
                print(f"  - Grid size: {auto_config['grid_size']}")
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
                'visibility_csv_path': auto_config['visibility_csv_path'],
                'total_simulation_steps': auto_config['total_simulation_steps'],
                'step_length': auto_config['step_length'],
                
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
            visibility_csv_path = f"{scenario_dir}/out_visibility/visibility_counts/visibility_counts_{scenario_name}.csv"
            output_dir = f"{scenario_dir}/out_visibility"
            
            self.config = {
                # Required parameters
                'bbox': BOUNDING_BOX,
                'FCO_share': FCO_SHARE,
                'FBO_share': FBO_SHARE,
                'visibility_csv_path': visibility_csv_path,
                'total_simulation_steps': TOTAL_SIMULATION_STEPS,
                'step_length': STEP_LENGTH,
                
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
        proj_in = Proj(proj='latlong', datum='WGS84')
        proj_out = Proj(proj='utm', zone=32, datum='WGS84')
        x, y = transform(proj_in, proj_out, lon, lat)
        return x, y
    
    def load_visibility_data(self):
        """
        Load visibility count data from CSV file.
        
        Returns:
            pandas.DataFrame: Visibility data with columns [x_coord, y_coord, visibility_count]
        """
        print(f"Loading visibility data: {self.config['visibility_csv_path']}")
        df = pd.read_csv(self.config['visibility_csv_path'])
        
        # Validate required columns
        required_cols = ['x_coord', 'y_coord', 'visibility_count']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found in CSV")
        
        print(f"Loaded {len(df)} visibility data points")
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
        if self.config['geojson_path'] and os.path.exists(self.config['geojson_path']):
            print("Loading road geometry from GeoJSON...")
            gdf1 = gpd.read_file(self.config['geojson_path'])
            # Filter for relevant types
            gdf1 = gdf1[gdf1['Type'].isin(['Junction', 'LaneBoundary', 'Gate', 'Signal'])]
            data['roads'] = gdf1.to_crs("EPSG:32632")
        else:
            print("No road geometry data available")
            data['roads'] = None
        
        # Load OpenStreetMap data
        print("Loading OpenStreetMap data...")
        
        # Get buildings
        if self.config['include_buildings']:
            try:
                buildings = ox.features_from_bbox(bbox=bbox, tags={'building': True})
                data['buildings'] = buildings.to_crs("EPSG:32632")
                print(f"Loaded {len(buildings)} buildings")
            except Exception as e:
                print(f"Warning: Could not load buildings - {e}")
                data['buildings'] = None
        else:
            data['buildings'] = None
        
        # Get parks
        if self.config['include_parks']:
            try:
                parks = ox.features_from_bbox(bbox=bbox, tags={'leisure': ['park', 'garden']})
                data['parks'] = parks.to_crs("EPSG:32632")
                print(f"Loaded {len(parks)} parks")
            except Exception as e:
                print(f"Warning: Could not load parks - {e}")
                data['parks'] = None
        else:
            data['parks'] = None
        
        # Get trees
        if self.config['include_trees']:
            try:
                trees = ox.features_from_bbox(bbox=bbox, tags={'natural': 'tree'})
                data['trees'] = trees.to_crs("EPSG:32632")
                print(f"Loaded {len(trees)} trees")
            except Exception as e:
                print(f"Warning: Could not load trees - {e}")
                data['trees'] = None
        else:
            data['trees'] = None
        
        # Get barriers
        if self.config['include_barriers']:
            try:
                barriers = ox.features_from_bbox(bbox=bbox, tags={'barrier': True})
                data['barriers'] = barriers.to_crs("EPSG:32632")
                print(f"Loaded {len(barriers)} barriers")
            except Exception as e:
                print(f"Warning: Could not load barriers - {e}")
                data['barriers'] = None
        else:
            data['barriers'] = None
        
        # Get public transport shelters
        if self.config['include_pt_shelters']:
            try:
                pt_shelters = ox.features_from_bbox(bbox=bbox, tags={'shelter_type': 'public_transport'})
                data['pt_shelters'] = pt_shelters.to_crs("EPSG:32632")
                print(f"Loaded {len(pt_shelters)} PT shelters")
            except Exception as e:
                print(f"Warning: Could not load PT shelters - {e}")
                data['pt_shelters'] = None
        else:
            data['pt_shelters'] = None
        
        return data
    
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
        
        print(f"Data grid size: {data_grid_size:.3f} m")
        print(f"Visualization grid size: {visualization_grid_size:.3f} m")
        
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
        
        print(f"Grid dimensions: {len(x_coords)} x {len(y_coords)}")
        print(f"X range: {x_coords[0]:.1f} to {x_coords[-1]:.1f}")
        print(f"Y range: {y_coords[0]:.1f} to {y_coords[-1]:.1f}")
        
        return x_coords, y_coords, data_grid_size
    
    def _plot_background_layers(self, ax, data, x_min, y_min):
        """Plot all background geospatial layers."""
        
        # Plot road space distribution
        if data['roads'] is not None:
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
            print("Direct mapping: grid sizes match")
            for _, row in df.iterrows():
                # Find closest grid indices for each data point
                x_idx = np.argmin(np.abs(x_coords - row['x_coord']))
                y_idx = np.argmin(np.abs(y_coords - row['y_coord']))
                
                # Ensure indices are within bounds
                if 0 <= x_idx < len(x_coords) and 0 <= y_idx < len(y_coords):
                    heatmap_data[x_idx, y_idx] = row['visibility_count']
        else:
            # Aggregate data to coarser grid
            print("Aggregating data to coarser visualization grid")
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
        
        print(f"Relative visibility heatmap data created: {len(x_coords)}x{len(y_coords)} grid")
        print(f"Non-zero cells: {np.sum(~np.isnan(heatmap_data))}")
        
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
        
        print(f"Dynamic figure size: {final_width:.1f} x {final_height:.1f} inches (geographic aspect ratio: {aspect_ratio:.2f})")
        
        # Plot geospatial background layers covering the full bounding box
        self._plot_background_layers(ax, geospatial_data, x_min, y_min)
        
        # Plot heatmap with proper extent (heatmap data extent, not bounding box)
        visualization_grid_size = self.config['visualization_grid_size']
        extent = [x_coords[0], x_coords[-1] + visualization_grid_size, 
                 y_coords[0], y_coords[-1] + visualization_grid_size]
        extent_translated = [x - x_min for x in extent[:2]] + [y - y_min for y in extent[2:]]
        
        # Use configured colormap for better scientific visualization
        cmap = plt.get_cmap(self.config['colormap'])  # Use configured colormap
        cmap.set_bad(color='white', alpha=0.0)  # Set NaN values to transparent white
        
        cax = ax.imshow(heatmap_data.T, origin='lower', cmap=cmap, 
                       extent=extent_translated, alpha=self.config['alpha'])
        
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
        cbar.set_label('Relative Visibility', rotation=270, labelpad=20)
        
        # Set labels (no title for cleaner appearance)
        ax.set_xlabel('Distance [m]')
        ax.set_ylabel('Distance [m]')
        # ax.set_title('Relative Visibility Heatmap')
        
        # Save figure
        output_prefix = f'FCO{self.config["FCO_share"]}%_FBO{self.config["FBO_share"]}%'
        output_filename = f'relative_visibility_heatmap_{output_prefix}.png'
        output_path = os.path.join(self.config['output_dir'], output_filename)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.config['dpi'], bbox_inches='tight')
        plt.close()
        
        print(f"Relative visibility heatmap saved to: {output_path}")
    
    # =============================
    # LEVEL OF VISIBILITY METHODS
    # =============================
    
    def calculate_lov_data(self, df):
        """
        Calculate Level of Visibility (LoV) from visibility counts.
        
        Args:
            df: DataFrame with visibility data
            
        Returns:
            tuple: (lov_data, max_lov, logging_info)
        """
        print("Calculating Level of Visibility (LoV) values...")
        
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
            ['Max. visibility count', np.max(df['visibility_count'].values)],
            ['Total simulation steps', self.config['total_simulation_steps']],
            ['Step Size', self.config['step_length']],
            ['LoV scale', f'0 - {max_lov}'],
            ['Max. LoV value', np.max(valid_lov) if len(valid_lov) > 0 else 0],
            ['Mean LoV value', np.mean(valid_lov) if len(valid_lov) > 0 else 0],
            ['Cells with observations', len(valid_lov)],
            ['Cells without observations', np.sum(np.isnan(lov_data))]
        ]
        
        print(f"LoV calculation completed:")
        print(f"  - Max LoV: {np.max(valid_lov) if len(valid_lov) > 0 else 0:.4f}")
        print(f"  - Mean LoV: {np.mean(valid_lov) if len(valid_lov) > 0 else 0:.4f}")
        print(f"  - LoV scale: 0 - {max_lov}")
        print(f"  - Cells with observations: {len(valid_lov)}")
        print(f"  - Cells without observations: {np.sum(np.isnan(lov_data))}")
        
        return lov_data, max_lov, logging_info
    
    def save_lov_logging_info(self, logging_info):
        """Save LoV logging information to CSV file."""
        output_prefix = f'FCO{self.config["FCO_share"]}%_FBO{self.config["FBO_share"]}%'
        log_dir = os.path.join(self.config['output_dir'], 'LoV_logging')
        os.makedirs(log_dir, exist_ok=True)
        
        log_path = os.path.join(log_dir, f'log_LoV_{output_prefix}.csv')
        with open(log_path, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['Description', 'Value'])
            csvwriter.writerows(logging_info)
        
        print(f"LoV logging information saved to: {log_path}")
    
    def aggregate_lov_data(self, df, lov_data, x_coords, y_coords, data_grid_size):
        """
        Aggregate LoV data to coarser visualization grid if needed.
        
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
            print("No aggregation needed: grid sizes match")
            return lov_data, df['visibility_count'].values
        
        # Aggregate to coarser grid
        print("Aggregating LoV data to coarser visualization grid")
        
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
        
        print(f"Aggregated LoV data: {np.sum(~np.isnan(aggregated_lov))} non-zero cells from {len(df)} original points")
        
        return aggregated_lov, aggregated_counts
    
    def plot_lov_heatmap(self, df, lov_data, max_lov, x_coords, y_coords, data_grid_size, geospatial_data):
        """
        Create and save the LoV heatmap visualization.
        
        Args:
            df: DataFrame with visibility data  
            lov_data: Calculated LoV values (possibly aggregated)
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
        self._plot_background_layers(ax, geospatial_data, x_min, y_min)
        
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
        
        im = ax.imshow(grid_data.T, origin='lower', extent=extent, cmap=cmap, norm=norm, 
                      alpha=alpha, interpolation='nearest')
        
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
        
        # Set axis limits to match the full bounding box
        ax.set_xlim(0, x_max - x_min)
        ax.set_ylim(0, y_max - y_min)
        
        # Set labels (no title for cleaner appearance)
        ax.set_xlabel('Distance [m]')
        ax.set_ylabel('Distance [m]')
        # ax.set_title('Level of Visibility (LoV) Heatmap')
        
        # Save figure
        output_prefix = f'FCO{self.config["FCO_share"]}%_FBO{self.config["FBO_share"]}%'
        output_filename = f'LoV_heatmap_{output_prefix}.png'
        output_path = os.path.join(self.config['output_dir'], output_filename)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.config['dpi'], bbox_inches='tight')
        plt.close()
        
        print(f"LoV heatmap saved to: {output_path}")
    
    # =============================
    # MAIN ANALYSIS METHOD
    # =============================
    
    def analyze_spatial_visibility(self):
        """
        Main method to perform spatial visibility analysis and generate heatmaps.
        """
        print("=== Spatial Visibility Analysis ===")
        print(f"Configuration:")
        print(f"  - Bounding box: {self.config['bbox']}")
        print(f"  - Visualization grid size: {self.config['visualization_grid_size']} m")
        print(f"  - FCO/FBO penetration: {self.config['FCO_share']}%/{self.config['FBO_share']}%")
        print(f"  - Visibility data: {self.config['visibility_csv_path']}")
        if self.config['geojson_path']:
            print(f"  - Road geometry: {self.config['geojson_path']}")
        else:
            print(f"  - Road geometry: OpenStreetMap data only")
        print(f"  - Output directory: {self.config['output_dir']}")
        print(f"  - Analyses enabled:")
        print(f"    - Relative Visibility: {'✓' if RELATIVE_VISIBILITY else '✗'}")
        print(f"    - Level of Visibility: {'✓' if LEVEL_OF_VISIBILITY else '✗'}")
        if LEVEL_OF_VISIBILITY:
            print(f"  - Simulation parameters: {self.config['total_simulation_steps']} steps, {self.config['step_length']}s step length")
        print()
        
        if not RELATIVE_VISIBILITY and not LEVEL_OF_VISIBILITY:
            print("No analysis enabled! Please set RELATIVE_VISIBILITY and/or LEVEL_OF_VISIBILITY to True.")
            return
        
        try:
            # Load visibility data
            df = self.load_visibility_data()
            
            # Check if we have valid data
            if self.config['FCO_share'] == 0 and self.config['FBO_share'] == 0:
                print("Warning: No visibility data to plot: FCO and FBO penetration rates are both set to 0%.")
                print("The heatmaps may not be meaningful.")
            
            # Load geospatial data
            geospatial_data = self.load_geospatial_data()
            
            # Create grid from data
            x_coords, y_coords, data_grid_size = self.create_grid_from_data(df)
            
            # Perform Relative Visibility Analysis
            if RELATIVE_VISIBILITY:
                print("\n--- Relative Visibility Analysis ---")
                print("Processing relative visibility data...")
                heatmap_data = self.create_relative_visibility_heatmap_data(df, x_coords, y_coords, data_grid_size)
                
                print("Generating relative visibility heatmap...")
                self.plot_relative_visibility_heatmap(heatmap_data, x_coords, y_coords, geospatial_data)
                print("✓ Relative Visibility analysis completed")
            
            # Perform Level of Visibility Analysis
            if LEVEL_OF_VISIBILITY:
                print("\n--- Level of Visibility Analysis ---")
                # Calculate LoV data
                lov_data, max_lov, logging_info = self.calculate_lov_data(df)
                
                # Aggregate LoV data if using coarser visualization grid
                lov_data_viz, aggregated_counts = self.aggregate_lov_data(df, lov_data, x_coords, y_coords, data_grid_size)
                
                # Save logging information
                self.save_lov_logging_info(logging_info)
                
                # Generate visualization
                print("Generating LoV heatmap...")
                self.plot_lov_heatmap(df, lov_data_viz, max_lov, x_coords, y_coords, data_grid_size, geospatial_data)
                print("✓ Level of Visibility analysis completed")
            
            print("\n=== Spatial Visibility Analysis completed successfully! ===")
            
        except Exception as e:
            print(f"Error during analysis: {e}")
            raise


def main():
    """Command line interface for spatial visibility analysis."""
    # Declare global variables at the start
    global RELATIVE_VISIBILITY, LEVEL_OF_VISIBILITY
    
    parser = argparse.ArgumentParser(description='Generate spatial visibility heatmaps from CSV data')
    parser.add_argument('--config', help='Path to JSON configuration file')
    
    # Analysis selection
    parser.add_argument('--relative-visibility', action='store_true', default=RELATIVE_VISIBILITY, 
                       help='Enable relative visibility analysis')
    parser.add_argument('--level-of-visibility', action='store_true', default=LEVEL_OF_VISIBILITY,
                       help='Enable Level of Visibility analysis')
    
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
    LEVEL_OF_VISIBILITY = args.level_of_visibility
    
    # Determine if we're using built-in configuration or command line configuration
    using_builtin_config = SCENARIO_OUTPUT_PATH is not None
    
    # Validate required parameters only if not using built-in configuration
    if not using_builtin_config and not args.config and not all([args.csv_path, args.bbox, args.fco_share is not None, args.fbo_share is not None]):
        parser.error("Either --config file, built-in SCENARIO_OUTPUT_PATH, or all required parameters (--csv-path, --bbox, --fco-share, --fbo-share) must be provided")
    
    # Validate LoV-specific parameters if LoV is enabled and not using built-in config
    if LEVEL_OF_VISIBILITY and not using_builtin_config and not args.config:
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
