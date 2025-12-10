#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# type: ignore
# See copilot-instructions.md for agent guidance
"""
VRU-Specific Detection Analysis Script

This script generates individual bicycle trajectory plots based on logged simulation data,
showing detection status over time and space. It extracts data from CSV log files instead 
of running during the simulation loop, allowing for post-processing analysis.

Features:
- Individual bicycle trajectory space-time diagrams
- Detection status visualization (detected vs undetected segments)
- Traffic light state visualization
- Trajectory statistics and detection rates
- Auto-detection of simulation parameters from log files

Author: FTO-Sim Development Team
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
from matplotlib.lines import Line2D  # Import Line2D properly
from pathlib import Path
import argparse
from datetime import datetime
import json
import re
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
import pyproj
from shapely.geometry import Point, box, Polygon, MultiPolygon, LineString
import shapely.ops
import geopandas as gpd
import osmnx as ox

# =============================
# CONFIGURATION CONSTANTS
# =============================

# 1. FEATURE CONFIGURATION - Primary User Settings
# =============================

# 2D Detection Plots
INDIVIDUAL_2D_DETECTION_PLOTS = True      # Generate individual 2D detection plots
FLOW_BASED_2D_DETECTION_PLOTS = True      # Generate 2D flow-based detection plots

# 2D Detected Object Redundancy Plots
INDIVIDUAL_2D_DETECTION_REDUNDANCY_PLOTS = True     # Generate individual 2D detection-redundancy plots
FLOW_BASED_2D_DETECTION_REDUNDANCY_PLOTS = True     # Generate flow-based 2D detection-redundancy plots

# 2D Occlusion Level Plots
INDIVIDUAL_2D_OCCLUSION_PLOTS = True      # Generate individual 2D bicycle occlusion level plots
FLOW_BASED_2D_OCCLUSION_PLOTS = True     # Generate flow-based 2D occlusion level plots

# 2D Conflict Plotss
INDIVIDUAL_2D_CONFLICT_PLOTS = False        # Generate individual 2D conflict plots
FLOW_BASED_2D_CONFLICT_PLOTS = False        # Generate flow-based 2D conflict plots

# 2D Plot Configuration
ENABLE_TRAFFIC_LIGHTS = True               # Include traffic light states in 2D plots

# 3D Plots
INDIVIDUAL_3D_DETECTION_PLOTS = False      # Generate individual 3D detection plots showing bicycle and observer trajectories
INDIVIDUAL_3D_CONFLICT_PLOTS = False        # Generate individual 3D conflict plots showing bicycle and foe trajectories

# Statistics
ENABLE_STATISTICS = False                   # Generate trajectory statistics and detection rate summaries

# =============================

# 2. SCENARIO CONFIGURATION
SCENARIO_OUTPUT_PATH = "outputs/test-CDR_FCO50%_FBO0%"  # Path to scenario output folder (set to None to use manual configuration)

# 3. TRAJECTORY ANALYSIS SETTINGS  
MIN_SEGMENT_LENGTH = 3      # Minimum segment length for bicycle trajectory analysis (data points)
MAX_GAP_BRIDGE = 10         # Maximum number of undetected frames to bridge between detected segments
STEP_LENGTH = 0.1           # Simulation step length in seconds (fallback value)

# 4. PLOT SETTINGS
DPI = 300                   # Resolution for saved plots
FIGURE_SIZE = (12, 8)       # Figure size in inches for 2D plots
FIGURE_SIZE_3D = (12, 8)    # Figure size in inches for 3D plots (same as 2D)
LEGEND_LOCATION = 'upper right'  # Legend position in plots
AXIS_LABEL_FONTSIZE = 12    # Font size for axis labels

# 5. 3D VISUALIZATION SETTINGS (only relevant if INDIVIDUAL_3D_DETECTION_PLOTS = True)
VIEW_ELEVATION = 35         # 3D plot elevation angle (degrees)
VIEW_AZIMUTH = 270          # 3D plot azimuth angle (degrees)
Z_AXIS_SCALE_FACTOR = 2.0   # Scale factor for z-axis relative to x/y axes

# 6. VRU VEHICLE TYPES
VRU_VEHICLE_TYPES = ["bicycle", "DEFAULT_BIKETYPE", "floating_bike_observer"]

# 7. OBSERVER VEHICLE TYPES  
OBSERVER_VEHICLE_TYPES = ["floating_car_observer", "floating_bike_observer"]

# 8. OCCLUSION LEVEL SCALE
def _get_occlusion_level_scale():
    """Return occlusion level scale with thresholds for categorization.
    
    Returns:
        List of tuples (category_name, min_percentage, max_percentage)
    """
    return [
        ("no_occlusion", 0, 0),           # Exactly 0% occlusion
        ("low_occlusion", 1, 39),         # 1-39% occlusion
        ("partial_occlusion", 40, 79),    # 40-79% occlusion
        ("heavy_occlusion", 80, 100)      # 80-100% occlusion
    ]

# 9. OCCLUSION COLOR PALETTE
def _get_occlusion_color_palette():
    """Return color palette for occlusion level visualization.
    
    Colors are generated dynamically based on the RdYlGn_r colormap
    (Red-Yellow-Green reversed: green for low occlusion, red for high occlusion).
    """
    return {
        'no_occlusion': '#006400',        # Dark Green (0% occlusion)
        'low_occlusion': '#90EE90',       # Light Green (low occlusion)
        'partial_occlusion': '#FFD700',   # Gold/Yellow (medium occlusion)
        'heavy_occlusion': '#FF4500'      # Orange/Red (high occlusion)
    }

# 10. REDUNDANCY COLOR PALETTE
def _get_redundancy_color_palette():
    """Return color palette for redundancy visualization."""
    return {
        0: 'black',           # Undetected
        1: '#40E0D0',         # Turquoise (single observer)
        2: '#20B2AA',         # Light Sea Green
        3: '#4682B4',         # Steel Blue
        4: '#2E8B57',         # Sea Green
        5: '#006400'          # Dark Green (5+ observers)
    }


# =============================
# OPTIONAL MANUAL CONFIGURATION (only needed if SCENARIO_OUTPUT_PATH = None)
# =============================

# Manual configuration (used only if SCENARIO_OUTPUT_PATH is None)
MANUAL_SCENARIO_PATH = "outputs/test-CDR_FCO10%_FBO0%"
MANUAL_FILE_TAG = "test-CDR"
MANUAL_FCO_SHARE = 10  # FCO penetration percentage
MANUAL_FBO_SHARE = 0    # FBO penetration percentage
MANUAL_STEP_LENGTH = 0.1  # Simulation step length in seconds


class VRUDetectionAnalyzer:
    """
    Main class for analyzing VRU (bicycle) detection trajectories from logged simulation data.
    """
    
    def __init__(self, config_file=None, **kwargs):
        """
        Initialize the analyzer with configuration.
        
        Args:
            config_file: Optional path to JSON configuration file
            **kwargs: Configuration overrides
        """
        self.config = self._build_configuration(config_file, **kwargs)
        self._ensure_output_directories()
        
    def _build_configuration(self, config_file, **kwargs):
        """Build configuration from various sources."""
        if config_file and os.path.exists(config_file):
            # Load from JSON file
            with open(config_file, 'r') as f:
                config = json.load(f)
        else:
            # Build configuration from auto-detection or manual settings
            # Use auto-detection if:
            # 1. scenario_path is provided in kwargs (command-line argument), OR
            # 2. SCENARIO_OUTPUT_PATH is set and no scenario_path in kwargs
            if kwargs.get('scenario_path') or (SCENARIO_OUTPUT_PATH and not kwargs.get('scenario_path')):
                # If scenario_path provided in kwargs, use it for auto-detection
                if kwargs.get('scenario_path'):
                    # Temporarily set SCENARIO_OUTPUT_PATH for auto-detection
                    original_path = SCENARIO_OUTPUT_PATH
                    globals()['SCENARIO_OUTPUT_PATH'] = kwargs['scenario_path']
                    config = self._auto_detect_configuration()
                    globals()['SCENARIO_OUTPUT_PATH'] = original_path
                else:
                    # Use SCENARIO_OUTPUT_PATH for auto-detection
                    config = self._auto_detect_configuration()
            else:
                config = self._get_manual_configuration(**kwargs)
        
        # Apply any command-line overrides (but don't override auto-detected values unless explicitly set)
        for key, value in kwargs.items():
            # Only override if the key is not scenario_path (already handled) and value is explicitly set
            if key != 'scenario_path' and value is not None:
                config[key] = value
                
        return config
    
    def _auto_detect_configuration(self):
        """Auto-detect configuration from scenario output directory."""
        # Make path relative to the script's parent directory, not current working directory
        script_dir = Path(__file__).parent  # Scripts directory
        project_dir = script_dir.parent      # FTO-Sim directory
        
        # Resolve the scenario path relative to project directory
        if SCENARIO_OUTPUT_PATH.startswith('../'):
            # Already relative to script dir, use as-is
            scenario_path = Path(SCENARIO_OUTPUT_PATH)
        elif SCENARIO_OUTPUT_PATH.startswith('outputs/'):
            # Make it relative to project directory
            scenario_path = project_dir / SCENARIO_OUTPUT_PATH
        else:
            # Absolute path or other format
            scenario_path = Path(SCENARIO_OUTPUT_PATH)
        
        if not scenario_path.exists():
            raise FileNotFoundError(f"Scenario output directory not found: {scenario_path}")
        
        # Parse scenario information from directory name
        scenario_name = scenario_path.name
        
        # Extract file tag and FCO/FBO shares from scenario name
        if '_FCO' in scenario_name and '_FBO' in scenario_name:
            # Extract file tag (part before _FCO)
            file_tag = scenario_name.split('_FCO')[0]
            
            # Extract FCO and FBO shares
            fco_match = re.search(r'FCO(\d+)%', scenario_name)
            fbo_match = re.search(r'FBO(\d+)%', scenario_name)
            
            fco_share = int(fco_match.group(1)) if fco_match else 100
            fbo_share = int(fbo_match.group(1)) if fbo_match else 0
        else:
            # Fallback parsing
            file_tag = scenario_name
            fco_share = 100
            fbo_share = 0
        
        # Auto-detect step length from log files
        step_length = self._detect_step_length(scenario_path)
        
        # Extract project name from scenario_path (last folder name)
        project_name = scenario_path.name
        
        # Create the VRU-specific detection output directory directly in scenario path
        vru_output_dir = scenario_path / 'out_VRU-specific_detection'
        
        return {
            'scenario_path': str(scenario_path),
            'project_name': project_name,
            'file_tag': file_tag,
            'fco_share': fco_share,
            'fbo_share': fbo_share,
            'step_length': step_length,
            'output_dir': str(vru_output_dir),
            'output_dir_3d': str(vru_output_dir),
            'output_dir_statistics': str(vru_output_dir),
            'min_segment_length': MIN_SEGMENT_LENGTH,
            'max_gap_bridge': MAX_GAP_BRIDGE,
            'dpi': DPI,
            'figure_size': FIGURE_SIZE,
            'figure_size_3d': FIGURE_SIZE_3D,
            'legend_location': LEGEND_LOCATION,
            'axis_label_fontsize': AXIS_LABEL_FONTSIZE,
            'individual_2d_detection_plots': INDIVIDUAL_2D_DETECTION_PLOTS,
            'flow_based_2d_detection_plots': FLOW_BASED_2D_DETECTION_PLOTS,
            'individual_2d_conflict_plots': INDIVIDUAL_2D_CONFLICT_PLOTS,
            'flow_based_2d_conflict_plots': FLOW_BASED_2D_CONFLICT_PLOTS,
            'individual_2d_detection_redundancy_plots': INDIVIDUAL_2D_DETECTION_REDUNDANCY_PLOTS,
            'flow_based_2d_detection_redundancy_plots': FLOW_BASED_2D_DETECTION_REDUNDANCY_PLOTS,
            'individual_2d_occlusion_plots': INDIVIDUAL_2D_OCCLUSION_PLOTS,
            'flow_based_2d_occlusion_plots': FLOW_BASED_2D_OCCLUSION_PLOTS,
            'individual_3d_detection_plots': INDIVIDUAL_3D_DETECTION_PLOTS,
            'individual_3d_conflict_plots': INDIVIDUAL_3D_CONFLICT_PLOTS,
            'show_3d_conflict_background': True,
            'enable_statistics': ENABLE_STATISTICS,
            'enable_traffic_lights': ENABLE_TRAFFIC_LIGHTS,
            'view_elevation': VIEW_ELEVATION,
            'view_azimuth': VIEW_AZIMUTH,
            'z_axis_scale_factor': Z_AXIS_SCALE_FACTOR,
            'occlusion_level_scale': _get_occlusion_level_scale()
        }
    
    def _get_manual_configuration(self, **kwargs):
        """Get manual configuration with overrides."""
        scenario_dir = kwargs.get('scenario_path', MANUAL_SCENARIO_PATH)
        
        # Extract project name from scenario_dir (last folder name)
        project_name = Path(scenario_dir).name
        
        # Create the VRU-specific detection output directory directly in scenario path
        vru_output_dir = Path(scenario_dir) / 'out_VRU-specific_detection'
        
        return {
            'scenario_path': scenario_dir,
            'project_name': project_name,
            'file_tag': kwargs.get('file_tag', MANUAL_FILE_TAG),
            'fco_share': kwargs.get('fco_share', MANUAL_FCO_SHARE),
            'fbo_share': kwargs.get('fbo_share', MANUAL_FBO_SHARE),
            'step_length': kwargs.get('step_length', MANUAL_STEP_LENGTH),
            'output_dir': kwargs.get('output_dir', str(vru_output_dir)),
            'output_dir_3d': kwargs.get('output_dir_3d', str(vru_output_dir)),
            'output_dir_statistics': kwargs.get('output_dir_statistics', str(vru_output_dir)),
            'min_segment_length': kwargs.get('min_segment_length', MIN_SEGMENT_LENGTH),
            'max_gap_bridge': kwargs.get('max_gap_bridge', MAX_GAP_BRIDGE),
            'dpi': kwargs.get('dpi', DPI),
            'legend_location': kwargs.get('legend_location', LEGEND_LOCATION),
            'axis_label_fontsize': kwargs.get('axis_label_fontsize', AXIS_LABEL_FONTSIZE),
            'figure_size': kwargs.get('figure_size', FIGURE_SIZE),
            'figure_size_3d': kwargs.get('figure_size_3d', FIGURE_SIZE_3D),
            'individual_2d_detection_plots': kwargs.get('individual_2d_detection_plots', INDIVIDUAL_2D_DETECTION_PLOTS),
            'flow_based_2d_detection_plots': kwargs.get('flow_based_2d_detection_plots', FLOW_BASED_2D_DETECTION_PLOTS),
            'individual_2d_conflict_plots': kwargs.get('individual_2d_conflict_plots', INDIVIDUAL_2D_CONFLICT_PLOTS),
            'flow_based_2d_conflict_plots': kwargs.get('flow_based_2d_conflict_plots', FLOW_BASED_2D_CONFLICT_PLOTS),
            'individual_2d_detection_redundancy_plots': kwargs.get('individual_2d_detection_redundancy_plots', INDIVIDUAL_2D_DETECTION_REDUNDANCY_PLOTS),
            'flow_based_2d_detection_redundancy_plots': kwargs.get('flow_based_2d_detection_redundancy_plots', FLOW_BASED_2D_DETECTION_REDUNDANCY_PLOTS),
            'individual_2d_occlusion_plots': kwargs.get('individual_2d_occlusion_plots', INDIVIDUAL_2D_OCCLUSION_PLOTS),
            'flow_based_2d_occlusion_plots': kwargs.get('flow_based_2d_occlusion_plots', FLOW_BASED_2D_OCCLUSION_PLOTS),
            'individual_3d_detection_plots': kwargs.get('individual_3d_detection_plots', INDIVIDUAL_3D_DETECTION_PLOTS),
            'individual_3d_conflict_plots': kwargs.get('individual_3d_conflict_plots', INDIVIDUAL_3D_CONFLICT_PLOTS),
            'show_3d_conflict_background': kwargs.get('show_3d_conflict_background', True),
            'enable_statistics': kwargs.get('enable_statistics', ENABLE_STATISTICS),
            'enable_traffic_lights': kwargs.get('enable_traffic_lights', ENABLE_TRAFFIC_LIGHTS),
            'view_elevation': kwargs.get('view_elevation', VIEW_ELEVATION),
            'view_azimuth': kwargs.get('view_azimuth', VIEW_AZIMUTH),
            'z_axis_scale_factor': kwargs.get('z_axis_scale_factor', Z_AXIS_SCALE_FACTOR),
            'occlusion_level_scale': kwargs.get('occlusion_level_scale', _get_occlusion_level_scale())
        }
    
    def _detect_step_length(self, scenario_path):
        """Detect simulation step length from trajectory log file header."""
        # Try to read step length from bicycle trajectory log header
        trajectory_file = scenario_path / 'out_logging' / f'log_bicycle_trajectories_{scenario_path.name}.csv'
        
        if trajectory_file.exists():
            try:
                with open(trajectory_file, 'r') as f:
                    # Read first 20 lines to find step length in header
                    for i, line in enumerate(f):
                        if i > 20:  # Stop after 20 lines
                            break
                        if '# Step length:' in line or '#Step length:' in line:
                            # Extract step length value (e.g., "# Step length: 0.1 seconds")
                            parts = line.split(':')
                            if len(parts) >= 2:
                                step_str = parts[1].split('seconds')[0].strip()
                                step_length = float(step_str)
                                return step_length
            except Exception:
                pass  # Silent fallback
        
        # Fallback to default
        return STEP_LENGTH

    def _detect_bounding_box(self, scenario_path):
        """Detect bounding box dimensions from log files."""
        # Try to find bounding box in summary log
        # First try with the exact folder name
        summary_log = scenario_path / 'out_logging' / f'summary_log_{scenario_path.name}.csv'
        
        # If that doesn't exist, look for any summary log file in the directory
        if not summary_log.exists():
            log_dir = scenario_path / 'out_logging'
            if log_dir.exists():
                # Find any file matching the summary_log pattern
                summary_files = list(log_dir.glob('summary_log_*.csv'))
                if summary_files:
                    summary_log = summary_files[0]  # Use the first one found
                else:
                    summary_log = None
            else:
                summary_log = None
        
        if summary_log and summary_log.exists():
            try:
                bbox_data = {}
                with open(summary_log, 'r') as f:
                    for line in f:
                        # Look for bounding box parameters in the CSV data
                        if 'Bounding box (north),' in line:
                            bbox_data['north'] = float(line.split(',')[1].strip())
                        elif 'Bounding box (south),' in line:
                            bbox_data['south'] = float(line.split(',')[1].strip())
                        elif 'Bounding box (east),' in line:
                            bbox_data['east'] = float(line.split(',')[1].strip())
                        elif 'Bounding box (west),' in line:
                            bbox_data['west'] = float(line.split(',')[1].strip())
                
                # Check if we have all required bounding box data
                if all(key in bbox_data for key in ['north', 'south', 'east', 'west']):
                    bbox = (bbox_data['north'], bbox_data['south'], bbox_data['east'], bbox_data['west'])
                    return bbox
                else:
                    missing_keys = [key for key in ['north', 'south', 'east', 'west'] if key not in bbox_data]
                    print(f"  ⚠ Missing bounding box parameters: {missing_keys}")
                    
            except Exception:
                pass  # Silent fallback
        
        # Fallback to hardcoded ETRR bounds
        bbox = (48.15050, 48.14905, 11.57100, 11.56790)
        return bbox
    
    def _ensure_output_directories(self):
        """Create output directory if it doesn't exist."""
        # Since all outputs go to the same directory now, just create once
        output_dir = self.config['output_dir']
        os.makedirs(output_dir, exist_ok=True)
    
    def load_trajectory_data(self):
        """Load bicycle trajectory data from CSV log file."""
        trajectory_file = Path(self.config['scenario_path']) / 'out_logging' / f'log_bicycle_trajectories_{Path(self.config["scenario_path"]).name}.csv'
        
        if not trajectory_file.exists():
            raise FileNotFoundError(f"Bicycle trajectory log file not found: {trajectory_file}")
        
        # Try reading without comment parameter to see if that's the issue
        with open(trajectory_file, 'r') as f:
            lines = f.readlines()
        
        # Find the first non-comment line (header)
        header_line_idx = None
        for i, line in enumerate(lines):
            if not line.strip().startswith('#') and line.strip():
                header_line_idx = i
                break
        
        if header_line_idx is None:
            raise ValueError("Could not find header line in CSV file")
        
        # Read CSV starting from the header line
        df = pd.read_csv(trajectory_file, skiprows=header_line_idx, na_values=[''], keep_default_na=False)
        
        # Filter for VRU vehicle types only
        df = df[df['vehicle_type'].isin(VRU_VEHICLE_TYPES)]
        
        print(f"✓ Loaded bicycle trajectory data ({len(df)} points, {df['vehicle_id'].nunique()} bicycles)")
        
        return df
    
    def load_detection_data(self):
        """Load bicycle detection data from CSV log file."""
        detection_file = Path(self.config['scenario_path']) / 'out_logging' / f'log_detections_{Path(self.config["scenario_path"]).name}.csv'
        
        if not detection_file.exists():
            print("⚠ No detection log file found - trajectories will show as undetected")
            return pd.DataFrame()
        
        # Read CSV, skipping comment lines
        df = pd.read_csv(detection_file, comment='#')
        
        print(f"✓ Loaded detection data ({len(df)} events)")
        
        return df
    
    def load_observer_trajectories(self):
        """Load observer vehicle trajectory data from CSV log file."""
        trajectory_file = Path(self.config['scenario_path']) / 'out_logging' / f'log_vehicle_trajectories_{Path(self.config["scenario_path"]).name}.csv'
        
        if not trajectory_file.exists():
            return pd.DataFrame()
        
        # Read CSV, skipping comment lines
        df = pd.read_csv(trajectory_file, comment='#')
        
        # Filter for observer vehicle types only
        df = df[df['vehicle_type'].isin(OBSERVER_VEHICLE_TYPES)]
        
        return df
    
    def load_foe_trajectories(self):
        """Load foe vehicle trajectories by identifying foes from conflict log.
        
        Strategy:
        1. Read conflict log to identify which vehicles are foes
        2. Load vehicle trajectory file
        3. Filter to only foe vehicle trajectories
        
        Returns:
            DataFrame with trajectories of all vehicles that appear as foes in conflicts
        """
        scenario_path = Path(self.config['scenario_path'])
        
        # Step 1: Load conflict log to identify foe vehicles
        conflict_file = scenario_path / 'out_logging' / f'log_conflicts_{scenario_path.name}.csv'
        
        if not conflict_file.exists():
            print("⚠ No conflict log file found - cannot identify foe vehicles")
            return pd.DataFrame()
        
        try:
            # Read conflict log
            with open(conflict_file, 'r') as f:
                lines = f.readlines()
            header_idx = next((i for i, l in enumerate(lines) if not l.strip().startswith('#') and l.strip()), 0)
            conflict_df = pd.read_csv(conflict_file, skiprows=header_idx)
            
            # Extract unique foe IDs
            if 'foe_id' not in conflict_df.columns:
                print("⚠ Conflict log missing 'foe_id' column")
                return pd.DataFrame()
            
            foe_ids = conflict_df['foe_id'].unique()
            
        except Exception as e:
            print(f"⚠ Error reading conflict log: {e}")
            return pd.DataFrame()
        
        # Step 2: Load vehicle trajectory file
        trajectory_file = scenario_path / 'out_logging' / f'log_vehicle_trajectories_{scenario_path.name}.csv'
        
        if not trajectory_file.exists():
            # Fallback: try bicycle trajectory file (foes might be bicycles)
            trajectory_file = scenario_path / 'out_logging' / f'log_bicycle_trajectories_{scenario_path.name}.csv'
            
            if not trajectory_file.exists():
                print("⚠ No vehicle trajectory log file found for foe vehicles")
                return pd.DataFrame()
        
        try:
            # Read trajectory CSV, skipping comment lines
            with open(trajectory_file, 'r') as f:
                lines = f.readlines()
            
            # Find the first non-comment line (header)
            header_line_idx = None
            for i, line in enumerate(lines):
                if not line.strip().startswith('#') and line.strip():
                    header_line_idx = i
                    break
            
            if header_line_idx is None:
                raise ValueError("Could not find header line in vehicle trajectory CSV file")
            
            # Read CSV starting from the header line
            all_trajectories = pd.read_csv(trajectory_file, skiprows=header_line_idx, na_values=[''], keep_default_na=False)
            
            # Step 3: Filter to only foe vehicles
            foe_trajectories = all_trajectories[all_trajectories['vehicle_id'].isin(foe_ids)]
            
            if foe_trajectories.empty:
                print("⚠ No trajectories found for identified foe vehicles")
                return pd.DataFrame()
            
            return foe_trajectories
            
        except Exception as e:
            print(f"⚠ Error loading vehicle trajectories: {e}")
            return pd.DataFrame()
    
    def load_geometry_data(self):
        """Load comprehensive geometry data for 3D visualization background using OSM data extraction."""
        import osmnx as ox
        
        # Get the bounding box from the summary log file
        scenario_path = Path(self.config['scenario_path'])
        bbox = self._detect_bounding_box(scenario_path)
        
        try:
            # Initialize coordinate transformer
            transformer = pyproj.Transformer.from_crs('EPSG:4326', 'EPSG:32632', always_xy=True)
            
            # Load road network (same as in main.py)
            try:
                G = ox.graph_from_bbox(bbox=bbox, network_type='all', simplify=True, retain_all=True)
                gdf1 = ox.graph_to_gdfs(G, nodes=False)  # road space distribution
                gdf1_proj = gdf1.to_crs("EPSG:32632")
            except Exception:
                gdf1_proj = None
            
            # Load buildings
            try:
                buildings = ox.features_from_bbox(bbox=bbox, tags={'building': True})
                buildings_proj = buildings.to_crs("EPSG:32632")
            except Exception:
                buildings_proj = None
                
            # Load parks
            try:
                parks = ox.features_from_bbox(bbox=bbox, tags={'leisure': 'park'})
                parks_proj = parks.to_crs("EPSG:32632")
            except Exception:
                parks_proj = None
                
            # Load trees
            try:
                trees = ox.features_from_bbox(bbox=bbox, tags={'natural': 'tree'})
                trees_proj = trees.to_crs("EPSG:32632")
                # Use same data for leaves (crown representation)
                leaves_proj = trees_proj
            except Exception:
                trees_proj = None
                leaves_proj = None
                
            # Load barriers
            try:
                barriers = ox.features_from_bbox(bbox=bbox, tags={'barrier': 'retaining_wall'})
                barriers_proj = barriers.to_crs("EPSG:32632")
            except Exception:
                barriers_proj = None
                
            # Load PT shelters
            try:
                PT_shelters = ox.features_from_bbox(bbox=bbox, tags={'shelter_type': 'public_transport'})
                PT_shelters_proj = PT_shelters.to_crs("EPSG:32632")
            except Exception:
                PT_shelters_proj = None
            
            # Summary
            element_counts = {
                'roads': len(gdf1_proj) if gdf1_proj is not None else 0,
                'buildings': len(buildings_proj) if buildings_proj is not None else 0,
                'parks': len(parks_proj) if parks_proj is not None else 0,
                'trees': len(trees_proj) if trees_proj is not None else 0,
                'barriers': len(barriers_proj) if barriers_proj is not None else 0,
                'pt_shelters': len(PT_shelters_proj) if PT_shelters_proj is not None else 0
            }
            
            return {
                'roads': gdf1_proj,
                'buildings': buildings_proj,
                'parks': parks_proj,
                'trees': trees_proj,
                'leaves': leaves_proj,
                'barriers': barriers_proj,
                'pt_shelters': PT_shelters_proj,
                'transformer': transformer,
                'bbox': bbox
            }
            
        except Exception as e:
            print(f"Warning: Failed to load comprehensive geometry data: {e}")
            print("  - Falling back to GeoJSON loading...")
            
            # Fallback to the original GeoJSON method
            return self._load_geojson_fallback()
    
    def _load_geojson_fallback(self):
        """Fallback method to load geometry from GeoJSON file."""
        scenario_name = Path(self.config['scenario_path']).name
        
        # Extract base scenario name (remove FCO/FBO suffixes)
        if '_FCO' in scenario_name:
            base_scenario = scenario_name.split('_FCO')[0]
        else:
            base_scenario = scenario_name
            
        # Look for GeoJSON file in simulation_examples
        script_dir = Path(__file__).parent
        parent_dir = script_dir.parent
        geojson_path = parent_dir / 'simulation_examples' / base_scenario / 'TUM_CentralCampus.geojson'
        
        # Fallback to standard location
        if not geojson_path.exists():
            geojson_path = parent_dir / 'simulation_examples' / 'ETRR_small_example' / 'TUM_CentralCampus.geojson'
        
        if not geojson_path.exists():
            print("Warning: No GeoJSON file found - 3D plots will not show background geometry")
            return None
        
        print(f"Loading geometry data from GeoJSON: {geojson_path}")
        
        try:
            # Load the GeoJSON data
            gdf = gpd.read_file(geojson_path)
            
            # Initialize coordinate transformer
            transformer = pyproj.Transformer.from_crs('EPSG:4326', 'EPSG:32632', always_xy=True)
            
            # Transform and separate different geometry types
            gdf_proj = gdf.to_crs('EPSG:32632')
            
            # Simple fallback - treat all as roads
            return {
                'roads': gdf_proj,
                'buildings': None,
                'parks': None,
                'trees': None,
                'leaves': None,
                'barriers': None,
                'pt_shelters': None,
                'transformer': transformer,
                'bbox': (48.15050, 48.14905, 11.57100, 11.56790)
            }
            
        except Exception as e:
            print(f"Warning: Failed to load GeoJSON fallback: {e}")
            return None
    
    def load_traffic_light_data(self):
        """Load traffic light data from CSV log file."""
        # Check if traffic light visualization is enabled
        if not self.config.get('enable_traffic_lights', True):
            return pd.DataFrame()
            
        tl_file = Path(self.config['scenario_path']) / 'out_logging' / f'log_traffic_lights_{Path(self.config["scenario_path"]).name}.csv'
        
        if not tl_file.exists():
            print("⚠ No traffic light log file found")
            return pd.DataFrame()
        
        # Read CSV, skipping comment lines
        df = pd.read_csv(tl_file, comment='#')
        
        print(f"✓ Loaded traffic light data ({len(df)} records)")
        
        return df
    
    def load_conflict_data(self):
        """Load conflict data from the scenario output directory."""
        
        if not self.config.get('enable_conflicts', True):
            return pd.DataFrame()
            
        conflict_file = Path(self.config['scenario_path']) / 'out_logging' / f'log_conflicts_{Path(self.config["scenario_path"]).name}.csv'
        
        if not conflict_file.exists():
            print("⚠ No conflict log file found")
            return pd.DataFrame()
        
        # Read CSV, skipping comment lines
        df = pd.read_csv(conflict_file, comment='#')
        
        # Validate expected columns (using actual column names from the log file)
        required_cols = ['time_step', 'bicycle_id', 'foe_id', 'ttc', 'pet', 'drac', 'x_coord', 'y_coord']
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            print(f"⚠ Conflict data missing columns: {missing_cols}")
            return pd.DataFrame()
        
        # Rename columns to match expected format in other methods
        # The log file only has bicycle position (x_coord, y_coord), not separate foe positions
        df = df.rename(columns={
            'ttc': 'TTC',
            'pet': 'PET', 
            'drac': 'DRAC',
            'x_coord': 'bicycle_x',
            'y_coord': 'bicycle_y'
        })
        
        # Add placeholder foe coordinates (not provided in the log file)
        # These won't be used for plotting since we only plot bicycle positions
        df['foe_x'] = 0.0
        df['foe_y'] = 0.0
        
        print(f"✓ Loaded conflict data ({len(df)} records)")
        
        return df
    
    def _identify_conflict_events(self, conflict_df):
        """Identify conflict events from consecutive conflict timesteps.
        
        A conflict event is defined as ≥2 consecutive timesteps with the same bicycle and foe.
        
        Args:
            conflict_df: DataFrame with conflict logs
            
        Returns:
            List of conflict event dictionaries
        """
        
        if conflict_df.empty:
            return []
        
        # Sort by bicycle_id, foe_id, time_step
        sorted_df = conflict_df.sort_values(['bicycle_id', 'foe_id', 'time_step']).reset_index(drop=True)
        
        events = []
        current_event_rows = []
        prev_bicycle = None
        prev_foe = None
        prev_time = None
        
        # Determine time step size from data (e.g., 0.1 seconds)
        if len(sorted_df) >= 2:
            time_diffs = sorted_df['time_step'].diff().dropna()
            step_size = time_diffs[time_diffs > 0].min() if len(time_diffs[time_diffs > 0]) > 0 else 0.1
        else:
            step_size = 0.1
        
        # Use a small tolerance for floating point comparison (half the step size)
        time_tolerance = step_size * 0.6
        
        for idx, row in sorted_df.iterrows():
            bicycle = row['bicycle_id']
            foe = row['foe_id']
            time = row['time_step']
            
            # Check if this row continues the current event
            if (bicycle == prev_bicycle and foe == prev_foe and 
                prev_time is not None and abs(time - (prev_time + step_size)) < time_tolerance):
                # Continue current event
                current_event_rows.append(row)
            else:
                # Finalize previous event if it meets criteria (≥2 consecutive steps)
                if len(current_event_rows) >= 2:
                    events.append(self._create_conflict_event(current_event_rows))
                
                # Start new event
                current_event_rows = [row]
            
            prev_bicycle = bicycle
            prev_foe = foe
            prev_time = time
        
        # Finalize last event
        if len(current_event_rows) >= 2:
            events.append(self._create_conflict_event(current_event_rows))
        
        print(f"✓ Identified {len(events)} conflict events from {len(conflict_df)} conflict records")
        
        return events
    
    def _create_conflict_event(self, event_rows):
        """Create a conflict event dictionary from consecutive conflict rows.
        
        Args:
            event_rows: List of DataFrame rows representing consecutive conflicts
            
        Returns:
            Dictionary with event metadata and representative point
        """
        
        # Find the point of maximum severity (minimum TTC if available, else use first point)
        event_df = pd.DataFrame(event_rows)
        
        # Determine representative point (max severity)
        if 'TTC' in event_df.columns and event_df['TTC'].notna().any():
            # Use minimum TTC as highest severity
            max_severity_idx = event_df['TTC'].idxmin()
        else:
            # Use first point as fallback
            max_severity_idx = event_df.index[0]
        
        rep_point = event_df.loc[max_severity_idx]
        
        # Determine dominant SSM
        dominant_ssm = self._get_dominant_ssm(event_df)
        
        event = {
            'bicycle_id': rep_point['bicycle_id'],
            'foe_id': rep_point['foe_id'],
            'time_step': rep_point['time_step'],
            'bicycle_x': rep_point['bicycle_x'],
            'bicycle_y': rep_point['bicycle_y'],
            'foe_x': rep_point['foe_x'],
            'foe_y': rep_point['foe_y'],
            'TTC': rep_point['TTC'],
            'PET': rep_point['PET'],
            'DRAC': rep_point['DRAC'],
            'dominant_ssm': dominant_ssm,
            'duration': len(event_rows),  # Number of consecutive timesteps
            'start_time': event_df['time_step'].min(),
            'end_time': event_df['time_step'].max()
        }
        
        return event
    
    def _get_dominant_ssm(self, conflict_df):
        """Determine the dominant SSM (TTC, PET, or DRAC) for a conflict event.
        
        Calculates the severity contribution from each SSM and returns the one
        with the highest contribution.
        
        Args:
            conflict_df: DataFrame with conflict measurements
            
        Returns:
            String: 'TTC', 'PET', or 'DRAC'
        """
        
        # Count valid (non-NaN, non-zero) measurements for each SSM
        ttc_count = conflict_df['TTC'].notna().sum()
        pet_count = conflict_df['PET'].notna().sum()
        drac_count = conflict_df['DRAC'].notna().sum()
        
        # Calculate average severity (lower is more severe for TTC/PET, higher for DRAC)
        # Normalize to make them comparable
        ttc_severity = 0
        pet_severity = 0
        drac_severity = 0
        
        if ttc_count > 0:
            # Lower TTC = higher severity; invert and normalize
            valid_ttc = conflict_df.loc[conflict_df['TTC'].notna(), 'TTC']
            ttc_severity = ttc_count / (valid_ttc.mean() + 0.001)  # Add small value to avoid div by zero
        
        if pet_count > 0:
            # Lower PET = higher severity; invert and normalize
            valid_pet = conflict_df.loc[conflict_df['PET'].notna(), 'PET']
            pet_severity = pet_count / (valid_pet.mean() + 0.001)
        
        if drac_count > 0:
            # Higher DRAC = higher severity
            valid_drac = conflict_df.loc[conflict_df['DRAC'].notna(), 'DRAC']
            drac_severity = drac_count * valid_drac.mean()
        
        # Return SSM with highest severity contribution
        severities = {'TTC': ttc_severity, 'PET': pet_severity, 'DRAC': drac_severity}
        dominant = max(severities, key=severities.get)
        
        return dominant
    
    def _calculate_conflict_detection_rates(self, conflict_events, detection_df):
        """Calculate detection rates for conflict events.
        
        Args:
            conflict_events: List of conflict event dictionaries
            detection_df: DataFrame with detection logs
            
        Returns:
            Dictionary with detection statistics
        """
        
        if not conflict_events:
            return {
                'total_events': 0,
                'temporal_detected': 0,
                'spatial_detected': 0,
                'spatiotemporal_detected': 0,
                'temporal_rate': 0.0,
                'spatial_rate': 0.0,
                'spatiotemporal_rate': 0.0
            }
        
        temporal_detected = 0
        spatial_detected = 0
        spatiotemporal_detected = 0
        
        for event in conflict_events:
            bicycle_id = event['bicycle_id']
            time_step = event['time_step']
            bicycle_x = event['bicycle_x']
            bicycle_y = event['bicycle_y']
            
            # Check temporal detection (bicycle detected at any location at this time)
            temporal_match = detection_df[
                (detection_df['bicycle_id'] == bicycle_id) &
                (detection_df['time_step'] == time_step)
            ]
            if not temporal_match.empty:
                temporal_detected += 1
            
            # Check spatial detection (bicycle detected at this location at any time)
            # Use 5m tolerance for spatial matching
            # Note: We can't do true spatial matching without coordinates in detection_df,
            # so we approximate by checking if the bicycle was detected at any nearby time
            spatial_tolerance_time = 1.0  # seconds
            spatial_match = detection_df[
                (detection_df['bicycle_id'] == bicycle_id) &
                (abs(detection_df['time_step'] - time_step) <= spatial_tolerance_time)
            ]
            if not spatial_match.empty:
                spatial_detected += 1
            
            # Check spatio-temporal detection (bicycle detected at this location AND time)
            # This is the same as temporal detection since we don't have spatial coords in detection_df
            spatiotemporal_match = detection_df[
                (detection_df['bicycle_id'] == bicycle_id) &
                (detection_df['time_step'] == time_step)
            ]
            if not spatiotemporal_match.empty:
                spatiotemporal_detected += 1
        
        total = len(conflict_events)
        
        return {
            'total_events': total,
            'temporal_detected': temporal_detected,
            'spatial_detected': spatial_detected,
            'spatiotemporal_detected': spatiotemporal_detected,
            'temporal_rate': temporal_detected / total if total > 0 else 0.0,
            'spatial_rate': spatial_detected / total if total > 0 else 0.0,
            'spatiotemporal_rate': spatiotemporal_detected / total if total > 0 else 0.0
        }
    
    def _calculate_conflict_detection_rates_improved(self, conflict_events, detection_df, bicycle_data):
        """Calculate detection rates for conflict events checking ALL time steps during conflicts.
        
        Uses the conflict log directly to ensure all conflict timesteps are counted.
        
        Temporal rate: % of ALL conflict time steps where bicycle was detected
        Spatial rate: % of ALL conflict distance where bicycle was detected
        Spatio-temporal rate: average of temporal and spatial
        
        Args:
            conflict_events: List of conflict event dictionaries
            detection_df: DataFrame with detection logs
            bicycle_data: DataFrame with full bicycle trajectory (for distance calculation)
            
        Returns:
            Dictionary with detection statistics
        """
        
        if not conflict_events:
            return {
                'total_events': 0,
                'temporal_rate': 0.0,
                'spatial_rate': 0.0,
                'spatiotemporal_rate': 0.0,
                'total_conflict_timesteps': 0,
                'detected_conflict_timesteps': 0,
                'total_conflict_distance': 0.0,
                'detected_conflict_distance': 0.0
            }
        
        bicycle_id = conflict_events[0]['bicycle_id']
        
        # Get ALL conflict timesteps from the original conflict DataFrame for this bicycle
        # This ensures we count every single conflict timestep, not just trajectory samples
        scenario_path = Path(self.config['scenario_path'])
        conflict_file = scenario_path / 'out_logging' / f'log_conflicts_{scenario_path.name}.csv'
        
        # Read conflict data directly
        with open(conflict_file, 'r') as f:
            lines = f.readlines()
        header_idx = next((i for i, l in enumerate(lines) if not l.strip().startswith('#') and l.strip()), 0)
        all_conflicts = pd.read_csv(conflict_file, skiprows=header_idx)
        
        # Filter for this bicycle's conflicts
        bicycle_conflicts = all_conflicts[all_conflicts['bicycle_id'] == bicycle_id].copy()
        
        if bicycle_conflicts.empty:
            return {
                'total_events': 0,
                'temporal_rate': 0.0,
                'spatial_rate': 0.0,
                'spatiotemporal_rate': 0.0,
                'total_conflict_timesteps': 0,
                'detected_conflict_timesteps': 0,
                'total_conflict_distance': 0.0,
                'detected_conflict_distance': 0.0
            }
        
        bicycle_detections = detection_df[detection_df['bicycle_id'] == bicycle_id].copy()
        
        # Initialize counters
        total_conflict_timesteps = 0
        detected_conflict_timesteps = 0
        total_conflict_distance = 0.0
        detected_conflict_distance = 0.0
        
        for event in conflict_events:
            start_time = event['start_time']
            end_time = event['end_time']
            foe_id = event['foe_id']
            
            # Get ALL conflict timesteps for this event from the conflict log
            event_conflicts = bicycle_conflicts[
                (bicycle_conflicts['time_step'] >= start_time) &
                (bicycle_conflicts['time_step'] <= end_time) &
                (bicycle_conflicts['foe_id'] == foe_id)  # Same foe as event
            ].copy().sort_values('time_step')
            
            if len(event_conflicts) == 0:
                continue
            
            # Temporal: Count every conflict timestep from conflict log
            for _, conflict_row in event_conflicts.iterrows():
                time_step = conflict_row['time_step']
                total_conflict_timesteps += 1
                
                # Check if bicycle was detected at this exact time step
                detected_at_timestep = bicycle_detections[
                    bicycle_detections['time_step'] == time_step
                ]
                
                if len(detected_at_timestep) > 0:
                    detected_conflict_timesteps += 1
            
            # Spatial: Calculate distance segment-by-segment based on detection at each timestep
            # Get trajectory points during this conflict event
            conflict_trajectory = bicycle_data[
                (bicycle_data['time_step'] >= start_time) &
                (bicycle_data['time_step'] <= end_time)
            ].copy().sort_values('time_step')
            
            if len(conflict_trajectory) < 2:
                continue
            
            # Mark which trajectory points were detected
            conflict_trajectory['detected'] = False
            for idx, row in conflict_trajectory.iterrows():
                time_step = row['time_step']
                detected_at_timestep = bicycle_detections[
                    bicycle_detections['time_step'] == time_step
                ]
                if len(detected_at_timestep) > 0:
                    conflict_trajectory.at[idx, 'detected'] = True
            
            # Calculate distance segment-by-segment
            # For each timestep, if detected, count the distance from PREVIOUS timestep to THIS one
            prev_distance = None
            for idx, row in conflict_trajectory.iterrows():
                current_distance = row['distance']
                
                if prev_distance is not None:
                    # Calculate segment distance
                    segment_distance = abs(current_distance - prev_distance)
                    total_conflict_distance += segment_distance
                    
                    # If current timestep is detected, count this segment as detected
                    if row['detected']:
                        detected_conflict_distance += segment_distance
                
                prev_distance = current_distance
        
        # Calculate rates
        temporal_rate = (detected_conflict_timesteps / total_conflict_timesteps * 100) if total_conflict_timesteps > 0 else 0.0
        spatial_rate = (detected_conflict_distance / total_conflict_distance * 100) if total_conflict_distance > 0 else 0.0
        spatiotemporal_rate = (temporal_rate + spatial_rate) / 2.0
        
        return {
            'total_events': len(conflict_events),
            'temporal_rate': temporal_rate,
            'spatial_rate': spatial_rate,
            'spatiotemporal_rate': spatiotemporal_rate,
            'total_conflict_timesteps': total_conflict_timesteps,
            'detected_conflict_timesteps': detected_conflict_timesteps,
            'total_conflict_distance': total_conflict_distance,
            'detected_conflict_distance': detected_conflict_distance
        }
    
    def _plot_individual_conflict_trajectory(self, bicycle_id, bicycle_data, conflict_events, 
                                             detection_df, traffic_light_df):
        """Generate individual conflict trajectory plot with conflict markers.
        
        Args:
            bicycle_id: ID of the bicycle
            bicycle_data: DataFrame with bicycle trajectory
            conflict_events: List of conflict event dictionaries for this bicycle
            detection_df: DataFrame with detection logs
            traffic_light_df: DataFrame with traffic light logs
        """
        
        # Prepare trajectory data (EXACTLY SAME AS _plot_individual_trajectory)
        bicycle_data = bicycle_data.sort_values('time_step').copy()
        
        # Get time range
        start_time_step = bicycle_data['time_step'].min()
        time_steps = bicycle_data['time_step'].values
        distances = bicycle_data['distance'].values
        
        # Create elapsed time (relative to bicycle start)
        elapsed_times = time_steps - start_time_step
        total_time = elapsed_times[-1] if len(elapsed_times) > 0 else 0
        
        # Get bicycle detections
        bicycle_detections = detection_df[detection_df['bicycle_id'] == bicycle_id]
        
        # Create detection timeline (same method as detection plots)
        detection_timeline = self._create_detection_timeline(time_steps, bicycle_detections, start_time_step)
        
        # Apply smoothing (same method as detection plots)
        detection_timeline = self._smooth_detection_timeline(detection_timeline)
        
        # Split into segments (same method as detection plots)
        segments = self._split_trajectory_segments(distances, elapsed_times, detection_timeline)
        
        # Get traffic light information (same method as detection plots)
        tl_info = self._get_bicycle_traffic_lights(bicycle_data, traffic_light_df)
        
        # Create figure (EXACTLY SAME AS _plot_individual_trajectory)
        fig, ax = plt.subplots(figsize=self.config['figure_size'])
        
        # Plot undetected segments (EXACTLY SAME)
        for segment in segments['undetected']:
            if len(segment) > 1:
                seg_distances, seg_times = zip(*segment)
                ax.plot(seg_times, seg_distances, color='black', linewidth=1.5, linestyle='solid')
        
        # Plot detected segments (EXACTLY SAME)
        for segment in segments['detected']:
            if len(segment) > 1:
                seg_distances, seg_times = zip(*segment)
                ax.plot(seg_times, seg_distances, color='darkturquoise', linewidth=1.5, linestyle='solid')
        
        # Plot traffic lights (EXACTLY SAME AS _plot_individual_trajectory)
        if tl_info:
            for tl_id, tl_data in tl_info.items():
                states = tl_data['states']
                avg_position = tl_data['avg_position']
                signal_index = tl_data['signal_index']
                
                # Plot horizontal line at traffic light position (dashed and thinner)
                ax.axhline(y=avg_position, color='black', linestyle='--', alpha=0.5, linewidth=0.5, zorder=1)
                
                # Plot state changes as colored segments on the horizontal line
                for i, state_change in enumerate(states):
                    signal_state = state_change['state']
                    
                    # Skip invalid states
                    if pd.isna(signal_state) or signal_state == '':
                        continue
                        
                    signal_state = str(signal_state).lower()
                    
                    # Map signal states to colors (including unknown)
                    if signal_state == 'unknown':
                        color = 'purple'
                    else:
                        color = {'r': 'red', 'y': 'orange', 'g': 'green'}.get(signal_state, 'gray')
                    
                    # Determine the time range for this state
                    start_time = state_change['elapsed_time']
                    end_time = states[i+1]['elapsed_time'] if i+1 < len(states) else total_time
                    
                    # Plot colored segment on the horizontal line (thinner and dashed)
                    if start_time <= total_time and end_time >= 0:
                        ax.plot([start_time, end_time], [avg_position, avg_position], 
                               color=color, linewidth=2, linestyle='--', alpha=0.8, zorder=5)
                
                # Add traffic light label at the right
                short_id = tl_id.split('_')[0] if '_' in tl_id else tl_id[:10]
                ax.text(ax.get_xlim()[1], avg_position, f'TL-{signal_index}\n{short_id}', 
                       fontsize=8, ha='left', va='center', rotation=0, alpha=0.8)
        
        # *** ONLY DIFFERENCE: Add conflict markers ***
        # Step 1: Collect all valid conflict positions
        conflict_positions = []
        for event in conflict_events:
            event_time = event['time_step']
            elapsed_time = event_time - start_time_step
            time_diffs = np.abs(time_steps - event_time)
            closest_idx = np.argmin(time_diffs)
            
            if closest_idx < len(distances):
                conflict_distance = distances[closest_idx]
                conflict_elapsed_time = elapsed_times[closest_idx]
                dominant_ssm = event['dominant_ssm']
                
                # Get the actual SSM value from the event (fields are uppercase)
                if dominant_ssm == 'TTC':
                    ssm_value = event.get('TTC', event.get('ttc', 0))
                    label = f"TTC={ssm_value:.1f}s"
                elif dominant_ssm == 'PET':
                    ssm_value = event.get('PET', event.get('pet', 0))
                    label = f"PET={ssm_value:.1f}s"
                elif dominant_ssm == 'DRAC':
                    ssm_value = event.get('DRAC', event.get('drac', 0))
                    label = f"DRAC={ssm_value:.1f}m/s²"
                else:
                    label = dominant_ssm
                
                conflict_positions.append({
                    'time': conflict_elapsed_time,
                    'distance': conflict_distance,
                    'label': label
                })
        
        # Sort conflicts by time to ensure chronological alternating placement
        conflict_positions.sort(key=lambda x: x['time'])
        
        # Step 2: Initial alternating label placement (above/below)
        label_placements = []
        for idx, conflict_info in enumerate(conflict_positions):
            place_above = (idx % 2 == 0)
            label_placements.append(place_above)
        
        # Step 3: Check for overlaps and adjust
        # Estimate label dimensions in data coordinates
        time_range = elapsed_times[-1] - elapsed_times[0]
        dist_range = distances.max() - distances.min()
        label_width_data = time_range * 0.08  # ~8% of time range
        label_height_data = dist_range * 0.03  # ~3% of distance range
        
        # Iteratively check for overlaps and adjust (multiple passes to handle chains)
        max_iterations = 5
        for iteration in range(max_iterations):
            any_overlap = False
            
            # Check each pair of consecutive conflicts for overlap
            for i in range(len(conflict_positions) - 1):
                curr = conflict_positions[i]
                next_conflict = conflict_positions[i + 1]
                
                curr_time = curr['time']
                curr_dist = curr['distance']
                next_time = next_conflict['time']
                next_dist = next_conflict['distance']
                
                # Calculate label bounding boxes accounting for 'far' adjustments
                curr_height_multiplier = 2 if label_placements[i] in ['above_far', 'below_far'] else 1
                next_height_multiplier = 2 if label_placements[i + 1] in ['above_far', 'below_far'] else 1
                
                if label_placements[i] in [True, 'above_far']:  # Current label is above
                    curr_label_top = curr_dist + label_height_data * curr_height_multiplier
                    curr_label_bottom = curr_dist
                else:  # Current label is below
                    curr_label_top = curr_dist
                    curr_label_bottom = curr_dist - label_height_data * curr_height_multiplier
                
                if label_placements[i + 1] in [True, 'above_far']:  # Next label is above
                    next_label_top = next_dist + label_height_data * next_height_multiplier
                    next_label_bottom = next_dist
                else:  # Next label is below
                    next_label_top = next_dist
                    next_label_bottom = next_dist - label_height_data * next_height_multiplier
                
                curr_label_left = curr_time - label_width_data / 2
                curr_label_right = curr_time + label_width_data / 2
                next_label_left = next_time - label_width_data / 2
                next_label_right = next_time + label_width_data / 2
                
                # Check for overlap (bounding box intersection)
                horizontal_overlap = not (curr_label_right < next_label_left or next_label_right < curr_label_left)
                vertical_overlap = not (curr_label_top < next_label_bottom or next_label_top < curr_label_bottom)
                
                if horizontal_overlap and vertical_overlap:
                    any_overlap = True
                    
                    # Overlap detected - move the second (further) label further away
                    if label_placements[i + 1] in [True, 'above_far']:
                        # Next label is above - keep it above but push it higher
                        label_placements[i + 1] = 'above_far'
                    else:
                        # Next label is below - keep it below but push it lower
                        label_placements[i + 1] = 'below_far'
            
            # If no overlaps detected in this pass, we're done
            if not any_overlap:
                break
        
        # Plot markers with adjusted label positions
        for idx, conflict_info in enumerate(conflict_positions):
            conflict_elapsed_time = conflict_info['time']
            conflict_distance = conflict_info['distance']
            label = conflict_info['label']
            
            # Plot conflict marker (hollow circle with firebrick edge)
            ax.scatter(conflict_elapsed_time, conflict_distance, 
                      s=80, marker='o', facecolors='none', 
                      edgecolors='firebrick', linewidth=2, zorder=10)
            
            # Place label based on adjusted placement
            placement = label_placements[idx]
            
            if placement == True or placement == 'above_far':
                # Above marker
                if placement == 'above_far':
                    ax.text(conflict_elapsed_time, conflict_distance, 
                           f"{label}\n\n", fontsize=8, color='firebrick',
                           ha='center', va='bottom')
                else:
                    ax.text(conflict_elapsed_time, conflict_distance, 
                           f"{label}\n", fontsize=8, color='firebrick',
                           ha='center', va='bottom')
            else:
                # Below marker
                if placement == 'below_far':
                    ax.text(conflict_elapsed_time, conflict_distance, 
                           f"\n\n{label}", fontsize=8, color='firebrick',
                           ha='center', va='top')
                else:
                    ax.text(conflict_elapsed_time, conflict_distance, 
                           f"\n{label}", fontsize=8, color='firebrick',
                           ha='center', va='top')
        
        # *** ONLY DIFFERENCE: Calculate conflict detection rates instead of trajectory detection rates ***
        conflict_detection_stats = self._calculate_conflict_detection_rates_improved(
            conflict_events, detection_df, bicycle_data
        )
        
        # *** ONLY DIFFERENCE: Info box shows conflict statistics ***
        info_text = (
            f"Bicycle: {bicycle_id}\n"
            f"Departure time: {start_time_step:.1f} s\n"
            f"Conflicts: {len(conflict_events)}\n"
            f"Temporal conflict detection rate: {conflict_detection_stats['temporal_rate']:.1f}%\n"
            f"Spatial conflict detection rate: {conflict_detection_stats['spatial_rate']:.1f}%\n"
            f"Spatio-temporal conflict detection rate: {conflict_detection_stats['spatiotemporal_rate']:.1f}%"
        )
        
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
        
        # Create legend (SAME AS _plot_individual_trajectory plus conflict marker)
        handles = [
            Line2D([0], [0], color='black', lw=2, label='Undetected'),
            Line2D([0], [0], color='darkturquoise', lw=2, label='Detected'),
        ]
        
        # Add conflict marker to legend (hollow circle)
        handles.append(
            Line2D([0], [0], marker='o', color='white', linestyle='None', 
                   markersize=8, markeredgewidth=2, markeredgecolor='firebrick', 
                   markerfacecolor='none', label='Conflict Event')
        )
        
        # Add traffic light legend items if any were plotted (SAME AS _plot_individual_trajectory)
        if tl_info:
            handles.extend([
                Line2D([0], [0], color='red', linestyle='--', alpha=0.7, label='Red TL'),
                Line2D([0], [0], color='orange', linestyle='--', alpha=0.7, label='Yellow TL'),
                Line2D([0], [0], color='green', linestyle='--', alpha=0.7, label='Green TL')
            ])
            
        ax.legend(handles=handles, loc='lower right', bbox_to_anchor=(0.99, 0.01))
        
        # Set labels and grid (EXACTLY SAME)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Space [m]')
        ax.grid(True)
        
        # Save plot in subdirectory (SAME STRUCTURE)
        output_subdir = os.path.join(self.config['output_dir'], '2D_conflict_individual')
        os.makedirs(output_subdir, exist_ok=True)
        output_filename = f'2D_conflict_individual_{self.config["file_tag"]}_FCO{self.config["fco_share"]}%_FBO{self.config["fbo_share"]}%_{bicycle_id}.png'
        output_path = os.path.join(output_subdir, output_filename)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.config['dpi'], bbox_inches='tight')
        plt.close(fig)
        
        print(f"  ✓ Saved individual 2D conflict plot: {output_filename}")
    
    def _process_flow_based_conflict_plots(self, trajectory_df, conflict_events, detection_df, traffic_light_df):
        """Generate flow-based conflict plots matching flow-based detection plots structure.
        
        Plots all bicycles per flow with detected/undetected segments, conflict markers,
        traffic light overlays, and flow-level conflict detection rates.
        
        Args:
            trajectory_df: DataFrame with all bicycle trajectories
            conflict_events: List of all conflict events
            detection_df: DataFrame with detection logs
            traffic_light_df: DataFrame with traffic light logs
        """
        
        print("\n=== Processing Flow-Based Conflict Plots ===")
        
        # Group conflicts by bicycle for quick lookup
        conflicts_by_bicycle = {}
        for event in conflict_events:
            bicycle_id = event['bicycle_id']
            if bicycle_id not in conflicts_by_bicycle:
                conflicts_by_bicycle[bicycle_id] = []
            conflicts_by_bicycle[bicycle_id].append(event)
        
        traj = trajectory_df.copy()
        traj['vehicle_id_str'] = traj['vehicle_id'].astype(str)
        traj['flow_id'] = traj['vehicle_id_str'].str.extract(r'(?i)(flow[_A-Za-z0-9-]*)', expand=False)

        flows = traj[traj['flow_id'].notna()]['flow_id'].unique()
        if len(flows) == 0:
            print("No explicit flow-tagged vehicle IDs found. Skipping flow-based conflict diagrams.")
            return

        print(f"Found {len(flows)} flows for flow-based conflict plotting")

        # For each flow, build diagram (SAME STRUCTURE AS flow-based detection)
        for flow_id in flows:
            flow_data = traj[traj['flow_id'] == flow_id].copy()
            if flow_data.empty:
                continue

            flow_data['time_step'] = pd.to_numeric(flow_data['time_step'], errors='coerce')
            flow_data['distance'] = pd.to_numeric(flow_data['distance'], errors='coerce')
            flow_data['abs_time_s'] = flow_data['time_step'].astype(float)

            valid_times = flow_data['abs_time_s'].dropna()
            if valid_times.empty:
                continue
            flow_start_time = float(valid_times.min())
            end_time = float(valid_times.max())
            start_time = flow_start_time

            try:
                first_distances = flow_data.sort_values('time_step').groupby('vehicle_id')['distance'].first().astype(float)
                flow_baseline = float(first_distances.min()) if len(first_distances) > 0 else 0.0
            except Exception:
                flow_baseline = 0.0

            out_dir = Path(self.config['output_dir'])
            out_dir.mkdir(parents=True, exist_ok=True)
            file_tag = self.config.get('file_tag', Path(self.config['scenario_path']).name)
            fco = int(self.config.get('fco_share', 0))
            fbo = int(self.config.get('fbo_share', 0))

            fig, ax = plt.subplots(figsize=self.config['figure_size'])
            ax.set_xlim(left=start_time, right=end_time)

            total_flow_distance = 0.0
            total_flow_detected_distance = 0.0
            total_flow_time = 0.0
            total_flow_detected_time = 0.0
            plotted_any = False
            flow_conflicts = []
            
            max_gap_seconds = float(self.config.get('max_continuous_gap_s', 5.0))

            # Plot each bicycle in the flow (SAME STRUCTURE AS detection plots)
            for vehicle_id, g in flow_data.groupby('vehicle_id'):
                g = g.reset_index(drop=True)
                abs_times = g['abs_time_s'].astype(float).values
                distances_arr = g['distance'].astype(float).values

                if len(abs_times) < 2:
                    continue

                # Collect conflicts for this bicycle
                if vehicle_id in conflicts_by_bicycle:
                    flow_conflicts.extend(conflicts_by_bicycle[vehicle_id])

                time_diffs = np.diff(abs_times)
                dist_diffs = np.diff(distances_arr)
                breaks_mask = (time_diffs < -1e-6) | (time_diffs > max_gap_seconds) | (dist_diffs < -1.0)
                break_idxs = np.where(breaks_mask)[0]

                spans = []
                start_idx = 0
                for b in break_idxs:
                    spans.append((start_idx, b))
                    start_idx = b + 1
                spans.append((start_idx, len(abs_times)-1))

                for span_idx, (sidx, eidx) in enumerate(spans):
                    sub_g = g.iloc[sidx:eidx+1]
                    if len(sub_g) < 2:
                        continue

                    sub_g_sorted = sub_g.sort_values('time_step').reset_index(drop=True)
                    det_events = detection_df[detection_df['bicycle_id'] == vehicle_id] if len(detection_df) > 0 else pd.DataFrame()
                    if not det_events.empty and 'time_step' in det_events.columns:
                        tmin = sub_g_sorted['time_step'].astype(float).min()
                        tmax = sub_g_sorted['time_step'].astype(float).max()
                        det_events = det_events[(det_events['time_step'] >= tmin) & (det_events['time_step'] <= tmax)]

                    bike_time_steps = sub_g_sorted['time_step'].astype(float).values
                    detection_timeline = self._create_detection_timeline(bike_time_steps, det_events, bike_time_steps[0])
                    smoothed = self._smooth_detection_timeline(detection_timeline)

                    distances = sub_g_sorted['distance'].astype(float).tolist()
                    times = sub_g_sorted['abs_time_s'].astype(float).tolist()
                    segments = self._split_trajectory_segments(distances, times, smoothed)

                    # Plot undetected segments
                    for seg in segments['undetected']:
                        if len(seg) > 1:
                            dists_s, times_s = zip(*seg)
                            adj_dists = [d - flow_baseline for d in dists_s]
                            ax.plot(times_s, adj_dists, color='black', linewidth=1.5, alpha=0.7)
                            
                            bike_total_distance = 0.0
                            bike_total_time = 0.0
                            for i in range(1, len(seg)):
                                dd = abs(dists_s[i] - dists_s[i-1])
                                dt = abs(times_s[i] - times_s[i-1])
                                bike_total_distance += dd
                                bike_total_time += dt

                            total_flow_distance += bike_total_distance
                            total_flow_time += bike_total_time

                    # Plot detected segments
                    for seg in segments['detected']:
                        if len(seg) > 1:
                            dists_s, times_s = zip(*seg)
                            adj_dists = [d - flow_baseline for d in dists_s]
                            ax.plot(times_s, adj_dists, color='darkturquoise', linewidth=1.5, alpha=0.7)

                            bike_total_distance = 0.0
                            bike_detected_distance = 0.0
                            bike_total_time = 0.0
                            bike_detected_time = 0.0
                            for i in range(1, len(seg)):
                                dd = abs(dists_s[i] - dists_s[i-1])
                                dt = abs(times_s[i] - times_s[i-1])
                                bike_total_distance += dd
                                bike_detected_distance += dd
                                bike_total_time += dt
                                bike_detected_time += dt

                            total_flow_distance += bike_total_distance
                            total_flow_detected_distance += bike_detected_distance
                            total_flow_time += bike_total_time
                            total_flow_detected_time += bike_detected_time
                    
                    plotted_any = True

                # *** CONFLICT MARKERS: Plot for this bicycle ***
                if vehicle_id in conflicts_by_bicycle:
                    # Get trajectory points for interpolation
                    vehicle_traj = g.sort_values('time_step')
                    traj_times = vehicle_traj['abs_time_s'].astype(float).values
                    traj_dists = vehicle_traj['distance'].astype(float).values
                    
                    for event in conflicts_by_bicycle[vehicle_id]:
                        event_time = event['time_step']
                        if event_time >= traj_times[0] and event_time <= traj_times[-1]:
                            event_distance = np.interp(event_time, traj_times, traj_dists)
                            adj_dist = event_distance - flow_baseline
                            
                            # Plot conflict marker (hollow circle like individual plots)
                            ax.scatter(event_time, adj_dist, 
                                      s=80, marker='o', facecolors='none', 
                                      edgecolors='firebrick', linewidth=2, zorder=10)

            if not plotted_any:
                print(f"No valid trajectories to plot for flow {flow_id}")
                plt.close(fig)
                continue

            # Calculate flow-level CONFLICT detection rates
            # Sum across all bicycles in the flow (same method as individual conflict plots)
            if flow_conflicts:
                flow_total_conflict_timesteps = 0
                flow_detected_conflict_timesteps = 0
                flow_total_conflict_distance = 0.0
                flow_detected_conflict_distance = 0.0
                
                # Get unique bicycles with conflicts in this flow
                bicycles_with_conflicts = set(event['bicycle_id'] for event in flow_conflicts)
                
                for bicycle_id in bicycles_with_conflicts:
                    # Get conflicts for this bicycle
                    bicycle_conflicts = [e for e in flow_conflicts if e['bicycle_id'] == bicycle_id]
                    
                    # Get bicycle trajectory data
                    bicycle_data = flow_data[flow_data['vehicle_id'] == bicycle_id].sort_values('time_step').copy()
                    
                    if bicycle_data.empty:
                        continue
                    
                    # Calculate detection rates for this bicycle using the improved method
                    conflict_stats = self._calculate_conflict_detection_rates_improved(
                        bicycle_conflicts, detection_df, bicycle_data
                    )
                    
                    # Accumulate across all bicycles in the flow
                    flow_total_conflict_timesteps += conflict_stats['total_conflict_timesteps']
                    flow_detected_conflict_timesteps += conflict_stats['detected_conflict_timesteps']
                    flow_total_conflict_distance += conflict_stats['total_conflict_distance']
                    flow_detected_conflict_distance += conflict_stats['detected_conflict_distance']
                
                # Calculate flow-level rates
                conflict_temporal_rate = (flow_detected_conflict_timesteps / flow_total_conflict_timesteps * 100) if flow_total_conflict_timesteps > 0 else 0.0
                conflict_spatial_rate = (flow_detected_conflict_distance / flow_total_conflict_distance * 100) if flow_total_conflict_distance > 0 else 0.0
                conflict_spatiotemporal_rate = (conflict_temporal_rate + conflict_spatial_rate) / 2.0
            else:
                conflict_temporal_rate = 0.0
                conflict_spatial_rate = 0.0
                conflict_spatiotemporal_rate = 0.0

            # Traffic lights (SAME AS detection plots)
            tl_info = {}
            required_cols = {'next_tl_id', 'next_tl_distance', 'next_tl_state', 'next_tl_index'}
            if required_cols.intersection(flow_data.columns):
                tl_rows = flow_data[flow_data['next_tl_id'].notna() & (flow_data['next_tl_id'] != '')].copy()
                if not tl_rows.empty:
                    tl_rows['abs_time_s'] = tl_rows['time_step'].astype(float)
                    for tl_id, tlg in tl_rows.groupby('next_tl_id'):
                        events = []
                        for _, row in tlg.iterrows():
                            state = row.get('next_tl_state')
                            rel_dist = row.get('next_tl_distance', np.nan)
                            bike_dist = row.get('distance', np.nan)
                            if pd.isna(state) or pd.isna(rel_dist) or pd.isna(bike_dist):
                                continue
                            t = float(row['abs_time_s'])
                            pos = float(bike_dist) + float(rel_dist)
                            events.append({'time': t, 'state': state, 'position': pos, 'signal_index': int(row.get('next_tl_index', 0))})

                        if not events:
                            continue

                        events = sorted(events, key=lambda x: x['time'])
                        positions = [e['position'] for e in events if not pd.isna(e['position'])]
                        avg_pos = float(np.median(positions)) if positions else np.nan

                        segments = []
                        for i, e in enumerate(events):
                            t0 = e['time']
                            state = e['state']
                            t1 = events[i+1]['time'] if i+1 < len(events) else end_time
                            if t1 <= t0:
                                continue
                            segments.append({'t0': t0, 't1': t1, 'state': state, 'position': avg_pos})

                        if segments:
                            tl_info[tl_id] = {'segments': segments, 'signal_index': events[0].get('signal_index', 0), 'avg_position': avg_pos}

            if tl_info:
                for tl_id, data in tl_info.items():
                    pos = data.get('avg_position', np.nan)
                    if not np.isnan(pos):
                        adj_pos = pos - flow_baseline
                        ax.axhline(y=adj_pos, xmin=0, xmax=1, color='black', linestyle='--', alpha=0.3, linewidth=0.6, zorder=1)
                        
                        segments = data.get('segments', [])
                        merged_segments = []
                        if segments:
                            current_color = {'r': 'red', 'y': 'yellow', 'g': 'green', 'G': 'green'}.get(str(segments[0]['state']).lower()[0], 'gray')
                            current_start = segments[0]['t0']
                            current_end = segments[0]['t1']
                            
                            for seg in segments[1:]:
                                seg_color = {'r': 'red', 'y': 'yellow', 'g': 'green', 'G': 'green'}.get(str(seg['state']).lower()[0], 'gray')
                                if seg_color == current_color and seg['t0'] <= current_end:
                                    current_end = max(current_end, seg['t1'])
                                else:
                                    merged_segments.append({'t0': current_start, 't1': current_end, 'color': current_color})
                                    current_color = seg_color
                                    current_start = seg['t0']
                                    current_end = seg['t1']
                            
                            merged_segments.append({'t0': current_start, 't1': current_end, 'color': current_color})
                        
                        for seg in merged_segments:
                            ax.plot([seg['t0'], seg['t1']], [adj_pos, adj_pos], 
                                   color=seg['color'], linewidth=2, linestyle='--', alpha=0.8, zorder=5)

            ax.set_xlabel('Simulation Time [s]')
            ax.set_ylabel('Space [m]')
            ax.grid(True)

            # Primary legend
            handles = [
                Line2D([0], [0], color='black', lw=2, label='Undetected'),
                Line2D([0], [0], color='darkturquoise', lw=2, label='Detected'),
                Line2D([0], [0], marker='o', color='white', linestyle='None', 
                       markersize=8, markeredgewidth=2, markeredgecolor='firebrick', 
                       markerfacecolor='none', label='Conflict Event')
            ]
            handles_tl = [
                Line2D([0], [0], color='red', lw=2, linestyle='--', label='Red TL'),
                Line2D([0], [0], color='yellow', lw=2, linestyle='--', label='Yellow TL'),
                Line2D([0], [0], color='green', lw=2, linestyle='--', label='Green TL')
            ]

            # Flow info legend (CONFLICT DETECTION RATES)
            info_lines = [
                f"Flow: {flow_id} ({flow_data['vehicle_id'].nunique()} bicycles)",
                f"Conflicts: {len(flow_conflicts)}",
                f"Temporal conflict detection rate: {conflict_temporal_rate:.1f}%",
                f"Spatial conflict detection rate: {conflict_spatial_rate:.1f}%",
                f"Spatio-temporal conflict detection rate: {conflict_spatiotemporal_rate:.1f}%"
            ]

            info_handles = [Line2D([0], [0], color='white', label=l) for l in info_lines]
            info_legend = ax.legend(handles=info_handles, loc='upper left', bbox_to_anchor=(0.01, 0.99), 
                                    fontsize=plt.rcParams['legend.fontsize'], framealpha=0.9,
                                    handlelength=0, handletextpad=0)
            ax.add_artist(info_legend)
            ax.legend(handles=handles + handles_tl, loc='lower right', bbox_to_anchor=(0.99, 0.01))

            out_dir_conflict = Path(self.config['output_dir']) / '2D_conflict_flow-based'
            out_dir_conflict.mkdir(parents=True, exist_ok=True)
            flow_plot_path = out_dir_conflict / f"2D_conflict_flow-based_{flow_id}_{file_tag}_FCO{fco}%_FBO{fbo}%.png"
            plt.savefig(str(flow_plot_path), dpi=self.config.get('dpi', DPI), bbox_inches='tight')
            plt.close(fig)

            print(f"  ✓ Saved flow-based conflict plot: {flow_plot_path.name}")
        
        print(f"\n✓ Generated {len(flows)} flow-based conflict plots")
    
    def process_bicycle_trajectories(self, trajectory_df, detection_df, traffic_light_df):
        """Process and plot individual 2D detection trajectory plots."""
        
        print("\n=== Processing Individual 2D Detection Plots ===")
        
        # Group trajectory data by bicycle
        bicycle_groups = trajectory_df.groupby('vehicle_id')
        num_bicycles = len(bicycle_groups)
        print(f"Found {num_bicycles} bicycles for individual 2D detection plotting")
        
        for bicycle_id, bicycle_data in bicycle_groups:
            
            # Sort by time step
            bicycle_data = bicycle_data.sort_values('time_step')
            
            # Extract trajectory information
            time_steps = bicycle_data['time_step'].values
            distances = bicycle_data['distance'].values
            
            # time_step is now in seconds (simulation time), not frame numbers
            # So elapsed_times is just the time difference, no multiplication needed
            start_time_step = time_steps[0]
            elapsed_times = time_steps - start_time_step
            
            # Calculate full trajectory distance and time (for detection rate denominators)
            total_trajectory_distance = distances[-1] - distances[0] if len(distances) > 0 else 0
            total_trajectory_time = elapsed_times[-1] if len(elapsed_times) > 0 else 0
            
            # Get detection status for this bicycle
            bicycle_detections = detection_df[detection_df['bicycle_id'] == bicycle_id] if len(detection_df) > 0 else pd.DataFrame()
            
            # Create detection timeline
            detection_timeline = self._create_detection_timeline(time_steps, bicycle_detections, start_time_step)
            
            # Apply detection smoothing
            smoothed_detection = self._smooth_detection_timeline(detection_timeline)
            
            # Split trajectory into detected/undetected segments
            # Create two versions: filtered for plotting, unfiltered for statistics
            segments_for_plot = self._split_trajectory_segments(distances, elapsed_times, smoothed_detection)
            segments_for_stats = self._split_trajectory_segments(distances, elapsed_times, smoothed_detection, apply_min_length_filter=False)
            
            # Calculate detection rates once (used by all plot types)
            detection_rates = self._calculate_detection_rates(segments_for_stats, total_trajectory_distance, total_trajectory_time)
            
            # Get traffic light information for this bicycle
            tl_info = self._get_bicycle_traffic_lights(bicycle_data, traffic_light_df)
            
            # Generate individual 2D detection plot
            self._plot_individual_trajectory(
                bicycle_id, segments_for_plot, tl_info,
                start_time_step, detection_rates
            )
        
        print(f"\n✓ Generated {num_bicycles} individual 2D detection plots")
    
    def process_bicycle_trajectories_redundancy(self, trajectory_df, traffic_light_df, detection_df):
        """Process and plot individual 2D detection-redundancy trajectory plots.
        
        Uses the same detection data source as detection plots for consistency.
        """
        
        print("\n=== Processing Individual 2D Detection-Redundancy Plots ===")
        
        # Group trajectory data by bicycle
        bicycle_groups = trajectory_df.groupby('vehicle_id')
        num_bicycles = len(bicycle_groups)
        print(f"Found {num_bicycles} bicycles for individual 2D detection-redundancy plotting")
        
        for bicycle_id, bicycle_data in bicycle_groups:
            
            # Sort by time step
            bicycle_data = bicycle_data.sort_values('time_step')
            
            # Extract trajectory information
            time_steps = bicycle_data['time_step'].values
            distances = bicycle_data['distance'].values
            start_time_step = time_steps[0]
            elapsed_times = time_steps - start_time_step
            
            # Calculate full trajectory distance and time (for detection rate denominators)
            total_trajectory_distance = distances[-1] - distances[0] if len(distances) > 0 else 0
            total_trajectory_time = elapsed_times[-1] if len(elapsed_times) > 0 else 0
            
            # Get traffic light information for this bicycle
            tl_info = self._get_bicycle_traffic_lights(bicycle_data, traffic_light_df)
            
            try:
                # Use the same detection data source as detection plots for consistency
                # Get detection status for this bicycle from detection_df
                bicycle_detections = detection_df[detection_df['bicycle_id'] == bicycle_id] if len(detection_df) > 0 else pd.DataFrame()
                
                # Create detection timeline (same method as detection plots)
                detection_timeline = self._create_detection_timeline(time_steps, bicycle_detections, start_time_step)
                
                # Extract redundancy values from CSV (for color coding only, not for detection status)
                if 'num_detecting_observers' in bicycle_data.columns:
                    redundancy_values = bicycle_data['num_detecting_observers'].values
                else:
                    # If no redundancy column, create one based on detection status
                    # All detected frames get redundancy level 1 (detected by at least 1 observer)
                    redundancy_values = np.zeros(len(detection_timeline), dtype=int)
                    redundancy_values[detection_timeline] = 1
                
                smoothed_detection = self._smooth_detection_timeline(detection_timeline)
                
                # Create smoothed redundancy values:
                # - If detection timeline shows detected AND redundancy is 0 → set to 1 (gap bridging)
                # - If detection timeline shows NOT detected → force redundancy to 0
                # - Otherwise preserve original redundancy value
                smoothed_redundancy = np.zeros(len(redundancy_values), dtype=int)
                for i in range(len(smoothed_redundancy)):
                    if smoothed_detection[i]:
                        # Detected: use original redundancy value, or 1 if it was 0 (gap bridged)
                        smoothed_redundancy[i] = max(1, redundancy_values[i])
                    else:
                        # Not detected: force to 0
                        smoothed_redundancy[i] = 0
                
                # Split trajectory by redundancy level (using smoothed values)
                # Create two versions: filtered for plotting, unfiltered for statistics
                redundancy_segments_for_plot = self._split_trajectory_segments_by_redundancy(
                    distances, elapsed_times, smoothed_redundancy
                )
                redundancy_segments_for_stats = self._split_trajectory_segments_by_redundancy(
                    distances, elapsed_times, smoothed_redundancy, apply_min_length_filter=False
                )
                
                # Create regular segments to calculate detection rates
                segments_for_stats = self._split_trajectory_segments(distances, elapsed_times, smoothed_detection, apply_min_length_filter=False)
                detection_rates = self._calculate_detection_rates(segments_for_stats, total_trajectory_distance, total_trajectory_time)
                
                # Generate redundancy plot
                self._plot_individual_trajectory_redundancy(
                    bicycle_id, redundancy_segments_for_plot, tl_info,
                    start_time_step, detection_rates
                )
                    
            except Exception as e:
                import traceback
                print(f"    Error generating redundancy plot for {bicycle_id}: {e}")
                traceback.print_exc()
        
        print(f"\n✓ Generated {num_bicycles} individual 2D detection-redundancy plots")
    
    def process_bicycle_occlusion_plots(self, trajectory_df, detection_df, traffic_light_df):
        """Process and plot individual 2D occlusion level plots."""
        
        print("\n=== Processing Individual 2D Occlusion Level Plots ===")
        
        bicycle_groups = trajectory_df.groupby('vehicle_id')
        num_bicycles = len(bicycle_groups)
        print(f"Found {num_bicycles} bicycles for individual 2D occlusion level plotting")
        
        for bicycle_id, bicycle_data in bicycle_groups:
            bicycle_data = bicycle_data.sort_values('time_step')
            
            # Get bicycle detections with occlusion data
            bicycle_detections = detection_df[detection_df['bicycle_id'] == bicycle_id] if len(detection_df) > 0 else pd.DataFrame()
            
            # Skip if no detections
            if bicycle_detections.empty:
                continue
            
            time_steps = bicycle_data['time_step'].values
            distances = bicycle_data['distance'].values
            start_time_step = time_steps[0]
            elapsed_times = time_steps - start_time_step
            
            total_trajectory_distance = distances[-1] - distances[0] if len(distances) > 0 else 0
            total_trajectory_time = elapsed_times[-1] if len(elapsed_times) > 0 else 0
            
            # Create detection timeline
            detection_timeline = self._create_detection_timeline(time_steps, bicycle_detections, start_time_step)
            smoothed_detection = self._smooth_detection_timeline(detection_timeline)
            
            # Get traffic light info
            tl_info = self._get_bicycle_traffic_lights(bicycle_data, traffic_light_df)
            
            # Create occlusion timeline (occlusion level for each detected time step)
            occlusion_timeline = self._create_occlusion_timeline(time_steps, bicycle_detections, start_time_step)
            
            # Split trajectory into segments by detection and occlusion level
            # Create two versions: filtered for plotting, unfiltered for statistics
            segments_for_plot = self._split_trajectory_segments(distances, elapsed_times, smoothed_detection)
            segments_for_stats = self._split_trajectory_segments(distances, elapsed_times, smoothed_detection, apply_min_length_filter=False)
            occlusion_segments = self._create_occlusion_segments(distances, elapsed_times, smoothed_detection, occlusion_timeline)
            
            # Calculate detection rates once (shared across all plot types)
            detection_rates = self._calculate_detection_rates(segments_for_stats, total_trajectory_distance, total_trajectory_time)
            
            # Generate plot
            self._plot_individual_occlusion_trajectory(
                bicycle_id, segments_for_plot, occlusion_segments, tl_info,
                start_time_step, detection_rates
            )
        
        print(f"\n✓ Generated {num_bicycles} individual 2D occlusion level plots")
    
    def _process_flow_based_occlusion_plots(self, trajectory_df, detection_df, traffic_light_df):
        """Create flow-based space-time diagrams with occlusion level color coding.
        
        Groups bicycles by flow and plots all trajectories on one diagram,
        with detected segments colored by occlusion level (no/low/partial/heavy).
        
        Args:
            trajectory_df: DataFrame with all bicycle trajectories
            detection_df: DataFrame with detection logs (including occlusion_level)
            traffic_light_df: DataFrame with traffic light logs
        """
        print("\n=== Processing Flow-Based 2D Occlusion Level Plots ===")
        
        traj = trajectory_df.copy()
        traj['vehicle_id_str'] = traj['vehicle_id'].astype(str)
        # Identify flows only when a 'flow' token exists in vehicle_id (case-insensitive)
        traj['flow_id'] = traj['vehicle_id_str'].str.extract(r'(?i)(flow[_A-Za-z0-9-]*)', expand=False)
        
        flows = traj[traj['flow_id'].notna()]['flow_id'].unique()
        if len(flows) == 0:
            print("No explicit flow-tagged vehicle IDs found. Skipping flow-based occlusion diagrams.")
            return
        
        print(f"Found {len(flows)} flows for flow-based occlusion plotting")
        
        # Get occlusion scale and color palette
        occlusion_scale = self.config.get('occlusion_level_scale', _get_occlusion_level_scale())
        occlusion_colors = _get_occlusion_color_palette()
        
        # For each flow, build diagram
        for flow_id in flows:
            flow_data = traj[traj['flow_id'] == flow_id].copy()
            bicycle_ids = flow_data['vehicle_id'].unique()
            
            if len(bicycle_ids) == 0:
                continue
            
            # Determine global spatial baseline (min distance across all bicycles in flow)
            spatial_baseline = flow_data['distance'].min()
            
            # Determine time extent for this flow
            flow_start_time = flow_data['time_step'].min()
            flow_end_time = flow_data['time_step'].max()
            time_padding = (flow_end_time - flow_start_time) * 0.05
            
            # Initialize flow-level detection metrics
            total_flow_distance = 0.0
            total_flow_detected_distance = 0.0
            total_flow_time = 0.0
            total_flow_detected_time = 0.0
            
            # Initialize occlusion category tracking (trajectory point counts)
            occlusion_points_by_category = {category_name: 0 for category_name, _, _ in occlusion_scale}
            total_detected_points = 0
            
            # Create figure
            fig, ax = plt.subplots(figsize=self.config['figure_size'])
            
            # Process each bicycle in the flow
            # Use same span-splitting logic as detection plots for consistency
            max_gap_seconds = float(self.config.get('max_continuous_gap_s', 5.0))
            
            for bicycle_id in bicycle_ids:
                bicycle_data = flow_data[flow_data['vehicle_id'] == bicycle_id].copy()
                
                if len(bicycle_data) < 2:
                    continue
                
                bicycle_data = bicycle_data.reset_index(drop=True)
                
                # Convert arrays (keep file order)
                abs_times = bicycle_data['time_step'].astype(float).values
                distances_arr = bicycle_data['distance'].astype(float).values
                
                if len(abs_times) < 2:
                    continue
                
                # Find break points where time decreases, gap exceeds threshold, or distance resets
                time_diffs = np.diff(abs_times)
                dist_diffs = np.diff(distances_arr)
                breaks_mask = (time_diffs < -1e-6) | (time_diffs > max_gap_seconds) | (dist_diffs < -1.0)
                break_idxs = np.where(breaks_mask)[0]
                
                # Build contiguous sub-trajectory index ranges
                spans = []
                start_idx = 0
                for b in break_idxs:
                    spans.append((start_idx, b))
                    start_idx = b + 1
                spans.append((start_idx, len(abs_times)-1))
                
                # Process each contiguous sub-trajectory
                vehicle_spans = []
                for span_idx, (sidx, eidx) in enumerate(spans):
                    sub_data = bicycle_data.iloc[sidx:eidx+1]
                    if len(sub_data) < 2:
                        continue
                    
                    sub_data_sorted = sub_data.sort_values('time_step').reset_index(drop=True)
                    
                    time_steps = sub_data_sorted['time_step'].astype(float).values
                    distances = sub_data_sorted['distance'].astype(float).values
                    start_time_step = time_steps[0]
                    
                    # Shift distances relative to spatial baseline
                    shifted_distances = distances - spatial_baseline
                    
                    # Get bicycle detections for this span
                    det_events = detection_df[detection_df['bicycle_id'] == bicycle_id]
                    if not det_events.empty and 'time_step' in det_events.columns:
                        tmin = time_steps.min()
                        tmax = time_steps.max()
                        det_events = det_events[(det_events['time_step'] >= tmin) & (det_events['time_step'] <= tmax)]
                    
                    # Create detection timeline
                    detection_timeline = self._create_detection_timeline(time_steps, det_events, start_time_step)
                    smoothed_detection = self._smooth_detection_timeline(detection_timeline)
                    
                    # Split into undetected and detected segments (using elapsed times for splitting)
                    elapsed_times = time_steps - start_time_step
                    segments = self._split_trajectory_segments(shifted_distances, elapsed_times, smoothed_detection)
                    
                    # Create occlusion timeline and segments
                    occlusion_timeline = self._create_occlusion_timeline(time_steps, det_events, start_time_step)
                    occlusion_segments = self._create_occlusion_segments(shifted_distances, elapsed_times, smoothed_detection, occlusion_timeline)
                    
                    span_info = {
                        'bicycle_id': bicycle_id,
                        'span_index': span_idx,
                        'first_time_s': float(time_steps[0]),
                        'first_distance': float(distances[0]),
                        'time_steps': time_steps,
                        'distances': distances,
                        'shifted_distances': shifted_distances,
                        'start_time_step': start_time_step,
                        'segments': segments,
                        'occlusion_segments': occlusion_segments,
                        'duration_s': float(time_steps[-1] - time_steps[0])
                    }
                    vehicle_spans.append(span_info)
                
                # Filter out stray mid-route spans (match detection plot logic)
                flow_start_dist_thresh = float(self.config.get('flow_start_distance_threshold_m', 1.0))
                primary_candidates = [s for s in vehicle_spans if s['first_distance'] <= flow_start_dist_thresh]
                primary_time = None
                if primary_candidates:
                    primary_time = min(s['first_time_s'] for s in primary_candidates)
                
                spans_to_plot = []
                for s in vehicle_spans:
                    ignored = False
                    if primary_time is not None and s['first_time_s'] < primary_time and s['first_distance'] > flow_start_dist_thresh:
                        ignored = True
                    if not ignored:
                        spans_to_plot.append(s)
                
                # Process only non-ignored spans
                for s in spans_to_plot:
                    time_steps = s['time_steps']
                    distances = s['distances']  # Original distances (not shifted)
                    start_time_step = s['start_time_step']
                    segments = s['segments']
                    occlusion_segments = s['occlusion_segments']
                    shifted_distances = s['shifted_distances']  # For plotting only
                    
                    # Plot undetected segments (black) - convert elapsed time back to simulation time
                    for segment in segments['undetected']:
                        if len(segment) > 1:
                            seg_distances, seg_elapsed_times = zip(*segment)
                            seg_sim_times = [t + start_time_step for t in seg_elapsed_times]
                            ax.plot(seg_sim_times, seg_distances, color='black', linewidth=1.5, linestyle='solid')
                    
                    # Plot detected segments colored by occlusion level - convert back to simulation time
                    for category_name, _, _ in occlusion_scale:
                        color = occlusion_colors.get(category_name, 'gray')
                        for segment in occlusion_segments[category_name]:
                            if len(segment) > 1:
                                seg_distances, seg_elapsed_times = zip(*segment)
                                seg_sim_times = [t + start_time_step for t in seg_elapsed_times]
                                ax.plot(seg_sim_times, seg_distances, color=color, linewidth=1.5, linestyle='solid')
                    
                    # Calculate detection metrics for this span using BASIC detected segments
                    # This ensures consistency with detection plots (use original distances, not shifted)
                    bike_total_distance = distances[-1] - distances[0]
                    bike_total_time = s['duration_s']
                    bike_detected_distance = 0.0
                    bike_detected_time = 0.0
                    
                    # Calculate from basic detected segments (NOT occlusion segments)
                    # to match detection plot calculation method
                    for segment in segments['detected']:
                        if len(segment) > 1:
                            # Segments use shifted_distances as coordinates, but differences are the same
                            seg_distance = segment[-1][0] - segment[0][0]
                            seg_time = segment[-1][1] - segment[0][1]
                            bike_detected_distance += seg_distance
                            bike_detected_time += seg_time
                    
                    # Track occlusion distribution for display (using occlusion segments)
                    for category_name, _, _ in occlusion_scale:
                        for segment in occlusion_segments[category_name]:
                            if len(segment) > 1:
                                # Track number of trajectory points per occlusion category
                                num_points = len(segment)
                                occlusion_points_by_category[category_name] += num_points
                                total_detected_points += num_points
                    
                    # Accumulate flow totals
                    total_flow_distance += bike_total_distance
                    total_flow_detected_distance += bike_detected_distance
                    total_flow_time += bike_total_time
                    total_flow_detected_time += bike_detected_time
            
            # Calculate flow detection rates using centralized method
            flow_detection_rates = self._calculate_flow_detection_rates(
                total_flow_distance, total_flow_detected_distance,
                total_flow_time, total_flow_detected_time
            )
            
            # Traffic lights: aggregate TL state events across all bicycles in the flow
            # and build a continuous timeline per TL so overlays span the full flow duration.
            tl_info = {}
            required_cols = {'next_tl_id', 'next_tl_distance', 'next_tl_state', 'next_tl_index'}
            if required_cols.intersection(flow_data.columns):
                # Collect all TL state observations from any bicycle in the flow
                tl_rows = flow_data[flow_data['next_tl_id'].notna() & (flow_data['next_tl_id'] != '')].copy()
                if not tl_rows.empty:
                    # time_step is now already in seconds, so abs_time_s is just a copy
                    tl_rows['abs_time_s'] = tl_rows['time_step'].astype(float)
                    for tl_id, tlg in tl_rows.groupby('next_tl_id'):
                        # For this TL, collect events (time, state, absolute position)
                        events = []
                        for _, row in tlg.iterrows():
                            state = row.get('next_tl_state')
                            rel_dist = row.get('next_tl_distance', np.nan)
                            bike_dist = row.get('distance', np.nan)
                            if pd.isna(state) or pd.isna(rel_dist) or pd.isna(bike_dist):
                                continue
                            t = float(row['abs_time_s'])
                            pos = float(bike_dist) + float(rel_dist)
                            events.append({'time': t, 'state': state, 'position': pos, 'signal_index': int(row.get('next_tl_index', 0))})

                        if not events:
                            continue

                        # Sort events by time
                        events = sorted(events, key=lambda x: x['time'])

                        # Use median of reported positions as the TL absolute position
                        positions = [e['position'] for e in events if not pd.isna(e['position'])]
                        avg_pos = float(np.median(positions)) if positions else np.nan

                        # Build continuous state segments: extend each observed state until next observed change
                        segments = []
                        for i, e in enumerate(events):
                            t0 = e['time']
                            state = e['state']
                            t1 = events[i+1]['time'] if i+1 < len(events) else flow_end_time
                            # skip degenerate zero-length
                            if t1 <= t0:
                                continue
                            segments.append({'t0': t0, 't1': t1, 'state': state, 'position': avg_pos})

                        if segments:
                            tl_info[tl_id] = {'segments': segments, 'signal_index': events[0].get('signal_index', 0), 'avg_position': avg_pos}

            # Fallback: use traffic_light_df to infer TLs if embedded data not present
            traffic_light_programs = {}
            if not traffic_light_df.empty and 'traffic_light_id' in traffic_light_df.columns:
                for tl_id, tlg in traffic_light_df.groupby('traffic_light_id'):
                    entries = []
                    for _, row in tlg.sort_values('time_step').iterrows():
                        # time_step is now already in seconds, no conversion needed
                        entries.append((row['time_step'], row.get('signal_states', row.get('signal_state', ''))))
                    traffic_light_programs[tl_id] = entries

            # Plot traffic lights similar to individual plots (horizontal colored segments at TL position)
            if tl_info:
                for tl_id, data in tl_info.items():
                    pos = data.get('avg_position', np.nan)
                    if not np.isnan(pos):
                        # Adjust position for flow baseline
                        adj_pos = pos - spatial_baseline if not np.isnan(pos) else pos
                        # horizontal faint baseline at TL distance (using adjusted position)
                        ax.axhline(y=adj_pos, xmin=0, xmax=1, color='black', linestyle='--', alpha=0.3, linewidth=0.6, zorder=1)
                        
                        # Merge consecutive segments with same color to avoid overlapping dashes appearing solid
                        segments = data.get('segments', [])
                        merged_segments = []
                        if segments:
                            current_color = {'r': 'red', 'y': 'yellow', 'g': 'green', 'G': 'green'}.get(str(segments[0]['state']).lower()[0], 'gray')
                            current_start = segments[0]['t0']
                            current_end = segments[0]['t1']
                            
                            for seg in segments[1:]:
                                seg_color = {'r': 'red', 'y': 'yellow', 'g': 'green', 'G': 'green'}.get(str(seg['state']).lower()[0], 'gray')
                                if seg_color == current_color and seg['t0'] <= current_end:
                                    # Same color and touching/overlapping - extend current segment
                                    current_end = max(current_end, seg['t1'])
                                else:
                                    # Different color or gap - save current and start new
                                    merged_segments.append({'t0': current_start, 't1': current_end, 'color': current_color})
                                    current_color = seg_color
                                    current_start = seg['t0']
                                    current_end = seg['t1']
                            
                            # Don't forget the last segment
                            merged_segments.append({'t0': current_start, 't1': current_end, 'color': current_color})
                        
                        # Now plot merged segments
                        for seg in merged_segments:
                            ax.plot([seg['t0'], seg['t1']], [adj_pos, adj_pos], 
                                   color=seg['color'], linewidth=2, linestyle='--', alpha=0.8, zorder=5)
            else:
                if traffic_light_programs and 'distance' in flow_data.columns:
                    approx_pos = (flow_data['distance'].min() + flow_data['distance'].max()) / 2 - spatial_baseline
                    ax.axhline(y=approx_pos, xmin=0, xmax=1, color='gray', linestyle='--', alpha=0.3)
                    
                    for tl_id, prog in traffic_light_programs.items():
                        # Build segments and merge consecutive ones with same color
                        segments = []
                        for i in range(len(prog)-1):
                            t0, state = prog[i]
                            t1 = prog[i+1][0]
                            color = {'r': 'red', 'y': 'yellow', 'g': 'green', 'G': 'green'}.get(str(state).lower()[0], 'gray') if state else 'gray'
                            segments.append({'t0': t0, 't1': t1, 'color': color})
                        
                        # Merge consecutive segments with same color
                        merged_segments = []
                        if segments:
                            current_color = segments[0]['color']
                            current_start = segments[0]['t0']
                            current_end = segments[0]['t1']
                            
                            for seg in segments[1:]:
                                if seg['color'] == current_color and seg['t0'] <= current_end:
                                    # Same color and touching/overlapping - extend
                                    current_end = max(current_end, seg['t1'])
                                else:
                                    # Different color or gap - save and start new
                                    merged_segments.append({'t0': current_start, 't1': current_end, 'color': current_color})
                                    current_color = seg['color']
                                    current_start = seg['t0']
                                    current_end = seg['t1']
                            
                            merged_segments.append({'t0': current_start, 't1': current_end, 'color': current_color})
                        
                        # Plot merged segments
                        for seg in merged_segments:
                            ax.plot([seg['t0'], seg['t1']], [approx_pos, approx_pos], 
                                   color=seg['color'], linewidth=2, linestyle='--', alpha=0.8, zorder=5)
            
            # Calculate occlusion distribution based on number of trajectory points per category
            occlusion_stats = {}
            for category_name, min_pct, max_pct in occlusion_scale:
                category_points = occlusion_points_by_category[category_name]
                # Calculate percentage of total detected points in this category
                percentage = (category_points / total_detected_points * 100) if total_detected_points > 0 else 0
                occlusion_stats[category_name] = {'points': category_points, 'percentage': percentage}
            
            # Info box with flow statistics (use pre-calculated detection rates)
            info_lines = [
                f"Flow: {flow_id}",
                f"Bicycles: {len(bicycle_ids)}",
                f"Spatio-temporal detection rate: {flow_detection_rates['spatiotemporal_rate']:.1f}%",
                "Occlusion distribution:"
            ]
            
            for category_name, min_pct, max_pct in occlusion_scale:
                display_name = category_name.replace('_', ' ').capitalize()
                stats = occlusion_stats[category_name]
                info_lines.append(f"  {display_name}: {stats['percentage']:.1f}%")
            
            info_text = "\n".join(info_lines)
            
            ax.text(0.01, 0.99, info_text,
                   transform=ax.transAxes,
                   verticalalignment='top',
                   horizontalalignment='left',
                   fontsize=plt.rcParams['legend.fontsize'],
                   bbox=dict(facecolor='white', edgecolor='black', alpha=0.8, boxstyle='round'))
            
            # Legend with occlusion categories and traffic lights
            handles = [Line2D([0], [0], color='black', lw=2, label='Undetected')]
            
            for category_name, min_pct, max_pct in occlusion_scale:
                color = occlusion_colors.get(category_name, 'gray')
                display_name = category_name.replace('_', ' ').capitalize()
                
                if min_pct == max_pct:
                    label = f'{display_name} ({min_pct}%)'
                else:
                    label = f'{display_name} ({min_pct}-{max_pct}%)'
                
                handles.append(Line2D([0], [0], color=color, lw=2, label=label))
            
            if self.config.get('enable_traffic_lights', True) and len(traffic_light_df) > 0:
                handles.extend([
                    Line2D([0], [0], color='red', linestyle='--', alpha=0.7, label='Red TL'),
                    Line2D([0], [0], color='orange', linestyle='--', alpha=0.7, label='Yellow TL'),
                    Line2D([0], [0], color='green', linestyle='--', alpha=0.7, label='Green TL')
                ])
            
            ax.legend(handles=handles, loc='lower right', bbox_to_anchor=(0.99, 0.01))
            
            # Set labels and grid
            ax.set_xlabel('Simulation Time [s]')
            ax.set_ylabel('Space [m]')
            ax.set_title(f'Flow-Based Occlusion Level Diagram: {flow_id}')
            ax.grid(True)
            
            # Set axis limits with padding (use simulation time)
            ax.set_xlim(flow_start_time - time_padding, flow_end_time + time_padding)
            
            # Save plot
            output_subdir = os.path.join(self.config['output_dir'], '2D_occlusion_flow-based')
            os.makedirs(output_subdir, exist_ok=True)
            output_filename = f'2D_occlusion_flow-based_{flow_id}_{self.config["file_tag"]}_FCO{self.config["fco_share"]}%_FBO{self.config["fbo_share"]}%.png'
            output_path = os.path.join(output_subdir, output_filename)
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=self.config['dpi'], bbox_inches='tight')
            plt.close(fig)
            
            print(f"  ✓ Saved flow-based 2D occlusion plot: {output_filename}")
        
        print(f"\n✓ Generated {len(flows)} flow-based 2D occlusion level plots")
    
    def process_3d_detection_plots(self, trajectory_df, detection_df, observer_df, geometry_data):
        """Process and generate 3D detection plots for bicycle trajectories."""
        
        print("\n=== Processing Individual 3D Detection Plots ===")
        
        if geometry_data is None:
            print("Warning: No geometry data available for 3D background")
            transformer = None
        else:
            transformer = geometry_data.get('transformer')
        
        if transformer is None:
            # Fallback transformer
            transformer = pyproj.Transformer.from_crs('EPSG:4326', 'EPSG:32632', always_xy=True)
            
        # Group trajectory data by bicycle
        bicycle_groups = trajectory_df.groupby('vehicle_id')
        num_bicycles = len(bicycle_groups)
        
        # Get list of bicycles that were actually detected
        detected_bicycle_ids = set(detection_df['bicycle_id'].unique()) if len(detection_df) > 0 else set()
        
        print(f"Found {num_bicycles} total bicycles, {len(detected_bicycle_ids)} were detected")
        
        trajectory_count = 0
        
        for bicycle_id, bicycle_data in bicycle_groups:
            
            # Skip bicycles that were never detected
            if bicycle_id not in detected_bicycle_ids:
                continue
            
            # Sort by time step
            bicycle_data = bicycle_data.sort_values('time_step')
            
            # Extract trajectory coordinates and times  
            time_steps = bicycle_data['time_step'].values
            x_coords = bicycle_data['x_coord'].values
            y_coords = bicycle_data['y_coord'].values
            
            # time_step is now in seconds (simulation time), not frame numbers
            # So elapsed_times is just the time difference, no multiplication needed
            start_time_step = time_steps[0]
            elapsed_times = time_steps - start_time_step
            
            # Create trajectory points for 3D plotting
            trajectory_3d = [(x, y, t) for x, y, t in zip(x_coords, y_coords, elapsed_times)]
            
            # Get detection status for this bicycle
            bicycle_detections = detection_df[detection_df['bicycle_id'] == bicycle_id] if len(detection_df) > 0 else pd.DataFrame()
            
            # Create detection timeline
            detection_timeline = self._create_detection_timeline(time_steps, bicycle_detections, start_time_step)
            
            # Apply detection smoothing
            smoothed_detection = self._smooth_detection_timeline(detection_timeline)
            
            # Split trajectory into detected/undetected segments for 3D plotting
            segments_3d = self._split_trajectory_segments_3d(trajectory_3d, smoothed_detection)
            
            # Get observer trajectories that detected this bicycle
            observer_trajectories = self._get_observer_trajectories_for_bicycle(
                bicycle_id, bicycle_detections, observer_df, start_time_step
            )
            
            # Generate the 3D plot
            self._plot_3d_detection_trajectory(
                bicycle_id, segments_3d, observer_trajectories, geometry_data, 
                start_time_step, elapsed_times[-1] if len(elapsed_times) > 0 else 0
            )
            
            trajectory_count += 1
        
        print(f"\n✓ Generated {trajectory_count} individual 3D detection plots")

    def _process_flow_based_from_logs(self, trajectory_df, detection_df, traffic_light_df):
        """Create flow-based space-time diagrams similar to the runtime version in main.py.

        Offline behavior: load all bicycle trajectories from `trajectory_df` and detection events
        from `detection_df`, group by flow id (only vehicle_ids containing an explicit 'flow' token),
        and produce a space-time diagram with detected/undetected segments, traffic light overlays,
        and conflict markers (if conflict logs are available).
        """
        print("\n=== Processing Flow-Based 2D Detection Plots ===")

        traj = trajectory_df.copy()
        traj['vehicle_id_str'] = traj['vehicle_id'].astype(str)
        # Identify flows only when a 'flow' token exists in vehicle_id (case-insensitive)
        traj['flow_id'] = traj['vehicle_id_str'].str.extract(r'(?i)(flow[_A-Za-z0-9-]*)', expand=False)

        flows = traj[traj['flow_id'].notna()]['flow_id'].unique()
        if len(flows) == 0:
            print("No explicit flow-tagged vehicle IDs found. Skipping flow-based diagrams.")
            return

        print(f"Found {len(flows)} flows for flow-based 2D detection plotting")

        # Try to load conflict log (optional)
        conflict_df = pd.DataFrame()
        conflicts_file = Path(self.config['scenario_path']) / 'out_logging' / f'log_conflicts_{Path(self.config["scenario_path"]).name}.csv'
        if conflicts_file.exists():
            try:
                with open(conflicts_file, 'r') as f:
                    # find header
                    lines = f.readlines()
                header_idx = next((i for i,l in enumerate(lines) if not l.strip().startswith('#') and l.strip()), 0)
                conflict_df = pd.read_csv(conflicts_file, skiprows=header_idx)
            except Exception:
                conflict_df = pd.DataFrame()

        # For each flow, build diagram
        for flow_id in flows:
            flow_data = traj[traj['flow_id'] == flow_id].copy()
            if flow_data.empty:
                continue

            # Ensure numeric time_step and distance columns
            flow_data['time_step'] = pd.to_numeric(flow_data['time_step'], errors='coerce')
            flow_data['distance'] = pd.to_numeric(flow_data['distance'], errors='coerce')

            # time_step is now already in seconds (simulation time), not frame numbers
            # So abs_time_s is just a copy of time_step
            flow_data['abs_time_s'] = flow_data['time_step'].astype(float)

            # Compute flow-level start/end times from earliest valid per-vehicle sample
            valid_times = flow_data['abs_time_s'].dropna()
            if valid_times.empty:
                continue
            # flow_start_time should be the earliest appearance of any bicycle in this flow
            flow_start_time = float(valid_times.min())
            end_time = float(valid_times.max())
            start_time = flow_start_time

            # Compute a flow-level spatial baseline so all bicycles align to a common origin.
            # Use each vehicle's earliest reported distance as its start; take the minimum
            # as the flow baseline (should usually be 0). Subtract this baseline when
            # plotting so space axis starts at 0 for the flow.
            # Compute the flow baseline (earliest reported distance per vehicle),
            # avoid groupby.apply deprecation by sorting first and taking first per group.
            try:
                first_distances = flow_data.sort_values('time_step').groupby('vehicle_id')['distance'].first().astype(float)
                flow_baseline = float(first_distances.min()) if len(first_distances) > 0 else 0.0
            except Exception:
                flow_baseline = 0.0

            # Prepare to collect per-sub-trajectory start info for diagnostics
            starts_info = []  # list of dicts: vehicle_id, span_index, first_time_s, first_distance
            # Prepare output names early to allow saving diagnostic CSVs during processing
            out_dir = Path(self.config['output_dir'])
            out_dir.mkdir(parents=True, exist_ok=True)
            file_tag = self.config.get('file_tag', Path(self.config['scenario_path']).name)
            fco = int(self.config.get('fco_share', 0))
            fbo = int(self.config.get('fbo_share', 0))

            # Prepare plot (time on x-axis, distance on y-axis to match individual trajectory plots)
            fig, ax = plt.subplots(figsize=self.config['figure_size'])

            # Warm-up line if configured (optional) - draw vertical warm-up time line
            warmup = self.config.get('delay', 0)
            if warmup and start_time < warmup:
                ax.axvline(x=warmup, color='firebrick', linestyle='--', alpha=0.5)
                ax.text(warmup, 0, 'simulation warm-up', color='firebrick', va='bottom', ha='left')

            # Set x-axis limits to start at the earliest bicycle appearance (flow_start_time)
            ax.set_xlim(left=start_time, right=end_time)

            # Track flow-level statistics
            total_flow_distance = 0.0
            total_flow_detected_distance = 0.0
            total_flow_time = 0.0
            total_flow_detected_time = 0.0
            plotted_any = False

            # Plot each bicycle in the flow with smoothed detection segments
            # To avoid concatenating re-used vehicle_ids or interleaved records, split
            # per-vehicle into contiguous sub-trajectories when there are large time gaps
            # or obvious resets. Make the split threshold configurable (seconds).
            max_gap_seconds = float(self.config.get('max_continuous_gap_s', 5.0))

            for vehicle_id, g in flow_data.groupby('vehicle_id'):
                # Use the original file/log order to detect separate instances of
                # the same vehicle_id. A decreasing time_step in file order
                # usually indicates a new instance or replay; likewise large
                # time gaps or sudden drops in cumulative distance indicate
                # a new traversal. We'll split on any of these conditions.
                g = g.reset_index(drop=True)

                # Convert arrays (keep file order)
                abs_times = g['abs_time_s'].astype(float).values
                distances_arr = g['distance'].astype(float).values

                if len(abs_times) < 2:
                    continue

                # Find break points where:
                #  - time decreases (new instance),
                #  - gap exceeds threshold (long pause), or
                #  - distance decreases substantially (reset)
                time_diffs = np.diff(abs_times)
                dist_diffs = np.diff(distances_arr)
                # boolean mask for breaks
                breaks_mask = (time_diffs < -1e-6) | (time_diffs > max_gap_seconds) | (dist_diffs < -1.0)
                break_idxs = np.where(breaks_mask)[0]

                # Build contiguous sub-trajectory index ranges
                spans = []
                start_idx = 0
                for b in break_idxs:
                    spans.append((start_idx, b))
                    start_idx = b + 1
                spans.append((start_idx, len(abs_times)-1))

                # For each contiguous sub-trajectory, collect data first then decide which
                # spans to plot. This lets us suppress stray mid-route spans that appear
                # earlier in time than the primary start (usually an artefact).
                vehicle_spans = []
                for span_idx, (sidx, eidx) in enumerate(spans):
                    sub_g = g.iloc[sidx:eidx+1]
                    if len(sub_g) < 2:
                        continue

                    # For timeline creation and plotting, sort the span chronologically
                    sub_g_sorted = sub_g.sort_values('time_step').reset_index(drop=True)

                    # Build detection timeline aligned to these chronologically-ordered samples
                    det_events = detection_df[detection_df['bicycle_id'] == vehicle_id] if len(detection_df) > 0 else pd.DataFrame()
                    if not det_events.empty and 'time_step' in det_events.columns:
                        tmin = sub_g_sorted['time_step'].astype(float).min()
                        tmax = sub_g_sorted['time_step'].astype(float).max()
                        det_events = det_events[(det_events['time_step'] >= tmin) & (det_events['time_step'] <= tmax)]

                    bike_time_steps = sub_g_sorted['time_step'].astype(float).values
                    detection_timeline = self._create_detection_timeline(bike_time_steps, det_events, bike_time_steps[0])
                    smoothed = self._smooth_detection_timeline(detection_timeline)

                    distances = sub_g_sorted['distance'].astype(float).tolist()
                    times = sub_g_sorted['abs_time_s'].astype(float).tolist()

                    segments = self._split_trajectory_segments(distances, times, smoothed)

                    span_info = {
                        'vehicle_id': vehicle_id,
                        'span_index': span_idx,
                        'first_time_s': float(times[0]),
                        'first_distance': float(distances[0]),
                        'last_time_s': float(times[-1]),
                        'n_points': len(times),
                        'duration_s': float(times[-1] - times[0]),
                        'segments': segments,
                        'times': times,
                        'distances': distances
                    }
                    vehicle_spans.append(span_info)

                # Decide which spans to plot: prefer spans that start near distance 0 as primary.
                flow_start_dist_thresh = float(self.config.get('flow_start_distance_threshold_m', 1.0))
                # Find earliest primary span for this vehicle (if any)
                primary_candidates = [s for s in vehicle_spans if s['first_distance'] <= flow_start_dist_thresh]
                primary_time = None
                if primary_candidates:
                    primary_time = min(s['first_time_s'] for s in primary_candidates)

                # Mark spans to ignore: spans that start before primary_time AND start mid-route
                spans_to_plot = []
                for s in vehicle_spans:
                    ignored = False
                    if primary_time is not None and s['first_time_s'] < primary_time and s['first_distance'] > flow_start_dist_thresh:
                        # This is a stray mid-route span that occurs earlier in time than the
                        # main near-zero-distance run -> ignore to avoid moving flow origin
                        ignored = True
                    s['ignored'] = ignored
                    # Append diagnostic start info for every span (including ignored)
                    starts_info.append({
                        'vehicle_id': s['vehicle_id'],
                        'span_index': s['span_index'],
                        'first_time_s': s['first_time_s'],
                        'first_distance': s['first_distance'],
                        'ignored': bool(ignored)
                    })
                    if not ignored:
                        spans_to_plot.append(s)

                # Now plot only non-ignored spans
                for s in spans_to_plot:
                    distances = s['distances']
                    times = s['times']
                    segments = s['segments']

                    # Calculate total trajectory distance and time (full span)
                    bike_total_distance = distances[-1] - distances[0]
                    bike_total_time = s['duration_s']
                    bike_detected_distance = 0.0
                    bike_detected_time = 0.0

                    # Plot undetected segments
                    for segment in segments['undetected']:
                        if len(segment) > 1:
                            distances_s, times_s = zip(*segment)
                            adj_distances = [d - flow_baseline for d in distances_s]
                            ax.plot(times_s, adj_distances, color='black', linewidth=1.5, linestyle='solid')
                    
                    # Plot detected segments and accumulate detected metrics
                    for segment in segments['detected']:
                        if len(segment) > 1:
                            distances_s, times_s = zip(*segment)
                            adj_distances = [d - flow_baseline for d in distances_s]
                            ax.plot(times_s, adj_distances, color='darkturquoise', linewidth=1.5, linestyle='solid')
                            # Match individual plot calculation
                            seg_distance = segment[-1][0] - segment[0][0]
                            seg_time = segment[-1][1] - segment[0][1]
                            bike_detected_distance += seg_distance
                            bike_detected_time += seg_time

                    total_flow_distance += bike_total_distance
                    total_flow_detected_distance += bike_detected_distance
                    total_flow_time += bike_total_time
                    total_flow_detected_time += bike_detected_time
                    plotted_any = True

                # Conflicts: intentionally not plotted for flow-based diagrams (user requested)

            if not plotted_any:
                print(f"No valid trajectories to plot for flow {flow_id}")
                plt.close(fig)
                continue

            # Update flow_start_time from starts_info if available
            if starts_info:
                starts_df = pd.DataFrame(starts_info)
                # Sort starts chronologically so first spans are the earliest in time
                if not starts_df.empty:
                    starts_df = starts_df.sort_values('first_time_s').reset_index(drop=True)
                    # Prefer spans that begin near the route origin (distance approx 0)
                    # to determine the flow's display start time. This avoids cases
                    # where a mid-route span with an earlier logged time (e.g. 0.1s)
                    # incorrectly becomes the flow origin.
                    start_dist_thresh = float(self.config.get('flow_start_distance_threshold_m', 1.0))
                    candidate = starts_df[starts_df['first_distance'] <= start_dist_thresh]
                    if not candidate.empty:
                        flow_start_time = float(candidate['first_time_s'].min())
                    else:
                        flow_start_time = float(starts_df['first_time_s'].min())
                    
                    # Add padding to x-axis limits per user request:
                    # Round start down and end up to improve visualization margins
                    import math
                    padding_interval = 5.0  # seconds
                    padded_start = math.floor(flow_start_time / padding_interval) * padding_interval
                    padded_end = math.ceil(end_time / padding_interval) * padding_interval
                    ax.set_xlim(left=padded_start, right=padded_end)

            # Traffic lights: aggregate TL state events across all bicycles in the flow
            # and build a continuous timeline per TL so overlays span the full flow duration.
            tl_info = {}
            required_cols = {'next_tl_id', 'next_tl_distance', 'next_tl_state', 'next_tl_index'}
            if required_cols.intersection(flow_data.columns):
                # Collect all TL state observations from any bicycle in the flow
                tl_rows = flow_data[flow_data['next_tl_id'].notna() & (flow_data['next_tl_id'] != '')].copy()
                if not tl_rows.empty:
                    # time_step is now already in seconds, so abs_time_s is just a copy
                    tl_rows['abs_time_s'] = tl_rows['time_step'].astype(float)
                    for tl_id, tlg in tl_rows.groupby('next_tl_id'):
                        # For this TL, collect events (time, state, absolute position)
                        events = []
                        for _, row in tlg.iterrows():
                            state = row.get('next_tl_state')
                            rel_dist = row.get('next_tl_distance', np.nan)
                            bike_dist = row.get('distance', np.nan)
                            if pd.isna(state) or pd.isna(rel_dist) or pd.isna(bike_dist):
                                continue
                            t = float(row['abs_time_s'])
                            pos = float(bike_dist) + float(rel_dist)
                            events.append({'time': t, 'state': state, 'position': pos, 'signal_index': int(row.get('next_tl_index', 0))})

                        if not events:
                            continue

                        # Sort events by time
                        events = sorted(events, key=lambda x: x['time'])

                        # Use median of reported positions as the TL absolute position
                        positions = [e['position'] for e in events if not pd.isna(e['position'])]
                        avg_pos = float(np.median(positions)) if positions else np.nan

                        # Build continuous state segments: extend each observed state until next observed change
                        segments = []
                        for i, e in enumerate(events):
                            t0 = e['time']
                            state = e['state']
                            t1 = events[i+1]['time'] if i+1 < len(events) else end_time
                            # skip degenerate zero-length
                            if t1 <= t0:
                                continue
                            segments.append({'t0': t0, 't1': t1, 'state': state, 'position': avg_pos})

                        if segments:
                            tl_info[tl_id] = {'segments': segments, 'signal_index': events[0].get('signal_index', 0), 'avg_position': avg_pos}

            # Fallback: use traffic_light_df to infer TLs if embedded data not present
            traffic_light_programs = {}
            if not traffic_light_df.empty and 'traffic_light_id' in traffic_light_df.columns:
                for tl_id, tlg in traffic_light_df.groupby('traffic_light_id'):
                    entries = []
                    for _, row in tlg.sort_values('time_step').iterrows():
                        # time_step is now already in seconds, no conversion needed
                        entries.append((row['time_step'], row.get('signal_states', row.get('signal_state', ''))))
                    traffic_light_programs[tl_id] = entries

            # Plot traffic lights similar to individual plots (horizontal colored segments at TL position)
            if tl_info:
                for tl_id, data in tl_info.items():
                    pos = data.get('avg_position', np.nan)
                    if not np.isnan(pos):
                        # Adjust position for flow baseline
                        adj_pos = pos - flow_baseline if not np.isnan(pos) else pos
                        # horizontal faint baseline at TL distance (using adjusted position)
                        ax.axhline(y=adj_pos, xmin=0, xmax=1, color='black', linestyle='--', alpha=0.3, linewidth=0.6, zorder=1)
                        
                        # Merge consecutive segments with same color to avoid overlapping dashes appearing solid
                        segments = data.get('segments', [])
                        merged_segments = []
                        if segments:
                            current_color = {'r': 'red', 'y': 'yellow', 'g': 'green', 'G': 'green'}.get(str(segments[0]['state']).lower()[0], 'gray')
                            current_start = segments[0]['t0']
                            current_end = segments[0]['t1']
                            
                            for seg in segments[1:]:
                                seg_color = {'r': 'red', 'y': 'yellow', 'g': 'green', 'G': 'green'}.get(str(seg['state']).lower()[0], 'gray')
                                if seg_color == current_color and seg['t0'] <= current_end:
                                    # Same color and touching/overlapping - extend current segment
                                    current_end = max(current_end, seg['t1'])
                                else:
                                    # Different color or gap - save current and start new
                                    merged_segments.append({'t0': current_start, 't1': current_end, 'color': current_color})
                                    current_color = seg_color
                                    current_start = seg['t0']
                                    current_end = seg['t1']
                            
                            # Don't forget the last segment
                            merged_segments.append({'t0': current_start, 't1': current_end, 'color': current_color})
                        
                        # Now plot merged segments
                        for seg in merged_segments:
                            ax.plot([seg['t0'], seg['t1']], [adj_pos, adj_pos], 
                                   color=seg['color'], linewidth=2, linestyle='--', alpha=0.8, zorder=5)
            else:
                if traffic_light_programs and 'distance' in flow_data.columns:
                    approx_pos = (flow_data['distance'].min() + flow_data['distance'].max()) / 2
                    ax.axhline(y=approx_pos, xmin=0, xmax=1, color='gray', linestyle='--', alpha=0.3)
                    
                    for tl_id, prog in traffic_light_programs.items():
                        # Build segments and merge consecutive ones with same color
                        segments = []
                        for i in range(len(prog)-1):
                            t0, state = prog[i]
                            t1 = prog[i+1][0]
                            color = {'r': 'red', 'y': 'yellow', 'g': 'green', 'G': 'green'}.get(str(state).lower()[0], 'gray') if state else 'gray'
                            segments.append({'t0': t0, 't1': t1, 'color': color})
                        
                        # Merge consecutive segments with same color
                        merged_segments = []
                        if segments:
                            current_color = segments[0]['color']
                            current_start = segments[0]['t0']
                            current_end = segments[0]['t1']
                            
                            for seg in segments[1:]:
                                if seg['color'] == current_color and seg['t0'] <= current_end:
                                    # Same color and touching/overlapping - extend
                                    current_end = max(current_end, seg['t1'])
                                else:
                                    # Different color or gap - save and start new
                                    merged_segments.append({'t0': current_start, 't1': current_end, 'color': current_color})
                                    current_color = seg['color']
                                    current_start = seg['t0']
                                    current_end = seg['t1']
                            
                            merged_segments.append({'t0': current_start, 't1': current_end, 'color': current_color})
                        
                        # Plot merged segments
                        for seg in merged_segments:
                            ax.plot([seg['t0'], seg['t1']], [approx_pos, approx_pos], 
                                   color=seg['color'], linewidth=2, linestyle='--', alpha=0.8, zorder=5)

            # Finalize plot appearance (time on x, distance on y)
            # Note: x-axis limits are set earlier after computing padded_start/padded_end from sub-trajectory starts
            # Do not override them here
            ax.set_xlabel('Simulation Time [s]')
            ax.set_ylabel('Space [m]')
            ax.grid(True)

            # Primary legend: trajectory / TL colors
            handles = [
                Line2D([0], [0], color='black', lw=2, label='Undetected'),
                Line2D([0], [0], color='darkturquoise', lw=2, label='Detected'),
            ]
            # TL legend entries
            handles_tl = [
                Line2D([0], [0], color='red', lw=2, linestyle='--', label='Red TL'),
                Line2D([0], [0], color='yellow', lw=2, linestyle='--', label='Yellow TL'),
                Line2D([0], [0], color='green', lw=2, linestyle='--', label='Green TL')
            ]

            # Calculate flow detection rates using centralized method
            flow_detection_rates = self._calculate_flow_detection_rates(
                total_flow_distance, total_flow_detected_distance,
                total_flow_time, total_flow_detected_time
            )

            # Secondary legend: flow info (detection rates) -> place top-left
            info_lines = [
            f"Flow: {flow_id} ({flow_data['vehicle_id'].nunique()} bicycles)",
            f"Temporal detection rate: {flow_detection_rates['time_rate']:.1f}%",
            f"Spatial detection rate: {flow_detection_rates['distance_rate']:.1f}%",
            f"Spatio-temporal detection rate: {flow_detection_rates['spatiotemporal_rate']:.1f}%"
            ]

            # Create flow-info legend in top-left using invisible handles
            # Use ax.add_artist() to prevent it from being replaced by the second legend
            # Match font size from individual trajectory plots
            # Set handlelength=0 and handletextpad=0 to remove blank space before text
            info_handles = [Line2D([0], [0], color='white', label=l) for l in info_lines]
            info_legend = ax.legend(handles=info_handles, loc='upper left', bbox_to_anchor=(0.01, 0.99), 
                                    fontsize=plt.rcParams['legend.fontsize'], framealpha=0.9,
                                    handlelength=0, handletextpad=0)
            ax.add_artist(info_legend)

            # Place trajectory + TL legend in bottom-right (same as individual plots)
            ax.legend(handles=handles + handles_tl, loc='lower right', bbox_to_anchor=(0.99, 0.01))

            # Ensure output subdirectory exists
            out_dir = Path(self.config['output_dir']) / '2D_detection_flow-based'
            out_dir.mkdir(parents=True, exist_ok=True)
            file_tag = self.config.get('file_tag', Path(self.config['scenario_path']).name)
            fco = int(self.config.get('fco_share', 0))
            fbo = int(self.config.get('fbo_share', 0))
            flow_plot_path = out_dir / f"2D_detection_flow-based_{flow_id}_{file_tag}_FCO{fco}%_FBO{fbo}%.png"
            plt.savefig(str(flow_plot_path), dpi=self.config.get('dpi', DPI), bbox_inches='tight')
            plt.close(fig)

            print(f"  ✓ Saved flow-based 2D detection plot: {flow_plot_path.name}")
        
        print(f"\n✓ Generated {len(flows)} flow-based 2D detection plots")
    
    def _process_flow_based_redundancy_from_logs(self, trajectory_df, detection_df, traffic_light_df):
        """Create flow-based redundancy space-time diagrams showing observer count per trajectory segment.
        
        Uses same detection data source as detection plots (detection_df) for consistency.
        """
        print("\n=== Processing Flow-Based 2D Detection-Redundancy Plots ===")
        
        traj = trajectory_df.copy()
        traj['vehicle_id_str'] = traj['vehicle_id'].astype(str)
        # Identify flows only when a 'flow' token exists in vehicle_id
        traj['flow_id'] = traj['vehicle_id_str'].str.extract(r'(?i)(flow[_A-Za-z0-9-]*)', expand=False)
        
        flows = traj[traj['flow_id'].notna()]['flow_id'].unique()
        if len(flows) == 0:
            print("No explicit flow-tagged vehicle IDs found. Skipping flow-based redundancy diagrams.")
            return
        
        print(f"Found {len(flows)} flows for flow-based 2D detection-redundancy plotting")
        
        # Get color palette for redundancy levels
        colors = _get_redundancy_color_palette()
        
        # For each flow, build redundancy diagram
        for flow_id in flows:
            flow_data = traj[traj['flow_id'] == flow_id].copy()
            if flow_data.empty:
                continue
            
            # Ensure numeric columns
            flow_data['time_step'] = pd.to_numeric(flow_data['time_step'], errors='coerce')
            flow_data['distance'] = pd.to_numeric(flow_data['distance'], errors='coerce')
            
            # Get time range
            valid_times = flow_data['time_step'].dropna()
            if valid_times.empty:
                continue
            flow_start_time = float(valid_times.min())
            end_time = float(valid_times.max())
            
            # Get spatial baseline
            try:
                first_distances = flow_data.sort_values('time_step').groupby('vehicle_id')['distance'].first().astype(float)
                flow_baseline = float(first_distances.min()) if len(first_distances) > 0 else 0.0
            except Exception:
                flow_baseline = 0.0
            
            # Create figure
            fig, ax = plt.subplots(figsize=self.config['figure_size'])
            ax.set_xlim(left=flow_start_time, right=end_time)
            
            # Initialize counters for detection rates
            total_flow_distance = 0.0
            total_flow_time = 0.0
            total_flow_detected_distance = 0.0
            total_flow_detected_time = 0.0
            
            # Plot each bicycle's trajectory with redundancy color coding
            # Use same span-splitting logic as detection plots for consistency
            max_gap_seconds = float(self.config.get('max_continuous_gap_s', 5.0))
            
            for vehicle_id, g in flow_data.groupby('vehicle_id'):
                g = g.reset_index(drop=True)
                
                # Convert arrays (keep file order)
                abs_times = g['time_step'].astype(float).values
                distances_arr = g['distance'].astype(float).values
                
                if len(abs_times) < 2:
                    continue
                
                # Find break points where time decreases, gap exceeds threshold, or distance resets
                time_diffs = np.diff(abs_times)
                dist_diffs = np.diff(distances_arr)
                breaks_mask = (time_diffs < -1e-6) | (time_diffs > max_gap_seconds) | (dist_diffs < -1.0)
                break_idxs = np.where(breaks_mask)[0]
                
                # Build contiguous sub-trajectory index ranges
                spans = []
                start_idx = 0
                for b in break_idxs:
                    spans.append((start_idx, b))
                    start_idx = b + 1
                spans.append((start_idx, len(abs_times)-1))
                
                # Process each contiguous sub-trajectory
                vehicle_spans = []
                for span_idx, (sidx, eidx) in enumerate(spans):
                    sub_g = g.iloc[sidx:eidx+1]
                    if len(sub_g) < 2:
                        continue
                    
                    sub_g_sorted = sub_g.sort_values('time_step').reset_index(drop=True)
                    
                    times = sub_g_sorted['time_step'].astype(float).values
                    distances = sub_g_sorted['distance'].astype(float).values - flow_baseline
                    
                    # Use same detection data source as detection plots (detection_df)
                    det_events = detection_df[detection_df['bicycle_id'] == vehicle_id] if len(detection_df) > 0 else pd.DataFrame()
                    if not det_events.empty and 'time_step' in det_events.columns:
                        tmin = times.min()
                        tmax = times.max()
                        det_events = det_events[(det_events['time_step'] >= tmin) & (det_events['time_step'] <= tmax)]
                    
                    detection_timeline = self._create_detection_timeline(times, det_events, times[0])
                    smoothed_detection = self._smooth_detection_timeline(detection_timeline)
                    
                    # Extract redundancy values from CSV (for color coding only)
                    if 'num_detecting_observers' in sub_g_sorted.columns:
                        redundancy_values = sub_g_sorted['num_detecting_observers'].values
                    else:
                        redundancy_values = np.zeros(len(detection_timeline), dtype=int)
                        redundancy_values[detection_timeline] = 1
                    
                    # Create smoothed redundancy values
                    smoothed_redundancy = np.zeros(len(redundancy_values), dtype=int)
                    for i in range(len(smoothed_redundancy)):
                        if smoothed_detection[i]:
                            smoothed_redundancy[i] = max(1, redundancy_values[i])
                        else:
                            smoothed_redundancy[i] = 0
                    
                    # Split into segments by redundancy level
                    redundancy_segments = self._split_trajectory_segments_by_redundancy(
                        distances.tolist(), times.tolist(), smoothed_redundancy
                    )
                    
                    # Create basic segments for consistent detection rate calculation
                    segments = self._split_trajectory_segments(
                        distances.tolist(), times.tolist(), smoothed_detection
                    )
                    
                    span_info = {
                        'vehicle_id': vehicle_id,
                        'span_index': span_idx,
                        'first_time_s': float(times[0]),
                        'first_distance': float(distances[0] + flow_baseline),
                        'times': times.tolist(),
                        'distances': distances.tolist(),
                        'segments': segments,
                        'redundancy_segments': redundancy_segments,
                        'duration_s': float(times[-1] - times[0])
                    }
                    vehicle_spans.append(span_info)
                
                # Filter out stray mid-route spans (match detection plot logic)
                flow_start_dist_thresh = float(self.config.get('flow_start_distance_threshold_m', 1.0))
                primary_candidates = [s for s in vehicle_spans if s['first_distance'] <= flow_start_dist_thresh]
                primary_time = None
                if primary_candidates:
                    primary_time = min(s['first_time_s'] for s in primary_candidates)
                
                spans_to_plot = []
                for s in vehicle_spans:
                    ignored = False
                    if primary_time is not None and s['first_time_s'] < primary_time and s['first_distance'] > flow_start_dist_thresh:
                        ignored = True
                    if not ignored:
                        spans_to_plot.append(s)
                
                # Process only non-ignored spans
                for s in spans_to_plot:
                    times = s['times']
                    distances = s['distances']
                    segments = s['segments']
                    redundancy_segments = s['redundancy_segments']
                    
                    # Calculate detection statistics for this span
                    # Use ORIGINAL total distance/time (baseline subtraction doesn't affect differences)
                    bike_total_distance = distances[-1] - distances[0]
                    bike_total_time = s['duration_s']
                    bike_detected_distance = 0.0
                    bike_detected_time = 0.0
                    
                    # Calculate detected distance/time from basic detected segments
                    # Use segments['detected'] for 100% consistency with detection and occlusion plots
                    for segment in segments['detected']:
                        if len(segment) > 1:
                            seg_distance = segment[-1][0] - segment[0][0]
                            seg_time = segment[-1][1] - segment[0][1]
                            bike_detected_distance += seg_distance
                            bike_detected_time += seg_time
                    
                    # Accumulate into flow totals
                    total_flow_distance += bike_total_distance
                    total_flow_detected_distance += bike_detected_distance
                    total_flow_time += bike_total_time
                    total_flow_detected_time += bike_detected_time
                    
                    # Plot each redundancy level
                    for redundancy_level in [0, 1, 2, 3, 4, 5]:
                        for segment in redundancy_segments[redundancy_level]:
                            if len(segment) > 1:
                                seg_distances, seg_times = zip(*segment)
                                ax.plot(seg_times, seg_distances, color=colors[redundancy_level],
                                       linewidth=1.5, linestyle='solid', alpha=0.8)
            
            # Calculate flow detection rates using centralized method
            flow_detection_rates = self._calculate_flow_detection_rates(
                total_flow_distance, total_flow_detected_distance,
                total_flow_time, total_flow_detected_time
            )
            
            # Add traffic light information if available (from bicycle trajectory embedded data)
            tl_info = {}
            if len(traffic_light_df) > 0:
                # Collect all traffic light state changes from all bicycles
                tl_all_states = {}  # tl_id -> list of all state changes from all bicycles
                
                for vehicle_id, g in flow_data.groupby('vehicle_id'):
                    if 'next_tl_state' not in g.columns or 'next_tl_distance' not in g.columns:
                        continue
                    
                    # Group by traffic light ID if available
                    if 'next_tl_id' in g.columns:
                        for tl_id in g['next_tl_id'].dropna().unique():
                            if tl_id not in tl_all_states:
                                tl_all_states[tl_id] = []
                            
                            tl_data = g[g['next_tl_id'] == tl_id].copy()
                            tl_data = tl_data.sort_values('time_step')
                            
                            prev_state = None
                            for _, row in tl_data.iterrows():
                                current_state = row['next_tl_state']
                                if pd.notna(current_state) and current_state != '' and current_state != prev_state:
                                    tl_position = row['distance'] - flow_baseline + row['next_tl_distance']
                                    tl_all_states[tl_id].append({
                                        'time': row['time_step'],
                                        'state': str(current_state).lower(),
                                        'position': tl_position
                                    })
                                    prev_state = current_state
                
                # Merge and deduplicate states from all bicycles for each traffic light
                for tl_id, all_states in tl_all_states.items():
                    if not all_states:
                        continue
                    
                    # Sort by time
                    all_states.sort(key=lambda x: x['time'])
                    
                    # Deduplicate consecutive states (keep unique state changes)
                    unique_states = []
                    prev_state = None
                    for state in all_states:
                        if state['state'] != prev_state:
                            unique_states.append(state)
                            prev_state = state['state']
                    
                    if unique_states:
                        avg_position = np.mean([s['position'] for s in unique_states])
                        tl_info[tl_id] = {'states': unique_states, 'avg_position': avg_position}
            
            # Plot traffic lights as horizontal colored segments
            if tl_info:
                # Get maximum time across all bicycles in the flow for traffic light plotting
                max_time_in_flow = flow_data['time_step'].max()
                
                for tl_id, tl_data in tl_info.items():
                    states = tl_data['states']
                    avg_position = tl_data['avg_position']
                    
                    # Plot horizontal line at traffic light position
                    ax.axhline(y=avg_position, color='black', linestyle='--', alpha=0.5, linewidth=0.5, zorder=1)
                    
                    # Plot state changes as colored segments
                    for i, state in enumerate(states):
                        signal_state = state['state']
                        start_time = state['time']
                        end_time = states[i+1]['time'] if i+1 < len(states) else max_time_in_flow
                        
                        # Map states to colors
                        color = {'r': 'red', 'y': 'orange', 'g': 'green'}.get(signal_state, 'gray')
                        
                        if start_time <= end_time:
                            ax.plot([start_time, end_time], [avg_position, avg_position],
                                   color=color, linewidth=2, linestyle='--', alpha=0.8, zorder=5)
            
            # Create legend with redundancy levels
            handles = [
                Line2D([0], [0], color=colors[0], lw=2, label='Undetected (0)'),
                Line2D([0], [0], color=colors[1], lw=2, label='1 Observer'),
                Line2D([0], [0], color=colors[2], lw=2, label='2 Observers'),
                Line2D([0], [0], color=colors[3], lw=2, label='3 Observers'),
                Line2D([0], [0], color=colors[4], lw=2, label='4 Observers'),
                Line2D([0], [0], color=colors[5], lw=2, label='5+ Observers')
            ]
            
            # Add traffic light legend items if any were plotted
            if tl_info:
                handles.extend([
                    Line2D([0], [0], color='red', linestyle='--', alpha=0.7, label='Red TL'),
                    Line2D([0], [0], color='orange', linestyle='--', alpha=0.7, label='Yellow TL'),
                    Line2D([0], [0], color='green', linestyle='--', alpha=0.7, label='Green TL')
                ])
            
            # Calculate flow detection rates using centralized method
            flow_detection_rates = self._calculate_flow_detection_rates(
                total_flow_distance, total_flow_detected_distance,
                total_flow_time, total_flow_detected_time
            )
            
            # Add flow info with detection rates in top-left
            num_bicycles = flow_data['vehicle_id'].nunique()
            info_lines = [
                f"Flow: {flow_id} ({num_bicycles} bicycles)",
                f"Temporal detection rate: {flow_detection_rates['time_rate']:.1f}%",
                f"Spatial detection rate: {flow_detection_rates['distance_rate']:.1f}%",
                f"Spatio-temporal detection rate: {flow_detection_rates['spatiotemporal_rate']:.1f}%"
            ]
            info_handles = [Line2D([0], [0], color='white', label=l) for l in info_lines]
            info_legend = ax.legend(handles=info_handles, loc='upper left', bbox_to_anchor=(0.01, 0.99),
                                   fontsize=plt.rcParams['legend.fontsize'], framealpha=0.9,
                                   handlelength=0, handletextpad=0)
            ax.add_artist(info_legend)
            
            # Place redundancy legend in bottom-right
            ax.legend(handles=handles, loc='lower right', bbox_to_anchor=(0.99, 0.01))
            
            # Set labels
            ax.set_xlabel('Simulation Time [s]')
            ax.set_ylabel('Space [m]')
            ax.grid(True)
            
            # Save plot
            out_dir = Path(self.config['output_dir']) / '2D_detection-redundancy_flow-based'
            out_dir.mkdir(parents=True, exist_ok=True)
            file_tag = self.config['file_tag']
            fco = int(self.config.get('fco_share', 0))
            fbo = int(self.config.get('fbo_share', 0))
            flow_plot_path = out_dir / f"2D_detection-redundancy_flow-based_{flow_id}_{file_tag}_FCO{fco}%_FBO{fbo}%.png"
            
            plt.tight_layout()
            plt.savefig(str(flow_plot_path), dpi=self.config.get('dpi', 150), bbox_inches='tight')
            plt.close(fig)
            
            print(f"  ✓ Saved flow-based 2D detection-redundancy plot: {flow_plot_path.name}")
        
        print(f"\n✓ Generated {len(flows)} flow-based 2D detection-redundancy plots")
    
    def _create_detection_timeline(self, time_steps, detection_df, start_time_step):
        """Create detection timeline for a bicycle."""
        detection_timeline = np.zeros(len(time_steps), dtype=bool)
        
        if len(detection_df) == 0:
            return detection_timeline
        
        # Match detection events with time steps
        for i, time_step in enumerate(time_steps):
            # Find detection events close to this time step
            detection_events = detection_df[
                np.abs(detection_df['time_step'] - time_step) <= (self.config['step_length'] * 10)
            ]
            
            if len(detection_events) > 0:
                detection_timeline[i] = True
        
        return detection_timeline
    
    def _create_occlusion_timeline(self, time_steps, detection_df, start_time_step):
        """Create occlusion level timeline for detected time steps."""
        occlusion_timeline = np.zeros(len(time_steps), dtype=float)
        
        if len(detection_df) == 0 or 'occlusion_level' not in detection_df.columns:
            return occlusion_timeline
        
        # Match detection events with time steps and extract occlusion levels
        for i, time_step in enumerate(time_steps):
            detection_events = detection_df[
                np.abs(detection_df['time_step'] - time_step) <= (self.config['step_length'] * 10)
            ]
            
            if len(detection_events) > 0:
                # Use average occlusion level if multiple observers
                occlusion_timeline[i] = detection_events['occlusion_level'].mean()
        
        return occlusion_timeline
    
    def _smooth_detection_timeline(self, detection_timeline):
        """Apply smoothing to detection timeline to bridge small gaps."""
        smoothed = detection_timeline.copy()
        
        # Apply gap bridging
        gap_counter = 0
        in_gap = False
        gap_start = 0
        
        for i in range(len(detection_timeline)):
            if detection_timeline[i]:
                if in_gap and gap_counter <= self.config['max_gap_bridge']:
                    # Bridge the gap
                    smoothed[gap_start:i] = True
                in_gap = False
                gap_counter = 0
            else:
                if not in_gap:
                    gap_start = i
                    in_gap = True
                gap_counter += 1
        
        return smoothed
    
    def _split_trajectory_segments(self, distances, times, detection_status, apply_min_length_filter=True):
        """Split trajectory into detected and undetected segments.
        
        Args:
            distances: List of distance values
            times: List of time values
            detection_status: Boolean array of detection status
            apply_min_length_filter: If True, filter out segments shorter than MIN_SEGMENT_LENGTH
        """
        segments = {'detected': [], 'undetected': []}
        
        if len(distances) == 0:
            return segments
        
        current_segment = []
        current_status = detection_status[0]
        
        for i in range(len(distances)):
            if detection_status[i] == current_status:
                current_segment.append((distances[i], times[i]))
            else:
                # Status changed, save current segment if long enough (or if filter disabled)
                if not apply_min_length_filter or len(current_segment) >= self.config['min_segment_length']:
                    segment_key = 'detected' if current_status else 'undetected'
                    segments[segment_key].append(current_segment)
                
                # Start new segment
                current_segment = [(distances[i], times[i])]
                current_status = detection_status[i]
        
        # Add final segment
        if not apply_min_length_filter or len(current_segment) >= self.config['min_segment_length']:
            segment_key = 'detected' if current_status else 'undetected'
            segments[segment_key].append(current_segment)
        
        return segments
    
    def _create_occlusion_segments(self, distances, times, detection_status, occlusion_timeline):
        """
        Split detected segments into occlusion level categories using configured scale.
        
        Args:
            distances: List of distance values
            times: List of time values
            detection_status: Boolean array of detection status
            occlusion_timeline: Array of occlusion levels (0-100%)
        
        Returns:
            Dictionary mapping category names to segment lists
        """
        # Get occlusion scale from configuration
        occlusion_scale = self.config.get('occlusion_level_scale', _get_occlusion_level_scale())
        
        # Initialize categories based on configured scale
        categories = {category_name: [] for category_name, _, _ in occlusion_scale}
        
        current_segment = []
        current_category = None
        
        for i in range(len(distances)):
            if not detection_status[i]:
                # Undetected - save any ongoing segment
                if current_segment and current_category:
                    categories[current_category].append(current_segment)
                current_segment = []
                current_category = None
                continue
            
            # Determine occlusion category based on configured scale
            occlusion = occlusion_timeline[i]
            category = None
            
            for category_name, min_pct, max_pct in occlusion_scale:
                if min_pct <= occlusion <= max_pct:
                    category = category_name
                    break
            
            # Fallback if no category matched
            if category is None:
                category = occlusion_scale[-1][0]
            
            if category == current_category:
                current_segment.append((distances[i], times[i]))
            else:
                # Category changed, save current segment
                if current_segment and current_category:
                    categories[current_category].append(current_segment)
                current_segment = [(distances[i], times[i])]
                current_category = category
        
        # Add final segment
        if current_segment and current_category:
            categories[current_category].append(current_segment)
        
        return categories
    
    def _split_trajectory_segments_by_redundancy(self, distances, times, redundancy_values, apply_min_length_filter=True):
        """
        Split trajectory into segments based on number of detecting observers.
        
        Args:
            distances: List of distance values along trajectory
            times: List of time values (elapsed time from bicycle start)
            redundancy_values: Array of num_detecting_observers for each point
            apply_min_length_filter: If True, filter out segments shorter than MIN_SEGMENT_LENGTH
            
        Returns:
            Dict with keys 0, 1, 2, 3, 4, 5 mapping to list of segments
            Each segment is a list of (distance, time) tuples
        """
        segments = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
        
        if len(distances) == 0:
            return segments
        
        current_segment = []
        current_redundancy = min(redundancy_values[0], 5)  # Cap at 5+ for visualization
        
        for i in range(len(distances)):
            redundancy = min(redundancy_values[i], 5)  # Cap at 5
            
            if redundancy == current_redundancy:
                current_segment.append((distances[i], times[i]))
            else:
                # Redundancy level changed, save current segment if long enough (or if filter disabled)
                if not apply_min_length_filter or len(current_segment) >= self.config['min_segment_length']:
                    segments[current_redundancy].append(current_segment)
                
                # Start new segment
                current_segment = [(distances[i], times[i])]
                current_redundancy = redundancy
        
        # Add final segment
        if not apply_min_length_filter or len(current_segment) >= self.config['min_segment_length']:
            segments[current_redundancy].append(current_segment)
        
        return segments
    
    def _split_trajectory_segments_3d(self, trajectory_3d, detection_status):
        """Split 3D trajectory into detected and undetected segments."""
        segments = {'detected': [], 'undetected': []}
        
        if len(trajectory_3d) == 0:
            return segments
        
        current_segment = []
        current_status = detection_status[0]
        
        for i in range(len(trajectory_3d)):
            if detection_status[i] == current_status:
                current_segment.append(trajectory_3d[i])
            else:
                # Status changed, save current segment if long enough
                if len(current_segment) >= self.config['min_segment_length']:
                    segment_key = 'detected' if current_status else 'undetected'
                    segments[segment_key].append(current_segment)
                
                # Start new segment
                current_segment = [trajectory_3d[i]]
                current_status = detection_status[i]
        
        # Add final segment
        if len(current_segment) >= self.config['min_segment_length']:
            segment_key = 'detected' if current_status else 'undetected'
            segments[segment_key].append(current_segment)
        
        return segments
    
    def _get_observer_trajectories_for_bicycle(self, bicycle_id, bicycle_detections, observer_df, start_time_step):
        """Get observer trajectories that detected the given bicycle."""
        observer_trajectories = {}
        
        if len(bicycle_detections) == 0 or len(observer_df) == 0:
            return observer_trajectories
        
        # Get unique observers that detected this bicycle
        detecting_observers = set(bicycle_detections['observer_id'].unique())
        
        for observer_id in detecting_observers:
            # Get observer trajectory data
            observer_data = observer_df[observer_df['vehicle_id'] == observer_id].copy()
            
            if len(observer_data) == 0:
                continue
                
            observer_data = observer_data.sort_values('time_step')
            
            # time_step is now in seconds, so elapsed time is just the difference
            observer_data['elapsed_time'] = observer_data['time_step'] - start_time_step
            
            # Only include points where elapsed time >= 0 (bicycle is active)
            observer_data = observer_data[observer_data['elapsed_time'] >= 0]
            
            if len(observer_data) == 0:
                continue
            
            # Create trajectory points
            trajectory_points = []
            for _, row in observer_data.iterrows():
                trajectory_points.append((row['x_coord'], row['y_coord'], row['elapsed_time']))
            
            # Get detection time periods for this observer-bicycle pair
            observer_detections = bicycle_detections[bicycle_detections['observer_id'] == observer_id]
            # time_step is now in seconds, so elapsed time is just the difference
            detection_times = set(det_row['time_step'] - start_time_step 
                                for _, det_row in observer_detections.iterrows())
            
            observer_trajectories[observer_id] = {
                'trajectory': trajectory_points,
                'detection_times': detection_times,
                'type': observer_data.iloc[0]['vehicle_type']
            }
        
        return observer_trajectories
    
    def _get_bicycle_traffic_lights(self, bicycle_data, traffic_light_df):
        """Extract traffic light information from bicycle trajectory data or infer from traffic light logs."""
        tl_info = {}
        
        # First try the original method - check if traffic light data is embedded in bicycle trajectory
        required_columns = ['next_tl_id', 'next_tl_distance', 'next_tl_state', 'next_tl_index']
        
        # If all required columns exist and have data, use the embedded approach
        if all(col in bicycle_data.columns for col in required_columns):
            tl_rows = bicycle_data[
                bicycle_data['next_tl_id'].notna() & 
                (bicycle_data['next_tl_id'] != '') &
                (bicycle_data['next_tl_id'] != '-') &  # Add check for dash character
                bicycle_data['next_tl_distance'].notna() &
                bicycle_data['next_tl_state'].notna() &
                bicycle_data['next_tl_index'].notna()
            ]
            
            if len(tl_rows) > 0:
                for tl_id in tl_rows['next_tl_id'].unique():
                    tl_data = tl_rows[tl_rows['next_tl_id'] == tl_id].copy()
                    
                    # time_step is now in seconds, so elapsed time is just the difference
                    start_time_step = bicycle_data['time_step'].min()
                    tl_data['elapsed_time'] = tl_data['time_step'] - start_time_step
                    
                    # Track state changes and position information
                    states = []
                    prev_state = None
                    prev_distance = None
                    
                    for _, row in tl_data.iterrows():
                        current_state = row['next_tl_state']
                        current_distance = row['next_tl_distance']
                        
                        # Skip rows with invalid data
                        if pd.isna(current_state) or pd.isna(current_distance) or current_state == '':
                            continue
                        
                        # Record state changes or significant distance changes
                        if (current_state != prev_state or 
                            (prev_distance is not None and abs(current_distance - prev_distance) > 5)):
                            
                            states.append({
                                'elapsed_time': row['elapsed_time'],
                                'tl_distance': current_distance,
                                'bicycle_distance': row['distance'],
                                'state': current_state,
                                'signal_index': row['next_tl_index']
                            })
                            
                        prev_state = current_state
                        prev_distance = current_distance
                    
                    if states:
                        # Calculate approximate traffic light position on bicycle's distance axis
                        for state in states:
                            state['tl_position'] = state['bicycle_distance'] + state['tl_distance']
                        
                        tl_info[tl_id] = {
                            'states': states,
                            'signal_index': states[0]['signal_index'],
                            'avg_position': np.mean([s['tl_position'] for s in states])
                        }
                
                return tl_info
        
        # Fallback method: Infer traffic light interaction from separate traffic light logs
        if len(traffic_light_df) == 0:
            print("    No traffic light information found (no embedded data, no separate traffic light log)")
            return tl_info
        
        # Try to infer traffic light interaction from bicycle trajectory and traffic light logs
        print("    Traffic light data not embedded in trajectory, attempting to infer from logs...")
        
        # Get the bicycle's time range
        start_time_step = bicycle_data['time_step'].min()
        end_time_step = bicycle_data['time_step'].max()
        
        # Get the bicycle's spatial range (assuming it might interact with nearby traffic lights)
        min_distance = bicycle_data['distance'].min()
        max_distance = bicycle_data['distance'].max()
        
        # Find traffic light states during the bicycle's trajectory period
        tl_during_bicycle = traffic_light_df[
            (traffic_light_df['time_step'] >= start_time_step) &
            (traffic_light_df['time_step'] <= end_time_step)
        ]
        
        if len(tl_during_bicycle) == 0:
            print("    No traffic light activity during bicycle trajectory period")
            return tl_info
        
        # For each unique traffic light, create approximate interaction data
        for tl_id in tl_during_bicycle['traffic_light_id'].unique():
            tl_data = tl_during_bicycle[tl_during_bicycle['traffic_light_id'] == tl_id].copy()
            
            # time_step is now in seconds, so elapsed time is just the difference
            tl_data['elapsed_time'] = tl_data['time_step'] - start_time_step
            
            # Parse signal states to track state changes
            states = []
            prev_signals = None
            
            for _, row in tl_data.iterrows():
                signal_states = row['signal_states']
                if pd.isna(signal_states) or signal_states == '':
                    continue
                
                # Only record when signal states change
                if signal_states != prev_signals:
                    # Estimate traffic light position as somewhere in the bicycle's path
                    # This is a rough approximation - in reality we'd need network topology
                    estimated_position = (min_distance + max_distance) / 2
                    
                    # Extract dominant signal state (most common character)
                    if signal_states:
                        signal_counts = {'r': signal_states.lower().count('r'),
                                       'y': signal_states.lower().count('y'), 
                                       'g': signal_states.lower().count('g')}
                        # Get dominant state with type-safe approach
                        try:
                            dominant_state = max(signal_counts, key=signal_counts.get)  # type: ignore # dict.get is valid for max()
                        except (ValueError, TypeError):
                            dominant_state = 'r'  # Fallback to red
                    else:
                        dominant_state = 'r'  # Default to red
                    
                    states.append({
                        'elapsed_time': row['elapsed_time'],
                        'tl_distance': 0,  # Unknown in this approximation
                        'bicycle_distance': estimated_position,
                        'state': dominant_state.upper(),
                        'signal_index': 0,  # Approximate
                        'tl_position': estimated_position
                    })
                    
                prev_signals = signal_states
            
            if states:
                tl_info[tl_id] = {
                    'states': states,
                    'signal_index': 0,
                    'avg_position': states[0]['tl_position']
                }
                
                print(f"    Inferred traffic light: {tl_id[:30]}... at ~{tl_info[tl_id]['avg_position']:.1f}m (estimated)")
        
        return tl_info
    
    def _calculate_detection_rates(self, segments_for_stats, total_distance, total_time):
        """Calculate detection rates from segments (central method used by all plot types).
        
        Args:
            segments_for_stats: Dictionary with 'detected' and 'undetected' segment lists (unfiltered)
            total_distance: Total trajectory distance
            total_time: Total trajectory time
            
        Returns:
            Dictionary with 'distance_rate', 'time_rate', 'spatiotemporal_rate', 
            'detected_distance', 'detected_time'
        """
        detected_distance = 0
        detected_time = 0
        
        for segment in segments_for_stats['detected']:
            if len(segment) > 1:
                seg_distance = segment[-1][0] - segment[0][0]
                seg_time = segment[-1][1] - segment[0][1]
                detected_distance += seg_distance
                detected_time += seg_time
        
        distance_detection_rate = (detected_distance / total_distance * 100) if total_distance > 0 else 0
        time_detection_rate = (detected_time / total_time * 100) if total_time > 0 else 0
        spatiotemporal_detection_rate = (distance_detection_rate + time_detection_rate) / 2
        
        return {
            'distance_rate': distance_detection_rate,
            'time_rate': time_detection_rate,
            'spatiotemporal_rate': spatiotemporal_detection_rate,
            'detected_distance': detected_distance,
            'detected_time': detected_time
        }
    
    def _calculate_flow_detection_rates(self, total_flow_distance, total_flow_detected_distance, 
                                       total_flow_time, total_flow_detected_time):
        """Calculate detection rates for flow-based plots (central method used by all flow-based plot types).
        
        Args:
            total_flow_distance: Sum of all bicycles' total distances in the flow
            total_flow_detected_distance: Sum of all bicycles' detected distances in the flow
            total_flow_time: Sum of all bicycles' total times in the flow
            total_flow_detected_time: Sum of all bicycles' detected times in the flow
            
        Returns:
            Dictionary with 'distance_rate', 'time_rate', 'spatiotemporal_rate'
        """
        distance_detection_rate = (total_flow_detected_distance / total_flow_distance * 100) if total_flow_distance > 0 else 0.0
        time_detection_rate = (total_flow_detected_time / total_flow_time * 100) if total_flow_time > 0 else 0.0
        spatiotemporal_detection_rate = (distance_detection_rate + time_detection_rate) / 2.0
        
        return {
            'distance_rate': distance_detection_rate,
            'time_rate': time_detection_rate,
            'spatiotemporal_rate': spatiotemporal_detection_rate,
            'total_distance': total_flow_distance,
            'total_detected_distance': total_flow_detected_distance,
            'total_time': total_flow_time,
            'total_detected_time': total_flow_detected_time
        }
    
    def _calculate_redundancy_statistics(self, redundancy_segments, total_distance, total_time):
        """Calculate distance/time coverage for each redundancy level and overall detection rates.
        
        Uses full trajectory distance/time as denominators (same as detection plots).
        """
        stats = {}
        
        # Calculate per-level statistics
        for level in [0, 1, 2, 3, 4, 5]:
            level_distance = 0
            level_time = 0
            
            for segment in redundancy_segments[level]:
                if len(segment) > 1:
                    # Match detection plot calculation
                    seg_distance = segment[-1][0] - segment[0][0]
                    seg_time = segment[-1][1] - segment[0][1]
                    level_distance += seg_distance
                    level_time += seg_time
            
            stats[level] = {
                'distance': level_distance,
                'time': level_time
            }
        
        # Calculate overall detection rates (detected = any redundancy level > 0)
        # Use full trajectory distance/time as denominators (passed as parameters)
        detected_distance = sum(stats[level]['distance'] for level in [1, 2, 3, 4, 5])
        detected_time = sum(stats[level]['time'] for level in [1, 2, 3, 4, 5])
        
        distance_detection_rate = (detected_distance / total_distance * 100) if total_distance > 0 else 0
        time_detection_rate = (detected_time / total_time * 100) if total_time > 0 else 0
        
        spatiotemporal_detection_rate = (distance_detection_rate + time_detection_rate) / 2
        
        stats['overall'] = {
            'distance_detection_rate': distance_detection_rate,
            'time_detection_rate': time_detection_rate,
            'spatiotemporal_detection_rate': spatiotemporal_detection_rate
        }
        
        return stats
    
    def _calculate_redundancy_breakdown(self, bicycle_data, bicycle_detections, time_steps, 
                                       start_time_step, distances, elapsed_times, 
                                       total_distance, total_steps):
        """Calculate detection rates broken down by redundancy level (0-5+ observers).
        
        Returns a dictionary with temporal, spatial, and spatiotemporal breakdowns.
        Each breakdown is a dict mapping redundancy level (0-5) to the percentage
        detected at that level.
        
        The sum of levels 1-5 should equal the overall detection rate.
        
        Args:
            bicycle_data: DataFrame with bicycle trajectory
            bicycle_detections: DataFrame with detection events for this bicycle
            time_steps: Array of time steps
            start_time_step: Start time for this bicycle
            distances: Array of distances along trajectory
            elapsed_times: Array of elapsed times
            total_distance: Total trajectory distance
            total_steps: Total number of time steps
            
        Returns:
            Dictionary with 'temporal', 'spatial', 'spatiotemporal' rate breakdowns (in %)
        """
        
        # Get redundancy values from trajectory data (or compute from detections)
        if 'num_detecting_observers' in bicycle_data.columns:
            redundancy_values = bicycle_data['num_detecting_observers'].values
        else:
            # Fallback: build redundancy timeline from detection events
            # Count unique observers at each time step
            redundancy_values = np.zeros(len(time_steps), dtype=int)
            if not bicycle_detections.empty:
                for time_step in time_steps:
                    observers_at_time = bicycle_detections[
                        bicycle_detections['time_step'] == time_step
                    ]['observer_id'].nunique()
                    time_idx = np.where(time_steps == time_step)[0]
                    if len(time_idx) > 0:
                        redundancy_values[time_idx[0]] = observers_at_time
        
        # Create detection timeline and apply smoothing (same as detection plots)
        detection_timeline = self._create_detection_timeline(time_steps, bicycle_detections, start_time_step)
        smoothed_detection = self._smooth_detection_timeline(detection_timeline)
        
        # Create smoothed redundancy: detected=1+, undetected=0
        # Important: redundancy values should match the smoothing behavior
        smoothed_redundancy = np.zeros(len(redundancy_values), dtype=int)
        for i in range(len(smoothed_redundancy)):
            if smoothed_detection[i]:
                # If detected after smoothing, keep redundancy value (minimum 1)
                smoothed_redundancy[i] = max(1, redundancy_values[i])
            else:
                # If undetected after smoothing, set to 0
                smoothed_redundancy[i] = 0
        
        # Split trajectory by redundancy level
        redundancy_segments = self._split_trajectory_segments_by_redundancy(
            distances, elapsed_times, smoothed_redundancy, apply_min_length_filter=False
        )
        
        # Initialize counters
        temporal_breakdown = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}  # Steps per level
        spatial_breakdown = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}  # Distance per level
        
        # Count steps and distance per redundancy level
        for level in range(6):
            for segment in redundancy_segments[level]:
                if len(segment) > 1:
                    # Temporal: count data points in this segment
                    temporal_breakdown[level] += len(segment)
                    
                    # Spatial: sum distances in this segment
                    seg_distance = segment[-1][0] - segment[0][0]
                    spatial_breakdown[level] += abs(seg_distance)
        
        # Calculate rates (as percentages of total)
        temporal_rate_breakdown = {}
        spatial_rate_breakdown = {}
        spatiotemporal_rate_breakdown = {}
        
        for level in range(6):
            temporal_rate_breakdown[level] = (temporal_breakdown[level] / total_steps * 100 
                                             if total_steps > 0 else 0)
            spatial_rate_breakdown[level] = (spatial_breakdown[level] / total_distance * 100 
                                            if total_distance > 0 else 0)
            spatiotemporal_rate_breakdown[level] = (
                temporal_rate_breakdown[level] + spatial_rate_breakdown[level]
            ) / 2
        
        return {
            'temporal': temporal_rate_breakdown,  # % of time at each redundancy level
            'spatial': spatial_rate_breakdown,    # % of distance at each redundancy level
            'spatiotemporal': spatiotemporal_rate_breakdown  # Average of temporal and spatial
        }
    
    def _calculate_occlusion_breakdown(self, bicycle_data, bicycle_detections, time_steps, 
                                       start_time_step, distances, elapsed_times, 
                                       total_distance, total_steps):
        """Calculate detection rates broken down by occlusion level.
        
        Returns a dictionary with temporal, spatial, and spatiotemporal breakdowns.
        Each breakdown is a dict mapping occlusion category name to the percentage
        detected at that level (relative to detected trajectory only, sums to 100%).
        
        Args:
            bicycle_data: DataFrame with bicycle trajectory
            bicycle_detections: DataFrame with detection events for this bicycle
            time_steps: Array of time steps
            start_time_step: Start time for this bicycle
            distances: Array of distances along trajectory
            elapsed_times: Array of elapsed times
            total_distance: Total trajectory distance
            total_steps: Total number of time steps
            
        Returns:
            Dictionary with 'temporal', 'spatial', 'spatiotemporal' rate breakdowns (in %)
        """
        
        # Get occlusion scale from configuration
        occlusion_scale = self.config.get('occlusion_level_scale', _get_occlusion_level_scale())
        
        # Create detection timeline
        detection_timeline = self._create_detection_timeline(time_steps, bicycle_detections, start_time_step)
        smoothed_detection = self._smooth_detection_timeline(detection_timeline)
        
        # Create occlusion timeline
        occlusion_timeline = self._create_occlusion_timeline(time_steps, bicycle_detections, start_time_step)
        
        # Split trajectory by occlusion level
        occlusion_segments = self._create_occlusion_segments(
            distances, elapsed_times, smoothed_detection, occlusion_timeline
        )
        
        # Initialize counters per occlusion category
        temporal_breakdown = {}  # Steps per category
        spatial_breakdown = {}   # Distance per category
        
        for category_name, _, _ in occlusion_scale:
            temporal_breakdown[category_name] = 0
            spatial_breakdown[category_name] = 0
        
        # Count steps and distance per occlusion category
        for category_name, _, _ in occlusion_scale:
            for segment in occlusion_segments[category_name]:
                if len(segment) > 1:
                    # Temporal: count data points in this segment
                    temporal_breakdown[category_name] += len(segment)
                    
                    # Spatial: sum distances in this segment
                    seg_distance = segment[-1][0] - segment[0][0]
                    spatial_breakdown[category_name] += abs(seg_distance)
        
        # Calculate rates as percentages of DETECTED trajectory (not total)
        # This ensures percentages add up to 100%
        total_detected_steps = sum(temporal_breakdown.values())
        total_detected_distance = sum(spatial_breakdown.values())
        
        temporal_rate_breakdown = {}
        spatial_rate_breakdown = {}
        spatiotemporal_rate_breakdown = {}
        
        for category_name, _, _ in occlusion_scale:
            temporal_rate_breakdown[category_name] = (
                temporal_breakdown[category_name] / total_detected_steps * 100 
                if total_detected_steps > 0 else 0
            )
            spatial_rate_breakdown[category_name] = (
                spatial_breakdown[category_name] / total_detected_distance * 100 
                if total_detected_distance > 0 else 0
            )
            spatiotemporal_rate_breakdown[category_name] = (
                temporal_rate_breakdown[category_name] + spatial_rate_breakdown[category_name]
            ) / 2
        
        # Sanity check: percentages should add up to ~100% (allowing small floating point error)
        total_temporal = sum(temporal_rate_breakdown.values())
        total_spatial = sum(spatial_rate_breakdown.values())
        
        if abs(total_temporal - 100.0) > 0.1 and total_detected_steps > 0:  # Allow 0.1% tolerance
            print(f"  Warning: Temporal occlusion percentages sum to {total_temporal:.2f}% (expected 100%)")
        if abs(total_spatial - 100.0) > 0.1 and total_detected_distance > 0:
            print(f"  Warning: Spatial occlusion percentages sum to {total_spatial:.2f}% (expected 100%)")
        
        return {
            'temporal': temporal_rate_breakdown,      # % of detected time at each occlusion level
            'spatial': spatial_rate_breakdown,        # % of detected distance at each occlusion level
            'spatiotemporal': spatiotemporal_rate_breakdown  # Average of temporal and spatial
        }
    
    def _format_redundancy_info_text(self, bicycle_id, start_time_step, stats_by_level):
        """Format info box text with redundancy statistics."""
        # Match the format of regular detection plots
        overall_stats = stats_by_level['overall']
        return (
            f"Bicycle: {bicycle_id}\n"
            f"Departure time: {start_time_step:.1f} s\n"
            f"Temporal detection rate: {overall_stats['time_detection_rate']:.1f}%\n"
            f"Spatial detection rate: {overall_stats['distance_detection_rate']:.1f}%\n"
            f"Spatio-temporal detection rate: {overall_stats['spatiotemporal_detection_rate']:.1f}%"
        )
    
    def _plot_individual_trajectory(self, bicycle_id, segments, tl_info, start_time_step, detection_rates):
        """Generate individual trajectory plot.
        
        Args:
            bicycle_id: ID of the bicycle
            segments: Dictionary with 'detected' and 'undetected' segment lists (filtered for plotting)
            tl_info: Traffic light information
            start_time_step: Start time in seconds
            detection_rates: Pre-calculated detection rates dictionary
        """
        
        fig, ax = plt.subplots(figsize=self.config['figure_size'])
        
        # Get total_time from detection_rates for traffic light plotting
        # Calculate from detected_time and time_rate
        total_time = detection_rates['detected_time'] / (detection_rates['time_rate'] / 100) if detection_rates['time_rate'] > 0 else 0
        
        # Plot undetected segments
        for segment in segments['undetected']:
            if len(segment) > 1:
                distances, times = zip(*segment)
                ax.plot(times, distances, color='black', linewidth=1.5, linestyle='solid', label='Undetected')
        
        # Plot detected segments (swap x and y axes)
        for segment in segments['detected']:
            if len(segment) > 1:
                distances, times = zip(*segment)
                ax.plot(times, distances, color='darkturquoise', linewidth=1.5, linestyle='solid', label='Detected')
        
        # Plot traffic light state changes as vertical lines
        if tl_info:
            for tl_id, tl_data in tl_info.items():
                states = tl_data['states']
                avg_position = tl_data['avg_position']
                signal_index = tl_data['signal_index']
                
                # Plot horizontal line at traffic light position (dashed and thinner)
                ax.axhline(y=avg_position, color='black', linestyle='--', alpha=0.5, linewidth=0.5, zorder=1)
                
                # Plot state changes as colored segments on the horizontal line
                for i, state_change in enumerate(states):
                    signal_state = state_change['state']
                    
                    # Skip invalid states
                    if pd.isna(signal_state) or signal_state == '':
                        continue
                        
                    signal_state = str(signal_state).lower()
                    
                    # Map signal states to colors (including unknown)
                    if signal_state == 'unknown':
                        color = 'purple'
                    else:
                        color = {'r': 'red', 'y': 'orange', 'g': 'green'}.get(signal_state, 'gray')
                    
                    # Determine the time range for this state
                    start_time = state_change['elapsed_time']
                    end_time = states[i+1]['elapsed_time'] if i+1 < len(states) else total_time
                    
                    # Plot colored segment on the horizontal line (thinner and dashed)
                    if start_time <= total_time and end_time >= 0:
                        ax.plot([start_time, end_time], [avg_position, avg_position], 
                               color=color, linewidth=2, linestyle='--', alpha=0.8, zorder=5)
                
                # Add traffic light label at the right (was top)
                short_id = tl_id.split('_')[0] if '_' in tl_id else tl_id[:10]
                ax.text(ax.get_xlim()[1], avg_position, f'TL-{signal_index}\n{short_id}', 
                       fontsize=8, ha='left', va='center', rotation=0, alpha=0.8)
        
        # Add information text box with updated terminology (use pre-calculated rates)
        # start_time_step is now already in seconds (simulation time)
        info_text = (
            f"Bicycle: {bicycle_id}\n"
            f"Departure time: {start_time_step:.1f} s\n"
            f"Temporal detection rate: {detection_rates['time_rate']:.1f}%\n"
            f"Spatial detection rate: {detection_rates['distance_rate']:.1f}%\n"
            f"Spatio-temporal detection rate: {detection_rates['spatiotemporal_rate']:.1f}%"
        )
        
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
        
        # Add legend
        handles = [
            Line2D([0], [0], color='black', lw=2, label='Undetected'),
            Line2D([0], [0], color='darkturquoise', lw=2, label='Detected'),
        ]
        
        # Add traffic light legend items if any were plotted
        if tl_info:
            handles.extend([
                Line2D([0], [0], color='red', linestyle='--', alpha=0.7, label='Red TL'),
                Line2D([0], [0], color='orange', linestyle='--', alpha=0.7, label='Yellow TL'),
                Line2D([0], [0], color='green', linestyle='--', alpha=0.7, label='Green TL')
            ])
            
        ax.legend(handles=handles, loc='lower right', bbox_to_anchor=(0.99, 0.01))
        
        # Set labels and grid (swap axis labels)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Space [m]')
        ax.grid(True)
        
        # Save plot in subdirectory
        output_subdir = os.path.join(self.config['output_dir'], '2D_detection_individual')
        os.makedirs(output_subdir, exist_ok=True)
        output_filename = f'2D_detection_individual_{self.config["file_tag"]}_FCO{self.config["fco_share"]}%_FBO{self.config["fbo_share"]}%_{bicycle_id}.png'
        output_path = os.path.join(output_subdir, output_filename)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.config['dpi'], bbox_inches='tight')
        plt.close(fig)
        
        print(f"  ✓ Saved individual 2D detection plot: {output_filename}")
    
    def _plot_individual_trajectory_redundancy(self, bicycle_id, redundancy_segments, tl_info, start_time_step, detection_rates):
        """Generate individual trajectory plot with redundancy color coding.
        
        Args:
            bicycle_id: ID of the bicycle
            redundancy_segments: Dictionary with segments by redundancy level (filtered for plotting)
            tl_info: Traffic light information
            start_time_step: Start time in seconds
            detection_rates: Pre-calculated detection rates dictionary
        """
        
        fig, ax = plt.subplots(figsize=self.config['figure_size'])
        
        # Get color palette
        colors = _get_redundancy_color_palette()
        
        # Calculate total_time from detection_rates for traffic light plotting
        total_time = detection_rates['detected_time'] / (detection_rates['time_rate'] / 100) if detection_rates['time_rate'] > 0 else 0
        
        # Plot segments for each redundancy level (0 through 5+)
        for redundancy_level in [0, 1, 2, 3, 4, 5]:
            for segment in redundancy_segments[redundancy_level]:
                if len(segment) > 1:
                    distances, times = zip(*segment)
                    ax.plot(times, distances, color=colors[redundancy_level], 
                           linewidth=1.5, linestyle='solid')
        
        # Plot traffic light state changes as horizontal colored segments (like regular detection plots)
        if tl_info:
            for tl_id, tl_data in tl_info.items():
                states = tl_data['states']
                avg_position = tl_data['avg_position']
                signal_index = tl_data['signal_index']
                
                # Plot horizontal line at traffic light position
                ax.axhline(y=avg_position, color='black', linestyle='--', alpha=0.5, linewidth=0.5, zorder=1)
                
                # Plot state changes as colored segments on the horizontal line
                for i, state_change in enumerate(states):
                    signal_state = state_change['state']
                    
                    # Skip invalid states
                    if pd.isna(signal_state) or signal_state == '':
                        continue
                    
                    signal_state = str(signal_state).lower()
                    
                    # Map traffic light states to colors
                    if signal_state == 'unknown':
                        color = 'purple'
                    else:
                        color = {'r': 'red', 'y': 'orange', 'g': 'green'}.get(signal_state, 'gray')
                    
                    # Determine the time range for this state
                    start_time = state_change['elapsed_time']
                    end_time = states[i+1]['elapsed_time'] if i+1 < len(states) else total_time
                    
                    # Plot colored segment on the horizontal line
                    if start_time <= total_time and end_time >= 0:
                        ax.plot([start_time, end_time], [avg_position, avg_position], 
                               color=color, linewidth=2, linestyle='--', alpha=0.8, zorder=5)
                
                # Add traffic light label at the right
                short_id = tl_id.split('_')[0] if '_' in tl_id else tl_id[:10]
                ax.text(ax.get_xlim()[1], avg_position, f'TL-{signal_index}\n{short_id}', 
                       fontsize=8, ha='left', va='center', rotation=0, alpha=0.8)
        
        # Add information text box (use pre-calculated detection rates)
        info_text = (
            f"Bicycle: {bicycle_id}\n"
            f"Departure time: {start_time_step:.1f} s\n"
            f"Temporal detection rate: {detection_rates['time_rate']:.1f}%\n"
            f"Spatial detection rate: {detection_rates['distance_rate']:.1f}%\n"
            f"Spatio-temporal detection rate: {detection_rates['spatiotemporal_rate']:.1f}%"
        )
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
        
        # Create legend
        handles = [
            Line2D([0], [0], color=colors[0], lw=2, label='Undetected (0)'),
            Line2D([0], [0], color=colors[1], lw=2, label='1 Observer'),
            Line2D([0], [0], color=colors[2], lw=2, label='2 Observers'),
            Line2D([0], [0], color=colors[3], lw=2, label='3 Observers'),
            Line2D([0], [0], color=colors[4], lw=2, label='4 Observers'),
            Line2D([0], [0], color=colors[5], lw=2, label='5+ Observers')
        ]
        
        # Add traffic light legend items if any were plotted
        if tl_info:
            handles.extend([
                Line2D([0], [0], color='red', linestyle='--', alpha=0.7, label='Red TL'),
                Line2D([0], [0], color='orange', linestyle='--', alpha=0.7, label='Yellow TL'),
                Line2D([0], [0], color='green', linestyle='--', alpha=0.7, label='Green TL')
            ])
        
        ax.legend(handles=handles, loc='lower right', bbox_to_anchor=(0.99, 0.01))
        
        # Set labels and grid
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Space [m]')
        ax.grid(True)
        
        # Save plot in subdirectory
        output_subdir = os.path.join(self.config['output_dir'], '2D_detection-redundancy_individual')
        os.makedirs(output_subdir, exist_ok=True)
        output_filename = f'2D_detection-redundancy_individual_{self.config["file_tag"]}_FCO{self.config["fco_share"]}%_FBO{self.config["fbo_share"]}%_{bicycle_id}.png'
        output_path = os.path.join(output_subdir, output_filename)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.config['dpi'], bbox_inches='tight')
        plt.close(fig)
        
        print(f"  ✓ Saved individual 2D detection-redundancy plot: {output_filename}")
    
    def _plot_individual_occlusion_trajectory(self, bicycle_id, segments, occlusion_segments, tl_info, start_time_step, detection_rates):
        """Generate individual occlusion level trajectory plot.
        
        Args:
            bicycle_id: ID of the bicycle
            segments: Dictionary with 'detected' and 'undetected' segment lists (filtered for plotting)
            occlusion_segments: Dictionary with segments by occlusion category (filtered for plotting)
            tl_info: Traffic light information
            start_time_step: Start time in seconds
            detection_rates: Pre-calculated detection rates dictionary
        """
        
        fig, ax = plt.subplots(figsize=self.config['figure_size'])
        
        # Calculate total_time from detection_rates for traffic light plotting
        total_time = detection_rates['detected_time'] / (detection_rates['time_rate'] / 100) if detection_rates['time_rate'] > 0 else 0
        
        # Plot undetected segments (black)
        for segment in segments['undetected']:
            if len(segment) > 1:
                distances, times = zip(*segment)
                ax.plot(times, distances, color='black', linewidth=1.5, linestyle='solid')
        
        # Get occlusion scale and generate colors
        occlusion_scale = self.config.get('occlusion_level_scale', _get_occlusion_level_scale())
        
        # Get color palette from configuration
        occlusion_colors = _get_occlusion_color_palette()
        
        # Plot detected segments colored by occlusion level
        for category_name, _, _ in occlusion_scale:
            color = occlusion_colors.get(category_name, 'gray')  # Fallback to gray if color not defined
            for segment in occlusion_segments[category_name]:
                if len(segment) > 1:
                    distances, times = zip(*segment)
                    ax.plot(times, distances, color=color, linewidth=1.5, linestyle='solid')
        
        # Plot traffic lights (same as detection plots)
        if tl_info:
            for tl_id, tl_data in tl_info.items():
                states = tl_data['states']
                avg_position = tl_data['avg_position']
                signal_index = tl_data['signal_index']
                
                ax.axhline(y=avg_position, color='black', linestyle='--', alpha=0.5, linewidth=0.5, zorder=1)
                
                for i, state_change in enumerate(states):
                    signal_state = state_change['state']
                    if pd.isna(signal_state) or signal_state == '':
                        continue
                    signal_state = str(signal_state).lower()
                    color = {'r': 'red', 'y': 'orange', 'g': 'green', 'unknown': 'purple'}.get(signal_state, 'gray')
                    
                    start_time = state_change['elapsed_time']
                    end_time = states[i+1]['elapsed_time'] if i+1 < len(states) else total_time
                    
                    if start_time <= total_time and end_time >= 0:
                        ax.plot([start_time, end_time], [avg_position, avg_position], 
                               color=color, linewidth=2, linestyle='--', alpha=0.8, zorder=5)
                
                short_id = tl_id.split('_')[0] if '_' in tl_id else tl_id[:10]
                ax.text(ax.get_xlim()[1], avg_position, f'TL-{signal_index}\n{short_id}', 
                       fontsize=8, ha='left', va='center', rotation=0, alpha=0.8)
        
        # Info box (use pre-calculated detection rates)
        info_text = (
            f"Bicycle: {bicycle_id}\n"
            f"Departure time: {start_time_step:.1f} s\n"
            f"Temporal detection rate: {detection_rates['time_rate']:.1f}%\n"
            f"Spatial detection rate: {detection_rates['distance_rate']:.1f}%\n"
            f"Spatio-temporal detection rate: {detection_rates['spatiotemporal_rate']:.1f}%"
        )
        
        ax.text(0.01, 0.99, info_text,
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='left',
                fontsize=plt.rcParams['legend.fontsize'],
                bbox=dict(facecolor='white', edgecolor='black', alpha=0.8, boxstyle='round'))
        
        # Legend with dynamic category labels
        handles = [Line2D([0], [0], color='black', lw=2, label='Undetected')]
        
        for category_name, min_pct, max_pct in occlusion_scale:
            color = occlusion_colors[category_name]
            
            # Format category name for display (replace underscores with spaces, capitalize)
            display_name = category_name.replace('_', ' ').capitalize()
            
            # Add percentage range to label
            if min_pct == max_pct:
                label = f'{display_name} ({min_pct}%)'
            else:
                label = f'{display_name} ({min_pct}-{max_pct}%)'
            
            handles.append(Line2D([0], [0], color=color, lw=2, label=label))
        
        if tl_info:
            handles.extend([
                Line2D([0], [0], color='red', linestyle='--', alpha=0.7, label='Red TL'),
                Line2D([0], [0], color='orange', linestyle='--', alpha=0.7, label='Yellow TL'),
                Line2D([0], [0], color='green', linestyle='--', alpha=0.7, label='Green TL')
            ])
        
        ax.legend(handles=handles, loc='lower right', bbox_to_anchor=(0.99, 0.01))
        
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Space [m]')
        ax.grid(True)
        
        # Save plot
        output_subdir = os.path.join(self.config['output_dir'], '2D_occlusion_individual')
        os.makedirs(output_subdir, exist_ok=True)
        output_filename = f'2D_occlusion_individual_{self.config["file_tag"]}_FCO{self.config["fco_share"]}%_FBO{self.config["fbo_share"]}%_{bicycle_id}.png'
        output_path = os.path.join(output_subdir, output_filename)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.config['dpi'], bbox_inches='tight')
        plt.close(fig)
        
        print(f"  ✓ Saved individual 2D occlusion plot: {output_filename}")
    
    def _plot_3d_detection_trajectory(self, bicycle_id, segments_3d, observer_trajectories, geometry_data, start_time_step, max_elapsed_time):
        """Create 3D detection plot for a single bicycle trajectory."""
        
        # Store geometry data for access by other methods
        self.current_geometry_data = geometry_data
        
        # Extract geometry data if available
        if geometry_data is not None:
            roads_proj = geometry_data.get('roads')
            buildings_proj = geometry_data.get('buildings') 
            parks_proj = geometry_data.get('parks')
            trees_proj = geometry_data.get('trees')
            leaves_proj = geometry_data.get('leaves')
            barriers_proj = geometry_data.get('barriers')
            pt_shelters_proj = geometry_data.get('pt_shelters')
            transformer = geometry_data.get('transformer')
        else:
            roads_proj = buildings_proj = parks_proj = None
            trees_proj = leaves_proj = barriers_proj = pt_shelters_proj = None
            transformer = None
        
        # Define coordinate bounds based on simulation bounding box, not just trajectory data
        if geometry_data is not None and 'bbox' in geometry_data:
            # Use the simulation bounding box for consistent visualization
            bbox = geometry_data['bbox']
            north, south, east, west = bbox
            
            # Transform bounding box to UTM coordinates 
            transformer = geometry_data.get('transformer')
            if transformer is None:
                transformer = pyproj.Transformer.from_crs('EPSG:4326', 'EPSG:32632', always_xy=True)
            
            # Transform bbox corners to get UTM bounds
            x_west, y_south = transformer.transform(west, south)
            x_east, y_north = transformer.transform(east, north)
            
            # Use simulation area bounds
            x_min_abs, x_max_abs = x_west, x_east
            y_min_abs, y_max_abs = y_south, y_north
            
            # Using simulation bounding box: ({north:.5f}, {south:.5f}, {east:.5f}, {west:.5f})
        else:
            # Fallback to trajectory-based bounds if no bounding box available
            all_points = []
            for segments_list in segments_3d.values():
                for segment in segments_list:
                    all_points.extend(segment)
            
            for obs_data in observer_trajectories.values():
                all_points.extend(obs_data['trajectory'])
            
            if not all_points:
                print(f"  Warning: No trajectory data for bicycle {bicycle_id}")
                return
            
            # Calculate coordinate bounds from trajectory data
            x_coords, y_coords, times = zip(*all_points)
            x_min_abs, x_max_abs = min(x_coords), max(x_coords)
            y_min_abs, y_max_abs = min(y_coords), max(y_coords)
        
        # Get time bounds: extend to when bicycle leaves bounding box (not just trajectory end)
        bicycle_points = []
        for segments_list in segments_3d.values():
            for segment in segments_list:
                bicycle_points.extend(segment)
        
        if bicycle_points:
            # Set time range for 3D plot based on full bicycle trajectory
            # Since spatial filtering happens during plotting, use full time range
            _, _, times = zip(*bicycle_points)
            t_max_trajectory = max(times)
            
            # Round up to next multiple of 5 for cleaner axis
            import math
            t_max_rounded = math.ceil(t_max_trajectory / 5.0) * 5  # Rounds up to next multiple of 5
            
            t_min = min(times)
            t_max = t_max_rounded
        else:
            t_min, t_max = 0, max_elapsed_time
        
        # Convert to relative coordinates (starting from 0)
        x_extent = x_max_abs - x_min_abs
        y_extent = y_max_abs - y_min_abs
        
        # Remove padding - use exact scene boundaries
        x_padding = 0
        y_padding = 0
        t_padding = 0
        
        # Set coordinate system: relative coordinates starting from 0
        x_min = -x_padding
        x_max = x_extent + x_padding
        y_min = -y_padding
        y_max = y_extent + y_padding
        base_z = t_min - t_padding
        top_z = t_max + t_padding
        
        # Define coordinate transformation functions
        def to_rel_x(x): return x - x_min_abs
        def to_rel_y(y): return y - y_min_abs
        
        # Use configured figure size instead of dynamic calculation
        fig_width, fig_height = self.config['figure_size_3d']
        
        # Create 3D figure with fixed size
        fig = plt.figure(figsize=(fig_width, fig_height))
        ax = fig.add_subplot(111, projection='3d')
        
        # Optimize subplot parameters to minimize white space
        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
        
        # Set axis limits
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(base_z, top_z)
        
        # Set axis labels
        ax.set_xlabel('Longitude [m]')
        ax.set_ylabel('Latitude [m]')
        ax.set_zlabel('Time (s)')
        
        # Configure 3D plot appearance
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis._axinfo['grid'].update(linestyle="--")
        ax.yaxis._axinfo['grid'].update(linestyle="--")
        ax.zaxis._axinfo['grid'].update(linestyle="--")
        
        # Set box aspect ratio to prevent Z-axis stretching
        # Limit Z-axis to be at most 60% of the smaller spatial dimension
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = top_z - base_z
        
        # Use smaller spatial dimension as reference
        spatial_ref = min(x_range, y_range)
        max_z_visual = spatial_ref * 0.6  # Z-axis at most 60% of smaller spatial dimension
        
        # Scale Z to not exceed this limit
        if z_range > max_z_visual:
            z_scale = max_z_visual / z_range
        else:
            z_scale = 1.0
        
        # Normalize all to max range for box aspect
        max_range = max(x_range, y_range, max_z_visual)
        aspect_x = x_range / max_range
        aspect_y = y_range / max_range
        aspect_z = (z_range * z_scale) / max_range
        
        ax.set_box_aspect([aspect_x, aspect_y, aspect_z])
        
        # Set view angle
        ax.view_init(elev=self.config['view_elevation'], azim=self.config['view_azimuth'])
        ax.set_axisbelow(True)
        
        # Create base plane
        base_vertices = [
            [x_min, y_min, base_z],
            [x_max, y_min, base_z],
            [x_max, y_max, base_z],
            [x_min, y_max, base_z]
        ]
        base_poly = Poly3DCollection([base_vertices], alpha=0.1)
        base_poly.set_facecolor('white')
        base_poly.set_edgecolor('gray')
        base_poly.set_sort_zpos(-2)
        ax.add_collection3d(base_poly)
        
        # Plot background geometry if available
        if geometry_data is not None:
            try:
                self._plot_comprehensive_background_geometry(ax, geometry_data, 
                                             x_min_abs, x_max_abs, y_min_abs, y_max_abs, 
                                             base_z=-0.1, to_rel_x=to_rel_x, to_rel_y=to_rel_y)
            except Exception as e:
                print(f"    Warning: Could not plot comprehensive background geometry: {e}")
        
        # Plot bicycle trajectory segments
        self._plot_bicycle_segments_3d(ax, segments_3d, base_z, to_rel_x, to_rel_y)
        
        # Plot observer trajectories (filtered to extended time range)
        self._plot_observer_trajectories_3d(ax, observer_trajectories, base_z, to_rel_x, to_rel_y, t_min, t_max)
        
        # Create legend
        handles = [
            Line2D([0], [0], color='black', linewidth=0, label=f'Bicycle ID: {bicycle_id}'),
            Line2D([0], [0], color='darkslateblue', linewidth=2, label='Bicycle Undetected'),
            Line2D([0], [0], color='cornflowerblue', linewidth=2, label='Bicycle Detected'),
            Line2D([0], [0], color='indianred', linewidth=2, label='Observer Vehicle'),
            Line2D([0], [0], color='darkred', linewidth=2, label='Observer Vehicle (Detecting)'),
            Line2D([0], [0], color='black', linestyle='--', label='Ground Projections')
        ]
        ax.legend(handles=handles, loc='upper left')
        
        # Save plot in subdirectory
        output_subdir = os.path.join(self.config['output_dir_3d'], '3D_detection_individual')
        os.makedirs(output_subdir, exist_ok=True)
        output_filename = f'3D_detection_individual_{self.config["file_tag"]}_FCO{self.config["fco_share"]}%_FBO{self.config["fbo_share"]}%_{bicycle_id}.png'
        output_path = os.path.join(output_subdir, output_filename)
        
        # Use tight layout and aggressive bbox trimming to minimize white space
        plt.tight_layout(pad=0.5)  # Reduce padding between elements
        plt.savefig(output_path, dpi=self.config['dpi'], bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        
        print(f"  ✓ Saved individual 3D detection plot: {output_filename}")
    
    def _plot_comprehensive_background_geometry(self, ax, geometry_data, x_min_abs, x_max_abs, y_min_abs, y_max_abs, base_z, to_rel_x, to_rel_y):
        """Plot comprehensive background geometry with proper colors matching main.py visualization."""
        
        # Create bounding box for clipping (using absolute coordinates)
        bbox = box(x_min_abs, y_min_abs, x_max_abs, y_max_abs)
        
        # Plot elements in the same order and style as main.py plot_geospatial_data()
        
        # 1. Plot roads (lightgray, alpha=0.5, zorder=1)
        roads_proj = geometry_data.get('roads')
        if roads_proj is not None:
            self._plot_geometry_layer(ax, roads_proj, bbox, base_z, to_rel_x, to_rel_y,
                                    facecolor='lightgray', edgecolor='lightgray', 
                                    alpha=0.5, linewidth=0.5, zorder=-1000)
        
        # 2. Plot parks (seagreen, alpha=0.5, zorder=2)
        parks_proj = geometry_data.get('parks')
        if parks_proj is not None:
            self._plot_geometry_layer(ax, parks_proj, bbox, base_z, to_rel_x, to_rel_y,
                                    facecolor='seagreen', edgecolor='black', 
                                    alpha=0.5, linewidth=0.5, zorder=-900)
        
        # 3. Plot buildings (darkgray, zorder=3)
        buildings_proj = geometry_data.get('buildings')
        if buildings_proj is not None:
            self._plot_geometry_layer(ax, buildings_proj, bbox, base_z, to_rel_x, to_rel_y,
                                    facecolor='darkgray', edgecolor='black', 
                                    alpha=1.0, linewidth=0.5, zorder=-800)
        
        # 4. Plot barriers (black, linewidth=1.0, zorder=4)
        barriers_proj = geometry_data.get('barriers')
        if barriers_proj is not None:
            self._plot_geometry_layer(ax, barriers_proj, bbox, base_z, to_rel_x, to_rel_y,
                                    facecolor='none', edgecolor='black', 
                                    alpha=1.0, linewidth=1.0, zorder=-700)
        
        # 5. Plot trees (forestgreen circles, zorder=5) - trunk + crown
        trees_proj = geometry_data.get('trees')
        leaves_proj = geometry_data.get('leaves')
        if trees_proj is not None:
            # Plot tree trunks (small circles)
            self._plot_point_features(ax, trees_proj, bbox, base_z, to_rel_x, to_rel_y,
                                    radius=0.5, facecolor='forestgreen', edgecolor='black',
                                    alpha=1.0, linewidth=0.5, zorder=-600)
            
            # Plot tree crowns (larger circles, semi-transparent)
            if leaves_proj is not None:
                self._plot_point_features(ax, leaves_proj, bbox, base_z, to_rel_x, to_rel_y,
                                        radius=2.5, facecolor='forestgreen', edgecolor='black',
                                        alpha=0.5, linewidth=0.5, zorder=-500)
        
        # 6. Plot PT shelters (lightgray, zorder=6)
        pt_shelters_proj = geometry_data.get('pt_shelters')
        if pt_shelters_proj is not None:
            self._plot_geometry_layer(ax, pt_shelters_proj, bbox, base_z, to_rel_x, to_rel_y,
                                    facecolor='lightgray', edgecolor='black', 
                                    alpha=1.0, linewidth=0.5, zorder=-400)
    
    def _plot_geometry_layer(self, ax, layer_proj, bbox, base_z, to_rel_x, to_rel_y,
                            facecolor, edgecolor, alpha, linewidth, zorder):
        """Plot a geometry layer (polygons or lines) with specified styling."""
        
        if layer_proj is None or len(layer_proj) == 0:
            return
            
        for _, feature in layer_proj.iterrows():
            if feature.geometry.intersects(bbox):
                clipped_geom = feature.geometry.intersection(bbox)
                
                if isinstance(clipped_geom, (MultiPolygon, Polygon)):
                    if isinstance(clipped_geom, MultiPolygon):
                        polygons = clipped_geom.geoms
                    else:
                        polygons = [clipped_geom]
                    
                    for polygon in polygons:
                        if hasattr(polygon, 'exterior'):
                            xs, ys = polygon.exterior.xy
                            # Transform to relative coordinates
                            xs_rel = [to_rel_x(x) for x in xs]
                            ys_rel = [to_rel_y(y) for y in ys]
                            verts = [(x, y, base_z) for x, y in zip(xs_rel, ys_rel)]
                            
                            poly = Poly3DCollection([verts], alpha=alpha)
                            if facecolor != 'none':
                                poly.set_facecolor(facecolor)
                            poly.set_edgecolor(edgecolor)
                            poly.set_linewidth(linewidth)
                            poly.set_sort_zpos(zorder)
                            ax.add_collection3d(poly)
                
                elif isinstance(clipped_geom, LineString):
                    xs, ys = clipped_geom.xy
                    # Transform to relative coordinates
                    xs_rel = [to_rel_x(x) for x in xs]
                    ys_rel = [to_rel_y(y) for y in ys]
                    ax.plot(xs_rel, ys_rel, [base_z]*len(xs_rel),
                           color=edgecolor, linewidth=linewidth, alpha=alpha, zorder=zorder)
    
    def _plot_point_features(self, ax, points_proj, bbox, base_z, to_rel_x, to_rel_y,
                            radius, facecolor, edgecolor, alpha, linewidth, zorder):
        """Plot point features (trees) as circles."""
        
        if points_proj is None or len(points_proj) == 0:
            return
            
        for _, point in points_proj.iterrows():
            if point.geometry.intersects(bbox):
                # Create circle around point
                circle = point.geometry.buffer(radius)
                clipped_circle = circle.intersection(bbox)
                
                if isinstance(clipped_circle, (Polygon, MultiPolygon)):
                    if isinstance(clipped_circle, MultiPolygon):
                        polygons = clipped_circle.geoms
                    else:
                        polygons = [clipped_circle]
                    
                    for polygon in polygons:
                        if hasattr(polygon, 'exterior'):
                            xs, ys = polygon.exterior.xy
                            # Transform to relative coordinates
                            xs_rel = [to_rel_x(x) for x in xs]
                            ys_rel = [to_rel_y(y) for y in ys]
                            verts = [(x, y, base_z) for x, y in zip(xs_rel, ys_rel)]
                            
                            poly = Poly3DCollection([verts], alpha=alpha)
                            poly.set_facecolor(facecolor)
                            poly.set_edgecolor(edgecolor)
                            poly.set_linewidth(linewidth)
                            poly.set_sort_zpos(zorder)
                            ax.add_collection3d(poly)
    
    def _plot_bicycle_segments_3d(self, ax, segments_3d, base_z, to_rel_x, to_rel_y):
        """Plot bicycle trajectory segments in 3D, filtered by spatial bounds."""
        
        # Create bounding box filter if geometry data is available
        bbox_filter = None
        if hasattr(self, 'current_geometry_data') and self.current_geometry_data and 'bbox' in self.current_geometry_data:
            from shapely.geometry import Point
            bbox = self.current_geometry_data['bbox']
            north, south, east, west = bbox
            
            # Transform bounding box to UTM coordinates 
            transformer = self.current_geometry_data.get('transformer')
            if transformer is None:
                import pyproj
                transformer = pyproj.Transformer.from_crs('EPSG:4326', 'EPSG:32632', always_xy=True)
            
            # Transform bbox corners to get UTM bounds
            x_west, y_south = transformer.transform(west, south)
            x_east, y_north = transformer.transform(east, north)
            
            # Create shapely polygon for filtering
            from shapely.geometry import Polygon
            bbox_coords = [(x_west, y_south), (x_east, y_south), (x_east, y_north), (x_west, y_north)]
            bbox_filter = Polygon(bbox_coords)
        
        # Plot detected and undetected segments
        colors = {'detected': 'cornflowerblue', 'undetected': 'darkslateblue'}
        
        for status, segment_list in segments_3d.items():
            color = colors[status]
            
            for segment in segment_list:
                if len(segment) > 1:
                    # Apply spatial filtering if bounding box is available
                    if bbox_filter is not None:
                        filtered_segment = [
                            (x, y, t) for x, y, t in segment 
                            if bbox_filter.contains(Point(x, y))
                        ]
                    else:
                        filtered_segment = segment
                    
                    if len(filtered_segment) > 1:
                        x_coords, y_coords, times = zip(*filtered_segment)
                        
                        # Transform coordinates to relative system
                        x_coords_rel = [to_rel_x(x) for x in x_coords]
                        y_coords_rel = [to_rel_y(y) for y in y_coords]
                        
                        # Plot 3D trajectory (highest priority for bicycle)
                        ax.plot(x_coords_rel, y_coords_rel, times, color=color, linewidth=2, alpha=1.0, zorder=2000)
                        
                        # Plot ground projection (highest priority for bicycle)
                        ax.plot(x_coords_rel, y_coords_rel, [base_z]*len(x_coords_rel),
                               color=color, linestyle='--', linewidth=2, alpha=1.0, zorder=2000)
                        
                        # Add projection planes
                        for i in range(len(filtered_segment)-1):
                            quad = [
                                (x_coords_rel[i], y_coords_rel[i], times[i]),
                                (x_coords_rel[i+1], y_coords_rel[i+1], times[i+1]),
                                (x_coords_rel[i+1], y_coords_rel[i+1], base_z),
                                (x_coords_rel[i], y_coords_rel[i], base_z)
                            ]
                            # Set alpha based on detection status
                            bicycle_alpha = 0.35 if status == 'detected' else 0.3
                            proj_plane = Poly3DCollection([quad], alpha=bicycle_alpha)  # Bicycle plane alpha based on detection
                            proj_plane.set_facecolor(color)
                            proj_plane.set_edgecolor('none')
                            proj_plane.set_sort_zpos(100)  # Lower value = front (bicycle projection planes in front)
                            ax.add_collection3d(proj_plane)
                        quad = [
                            (x_coords_rel[i], y_coords_rel[i], times[i]),
                            (x_coords_rel[i+1], y_coords_rel[i+1], times[i+1]),
                            (x_coords_rel[i+1], y_coords_rel[i+1], base_z),
                            (x_coords_rel[i], y_coords_rel[i], base_z)
                        ]
                        # Set alpha based on detection status  
                        bicycle_alpha = 0.35 if status == 'detected' else 0.3
                        proj_plane = Poly3DCollection([quad], alpha=bicycle_alpha)  # Bicycle plane alpha based on detection
                        proj_plane.set_facecolor(color)
                        proj_plane.set_edgecolor('none')
                        proj_plane.set_sort_zpos(100)  # Lower value = front (bicycle projection planes in front)
                        ax.add_collection3d(proj_plane)
    
    def _plot_observer_trajectories_3d(self, ax, observer_trajectories, base_z, to_rel_x, to_rel_y, t_min, t_max):
        """Plot observer trajectories in 3D, filtered to time range and spatial bounding box."""
        
        # Set up spatial filtering using the same bbox as bicycle trajectories
        bbox_filter = None
        if hasattr(self, 'current_geometry_data') and self.current_geometry_data and 'bbox' in self.current_geometry_data:
            from shapely.geometry import Point, Polygon
            bbox = self.current_geometry_data['bbox']
            north, south, east, west = bbox
            
            # Transform bounding box to UTM coordinates 
            transformer = self.current_geometry_data.get('transformer')
            if transformer is None:
                import pyproj
                transformer = pyproj.Transformer.from_crs('EPSG:4326', 'EPSG:32632', always_xy=True)
            
            # Transform bbox corners to get UTM bounds
            x_west, y_south = transformer.transform(west, south)
            x_east, y_north = transformer.transform(east, north)
            
            # Create shapely polygon for filtering
            bbox_coords = [(x_west, y_south), (x_east, y_south), (x_east, y_north), (x_west, y_north)]
            bbox_filter = Polygon(bbox_coords)
        
        for observer_id, obs_data in observer_trajectories.items():
            trajectory = obs_data['trajectory']
            detection_times = obs_data['detection_times']
            
            # Filter trajectory points to time range first
            time_filtered_trajectory = [(x, y, t) for x, y, t in trajectory if t_min <= t <= t_max]
            
            # Apply spatial filtering if bounding box is available
            if bbox_filter is not None:
                spatially_filtered_trajectory = [
                    (x, y, t) for x, y, t in time_filtered_trajectory 
                    if bbox_filter.contains(Point(x, y))
                ]
            else:
                spatially_filtered_trajectory = time_filtered_trajectory
            
            if len(spatially_filtered_trajectory) > 1:
                
                # Split spatially filtered observer trajectory into detecting/non-detecting segments
                detecting_segments = []
                non_detecting_segments = []
                
                current_segment = []
                current_detecting = None
                
                for point in spatially_filtered_trajectory:
                    x, y, t = point
                    is_detecting = any(abs(t - dt) < self.config['step_length'] for dt in detection_times)
                    
                    if current_detecting is None:
                        current_detecting = is_detecting
                    
                    if is_detecting == current_detecting:
                        current_segment.append((to_rel_x(x), to_rel_y(y), t))
                    else:
                        if len(current_segment) > 1:
                            if current_detecting:
                                detecting_segments.append(current_segment)
                            else:
                                non_detecting_segments.append(current_segment)
                        
                        current_segment = [(to_rel_x(x), to_rel_y(y), t)]
                        current_detecting = is_detecting
                
                # Add final segment
                if len(current_segment) > 1:
                    if current_detecting:
                        detecting_segments.append(current_segment)
                    else:
                        non_detecting_segments.append(current_segment)
                
                # Plot segments with projection planes
                for segments, color, is_detecting in [(non_detecting_segments, 'indianred', False), (detecting_segments, 'darkred', True)]:
                    for segment in segments:
                        if len(segment) > 1:
                            x_coords, y_coords, times = zip(*segment)
                            
                            # Plot 3D trajectory (secondary priority for observer)
                            ax.plot(x_coords, y_coords, times, color=color, linewidth=2, alpha=1.0, zorder=1000)
                            
                            # Plot ground projection (secondary priority for observer)
                            ax.plot(x_coords, y_coords, [base_z]*len(x_coords),
                                   color=color, linestyle='--', linewidth=2, alpha=0.7, zorder=1000)
                            
                            # Add projection planes for this segment with different transparency based on detection status
                            plane_alpha = 0.1 if is_detecting else 0.07  # Observer alpha values
                            for i in range(len(segment)-1):
                                quad = [
                                    (x_coords[i], y_coords[i], times[i]),
                                    (x_coords[i+1], y_coords[i+1], times[i+1]),
                                    (x_coords[i+1], y_coords[i+1], base_z),
                                    (x_coords[i], y_coords[i], base_z)
                                ]
                                proj_plane = Poly3DCollection([quad], alpha=plane_alpha)
                                proj_plane.set_facecolor(color)
                                proj_plane.set_edgecolor('none')
                                proj_plane.set_sort_zpos(1000)  # Higher value = back (observer projection planes behind bicycle)
                                ax.add_collection3d(proj_plane)
    
    def _plot_3d_conflict_event(self, conflict_event, bicycle_data, foe_data, detection_df, geometry_data):
        """Create 3D conflict plot for a single conflict event.
        
        Shows bicycle and foe trajectories in 3D space-time with conflict marker.
        Matches the visual style of _plot_3d_detection_trajectory exactly.
        
        Args:
            conflict_event: Dictionary with conflict event data
            bicycle_data: DataFrame with bicycle trajectory
            foe_data: DataFrame with foe vehicle trajectory
            detection_df: DataFrame with detection logs
            geometry_data: Dictionary with 3D background geometry
        """
        
        bicycle_id = conflict_event['bicycle_id']
        foe_id = conflict_event['foe_id']
        conflict_time = conflict_event['time_step']
        conflict_x = conflict_event['bicycle_x']
        conflict_y = conflict_event['bicycle_y']
        
        # Store geometry data for access by other methods
        self.current_geometry_data = geometry_data
        
        # Get bicycle's full trajectory (entire trajectory, not just around conflict)
        bicycle_window = bicycle_data.sort_values('time_step').copy()
        
        # Get bicycle's full trajectory time range
        bicycle_start = bicycle_window['time_step'].min()
        bicycle_end = bicycle_window['time_step'].max()
        
        # Filter foe trajectory to only the part overlapping with bicycle trajectory timeframe
        foe_window = foe_data[
            (foe_data['time_step'] >= bicycle_start) &
            (foe_data['time_step'] <= bicycle_end)
        ].sort_values('time_step').copy()
        
        if bicycle_window.empty:
            print(f"  Warning: No bicycle trajectory data around conflict at t={conflict_time:.1f}s")
            return
        
        if foe_window.empty:
            print(f"  Warning: No foe trajectory data around conflict at t={conflict_time:.1f}s")
            return
        
        # Get start time for relative coordinates
        start_time = bicycle_window['time_step'].min()
        
        # Create 3D trajectory points (x, y, elapsed_time)
        bicycle_3d = [(row['x_coord'], row['y_coord'], row['time_step'] - start_time) 
                      for _, row in bicycle_window.iterrows()]
        foe_3d = [(row['x_coord'], row['y_coord'], row['time_step'] - start_time) 
                  for _, row in foe_window.iterrows()]
        
        # Get detection status for bicycle
        bicycle_detections = detection_df[detection_df['bicycle_id'] == bicycle_id] if len(detection_df) > 0 else pd.DataFrame()
        time_steps = bicycle_window['time_step'].values
        detection_timeline = self._create_detection_timeline(time_steps, bicycle_detections, start_time)
        smoothed_detection = self._smooth_detection_timeline(detection_timeline)
        
        # Split bicycle trajectory by detection status (detected/undetected)
        bicycle_segments_3d = self._split_trajectory_segments_3d(bicycle_3d, smoothed_detection)
        
        # Foe trajectory as single segment (no detection status)
        foe_segments_3d = {'detected': [], 'undetected': [foe_3d]}
        
        # === Coordinate system setup (matching _plot_3d_detection_trajectory exactly) ===
        
        # Use simulation bounding box if available
        if geometry_data is not None and 'bbox' in geometry_data:
            bbox = geometry_data['bbox']
            north, south, east, west = bbox
            
            transformer = geometry_data.get('transformer')
            if transformer is None:
                import pyproj
                transformer = pyproj.Transformer.from_crs('EPSG:4326', 'EPSG:32632', always_xy=True)
            
            # Transform bbox corners to get UTM bounds
            x_west, y_south = transformer.transform(west, south)
            x_east, y_north = transformer.transform(east, north)
            
            x_min_abs, x_max_abs = x_west, x_east
            y_min_abs, y_max_abs = y_south, y_north
        else:
            # Fallback to trajectory-based bounds
            all_points = bicycle_3d + foe_3d
            
            if not all_points:
                print(f"  Warning: No trajectory data for conflict at t={conflict_time:.1f}s")
                return
            
            x_coords, y_coords, times = zip(*all_points)
            x_min_abs, x_max_abs = min(x_coords), max(x_coords)
            y_min_abs, y_max_abs = min(y_coords), max(y_coords)
        
        # Get time bounds (matching 3D detection plot logic)
        _, _, times = zip(*bicycle_3d)
        t_max_trajectory = max(times)
        
        # Round up to next multiple of 5 for cleaner axis
        import math
        t_max_rounded = math.ceil(t_max_trajectory / 5.0) * 5
        
        t_min = min(times)
        t_max = t_max_rounded
        
        # Convert to relative coordinates (starting from 0)
        x_extent = x_max_abs - x_min_abs
        y_extent = y_max_abs - y_min_abs
        
        # No padding - use exact scene boundaries
        x_padding = 0
        y_padding = 0
        t_padding = 0
        
        # Set coordinate system
        x_min = -x_padding
        x_max = x_extent + x_padding
        y_min = -y_padding
        y_max = y_extent + y_padding
        base_z = t_min - t_padding
        top_z = t_max + t_padding
        
        # Define coordinate transformation functions
        def to_rel_x(x): return x - x_min_abs
        def to_rel_y(y): return y - y_min_abs
        
        # === Figure setup (use fixed figure size like 2D plots) ===
        
        # Use configured figure size instead of dynamic calculation
        fig_width, fig_height = self.config['figure_size_3d']
        
        # Create 3D figure
        fig = plt.figure(figsize=(fig_width, fig_height))
        ax = fig.add_subplot(111, projection='3d')
        
        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
        
        # Set axis limits
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_zlim(base_z, top_z)
        
        # Set axis labels
        ax.set_xlabel('Longitude [m]')
        ax.set_ylabel('Latitude [m]')
        ax.set_zlabel('Time (s)')
        
        # Configure 3D plot appearance
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis._axinfo['grid'].update(linestyle="--")
        ax.yaxis._axinfo['grid'].update(linestyle="--")
        ax.zaxis._axinfo['grid'].update(linestyle="--")
        
        # Set box aspect ratio to prevent Z-axis stretching
        # Limit Z-axis to be at most 60% of the smaller spatial dimension
        x_range = x_max - x_min
        y_range = y_max - y_min
        z_range = top_z - base_z
        
        # Use smaller spatial dimension as reference
        spatial_ref = min(x_range, y_range)
        max_z_visual = spatial_ref * 0.6  # Z-axis at most 60% of smaller spatial dimension
        
        # Scale Z to not exceed this limit
        if z_range > max_z_visual:
            z_scale = max_z_visual / z_range
        else:
            z_scale = 1.0
        
        # Normalize all to max range for box aspect
        max_range = max(x_range, y_range, max_z_visual)
        aspect_x = x_range / max_range
        aspect_y = y_range / max_range
        aspect_z = (z_range * z_scale) / max_range
        
        ax.set_box_aspect([aspect_x, aspect_y, aspect_z])
        
        # Set view angle
        ax.view_init(elev=self.config['view_elevation'], azim=self.config['view_azimuth'])
        ax.set_axisbelow(True)
        
        # Create base plane
        base_vertices = [
            [x_min, y_min, base_z],
            [x_max, y_min, base_z],
            [x_max, y_max, base_z],
            [x_min, y_max, base_z]
        ]
        base_poly = Poly3DCollection([base_vertices], alpha=0.1)
        base_poly.set_facecolor('white')
        base_poly.set_edgecolor('gray')
        base_poly.set_sort_zpos(-2)
        ax.add_collection3d(base_poly)
        
        # === Plot background geometry (optional, controlled by config) ===
        if geometry_data is not None and self.config.get('show_3d_conflict_background', True):
            try:
                self._plot_comprehensive_background_geometry(ax, geometry_data,
                                             x_min_abs, x_max_abs, y_min_abs, y_max_abs,
                                             base_z=-0.1, to_rel_x=to_rel_x, to_rel_y=to_rel_y)
            except Exception as e:
                print(f"    Warning: Could not plot background geometry: {e}")
        
        # === Plot bicycle trajectory segments (matching _plot_bicycle_segments_3d style) ===
        colors = {'detected': 'cornflowerblue', 'undetected': 'darkslateblue'}
        
        for status, segment_list in bicycle_segments_3d.items():
            color = colors[status]
            
            for segment in segment_list:
                if len(segment) > 1:
                    # Apply spatial filtering if bounding box available
                    bbox_filter = None
                    if geometry_data and 'bbox' in geometry_data:
                        from shapely.geometry import Point, Polygon
                        bbox_coords = [(x_min_abs, y_min_abs), (x_max_abs, y_min_abs), 
                                     (x_max_abs, y_max_abs), (x_min_abs, y_max_abs)]
                        bbox_filter = Polygon(bbox_coords)
                    
                    if bbox_filter:
                        filtered_segment = [
                            (x, y, t) for x, y, t in segment 
                            if bbox_filter.contains(Point(x, y))
                        ]
                    else:
                        filtered_segment = segment
                    
                    if len(filtered_segment) > 1:
                        x_coords, y_coords, times = zip(*filtered_segment)
                        x_coords_rel = [to_rel_x(x) for x in x_coords]
                        y_coords_rel = [to_rel_y(y) for y in y_coords]
                        
                        # 3D trajectory
                        ax.plot(x_coords_rel, y_coords_rel, times, color=color, linewidth=2, alpha=1.0, zorder=2000)
                        
                        # Ground projection
                        ax.plot(x_coords_rel, y_coords_rel, [base_z]*len(x_coords_rel),
                               color=color, linestyle='--', linewidth=2, alpha=1.0, zorder=2000)
                        
                        # Add projection planes
                        bicycle_alpha = 0.35 if status == 'detected' else 0.3
                        for i in range(len(filtered_segment)-1):
                            quad = [
                                (x_coords_rel[i], y_coords_rel[i], times[i]),
                                (x_coords_rel[i+1], y_coords_rel[i+1], times[i+1]),
                                (x_coords_rel[i+1], y_coords_rel[i+1], base_z),
                                (x_coords_rel[i], y_coords_rel[i], base_z)
                            ]
                            proj_plane = Poly3DCollection([quad], alpha=bicycle_alpha)
                            proj_plane.set_facecolor(color)
                            proj_plane.set_edgecolor('none')
                            proj_plane.set_sort_zpos(100)
                            ax.add_collection3d(proj_plane)
        
        # === Plot foe trajectory (matching observer trajectory style) ===
        for segment in foe_segments_3d['undetected']:
            if len(segment) > 1:
                # Apply spatial filtering
                bbox_filter = None
                if geometry_data and 'bbox' in geometry_data:
                    from shapely.geometry import Point, Polygon
                    bbox_coords = [(x_min_abs, y_min_abs), (x_max_abs, y_min_abs), 
                                 (x_max_abs, y_max_abs), (x_min_abs, y_max_abs)]
                    bbox_filter = Polygon(bbox_coords)
                
                if bbox_filter:
                    filtered_segment = [
                        (x, y, t) for x, y, t in segment 
                        if bbox_filter.contains(Point(x, y))
                    ]
                else:
                    filtered_segment = segment
                
                if len(filtered_segment) > 1:
                    x_coords, y_coords, times = zip(*filtered_segment)
                    x_coords_rel = [to_rel_x(x) for x in x_coords]
                    y_coords_rel = [to_rel_y(y) for y in y_coords]
                    
                    # 3D trajectory (black for foe)
                    ax.plot(x_coords_rel, y_coords_rel, times, color='black', linewidth=2, alpha=1.0, zorder=1500)
                    
                    # Ground projection
                    ax.plot(x_coords_rel, y_coords_rel, [base_z]*len(x_coords_rel),
                           color='black', linestyle='--', linewidth=2, alpha=0.7, zorder=1500)
                    
                    # Add projection planes (foe uses increased alpha)
                    foe_alpha = 0.15
                    for i in range(len(filtered_segment)-1):
                        quad = [
                            (x_coords_rel[i], y_coords_rel[i], times[i]),
                            (x_coords_rel[i+1], y_coords_rel[i+1], times[i+1]),
                            (x_coords_rel[i+1], y_coords_rel[i+1], base_z),
                            (x_coords_rel[i], y_coords_rel[i], base_z)
                        ]
                        proj_plane = Poly3DCollection([quad], alpha=foe_alpha)
                        proj_plane.set_facecolor('black')
                        proj_plane.set_edgecolor('none')
                        proj_plane.set_sort_zpos(1000)
                        ax.add_collection3d(proj_plane)
        
        # === Plot conflict marker LAST (after all other elements for maximum visibility) ===
        conflict_t_rel = conflict_time - start_time
        conflict_x_rel = to_rel_x(conflict_x)
        conflict_y_rel = to_rel_y(conflict_y)
        
        # Main conflict marker in 3D space (drawn LAST to appear on top)
        ax.scatter([conflict_x_rel], [conflict_y_rel], [conflict_t_rel],
                  s=200, marker='o', facecolors='none', edgecolors='firebrick',
                  linewidth=1.5, zorder=10000, depthshade=False, label='Conflict Event')
        
        # Ground projection of conflict point (same style as 3D marker, plotted above projection planes)
        ax.scatter([conflict_x_rel], [conflict_y_rel], [base_z],
                  s=200, marker='o', facecolors='none', edgecolors='firebrick',
                  linewidth=1.5, zorder=5000, depthshade=False)
        
        # Vertical dashed line connecting conflict point to ground projection
        ax.plot([conflict_x_rel, conflict_x_rel], [conflict_y_rel, conflict_y_rel],
               [base_z, conflict_t_rel], color='firebrick', linestyle='--', 
               linewidth=1.5, alpha=0.5, zorder=2500)
        
        # === Create legend (matching 3D detection plot style) ===
        dominant_ssm = conflict_event['dominant_ssm']
        if dominant_ssm == 'TTC':
            ssm_value = conflict_event.get('TTC', 0)
            ssm_label = f"TTC={ssm_value:.1f}s"
        elif dominant_ssm == 'PET':
            ssm_value = conflict_event.get('PET', 0)
            ssm_label = f"PET={ssm_value:.1f}s"
        elif dominant_ssm == 'DRAC':
            ssm_value = conflict_event.get('DRAC', 0)
            ssm_label = f"DRAC={ssm_value:.1f}m/s²"
        else:
            ssm_label = dominant_ssm
        
        handles = [
            Line2D([0], [0], color='black', linewidth=0, label=f'Conflict at t={conflict_time:.1f}s'),
            Line2D([0], [0], color='black', linewidth=0, label=f'SSM: {ssm_label}'),
            Line2D([0], [0], color='black', linewidth=0, label=f'Bicycle: {bicycle_id}'),
            Line2D([0], [0], color='black', linewidth=0, label=f'Foe: {foe_id}'),
            Line2D([0], [0], color='darkslateblue', linewidth=2, label='Bicycle Undetected'),
            Line2D([0], [0], color='cornflowerblue', linewidth=2, label='Bicycle Detected'),
            Line2D([0], [0], color='black', linewidth=2, label=f'Foe Vehicle'),
            Line2D([0], [0], marker='o', color='white', markerfacecolor='none', 
                   markeredgecolor='firebrick', markersize=10, linewidth=0, label='Conflict Point'),
            Line2D([0], [0], color='black', linestyle='--', label='Ground Projections')
        ]
        ax.legend(handles=handles, loc='upper left')
        
        # === Save plot ===
        output_subdir = os.path.join(self.config['output_dir'], '3D_conflict_individual')
        os.makedirs(output_subdir, exist_ok=True)
        output_filename = f'3D_conflict_individual_{self.config["file_tag"]}_FCO{self.config["fco_share"]}%_FBO{self.config["fbo_share"]}%_{bicycle_id}_t{conflict_time:.1f}s.png'
        output_path = os.path.join(output_subdir, output_filename)
        
        plt.tight_layout(pad=0.5)
        plt.savefig(output_path, dpi=self.config['dpi'], bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        
        print(f"  ✓ Saved 3D conflict plot: {output_filename}")
    
    def process_3d_conflict_plots(self, trajectory_df, conflict_events, detection_df, foe_trajectory_df, geometry_data):
        """Process and generate 3D conflict plots for all conflict events."""
        
        if not conflict_events:
            print("  No conflict events to plot")
            return
        
        print(f"\n=== Processing Individual 3D Conflict Plots ===")
        print(f"Found {len(conflict_events)} conflict events")
        
        # Group conflicts by bicycle for efficiency
        conflicts_by_bicycle = {}
        for event in conflict_events:
            bicycle_id = event['bicycle_id']
            if bicycle_id not in conflicts_by_bicycle:
                conflicts_by_bicycle[bicycle_id] = []
            conflicts_by_bicycle[bicycle_id].append(event)
        
        plot_count = 0
        
        for bicycle_id, bicycle_conflicts in conflicts_by_bicycle.items():
            # Get bicycle trajectory
            bicycle_data = trajectory_df[trajectory_df['vehicle_id'] == bicycle_id]
            
            if bicycle_data.empty:
                print(f"  Warning: No trajectory data for bicycle {bicycle_id}")
                continue
            
            # Process each conflict for this bicycle
            for conflict_event in bicycle_conflicts:
                foe_id = conflict_event['foe_id']
                
                # Get foe trajectory (try both vehicle and bicycle trajectory files)
                foe_data = foe_trajectory_df[foe_trajectory_df['vehicle_id'] == foe_id]
                
                if foe_data.empty:
                    # Try bicycle trajectory file as fallback
                    foe_data = trajectory_df[trajectory_df['vehicle_id'] == foe_id]
                
                if foe_data.empty:
                    print(f"  Warning: No trajectory data for foe {foe_id}")
                    continue
                
                # Generate 3D conflict plot
                try:
                    self._plot_3d_conflict_event(conflict_event, bicycle_data, foe_data, 
                                                detection_df, geometry_data)
                    plot_count += 1
                except Exception as e:
                    print(f"  Error plotting conflict at t={conflict_event['time_step']:.1f}s: {e}")
        
        print(f"\n✓ Generated {plot_count} individual 3D conflict plots")
    
    def analyze_vru_trajectories(self):
        """Main method to perform VRU trajectory analysis."""
        try:
            # Initialization phase
            print("=== Initialization ===")
            print("✓ Auto-detected parameters: {}, FCO: {}%, FBO: {}%, step length: {}s".format(
                self.config['file_tag'], 
                self.config['fco_share'], 
                self.config['fbo_share'], 
                self.config['step_length']
            ))
            print(f"✓ Output directory: {self.config['output_dir']}")
            
            # Load core data
            trajectory_df = self.load_trajectory_data()
            detection_df = self.load_detection_data()
            traffic_light_df = self.load_traffic_light_data()
            
            # Load conflict data if enabled
            conflict_df = pd.DataFrame()
            conflict_events = []
            if (self.config.get('individual_2d_conflict_plots', False) or 
                self.config.get('flow_based_2d_conflict_plots', False) or 
                self.config.get('individual_3d_conflict_plots', False)):
                conflict_df = self.load_conflict_data()
                if not conflict_df.empty:
                    conflict_events = self._identify_conflict_events(conflict_df)
            
            # Legacy: Load conflict data if flow-based plots are enabled (for backward compatibility)
            elif self.config.get('flow_based_2d_detection_plots', False) or self.config.get('flow_based_2d_detection_redundancy_plots', False):
                conflicts_file = Path(self.config['scenario_path']) / 'out_logging' / f'log_conflicts_{Path(self.config["scenario_path"]).name}.csv'
                if conflicts_file.exists():
                    try:
                        with open(conflicts_file, 'r') as f:
                            lines = f.readlines()
                        header_idx = next((i for i,l in enumerate(lines) if not l.strip().startswith('#') and l.strip()), 0)
                        conflict_df = pd.read_csv(conflicts_file, skiprows=header_idx)
                        print(f"✓ Loaded conflict data ({len(conflict_df)} events)")
                    except Exception:
                        pass
            
            # Report traffic light status
            if self.config.get('enable_traffic_lights', True) and len(traffic_light_df) > 0:
                print(f"✓ Traffic light visualization: enabled")
            
            # Load 3D data if needed
            observer_df = pd.DataFrame()
            geometry_data = None
            
            if self.config.get('individual_3d_detection_plots', False):
                observer_df = self.load_observer_trajectories()
                if len(observer_df) > 0:
                    # Count vehicle types
                    type_counts = observer_df['vehicle_type'].value_counts().to_dict()
                    type_str = ", ".join([f"{count} {vtype.replace('floating_', '').replace('_observer', '')}{'s' if count > 1 else ''}" 
                                         for vtype, count in type_counts.items()])
                    print(f"✓ Loaded observer trajectories ({len(observer_df)} points, {type_str})")
                geometry_data = self.load_geometry_data()
                if geometry_data:
                    counts = {
                        'roads': len(geometry_data['roads']) if geometry_data.get('roads') is not None else 0,
                        'buildings': len(geometry_data['buildings']) if geometry_data.get('buildings') is not None else 0,
                        'trees': len(geometry_data['trees']) if geometry_data.get('trees') is not None else 0
                    }
                    print(f"✓ Loaded 3D scene geometry ({counts['roads']} roads, {counts['buildings']} buildings, {counts['trees']} trees)")
            
            # Analysis phase
            print("\n=== VRU-Specific Detection Analysis ===")
            
            # List enabled and disabled features
            print("Plot types:")
            print(f"  - Individual 2D detection: {'ENABLED' if self.config.get('individual_2d_detection_plots', True) else 'DISABLED'}")
            print(f"  - Flow-based 2D detection: {'ENABLED' if self.config.get('flow_based_2d_detection_plots', False) else 'DISABLED'}")
            print(f"  - Individual 3D detection: {'ENABLED' if self.config.get('individual_3d_detection_plots', False) else 'DISABLED'}")
            print(f"  - Individual 2D detection-redundancy: {'ENABLED' if self.config.get('individual_2d_detection_redundancy_plots', False) else 'DISABLED'}")
            print(f"  - Flow-based 2D detection-redundancy: {'ENABLED' if self.config.get('flow_based_2d_detection_redundancy_plots', False) else 'DISABLED'}")
            print(f"  - Individual 2D occlusion: {'ENABLED' if self.config.get('individual_2d_occlusion_plots', False) else 'DISABLED'}")
            print(f"  - Flow-based 2D occlusion: {'ENABLED' if self.config.get('flow_based_2d_occlusion_plots', False) else 'DISABLED'}")
            print(f"  - Individual 2D conflict: {'ENABLED' if self.config.get('individual_2d_conflict_plots', False) else 'DISABLED'}")
            print(f"  - Flow-based 2D conflict: {'ENABLED' if self.config.get('flow_based_2d_conflict_plots', False) else 'DISABLED'}")
            print(f"  - Individual 3D conflict: {'ENABLED' if self.config.get('individual_3d_conflict_plots', False) else 'DISABLED'}")
            
            # Process individual 2D detection plots if enabled
            if self.config.get('individual_2d_detection_plots', True):
                self.process_bicycle_trajectories(trajectory_df, detection_df, traffic_light_df)
            
            # Process flow-based 2D detection plots if enabled
            if self.config.get('flow_based_2d_detection_plots', False) and len(trajectory_df) > 0:
                try:
                    self._process_flow_based_from_logs(trajectory_df, detection_df, traffic_light_df)
                except AttributeError:
                    print("    Warning: flow-based processing function not implemented in this script.")
            
            # Process 3D detection plots if enabled
            if self.config.get('individual_3d_detection_plots', False) and len(trajectory_df) > 0:
                self.process_3d_detection_plots(trajectory_df, detection_df, observer_df, geometry_data)
            
            # Process individual 2D detection-redundancy plots if enabled
            if self.config.get('individual_2d_detection_redundancy_plots', False):
                self.process_bicycle_trajectories_redundancy(trajectory_df, traffic_light_df, detection_df)
            
            # Process flow-based 2D detection-redundancy plots if enabled
            if self.config.get('flow_based_2d_detection_redundancy_plots', False) and len(trajectory_df) > 0:
                try:
                    self._process_flow_based_redundancy_from_logs(trajectory_df, detection_df, traffic_light_df)
                except AttributeError:
                    print("    Warning: flow-based redundancy processing function not implemented yet.")
            
            # Process individual 2D occlusion plots if enabled
            if self.config.get('individual_2d_occlusion_plots', False):
                self.process_bicycle_occlusion_plots(trajectory_df, detection_df, traffic_light_df)
            
            # Process flow-based 2D occlusion plots if enabled
            if self.config.get('flow_based_2d_occlusion_plots', False):
                self._process_flow_based_occlusion_plots(trajectory_df, detection_df, traffic_light_df)
            
            # Process individual 2D conflict plots if enabled
            if self.config.get('individual_2d_conflict_plots', False) and conflict_events:
                print("\n=== Processing Individual 2D Conflict Plots ===")
                # Group conflicts by bicycle
                conflicts_by_bicycle = {}
                for event in conflict_events:
                    bicycle_id = event['bicycle_id']
                    if bicycle_id not in conflicts_by_bicycle:
                        conflicts_by_bicycle[bicycle_id] = []
                    conflicts_by_bicycle[bicycle_id].append(event)
                
                print(f"Found {len(conflicts_by_bicycle)} bicycles with conflicts")
                
                # Plot each bicycle with conflicts
                for bicycle_id, bicycle_conflicts in conflicts_by_bicycle.items():
                    bicycle_data = trajectory_df[trajectory_df['vehicle_id'] == bicycle_id]
                    if not bicycle_data.empty:
                        self._plot_individual_conflict_trajectory(
                            bicycle_id, bicycle_data, bicycle_conflicts, 
                            detection_df, traffic_light_df
                        )
                
                print(f"\n✓ Generated {len(conflicts_by_bicycle)} individual conflict plots")
            
            # Process flow-based 2D conflict plots if enabled
            if self.config.get('flow_based_2d_conflict_plots', False) and conflict_events:
                self._process_flow_based_conflict_plots(trajectory_df, conflict_events, detection_df, traffic_light_df)
            
            # Process individual 3D conflict plots if enabled
            if self.config.get('individual_3d_conflict_plots', False) and conflict_events:
                # Load foe trajectories
                foe_trajectory_df = self.load_foe_trajectories()
                
                # Load geometry data if not already loaded and background is enabled
                if geometry_data is None and self.config.get('show_3d_conflict_background', True):
                    geometry_data = self.load_geometry_data()
                
                self.process_3d_conflict_plots(trajectory_df, conflict_events, detection_df, 
                                              foe_trajectory_df, geometry_data)
            
            # Process statistics if enabled
            if self.config.get('enable_statistics', True):
                statistics_results = self.calculate_detection_statistics(trajectory_df, detection_df, conflict_events)
                self.export_statistics_data(statistics_results)
            
            print("\n=== VRU-Specific Detection Analysis completed successfully! ===")
            
        except Exception as e:
            print(f"\nAnalysis failed: {e}")
            raise
    
    def calculate_segment_distance(self, segment):
        """Helper function to calculate total distance of a segment using coordinates"""
        if len(segment) < 2:
            return 0.0
        
        # Calculate Euclidean distance between consecutive points
        dx = np.diff(segment['x_coord'].values)
        dy = np.diff(segment['y_coord'].values)
        distances = np.sqrt(dx**2 + dy**2)
        return np.sum(distances)
    
    def calculate_detection_statistics(self, trajectory_df, detection_df, conflict_events=None):
        """
        Calculate detection statistics for three layers:
        1. Individual bicycle level
        2. Flow-based level (mean values per flow)
        3. System-wide level (overall mean values)
        
        Args:
            trajectory_df: DataFrame with bicycle trajectories
            detection_df: DataFrame with detection logs
            conflict_events: List of conflict event dictionaries (optional)
        """
        from datetime import datetime
        
        print("\n=== Statistics ===")
        
        # Initialize dictionaries for storing metrics
        bicycle_metrics = {}
        flow_metrics = {}

        # Quick check: do any vehicle IDs contain a 'flow' token? If not, we'll skip flow-level aggregation.
        try:
            has_flow_tokens = trajectory_df['vehicle_id'].astype(str).str.contains(r'flow', case=False, na=False).any()
        except Exception:
            has_flow_tokens = False

        if not has_flow_tokens:
            print("Note: No 'flow' tokens detected in vehicle IDs. Flow-based aggregation will be skipped.")
        
        # Process each individual bicycle
        bicycle_groups = trajectory_df.groupby('vehicle_id')
        
        for bicycle_id, bicycle_data in bicycle_groups:
            # Extract flow ID from bicycle ID only if it contains an explicit 'flow' token (case-insensitive)
            flow_match = re.search(r'(?i)flow[_A-Za-z0-9-]*', str(bicycle_id))
            flow_id = flow_match.group(0) if flow_match else None
            
            # Sort by time step
            bicycle_data = bicycle_data.sort_values('time_step')
            
            # Calculate overall temporal detection rates
            total_steps = len(bicycle_data)
            
            # Get detection status for this bicycle
            bicycle_detections = detection_df[detection_df['bicycle_id'] == bicycle_id] if len(detection_df) > 0 else pd.DataFrame()
            
            # Create detection timeline
            time_steps = bicycle_data['time_step'].values
            start_time_step = time_steps[0]
            detection_timeline = self._create_detection_timeline(time_steps, bicycle_detections, start_time_step)
            
            # Apply detection smoothing
            smoothed_detection = self._smooth_detection_timeline(detection_timeline)
            
            # Count detected time steps
            detected_steps = sum(smoothed_detection)
            temporal_rate = (detected_steps / total_steps) * 100 if total_steps > 0 else 0
            
            # Calculate overall spatial detection rates using coordinates
            total_distance = self.calculate_segment_distance(bicycle_data)
            
            # Calculate detected distance using segments
            distances = bicycle_data['distance'].values if 'distance' in bicycle_data.columns else np.arange(len(bicycle_data))
            # time_step is now in seconds, so elapsed time is just the difference
            elapsed_times = time_steps - start_time_step
            
            # Split trajectory into detected/undetected segments
            segments = self._split_trajectory_segments(distances, elapsed_times, smoothed_detection)
            
            # Calculate detected distance
            total_detected_distance = 0
            for segment in segments['detected']:
                if len(segment) > 1:
                    seg_distance = segment[-1][0] - segment[0][0]
                    total_detected_distance += abs(seg_distance)
            
            spatial_rate = (total_detected_distance / total_distance) * 100 if total_distance > 0 else 0
            
            # Calculate spatio-temporal rate
            spatiotemporal_rate = (temporal_rate + spatial_rate) / 2
            
            # ===== Calculate redundancy-level breakdown =====
            redundancy_breakdown = self._calculate_redundancy_breakdown(
                bicycle_data, bicycle_detections, time_steps, start_time_step,
                distances, elapsed_times, total_distance, total_steps
            )
            
            # ===== Calculate occlusion-level breakdown =====
            occlusion_breakdown = self._calculate_occlusion_breakdown(
                bicycle_data, bicycle_detections, time_steps, start_time_step,
                distances, elapsed_times, total_distance, total_steps
            )
            
            # Calculate important area metrics (critical interaction areas)
            important_area_data = bicycle_data[bicycle_data['in_test_area'] == 1].copy() if 'in_test_area' in bicycle_data.columns else pd.DataFrame()
            
            if not important_area_data.empty:
                # Important area temporal detection rate
                important_area_steps = len(important_area_data)
                important_area_detected_steps = len(important_area_data[important_area_data['is_detected'] == 1]) if 'is_detected' in important_area_data.columns else 0
                important_temporal_rate = (important_area_detected_steps / important_area_steps * 100 
                                        if important_area_steps > 0 else 0)
                
                # Important area spatial detection rate
                # Create segments based on discontinuities in time_step for important areas
                important_area_data = important_area_data.sort_values('time_step')
                important_area_data = important_area_data.copy()
                important_area_data['time_diff'] = important_area_data['time_step'].diff()
                important_area_data['segment_id'] = (important_area_data['time_diff'] > 1).cumsum()
                
                # Group by continuous segments and calculate distances
                important_segments = important_area_data.groupby('segment_id')
                total_important_distance = 0
                total_important_detected_distance = 0
                
                for segment_id, segment in important_segments:
                    if len(segment) > 1:
                        # Calculate total distance for this segment
                        segment_distance = self.calculate_segment_distance(segment)
                        total_important_distance += segment_distance
                        
                        # Calculate detected distance within this segment
                        detected_subsegments = segment[segment['is_detected'] == 1] if 'is_detected' in segment.columns else pd.DataFrame()
                        if not detected_subsegments.empty and len(detected_subsegments) > 1:
                            detected_distance = self.calculate_segment_distance(detected_subsegments)
                            total_important_detected_distance += detected_distance
                
                important_spatial_rate = (total_important_detected_distance / total_important_distance * 100 
                                        if total_important_distance > 0 else 0)
                important_spatiotemporal_rate = (important_temporal_rate + important_spatial_rate) / 2
            else:
                # No important area data available
                important_temporal_rate = 0
                important_spatial_rate = 0
                important_spatiotemporal_rate = 0
                important_area_steps = 0
                important_area_detected_steps = 0
                total_important_distance = 0
                total_important_detected_distance = 0
            
            # Store individual bicycle metrics
            bicycle_metrics[bicycle_id] = {
                'temporal_rate': temporal_rate,
                'spatial_rate': spatial_rate,
                'spatiotemporal_rate': spatiotemporal_rate,
                'total_distance': total_distance,
                'total_time_steps': total_steps,
                'detected_distance': total_detected_distance,
                'detected_steps': detected_steps,
                # Redundancy breakdown
                'redundancy_temporal': redundancy_breakdown['temporal'],
                'redundancy_spatial': redundancy_breakdown['spatial'],
                'redundancy_spatiotemporal': redundancy_breakdown['spatiotemporal'],
                # Occlusion breakdown
                'occlusion_temporal': occlusion_breakdown['temporal'],
                'occlusion_spatial': occlusion_breakdown['spatial'],
                'occlusion_spatiotemporal': occlusion_breakdown['spatiotemporal'],
                'important_temporal_rate': important_temporal_rate,
                'important_spatial_rate': important_spatial_rate,
                'important_spatiotemporal_rate': important_spatiotemporal_rate,
                'important_total_steps': important_area_steps,
                'important_detected_steps': important_area_detected_steps,
                'important_total_distance': total_important_distance,
                'important_detected_distance': total_important_detected_distance,
                'flow_id': flow_id if flow_id is not None else '',
                # Initialize conflict metrics (will be updated later if conflicts exist)
                'num_conflicts': 0,
                'conflict_temporal_rate': 0.0,
                'conflict_spatial_rate': 0.0,
                'conflict_spatiotemporal_rate': 0.0
            }
            # Initialize or update flow metrics only if we detected a flow token for this bicycle
            if flow_id is not None:
                if flow_id not in flow_metrics:
                    # Get occlusion scale for initialization
                    occlusion_scale = self.config.get('occlusion_level_scale', _get_occlusion_level_scale())
                    
                    flow_metrics[flow_id] = {
                        'bicycles': [],
                        'total_steps': 0,
                        'detected_steps': 0,
                        'total_distance': 0,
                        'detected_distance': 0,
                        # Initialize redundancy breakdown for flows
                        'redundancy_temporal': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
                        'redundancy_spatial': {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
                        # Initialize occlusion breakdown for flows
                        'occlusion_temporal': {},
                        'occlusion_spatial': {},
                        'important_total_steps': 0,
                        'important_detected_steps': 0,
                        'important_total_distance': 0,
                        'important_detected_distance': 0
                    }
                    
                    # Initialize occlusion categories dynamically from scale
                    for category_name, _, _ in occlusion_scale:
                        flow_metrics[flow_id]['occlusion_temporal'][category_name] = 0
                        flow_metrics[flow_id]['occlusion_spatial'][category_name] = 0

                # Aggregate metrics per flow
                flow_metrics[flow_id]['bicycles'].append(bicycle_id)
                flow_metrics[flow_id]['total_steps'] += total_steps
                flow_metrics[flow_id]['detected_steps'] += detected_steps
                flow_metrics[flow_id]['total_distance'] += total_distance
                flow_metrics[flow_id]['detected_distance'] += total_detected_distance
                
                # Aggregate redundancy breakdown at flow level (sum raw counts)
                for level in range(6):
                    # Convert percentages back to counts using bicycle's totals
                    temporal_steps = redundancy_breakdown['temporal'][level] * total_steps / 100
                    spatial_dist = redundancy_breakdown['spatial'][level] * total_distance / 100
                    
                    flow_metrics[flow_id]['redundancy_temporal'][level] += temporal_steps
                    flow_metrics[flow_id]['redundancy_spatial'][level] += spatial_dist
                
                # Aggregate occlusion breakdown at flow level
                # Occlusion percentages are relative to detected trajectory, so use detected totals
                occlusion_scale = self.config.get('occlusion_level_scale', _get_occlusion_level_scale())
                for category_name, _, _ in occlusion_scale:
                    # Convert percentages to counts (percentages are relative to detected only)
                    temporal_steps = (
                        bicycle_metrics[bicycle_id]['occlusion_temporal'][category_name] * 
                        detected_steps / 100
                    )
                    spatial_dist = (
                        bicycle_metrics[bicycle_id]['occlusion_spatial'][category_name] * 
                        total_detected_distance / 100
                    )
                    
                    flow_metrics[flow_id]['occlusion_temporal'][category_name] += temporal_steps
                    flow_metrics[flow_id]['occlusion_spatial'][category_name] += spatial_dist
                
                flow_metrics[flow_id]['important_total_steps'] += important_area_steps
                flow_metrics[flow_id]['important_detected_steps'] += important_area_detected_steps
                flow_metrics[flow_id]['important_total_distance'] += total_important_distance
                flow_metrics[flow_id]['important_detected_distance'] += total_important_detected_distance
        
        # Calculate conflict detection rates if conflict events provided
        if conflict_events:
            # Group conflicts by bicycle
            conflicts_by_bicycle = {}
            for event in conflict_events:
                bicycle_id = event['bicycle_id']
                if bicycle_id not in conflicts_by_bicycle:
                    conflicts_by_bicycle[bicycle_id] = []
                conflicts_by_bicycle[bicycle_id].append(event)
            
            # Update bicycle metrics with conflict statistics
            for bicycle_id, bicycle_conflicts in conflicts_by_bicycle.items():
                if bicycle_id in bicycle_metrics:
                    # Calculate detection rates for this bicycle's conflicts
                    conflict_stats = self._calculate_conflict_detection_rates(bicycle_conflicts, detection_df)
                    
                    bicycle_metrics[bicycle_id]['num_conflicts'] = conflict_stats['total_events']
                    bicycle_metrics[bicycle_id]['conflict_temporal_rate'] = conflict_stats['temporal_rate'] * 100
                    bicycle_metrics[bicycle_id]['conflict_spatial_rate'] = conflict_stats['spatial_rate'] * 100
                    bicycle_metrics[bicycle_id]['conflict_spatiotemporal_rate'] = conflict_stats['spatiotemporal_rate'] * 100
        
        # Calculate flow-based detection rates (including redundancy breakdown rates)
        for flow_id in flow_metrics:
            metrics = flow_metrics[flow_id]
            metrics['temporal_rate'] = (metrics['detected_steps'] / metrics['total_steps'] * 100 
                                      if metrics['total_steps'] > 0 else 0)
            metrics['spatial_rate'] = (metrics['detected_distance'] / metrics['total_distance'] * 100 
                                     if metrics['total_distance'] > 0 else 0)
            metrics['spatiotemporal_rate'] = (metrics['temporal_rate'] + metrics['spatial_rate']) / 2
            
            # Calculate redundancy rates for flows (convert counts back to percentages)
            metrics['redundancy_temporal_rate'] = {}
            metrics['redundancy_spatial_rate'] = {}
            metrics['redundancy_spatiotemporal_rate'] = {}
            
            for level in range(6):
                metrics['redundancy_temporal_rate'][level] = (
                    metrics['redundancy_temporal'][level] / metrics['total_steps'] * 100 
                    if metrics['total_steps'] > 0 else 0
                )
                metrics['redundancy_spatial_rate'][level] = (
                    metrics['redundancy_spatial'][level] / metrics['total_distance'] * 100 
                    if metrics['total_distance'] > 0 else 0
                )
                metrics['redundancy_spatiotemporal_rate'][level] = (
                    metrics['redundancy_temporal_rate'][level] + metrics['redundancy_spatial_rate'][level]
                ) / 2
            
            # Calculate occlusion rates for flows (relative to detected trajectory)
            metrics['occlusion_temporal_rate'] = {}
            metrics['occlusion_spatial_rate'] = {}
            metrics['occlusion_spatiotemporal_rate'] = {}
            
            total_detected_steps_flow = metrics['detected_steps']
            total_detected_distance_flow = metrics['detected_distance']
            
            for category_name in metrics['occlusion_temporal'].keys():
                metrics['occlusion_temporal_rate'][category_name] = (
                    metrics['occlusion_temporal'][category_name] / total_detected_steps_flow * 100 
                    if total_detected_steps_flow > 0 else 0
                )
                metrics['occlusion_spatial_rate'][category_name] = (
                    metrics['occlusion_spatial'][category_name] / total_detected_distance_flow * 100 
                    if total_detected_distance_flow > 0 else 0
                )
                metrics['occlusion_spatiotemporal_rate'][category_name] = (
                    metrics['occlusion_temporal_rate'][category_name] + 
                    metrics['occlusion_spatial_rate'][category_name]
                ) / 2
            
            metrics['important_temporal_rate'] = (metrics['important_detected_steps'] / metrics['important_total_steps'] * 100 
                                                if metrics['important_total_steps'] > 0 else 0)
            metrics['important_spatial_rate'] = (metrics['important_detected_distance'] / metrics['important_total_distance'] * 100 
                                               if metrics['important_total_distance'] > 0 else 0)
            metrics['important_spatiotemporal_rate'] = (metrics['important_temporal_rate'] + metrics['important_spatial_rate']) / 2
        
        # Calculate system-wide statistics
        system_metrics = self._calculate_system_wide_metrics(bicycle_metrics, flow_metrics, conflict_events)
        
        # Package results
        results = {
            'individual': bicycle_metrics,
            'flow_based': flow_metrics,
            'system_wide': system_metrics,
            'summary': {
                'total_bicycles': len(bicycle_metrics),
                'total_flows': len(flow_metrics),
                'total_conflicts': len(conflict_events) if conflict_events else 0,
                'bicycles_with_conflicts': len([m for m in bicycle_metrics.values() if m['num_conflicts'] > 0]),
                'analysis_timestamp': datetime.now().isoformat(),
                'configuration': {
                    'scenario_path': self.config['scenario_path'],
                    'file_tag': self.config['file_tag'],
                    'fco_share': self.config['fco_share'],
                    'fbo_share': self.config['fbo_share'],
                    'step_length': self.config['step_length']
                }
            }
        }
        
        return results
    
    def _calculate_system_wide_metrics(self, bicycle_metrics, flow_metrics, conflict_events=None):
        """Calculate system-wide aggregated metrics
        
        Args:
            bicycle_metrics: Dictionary of per-bicycle metrics
            flow_metrics: Dictionary of per-flow metrics
            conflict_events: List of conflict event dictionaries (optional)
        """
        if not bicycle_metrics:
            return {}
        
        # Calculate averages for individual bicycle metrics
        avg_temporal_rate = np.mean([m['temporal_rate'] for m in bicycle_metrics.values()])
        avg_spatial_rate = np.mean([m['spatial_rate'] for m in bicycle_metrics.values()])
        avg_spatiotemporal_rate = np.mean([m['spatiotemporal_rate'] for m in bicycle_metrics.values()])
        
        # Calculate averages for important area metrics
        avg_important_temporal_rate = np.mean([m['important_temporal_rate'] for m in bicycle_metrics.values()])
        avg_important_spatial_rate = np.mean([m['important_spatial_rate'] for m in bicycle_metrics.values()])
        avg_important_spatiotemporal_rate = np.mean([m['important_spatiotemporal_rate'] for m in bicycle_metrics.values()])
        
        # Calculate conflict averages if available
        bicycles_with_conflicts = [m for m in bicycle_metrics.values() if m['num_conflicts'] > 0]
        if bicycles_with_conflicts:
            avg_conflict_temporal_rate = np.mean([m['conflict_temporal_rate'] for m in bicycles_with_conflicts])
            avg_conflict_spatial_rate = np.mean([m['conflict_spatial_rate'] for m in bicycles_with_conflicts])
            avg_conflict_spatiotemporal_rate = np.mean([m['conflict_spatiotemporal_rate'] for m in bicycles_with_conflicts])
            total_conflicts = sum(m['num_conflicts'] for m in bicycle_metrics.values())
        else:
            avg_conflict_temporal_rate = 0.0
            avg_conflict_spatial_rate = 0.0
            avg_conflict_spatiotemporal_rate = 0.0
            total_conflicts = 0
        
        # Calculate averages for flow-based metrics
        avg_flow_temporal_rate = np.mean([m['temporal_rate'] for m in flow_metrics.values()]) if flow_metrics else 0
        avg_flow_spatial_rate = np.mean([m['spatial_rate'] for m in flow_metrics.values()]) if flow_metrics else 0
        avg_flow_spatiotemporal_rate = np.mean([m['spatiotemporal_rate'] for m in flow_metrics.values()]) if flow_metrics else 0
        
        # Calculate averages for flow-based important area metrics
        avg_flow_important_temporal_rate = np.mean([m['important_temporal_rate'] for m in flow_metrics.values()]) if flow_metrics else 0
        avg_flow_important_spatial_rate = np.mean([m['important_spatial_rate'] for m in flow_metrics.values()]) if flow_metrics else 0
        avg_flow_important_spatiotemporal_rate = np.mean([m['important_spatiotemporal_rate'] for m in flow_metrics.values()]) if flow_metrics else 0
        
        # Calculate cumulative system-wide detection rates
        total_system_steps = sum(metrics['total_time_steps'] for metrics in bicycle_metrics.values())
        total_system_detected_steps = sum(metrics['detected_steps'] for metrics in bicycle_metrics.values())
        total_system_distance = sum(metrics['total_distance'] for metrics in bicycle_metrics.values())
        total_system_detected_distance = sum(metrics['detected_distance'] for metrics in bicycle_metrics.values())
        
        # Calculate cumulative system-wide important area rates
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
        
        # ===== NEW: Calculate system-wide redundancy breakdown =====
        
        # Method 1: Average redundancy rates across individuals
        avg_redundancy_temporal = {level: 0.0 for level in range(6)}
        avg_redundancy_spatial = {level: 0.0 for level in range(6)}
        avg_redundancy_spatiotemporal = {level: 0.0 for level in range(6)}
        
        for level in range(6):
            temporal_rates = [m['redundancy_temporal'][level] for m in bicycle_metrics.values()]
            spatial_rates = [m['redundancy_spatial'][level] for m in bicycle_metrics.values()]
            spatiotemp_rates = [m['redundancy_spatiotemporal'][level] for m in bicycle_metrics.values()]
            
            avg_redundancy_temporal[level] = np.mean(temporal_rates)
            avg_redundancy_spatial[level] = np.mean(spatial_rates)
            avg_redundancy_spatiotemporal[level] = np.mean(spatiotemp_rates)
        
        # Method 2: Cumulative redundancy (sum all steps/distance per level across all bicycles)
        # This gives the absolute contribution of each redundancy level to the system total
        cumulative_redundancy_temporal_steps = {level: 0 for level in range(6)}
        cumulative_redundancy_spatial_distance = {level: 0.0 for level in range(6)}
        
        for metrics in bicycle_metrics.values():
            for level in range(6):
                # Convert percentages back to counts using bicycle's totals
                temporal_steps = metrics['redundancy_temporal'][level] * metrics['total_time_steps'] / 100
                spatial_dist = metrics['redundancy_spatial'][level] * metrics['total_distance'] / 100
                
                cumulative_redundancy_temporal_steps[level] += temporal_steps
                cumulative_redundancy_spatial_distance[level] += spatial_dist
        
        # Convert cumulative counts back to system-wide percentages
        cumulative_redundancy_temporal_rate = {}
        cumulative_redundancy_spatial_rate = {}
        cumulative_redundancy_spatiotemporal_rate = {}
        
        for level in range(6):
            cumulative_redundancy_temporal_rate[level] = (
                cumulative_redundancy_temporal_steps[level] / total_system_steps * 100
                if total_system_steps > 0 else 0
            )
            cumulative_redundancy_spatial_rate[level] = (
                cumulative_redundancy_spatial_distance[level] / total_system_distance * 100
                if total_system_distance > 0 else 0
            )
            cumulative_redundancy_spatiotemporal_rate[level] = (
                cumulative_redundancy_temporal_rate[level] + cumulative_redundancy_spatial_rate[level]
            ) / 2
        
        # Method 3: Flow-based redundancy averages (if flows exist)
        if flow_metrics:
            avg_flow_redundancy_temporal = {level: 0.0 for level in range(6)}
            avg_flow_redundancy_spatial = {level: 0.0 for level in range(6)}
            avg_flow_redundancy_spatiotemporal = {level: 0.0 for level in range(6)}
            
            for level in range(6):
                flow_temporal_rates = [m.get('redundancy_temporal_rate', {}).get(level, 0) for m in flow_metrics.values()]
                flow_spatial_rates = [m.get('redundancy_spatial_rate', {}).get(level, 0) for m in flow_metrics.values()]
                flow_spatiotemp_rates = [m.get('redundancy_spatiotemporal_rate', {}).get(level, 0) for m in flow_metrics.values()]
                
                avg_flow_redundancy_temporal[level] = np.mean(flow_temporal_rates)
                avg_flow_redundancy_spatial[level] = np.mean(flow_spatial_rates)
                avg_flow_redundancy_spatiotemporal[level] = np.mean(flow_spatiotemp_rates)
        else:
            avg_flow_redundancy_temporal = {level: 0.0 for level in range(6)}
            avg_flow_redundancy_spatial = {level: 0.0 for level in range(6)}
            avg_flow_redundancy_spatiotemporal = {level: 0.0 for level in range(6)}
        
        # ===== Calculate system-wide occlusion breakdown =====
        
        occlusion_scale = self.config.get('occlusion_level_scale', _get_occlusion_level_scale())
        
        # Method 1: Average occlusion rates across individuals
        avg_occlusion_temporal = {}
        avg_occlusion_spatial = {}
        avg_occlusion_spatiotemporal = {}
        
        for category_name, _, _ in occlusion_scale:
            temporal_rates = [
                m.get('occlusion_temporal', {}).get(category_name, 0) 
                for m in bicycle_metrics.values()
            ]
            spatial_rates = [
                m.get('occlusion_spatial', {}).get(category_name, 0) 
                for m in bicycle_metrics.values()
            ]
            spatiotemp_rates = [
                m.get('occlusion_spatiotemporal', {}).get(category_name, 0) 
                for m in bicycle_metrics.values()
            ]
            
            avg_occlusion_temporal[category_name] = np.mean(temporal_rates)
            avg_occlusion_spatial[category_name] = np.mean(spatial_rates)
            avg_occlusion_spatiotemporal[category_name] = np.mean(spatiotemp_rates)
        
        # Method 2: Cumulative occlusion (sum all detected steps/distance per category)
        cumulative_occlusion_temporal_steps = {}
        cumulative_occlusion_spatial_distance = {}
        
        for category_name, _, _ in occlusion_scale:
            cumulative_occlusion_temporal_steps[category_name] = 0
            cumulative_occlusion_spatial_distance[category_name] = 0
        
        for metrics in bicycle_metrics.values():
            total_detected_steps = metrics['detected_steps']
            total_detected_distance = metrics['detected_distance']
            
            for category_name, _, _ in occlusion_scale:
                # Convert percentages back to counts
                temporal_steps = (
                    metrics.get('occlusion_temporal', {}).get(category_name, 0) * 
                    total_detected_steps / 100
                )
                spatial_dist = (
                    metrics.get('occlusion_spatial', {}).get(category_name, 0) * 
                    total_detected_distance / 100
                )
                
                cumulative_occlusion_temporal_steps[category_name] += temporal_steps
                cumulative_occlusion_spatial_distance[category_name] += spatial_dist
        
        # Convert cumulative counts to system-wide percentages (of total detected)
        cumulative_occlusion_temporal_rate = {}
        cumulative_occlusion_spatial_rate = {}
        cumulative_occlusion_spatiotemporal_rate = {}
        
        for category_name, _, _ in occlusion_scale:
            cumulative_occlusion_temporal_rate[category_name] = (
                cumulative_occlusion_temporal_steps[category_name] / total_system_detected_steps * 100
                if total_system_detected_steps > 0 else 0
            )
            cumulative_occlusion_spatial_rate[category_name] = (
                cumulative_occlusion_spatial_distance[category_name] / total_system_detected_distance * 100
                if total_system_detected_distance > 0 else 0
            )
            cumulative_occlusion_spatiotemporal_rate[category_name] = (
                cumulative_occlusion_temporal_rate[category_name] + 
                cumulative_occlusion_spatial_rate[category_name]
            ) / 2
        
        # Method 3: Flow-based occlusion averages (if flows exist)
        if flow_metrics:
            avg_flow_occlusion_temporal = {}
            avg_flow_occlusion_spatial = {}
            avg_flow_occlusion_spatiotemporal = {}
            
            for category_name, _, _ in occlusion_scale:
                flow_temporal_rates = [
                    m.get('occlusion_temporal_rate', {}).get(category_name, 0) 
                    for m in flow_metrics.values()
                ]
                flow_spatial_rates = [
                    m.get('occlusion_spatial_rate', {}).get(category_name, 0) 
                    for m in flow_metrics.values()
                ]
                flow_spatiotemp_rates = [
                    m.get('occlusion_spatiotemporal_rate', {}).get(category_name, 0) 
                    for m in flow_metrics.values()
                ]
                
                avg_flow_occlusion_temporal[category_name] = np.mean(flow_temporal_rates)
                avg_flow_occlusion_spatial[category_name] = np.mean(flow_spatial_rates)
                avg_flow_occlusion_spatiotemporal[category_name] = np.mean(flow_spatiotemp_rates)
        else:
            avg_flow_occlusion_temporal = {cat[0]: 0.0 for cat in occlusion_scale}
            avg_flow_occlusion_spatial = {cat[0]: 0.0 for cat in occlusion_scale}
            avg_flow_occlusion_spatiotemporal = {cat[0]: 0.0 for cat in occlusion_scale}
        
        return {
            # Individual bicycle averages
            'avg_individual_temporal_rate': avg_temporal_rate,
            'avg_individual_spatial_rate': avg_spatial_rate,
            'avg_individual_spatiotemporal_rate': avg_spatiotemporal_rate,
            
            # Individual bicycle important area averages
            'avg_individual_important_temporal_rate': avg_important_temporal_rate,
            'avg_individual_important_spatial_rate': avg_important_spatial_rate,
            'avg_individual_important_spatiotemporal_rate': avg_important_spatiotemporal_rate,
            
            # Conflict detection averages (for bicycles with conflicts)
            'avg_conflict_temporal_rate': avg_conflict_temporal_rate,
            'avg_conflict_spatial_rate': avg_conflict_spatial_rate,
            'avg_conflict_spatiotemporal_rate': avg_conflict_spatiotemporal_rate,
            'total_conflicts': total_conflicts,
            'bicycles_with_conflicts': len(bicycles_with_conflicts),
            
            # Flow-based averages
            'avg_flow_temporal_rate': avg_flow_temporal_rate,
            'avg_flow_spatial_rate': avg_flow_spatial_rate,
            'avg_flow_spatiotemporal_rate': avg_flow_spatiotemporal_rate,
            
            # Flow-based important area averages
            'avg_flow_important_temporal_rate': avg_flow_important_temporal_rate,
            'avg_flow_important_spatial_rate': avg_flow_important_spatial_rate,
            'avg_flow_important_spatiotemporal_rate': avg_flow_important_spatiotemporal_rate,
            
            'avg_bicycles_per_flow': np.mean([len(m['bicycles']) for m in flow_metrics.values()]) if flow_metrics else 0,
            
            # System-wide cumulative rates
            'overall_temporal_rate': overall_temporal_rate,
            'overall_spatial_rate': overall_spatial_rate,
            'overall_spatiotemporal_rate': overall_spatiotemporal_rate,
            
            # System-wide cumulative important area rates
            'overall_important_temporal_rate': overall_important_temporal_rate,
            'overall_important_spatial_rate': overall_important_spatial_rate,
            'overall_important_spatiotemporal_rate': overall_important_spatiotemporal_rate,
            
            # System-wide totals
            'total_system_steps': total_system_steps,
            'total_system_detected_steps': total_system_detected_steps,
            'total_system_distance': total_system_distance,
            'total_system_detected_distance': total_system_detected_distance,
            
            # System-wide important area totals
            'total_system_important_steps': total_system_important_steps,
            'total_system_important_detected_steps': total_system_important_detected_steps,
            'total_system_important_distance': total_system_important_distance,
            'total_system_important_detected_distance': total_system_important_detected_distance,
            
            # ===== NEW: Redundancy breakdown metrics =====
            
            # Individual-level averages (mean redundancy rates across all bicycles)
            'avg_individual_redundancy_temporal': avg_redundancy_temporal,
            'avg_individual_redundancy_spatial': avg_redundancy_spatial,
            'avg_individual_redundancy_spatiotemporal': avg_redundancy_spatiotemporal,
            
            # Flow-level averages (mean redundancy rates across all flows)
            'avg_flow_redundancy_temporal': avg_flow_redundancy_temporal,
            'avg_flow_redundancy_spatial': avg_flow_redundancy_spatial,
            'avg_flow_redundancy_spatiotemporal': avg_flow_redundancy_spatiotemporal,
            
            # System-wide cumulative (total contribution of each level to system detection)
            'cumulative_redundancy_temporal_rate': cumulative_redundancy_temporal_rate,
            'cumulative_redundancy_spatial_rate': cumulative_redundancy_spatial_rate,
            'cumulative_redundancy_spatiotemporal_rate': cumulative_redundancy_spatiotemporal_rate,
            
            # ===== Occlusion breakdown metrics =====
            
            # Individual-level averages
            'avg_individual_occlusion_temporal': avg_occlusion_temporal,
            'avg_individual_occlusion_spatial': avg_occlusion_spatial,
            'avg_individual_occlusion_spatiotemporal': avg_occlusion_spatiotemporal,
            
            # Flow-level averages
            'avg_flow_occlusion_temporal': avg_flow_occlusion_temporal,
            'avg_flow_occlusion_spatial': avg_flow_occlusion_spatial,
            'avg_flow_occlusion_spatiotemporal': avg_flow_occlusion_spatiotemporal,
            
            # System-wide cumulative
            'cumulative_occlusion_temporal_rate': cumulative_occlusion_temporal_rate,
            'cumulative_occlusion_spatial_rate': cumulative_occlusion_spatial_rate,
            'cumulative_occlusion_spatiotemporal_rate': cumulative_occlusion_spatiotemporal_rate
        }
    
    def export_statistics_data(self, statistics_results):
        """Export statistics data as a comprehensive CSV and summary text report"""
        
        # Use the configured statistics output directory
        stats_dir = self.config['output_dir_statistics']
        
        # Generate file prefix
        file_prefix = f"detection_rates_{self.config['file_tag']}_FCO{self.config['fco_share']}%_FBO{self.config['fbo_share']}%"
        
        # Export comprehensive CSV with all data
        self._export_comprehensive_csv(statistics_results, stats_dir, file_prefix)
        
        # Export concise text summary
        self._export_summary_report(statistics_results, stats_dir, file_prefix)
        
        print(f"✓ Statistics exported to: {stats_dir}")
        print(f"  - Comprehensive data: {file_prefix}_data.csv")
        print(f"  - Summary report: {file_prefix}_summary.txt")
    
    def _export_comprehensive_csv(self, results, output_dir, file_prefix):
        """Export all statistics data in a single comprehensive CSV file"""
        
        csv_file = os.path.join(output_dir, f"{file_prefix}_data.csv")
        csv_data = []
        
        # Add individual bicycle data with level indicator
        for bicycle_id, metrics in results['individual'].items():
            row = {
                'analysis_level': 'individual',
                'identifier': bicycle_id,
                'flow_id': metrics['flow_id'],
                'num_bicycles': 1,
                'temporal_rate': metrics['temporal_rate'],
                'spatial_rate': metrics['spatial_rate'],
                'spatiotemporal_rate': metrics['spatiotemporal_rate'],
                'important_temporal_rate': metrics['important_temporal_rate'],
                'important_spatial_rate': metrics['important_spatial_rate'],
                'important_spatiotemporal_rate': metrics['important_spatiotemporal_rate'],
                'total_time_steps': metrics['total_time_steps'],
                'detected_steps': metrics['detected_steps'],
                'total_distance': metrics['total_distance'],
                'detected_distance': metrics['detected_distance'],
                'important_total_steps': metrics['important_total_steps'],
                'important_detected_steps': metrics['important_detected_steps'],
                'important_total_distance': metrics['important_total_distance'],
                'important_detected_distance': metrics['important_detected_distance'],
                'num_conflicts': metrics.get('num_conflicts', 0),
                'conflict_temporal_rate': metrics.get('conflict_temporal_rate', 0.0),
                'conflict_spatial_rate': metrics.get('conflict_spatial_rate', 0.0),
                'conflict_spatiotemporal_rate': metrics.get('conflict_spatiotemporal_rate', 0.0)
            }
            
            # Add redundancy breakdown columns
            for level in range(6):
                row[f'redundancy_{level}_temporal_rate'] = metrics['redundancy_temporal'][level]
                row[f'redundancy_{level}_spatial_rate'] = metrics['redundancy_spatial'][level]
                row[f'redundancy_{level}_spatiotemporal_rate'] = metrics['redundancy_spatiotemporal'][level]
            
            # Add occlusion breakdown columns
            occlusion_scale = self.config.get('occlusion_level_scale', _get_occlusion_level_scale())
            for category_name, _, _ in occlusion_scale:
                row[f'occlusion_level_{category_name}_temporal_pct'] = metrics.get('occlusion_temporal', {}).get(category_name, 0.0)
                row[f'occlusion_level_{category_name}_spatial_pct'] = metrics.get('occlusion_spatial', {}).get(category_name, 0.0)
                row[f'occlusion_level_{category_name}_spatiotemporal_pct'] = metrics.get('occlusion_spatiotemporal', {}).get(category_name, 0.0)
            
            csv_data.append(row)
        
        # Add flow-based data
        for flow_id, metrics in results['flow_based'].items():
            row = {
                'analysis_level': 'flow',
                'identifier': flow_id,
                'flow_id': flow_id,
                'num_bicycles': len(metrics['bicycles']),
                'temporal_rate': metrics['temporal_rate'],
                'spatial_rate': metrics['spatial_rate'],
                'spatiotemporal_rate': metrics['spatiotemporal_rate'],
                'important_temporal_rate': metrics['important_temporal_rate'],
                'important_spatial_rate': metrics['important_spatial_rate'],
                'important_spatiotemporal_rate': metrics['important_spatiotemporal_rate'],
                'total_time_steps': metrics['total_steps'],
                'detected_steps': metrics['detected_steps'],
                'total_distance': metrics['total_distance'],
                'detected_distance': metrics['detected_distance'],
                'important_total_steps': metrics['important_total_steps'],
                'important_detected_steps': metrics['important_detected_steps'],
                'important_total_distance': metrics['important_total_distance'],
                'important_detected_distance': metrics['important_detected_distance']
            }
            
            # Add redundancy breakdown columns for flows
            for level in range(6):
                row[f'redundancy_{level}_temporal_rate'] = metrics.get('redundancy_temporal_rate', {}).get(level, 0.0)
                row[f'redundancy_{level}_spatial_rate'] = metrics.get('redundancy_spatial_rate', {}).get(level, 0.0)
                row[f'redundancy_{level}_spatiotemporal_rate'] = metrics.get('redundancy_spatiotemporal_rate', {}).get(level, 0.0)
            
            # Add occlusion breakdown columns for flows
            occlusion_scale = self.config.get('occlusion_level_scale', _get_occlusion_level_scale())
            for category_name, _, _ in occlusion_scale:
                row[f'occlusion_level_{category_name}_temporal_pct'] = metrics.get('occlusion_temporal_rate', {}).get(category_name, 0.0)
                row[f'occlusion_level_{category_name}_spatial_pct'] = metrics.get('occlusion_spatial_rate', {}).get(category_name, 0.0)
                row[f'occlusion_level_{category_name}_spatiotemporal_pct'] = metrics.get('occlusion_spatiotemporal_rate', {}).get(category_name, 0.0)
            
            csv_data.append(row)
        
        # Add system-wide data
        system_wide = results['system_wide']
        row_individual_avg = {
            'analysis_level': 'system_wide_individual_avg',
            'identifier': 'system_average',
            'flow_id': 'all_flows',
            'num_bicycles': results['summary']['total_bicycles'],
            'temporal_rate': system_wide['avg_individual_temporal_rate'],
            'spatial_rate': system_wide['avg_individual_spatial_rate'],
            'spatiotemporal_rate': system_wide['avg_individual_spatiotemporal_rate'],
            'important_temporal_rate': system_wide['avg_individual_important_temporal_rate'],
            'important_spatial_rate': system_wide['avg_individual_important_spatial_rate'],
            'important_spatiotemporal_rate': system_wide['avg_individual_important_spatiotemporal_rate'],
            'total_time_steps': system_wide['total_system_steps'],
            'detected_steps': system_wide['total_system_detected_steps'],
            'total_distance': system_wide['total_system_distance'],
            'detected_distance': system_wide['total_system_detected_distance'],
            'important_total_steps': system_wide['total_system_important_steps'],
            'important_detected_steps': system_wide['total_system_important_detected_steps'],
            'important_total_distance': system_wide['total_system_important_distance'],
            'important_detected_distance': system_wide['total_system_important_detected_distance']
        }
        for level in range(6):
            row_individual_avg[f'redundancy_{level}_temporal_rate'] = system_wide['avg_individual_redundancy_temporal'][level]
            row_individual_avg[f'redundancy_{level}_spatial_rate'] = system_wide['avg_individual_redundancy_spatial'][level]
            row_individual_avg[f'redundancy_{level}_spatiotemporal_rate'] = system_wide['avg_individual_redundancy_spatiotemporal'][level]
        occlusion_scale = self.config.get('occlusion_level_scale', _get_occlusion_level_scale())
        for category_name, _, _ in occlusion_scale:
            row_individual_avg[f'occlusion_level_{category_name}_temporal_pct'] = system_wide['avg_individual_occlusion_temporal'][category_name]
            row_individual_avg[f'occlusion_level_{category_name}_spatial_pct'] = system_wide['avg_individual_occlusion_spatial'][category_name]
            row_individual_avg[f'occlusion_level_{category_name}_spatiotemporal_pct'] = system_wide['avg_individual_occlusion_spatiotemporal'][category_name]
        csv_data.append(row_individual_avg)
        
        row_flow_avg = {
            'analysis_level': 'system_wide_flow_avg',
            'identifier': 'flow_average',
            'flow_id': 'all_flows',
            'num_bicycles': system_wide['avg_bicycles_per_flow'],
            'temporal_rate': system_wide['avg_flow_temporal_rate'],
            'spatial_rate': system_wide['avg_flow_spatial_rate'],
            'spatiotemporal_rate': system_wide['avg_flow_spatiotemporal_rate'],
            'important_temporal_rate': system_wide['avg_flow_important_temporal_rate'],
            'important_spatial_rate': system_wide['avg_flow_important_spatial_rate'],
            'important_spatiotemporal_rate': system_wide['avg_flow_important_spatiotemporal_rate'],
            'total_time_steps': system_wide['total_system_steps'],
            'detected_steps': system_wide['total_system_detected_steps'],
            'total_distance': system_wide['total_system_distance'],
            'detected_distance': system_wide['total_system_detected_distance'],
            'important_total_steps': system_wide['total_system_important_steps'],
            'important_detected_steps': system_wide['total_system_important_detected_steps'],
            'important_total_distance': system_wide['total_system_important_distance'],
            'important_detected_distance': system_wide['total_system_important_detected_distance']
        }
        for level in range(6):
            row_flow_avg[f'redundancy_{level}_temporal_rate'] = system_wide['avg_flow_redundancy_temporal'][level]
            row_flow_avg[f'redundancy_{level}_spatial_rate'] = system_wide['avg_flow_redundancy_spatial'][level]
            row_flow_avg[f'redundancy_{level}_spatiotemporal_rate'] = system_wide['avg_flow_redundancy_spatiotemporal'][level]
        occlusion_scale = self.config.get('occlusion_level_scale', _get_occlusion_level_scale())
        for category_name, _, _ in occlusion_scale:
            row_flow_avg[f'occlusion_level_{category_name}_temporal_pct'] = system_wide['avg_flow_occlusion_temporal'][category_name]
            row_flow_avg[f'occlusion_level_{category_name}_spatial_pct'] = system_wide['avg_flow_occlusion_spatial'][category_name]
            row_flow_avg[f'occlusion_level_{category_name}_spatiotemporal_pct'] = system_wide['avg_flow_occlusion_spatiotemporal'][category_name]
        csv_data.append(row_flow_avg)
        
        row_cumulative = {
            'analysis_level': 'system_wide_cumulative',
            'identifier': 'cumulative_total',
            'flow_id': 'all_flows',
            'num_bicycles': results['summary']['total_bicycles'],
            'temporal_rate': system_wide['overall_temporal_rate'],
            'spatial_rate': system_wide['overall_spatial_rate'],
            'spatiotemporal_rate': system_wide['overall_spatiotemporal_rate'],
            'important_temporal_rate': system_wide['overall_important_temporal_rate'],
            'important_spatial_rate': system_wide['overall_important_spatial_rate'],
            'important_spatiotemporal_rate': system_wide['overall_important_spatiotemporal_rate'],
            'total_time_steps': system_wide['total_system_steps'],
            'detected_steps': system_wide['total_system_detected_steps'],
            'total_distance': system_wide['total_system_distance'],
            'detected_distance': system_wide['total_system_detected_distance'],
            'important_total_steps': system_wide['total_system_important_steps'],
            'important_detected_steps': system_wide['total_system_important_detected_steps'],
            'important_total_distance': system_wide['total_system_important_distance'],
            'important_detected_distance': system_wide['total_system_important_detected_distance']
        }
        for level in range(6):
            row_cumulative[f'redundancy_{level}_temporal_rate'] = system_wide['cumulative_redundancy_temporal_rate'][level]
            row_cumulative[f'redundancy_{level}_spatial_rate'] = system_wide['cumulative_redundancy_spatial_rate'][level]
            row_cumulative[f'redundancy_{level}_spatiotemporal_rate'] = system_wide['cumulative_redundancy_spatiotemporal_rate'][level]
        occlusion_scale = self.config.get('occlusion_level_scale', _get_occlusion_level_scale())
        for category_name, _, _ in occlusion_scale:
            row_cumulative[f'occlusion_level_{category_name}_temporal_pct'] = system_wide['cumulative_occlusion_temporal_rate'][category_name]
            row_cumulative[f'occlusion_level_{category_name}_spatial_pct'] = system_wide['cumulative_occlusion_spatial_rate'][category_name]
            row_cumulative[f'occlusion_level_{category_name}_spatiotemporal_pct'] = system_wide['cumulative_occlusion_spatiotemporal_rate'][category_name]
        csv_data.append(row_cumulative)
        
        # Add conflict-specific system-wide data if conflicts exist
        if system_wide.get('total_conflicts', 0) > 0:
            csv_data.append({
                'analysis_level': 'system_wide_conflicts',
                'identifier': 'conflict_average',
                'flow_id': 'all_flows',
                'num_bicycles': system_wide.get('bicycles_with_conflicts', 0),
                'temporal_rate': system_wide.get('avg_conflict_temporal_rate', 0.0),
                'spatial_rate': system_wide.get('avg_conflict_spatial_rate', 0.0),
                'spatiotemporal_rate': system_wide.get('avg_conflict_spatiotemporal_rate', 0.0),
                'important_temporal_rate': 0.0,
                'important_spatial_rate': 0.0,
                'important_spatiotemporal_rate': 0.0,
                'total_time_steps': 0,
                'detected_steps': 0,
                'total_distance': 0,
                'detected_distance': 0,
                'important_total_steps': 0,
                'important_detected_steps': 0,
                'important_total_distance': 0,
                'important_detected_distance': 0,
                'num_conflicts': system_wide.get('total_conflicts', 0),
                'conflict_temporal_rate': system_wide.get('avg_conflict_temporal_rate', 0.0),
                'conflict_spatial_rate': system_wide.get('avg_conflict_spatial_rate', 0.0),
                'conflict_spatiotemporal_rate': system_wide.get('avg_conflict_spatiotemporal_rate', 0.0)
            })
        
        # Convert to DataFrame and save
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_file, index=False)
    
    def _export_summary_report(self, results, output_dir, file_prefix):
        """Export comprehensive human-readable summary report with detailed per-bicycle and per-flow information"""
        
        report_file = os.path.join(output_dir, f"{file_prefix}_summary.txt")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            # =====================================================
            # HEADER
            # =====================================================
            f.write('═══════════════════════════════════════════════════════════════════\n')
            f.write('VRU DETECTION ANALYSIS - COMPREHENSIVE SUMMARY REPORT\n')
            f.write('═══════════════════════════════════════════════════════════════════\n')
            config = results['summary']['configuration']
            f.write(f'Scenario:     {config["file_tag"]}\n')
            f.write(f'FCO Share:    {config["fco_share"]}%\n')
            f.write(f'FBO Share:    {config["fbo_share"]}%\n')
            f.write(f'Step Length:  {config["step_length"]}s\n')
            f.write(f'Generated:    {results["summary"]["analysis_timestamp"][:19]}\n')
            f.write('═══════════════════════════════════════════════════════════════════\n')
            f.write('\n')
            f.write('METRIC DEFINITIONS:\n')
            f.write('─────────────────────────────────────────────────────────────────\n')
            f.write('Detection Rates:\n')
            f.write('  • Temporal:         % of time steps where VRU was detected\n')
            f.write('  • Spatial:          % of trajectory distance that was detected\n')
            f.write('  • Spatio-temporal:  Average of temporal and spatial rates\n')
            f.write('\n')
            f.write('Redundancy Distribution:\n')
            f.write('  Shows the percentage of total trajectory at each observer count level.\n')
            f.write('  • Level 0: Undetected segments (no observers)\n')
            f.write('  • Levels 1-5: Detected by 1, 2, 3, 4, 5, or 5+ observers\n')
            f.write('  Note: Sum of levels 0-5 = 100% of total trajectory.\n')
            f.write('\n')
            f.write('Occlusion Level Distribution:\n')
            f.write('  Shows the percentage of DETECTED trajectory at each occlusion level.\n')
            f.write('  • No occlusion (0%):       Clear line of sight\n')
            f.write('  • Low occlusion (1-39%):   Slight obstruction\n')
            f.write('  • Partial (40-79%):        Moderate obstruction\n')
            f.write('  • Heavy (80-100%):         Severe obstruction\n')
            f.write('  Note: Percentages sum to 100% of detected segments only (excludes\n')
            f.write('        undetected portions).\n')
            f.write('═══════════════════════════════════════════════════════════════════\n\n')
            
            system_wide = results['system_wide']
            
            # =====================================================
            # SCENARIO OVERVIEW
            # =====================================================
            f.write('SCENARIO OVERVIEW:\n')
            f.write('─────────────────────────────────────────────────────────────────\n')
            f.write(f'  Total VRUs (Bicycles):      {results["summary"]["total_bicycles"]:>5}\n')
            f.write(f'  Total Flows:                {results["summary"]["total_flows"]:>5}\n')
            f.write(f'  Avg VRUs per Flow:          {system_wide["avg_bicycles_per_flow"]:>5.1f}\n')
            f.write(f'  Total Distance Traveled:    {system_wide["total_system_distance"]:>8.0f} m\n')
            f.write(f'  Total Simulation Steps:     {system_wide["total_system_steps"]:>8}\n')
            
            if results["summary"].get("total_conflicts", 0) > 0:
                f.write(f'  Total Conflict Events:      {results["summary"]["total_conflicts"]:>5}\n')
                f.write(f'  VRUs with Conflicts:        {results["summary"]["bicycles_with_conflicts"]:>5}\n')
            
            if system_wide["total_system_important_steps"] > 0:
                f.write(f'  Critical Area Distance:     {system_wide["total_system_important_distance"]:>8.0f} m\n')
                f.write(f'  Critical Area Steps:        {system_wide["total_system_important_steps"]:>8}\n')
            
            f.write('\n\n')
            
            # =====================================================
            # SYSTEM-WIDE SUMMARY
            # =====================================================
            f.write('SYSTEM-WIDE DETECTION RATES:\n')
            f.write('═══════════════════════════════════════════════════════════════════\n')
            f.write('Aggregation Method     │  Temporal │  Spatial  │ Spatio-temporal\n')
            f.write('───────────────────────┼───────────┼───────────┼────────────────\n')
            f.write(f'Individual Average     │  {system_wide["avg_individual_temporal_rate"]:>6.1f}%  │  {system_wide["avg_individual_spatial_rate"]:>6.1f}%  │     {system_wide["avg_individual_spatiotemporal_rate"]:>6.1f}%\n')
            f.write(f'Flow Average           │  {system_wide["avg_flow_temporal_rate"]:>6.1f}%  │  {system_wide["avg_flow_spatial_rate"]:>6.1f}%  │     {system_wide["avg_flow_spatiotemporal_rate"]:>6.1f}%\n')
            f.write(f'Cumulative Total       │  {system_wide["overall_temporal_rate"]:>6.1f}%  │  {system_wide["overall_spatial_rate"]:>6.1f}%  │     {system_wide["overall_spatiotemporal_rate"]:>6.1f}%\n')
            f.write('\n')
            
            # System-wide redundancy distribution
            f.write('SYSTEM-WIDE REDUNDANCY DISTRIBUTION (% of total trajectory):\n')
            f.write('Observer Count     │  Temporal │  Spatial  │ Spatio-temporal\n')
            f.write('───────────────────┼───────────┼───────────┼────────────────\n')
            for level in range(6):
                label = f'{level} Observer{"s" if level > 1 else " "}'
                f.write(f'{label:<18} │  {system_wide["cumulative_redundancy_temporal_rate"][level]:>6.1f}%  │  '
                       f'{system_wide["cumulative_redundancy_spatial_rate"][level]:>6.1f}%  │     '
                       f'{system_wide["cumulative_redundancy_spatiotemporal_rate"][level]:>6.1f}%\n')
            f.write('\n')
            
            # System-wide occlusion distribution
            f.write('SYSTEM-WIDE OCCLUSION LEVEL DISTRIBUTION (% of detected trajectory):\n')
            f.write('Occlusion Level        │  Temporal │  Spatial  │ Spatio-temporal\n')
            f.write('───────────────────────┼───────────┼───────────┼────────────────\n')
            occlusion_scale = self.config.get('occlusion_level_scale', _get_occlusion_level_scale())
            for category_name, min_pct, max_pct in occlusion_scale:
                display_name = category_name.replace('_', ' ').capitalize()
                f.write(f'{display_name:<22} │  {system_wide["cumulative_occlusion_temporal_rate"][category_name]:>6.1f}%  │  '
                       f'{system_wide["cumulative_occlusion_spatial_rate"][category_name]:>6.1f}%  │     '
                       f'{system_wide["cumulative_occlusion_spatiotemporal_rate"][category_name]:>6.1f}%\n')
            f.write('\n')
            
            # System-wide critical areas
            if system_wide["total_system_important_steps"] > 0:
                f.write('SYSTEM-WIDE CRITICAL INTERACTION AREA DETECTION:\n')
                f.write('Aggregation Method     │  Temporal │  Spatial  │ Spatio-temporal\n')
                f.write('───────────────────────┼───────────┼───────────┼────────────────\n')
                f.write(f'Individual Average     │  {system_wide["avg_individual_important_temporal_rate"]:>6.1f}%  │  {system_wide["avg_individual_important_spatial_rate"]:>6.1f}%  │     {system_wide["avg_individual_important_spatiotemporal_rate"]:>6.1f}%\n')
                f.write(f'Flow Average           │  {system_wide["avg_flow_important_temporal_rate"]:>6.1f}%  │  {system_wide["avg_flow_important_spatial_rate"]:>6.1f}%  │     {system_wide["avg_flow_important_spatiotemporal_rate"]:>6.1f}%\n')
                f.write(f'Cumulative Total       │  {system_wide["overall_important_temporal_rate"]:>6.1f}%  │  {system_wide["overall_important_spatial_rate"]:>6.1f}%  │     {system_wide["overall_important_spatiotemporal_rate"]:>6.1f}%\n')
                f.write('\n')
            
            # System-wide conflicts
            if system_wide.get('total_conflicts', 0) > 0:
                f.write('SYSTEM-WIDE CONFLICT DETECTION (VRUs with conflicts only):\n')
                f.write('Metric                 │      Rate\n')
                f.write('───────────────────────┼───────────\n')
                f.write(f'Temporal Detection     │    {system_wide.get("avg_conflict_temporal_rate", 0.0):>6.1f}%\n')
                f.write(f'Spatial Detection      │    {system_wide.get("avg_conflict_spatial_rate", 0.0):>6.1f}%\n')
                f.write(f'Spatio-temporal        │    {system_wide.get("avg_conflict_spatiotemporal_rate", 0.0):>6.1f}%\n')
                f.write(f'Total Conflicts:       {system_wide.get("total_conflicts", 0)}\n')
                f.write(f'VRUs with Conflicts:   {system_wide.get("bicycles_with_conflicts", 0)}\n')
                f.write('\n')
            
            f.write('\n')
            
            # =====================================================
            # FLOW-BASED DETAILS
            # =====================================================
            if results['flow_based']:
                f.write('FLOW-BASED DETAILS:\n')
                f.write('═══════════════════════════════════════════════════════════════════\n\n')
                
                # Sort flows by spatio-temporal rate (descending)
                flows_sorted = sorted(results['flow_based'].items(), 
                                    key=lambda x: x[1]['spatiotemporal_rate'], 
                                    reverse=True)
                
                for flow_id, metrics in flows_sorted:
                    f.write(f'Flow: {flow_id}\n')
                    f.write('─────────────────────────────────────────────────────────────────\n')
                    f.write(f'  Bicycles in Flow:   {len(metrics["bicycles"])}\n')
                    f.write(f'  Total Distance:     {metrics["total_distance"]:.0f} m\n')
                    f.write(f'  Total Time Steps:   {metrics["total_steps"]}\n')
                    f.write('\n')
                    
                    # Detection rates
                    f.write('  Overall Detection Rates:\n')
                    f.write(f'    Temporal:         {metrics["temporal_rate"]:>6.1f}%\n')
                    f.write(f'    Spatial:          {metrics["spatial_rate"]:>6.1f}%\n')
                    f.write(f'    Spatio-temporal:  {metrics["spatiotemporal_rate"]:>6.1f}%\n')
                    f.write('\n')
                    
                    # Redundancy distribution
                    if 'redundancy_temporal_rate' in metrics:
                        f.write('  Redundancy Distribution:\n')
                        f.write('    Observer Count │  Temporal │  Spatial  │ Spatio-temporal\n')
                        f.write('    ───────────────┼───────────┼───────────┼────────────────\n')
                        for level in range(6):
                            label = f'{level} obs'
                            temp_rate = metrics['redundancy_temporal_rate'].get(level, 0.0)
                            spat_rate = metrics['redundancy_spatial_rate'].get(level, 0.0)
                            st_rate = metrics['redundancy_spatiotemporal_rate'].get(level, 0.0)
                            f.write(f'    {label:<14} │  {temp_rate:>6.1f}%  │  {spat_rate:>6.1f}%  │     {st_rate:>6.1f}%\n')
                        f.write('\n')
                    
                    # Occlusion level distribution
                    if 'occlusion_temporal_rate' in metrics:
                        f.write('  Occlusion Level Distribution (within detected segments):\n')
                        f.write('    Occlusion Level    │  Temporal │  Spatial  │ Spatio-temporal\n')
                        f.write('    ───────────────────┼───────────┼───────────┼────────────────\n')
                        occlusion_scale = self.config.get('occlusion_level_scale', _get_occlusion_level_scale())
                        for category_name, min_pct, max_pct in occlusion_scale:
                            display_name = category_name.replace('_', ' ').capitalize()
                            temp_rate = metrics['occlusion_temporal_rate'].get(category_name, 0.0)
                            spat_rate = metrics['occlusion_spatial_rate'].get(category_name, 0.0)
                            st_rate = metrics['occlusion_spatiotemporal_rate'].get(category_name, 0.0)
                            f.write(f'    {display_name:<18} │  {temp_rate:>6.1f}%  │  {spat_rate:>6.1f}%  │     {st_rate:>6.1f}%\n')
                        f.write('\n')
                    
                    # Critical area detection
                    if metrics['important_total_steps'] > 0:
                        f.write('  Critical Interaction Area Detection:\n')
                        f.write(f'    Temporal:         {metrics["important_temporal_rate"]:>6.1f}%\n')
                        f.write(f'    Spatial:          {metrics["important_spatial_rate"]:>6.1f}%\n')
                        f.write(f'    Spatio-temporal:  {metrics["important_spatiotemporal_rate"]:>6.1f}%\n')
                        f.write(f'    Total Steps:      {metrics["important_total_steps"]}\n')
                        f.write(f'    Total Distance:   {metrics["important_total_distance"]:.0f} m\n')
                        f.write('\n')
                    
                    f.write('\n')
            
            # =====================================================
            # INDIVIDUAL BICYCLE DETAILS
            # =====================================================
            f.write('INDIVIDUAL BICYCLE DETAILS:\n')
            f.write('═══════════════════════════════════════════════════════════════════\n\n')
            
            # Sort bicycles by spatio-temporal rate (descending)
            bicycles_sorted = sorted(results['individual'].items(), 
                                   key=lambda x: x[1]['spatiotemporal_rate'], 
                                   reverse=True)
            
            for bicycle_id, metrics in bicycles_sorted:
                f.write(f'Bicycle: {bicycle_id}\n')
                f.write('─────────────────────────────────────────────────────────────────\n')
                if metrics['flow_id']:
                    f.write(f'  Flow:               {metrics["flow_id"]}\n')
                f.write(f'  Total Distance:     {metrics["total_distance"]:.0f} m\n')
                f.write(f'  Total Time Steps:   {metrics["total_time_steps"]}\n')
                f.write('\n')
                
                # Detection rates
                f.write('  Overall Detection Rates:\n')
                f.write(f'    Temporal:         {metrics["temporal_rate"]:>6.1f}%\n')
                f.write(f'    Spatial:          {metrics["spatial_rate"]:>6.1f}%\n')
                f.write(f'    Spatio-temporal:  {metrics["spatiotemporal_rate"]:>6.1f}%\n')
                f.write('\n')
                
                # Redundancy distribution
                if 'redundancy_temporal' in metrics:
                    f.write('  Redundancy Distribution:\n')
                    f.write('    Observer Count │  Temporal │  Spatial  │ Spatio-temporal\n')
                    f.write('    ───────────────┼───────────┼───────────┼────────────────\n')
                    for level in range(6):
                        label = f'{level} obs'
                        temp_rate = metrics['redundancy_temporal'].get(level, 0.0)
                        spat_rate = metrics['redundancy_spatial'].get(level, 0.0)
                        st_rate = metrics['redundancy_spatiotemporal'].get(level, 0.0)
                        f.write(f'    {label:<14} │  {temp_rate:>6.1f}%  │  {spat_rate:>6.1f}%  │     {st_rate:>6.1f}%\n')
                    f.write('\n')
                
                # Occlusion level distribution
                if 'occlusion_temporal' in metrics:
                    f.write('  Occlusion Level Distribution:\n')
                    f.write('    Occlusion Level    │  Temporal │  Spatial  │ Spatio-temporal\n')
                    f.write('    ───────────────────┼───────────┼───────────┼────────────────\n')
                    occlusion_scale = self.config.get('occlusion_level_scale', _get_occlusion_level_scale())
                    for category_name, min_pct, max_pct in occlusion_scale:
                        display_name = category_name.replace('_', ' ').capitalize()
                        temp_rate = metrics['occlusion_temporal'].get(category_name, 0.0)
                        spat_rate = metrics['occlusion_spatial'].get(category_name, 0.0)
                        st_rate = metrics['occlusion_spatiotemporal'].get(category_name, 0.0)
                        f.write(f'    {display_name:<18} │  {temp_rate:>6.1f}%  │  {spat_rate:>6.1f}%  │     {st_rate:>6.1f}%\n')
                    f.write('\n')
                
                # Critical area detection
                if metrics['important_total_steps'] > 0:
                    f.write('  Critical Interaction Area Detection:\n')
                    f.write(f'    Temporal:         {metrics["important_temporal_rate"]:>6.1f}%\n')
                    f.write(f'    Spatial:          {metrics["important_spatial_rate"]:>6.1f}%\n')
                    f.write(f'    Spatio-temporal:  {metrics["important_spatiotemporal_rate"]:>6.1f}%\n')
                    f.write(f'    Total Steps:      {metrics["important_total_steps"]}\n')
                    f.write(f'    Total Distance:   {metrics["important_total_distance"]:.0f} m\n')
                    f.write('\n')
                
                # Conflict detection
                if metrics['num_conflicts'] > 0:
                    f.write('  Conflict Detection:\n')
                    f.write(f'    Temporal:         {metrics["conflict_temporal_rate"]:>6.1f}%\n')
                    f.write(f'    Spatial:          {metrics["conflict_spatial_rate"]:>6.1f}%\n')
                    f.write(f'    Spatio-temporal:  {metrics["conflict_spatiotemporal_rate"]:>6.1f}%\n')
                    f.write(f'    Total Conflicts:  {metrics["num_conflicts"]}\n')
                    f.write('\n')
                
                f.write('\n')
            
            # =====================================================
            # FOOTER
            # =====================================================
            f.write('═══════════════════════════════════════════════════════════════════\n')
            f.write('For machine-readable data, see the accompanying CSV file:\n')
            f.write(f'  {file_prefix}_data.csv\n')
            f.write('═══════════════════════════════════════════════════════════════════\n')
    
    def save_config(self, filename):
        """Save current configuration to JSON file."""
        with open(filename, 'w') as f:
            json.dump(self.config, f, indent=2)
        print(f"Configuration saved to: {filename}")


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(
        description='VRU-Specific Detection Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Auto-detect from default scenario
  python evaluation_VRU_specific_detection.py
  
  # Specify custom scenario path
  python evaluation_VRU_specific_detection.py --scenario-path outputs/my_scenario
  
  # Override FCO/FBO shares
  python evaluation_VRU_specific_detection.py --fco-share 75 --fbo-share 25
  
  # Use custom configuration file
  python evaluation_VRU_specific_detection.py --config my_config.json
        '''
    )
    
    # Configuration source
    parser.add_argument('--config', help='Path to JSON configuration file')
    
    # Feature toggles
    parser.add_argument('--enable-2d-plots', action='store_true', help='Generate individual 2D bicycle detection plots')
    parser.add_argument('--disable-2d-plots', action='store_true', help='Disable individual 2D bicycle detection plots')
    parser.add_argument('--enable-flow-based-plots', action='store_true', help='Generate flow-based 2D detection plots')
    parser.add_argument('--disable-flow-based-plots', action='store_true', help='Disable flow-based 2D detection plots')
    parser.add_argument('--enable-redundancy-plots', action='store_true', help='Generate individual 2D detection-redundancy plots')
    parser.add_argument('--disable-redundancy-plots', action='store_true', help='Disable individual 2D detection-redundancy plots')
    parser.add_argument('--enable-flow-redundancy-plots', action='store_true', help='Generate flow-based 2D detection-redundancy plots')
    parser.add_argument('--disable-flow-redundancy-plots', action='store_true', help='Disable flow-based 2D detection-redundancy plots')
    parser.add_argument('--enable-occlusion-plots', action='store_true', help='Generate individual 2D occlusion level plots')
    parser.add_argument('--disable-occlusion-plots', action='store_true', help='Disable individual 2D occlusion level plots')
    parser.add_argument('--enable-flow-occlusion-plots', action='store_true', help='Generate flow-based 2D occlusion level plots')
    parser.add_argument('--disable-flow-occlusion-plots', action='store_true', help='Disable flow-based 2D occlusion level plots')
    parser.add_argument('--enable-3d-plots', action='store_true', help='Generate 3D detection plots')
    parser.add_argument('--disable-3d-plots', action='store_true', help='Disable 3D detection plots')
    parser.add_argument('--enable-3d-conflict-plots', action='store_true', help='Generate individual 3D conflict plots')
    parser.add_argument('--disable-3d-conflict-plots', action='store_true', help='Disable individual 3D conflict plots')
    parser.add_argument('--show-3d-conflict-background', action='store_true', help='Show background geometry in 3D conflict plots')
    parser.add_argument('--disable-3d-conflict-background', action='store_true', help='Disable background geometry in 3D conflict plots')
    parser.add_argument('--enable-statistics', action='store_true', help='Generate statistics summaries')
    parser.add_argument('--disable-statistics', action='store_true', help='Disable statistics summaries')
    
    # Scenario parameters
    parser.add_argument('--scenario-path', help='Path to scenario output directory')
    parser.add_argument('--file-tag', help='File tag for output naming')
    parser.add_argument('--fco-share', type=int, help='FCO penetration percentage')
    parser.add_argument('--fbo-share', type=int, help='FBO penetration percentage')
    
    # Analysis parameters
    parser.add_argument('--step-length', type=float, help='Simulation step length in seconds')
    parser.add_argument('--min-segment-length', type=int, help='Minimum segment length for trajectory analysis')
    parser.add_argument('--max-gap-bridge', type=int, help='Maximum gap to bridge in detection timeline')
    
    # Occlusion configuration
    parser.add_argument('--occlusion-scale', help='Custom occlusion level scale (JSON format: [["name", min, max], ...])')
    
    # 3D plotting parameters (only relevant if 3D plots are enabled)
    parser.add_argument('--3d-output-dir', help='Output directory for 3D plots')
    parser.add_argument('--view-elevation', type=float, help='3D plot elevation angle (degrees)')
    parser.add_argument('--view-azimuth', type=float, help='3D plot azimuth angle (degrees)')
    
    # Optional parameters
    parser.add_argument('--output-dir', default=None, help='Output directory for 2D plots')
    parser.add_argument('--save-config', help='Save configuration to JSON file')
    
    args = parser.parse_args()
    
    # Build configuration overrides
    config_kwargs = {}
    
    # Feature toggles (handle enable/disable pairs)
    if args.enable_2d_plots:
        config_kwargs['individual_2d_detection_plots'] = True
    elif args.disable_2d_plots:
        config_kwargs['individual_2d_detection_plots'] = False
        
    if args.enable_flow_based_plots:
        config_kwargs['flow_based_2d_detection_plots'] = True
    elif args.disable_flow_based_plots:
        config_kwargs['flow_based_2d_detection_plots'] = False
        
    if args.enable_redundancy_plots:
        config_kwargs['individual_2d_detection_redundancy_plots'] = True
    elif args.disable_redundancy_plots:
        config_kwargs['individual_2d_detection_redundancy_plots'] = False
        
    if args.enable_flow_redundancy_plots:
        config_kwargs['flow_based_2d_detection_redundancy_plots'] = True
    elif args.disable_flow_redundancy_plots:
        config_kwargs['flow_based_2d_detection_redundancy_plots'] = False
    
    if args.enable_occlusion_plots:
        config_kwargs['individual_2d_occlusion_plots'] = True
    elif args.disable_occlusion_plots:
        config_kwargs['individual_2d_occlusion_plots'] = False
    
    if args.enable_flow_occlusion_plots:
        config_kwargs['flow_based_2d_occlusion_plots'] = True
    elif args.disable_flow_occlusion_plots:
        config_kwargs['flow_based_2d_occlusion_plots'] = False
        
    if args.enable_3d_plots:
        config_kwargs['individual_3d_detection_plots'] = True
    elif args.disable_3d_plots:
        config_kwargs['individual_3d_detection_plots'] = False
    
    if args.enable_3d_conflict_plots:
        config_kwargs['individual_3d_conflict_plots'] = True
    elif args.disable_3d_conflict_plots:
        config_kwargs['individual_3d_conflict_plots'] = False
    
    if args.show_3d_conflict_background:
        config_kwargs['show_3d_conflict_background'] = True
    elif args.disable_3d_conflict_background:
        config_kwargs['show_3d_conflict_background'] = False
        
    if args.enable_statistics:
        config_kwargs['enable_statistics'] = True
    elif args.disable_statistics:
        config_kwargs['enable_statistics'] = False
    
    # Other configuration parameters
    if args.scenario_path:
        config_kwargs['scenario_path'] = args.scenario_path
    if args.file_tag:
        config_kwargs['file_tag'] = args.file_tag
    if args.fco_share is not None:
        config_kwargs['fco_share'] = args.fco_share
    if args.fbo_share is not None:
        config_kwargs['fbo_share'] = args.fbo_share
    if args.step_length:
        config_kwargs['step_length'] = args.step_length
    if args.min_segment_length:
        config_kwargs['min_segment_length'] = args.min_segment_length
    if args.max_gap_bridge:
        config_kwargs['max_gap_bridge'] = args.max_gap_bridge
    if args.output_dir:
        config_kwargs['output_dir'] = args.output_dir
    if getattr(args, '3d_output_dir', None):
        config_kwargs['output_dir_3d'] = getattr(args, '3d_output_dir')
    if args.view_elevation:
        config_kwargs['view_elevation'] = args.view_elevation
    if args.view_azimuth:
        config_kwargs['view_azimuth'] = args.view_azimuth
    
    # Parse occlusion scale if provided
    if args.occlusion_scale:
        import json
        try:
            custom_scale = json.loads(args.occlusion_scale)
            config_kwargs['occlusion_level_scale'] = custom_scale
        except json.JSONDecodeError as e:
            print(f"Error parsing occlusion scale JSON: {e}")
            return 1
    
    # Initialize analyzer
    try:
        analyzer = VRUDetectionAnalyzer(config_file=args.config, **config_kwargs)
    except Exception as e:
        print(f"Configuration error: {e}")
        return 1
    
    # Save configuration if requested
    if args.save_config:
        analyzer.save_config(args.save_config)
        print(f"Configuration saved to: {args.save_config}")
    
    # Perform analysis
    try:
        analyzer.analyze_vru_trajectories()
        return 0
    except Exception as e:
        print(f"Analysis failed: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
