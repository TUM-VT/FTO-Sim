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

# ANALYSIS FEATURE TOGGLES - Configure which analyses to perform
ENABLE_2D_PLOTS = False      # Generate individual 2D bicycle trajectory plots (space-time diagrams)
ENABLE_TRAFFIC_LIGHTS = False  # Include traffic light states in 2D trajectory plots

ENABLE_3D_PLOTS = True     # Generate 3D detection plots with observer trajectories and scene geometry

ENABLE_STATISTICS = True    # Generate trajectory statistics and detection rate summaries

# =============================

# 2. SCENARIO CONFIGURATION
SCENARIO_OUTPUT_PATH = "outputs/ETRR_single-FCO"  # Path to scenario output folder (set to None to use manual configuration)

# 3. TRAJECTORY ANALYSIS SETTINGS  
MIN_SEGMENT_LENGTH = 3      # Minimum segment length for bicycle trajectory analysis (data points)
MAX_GAP_BRIDGE = 10         # Maximum number of undetected frames to bridge between detected segments
STEP_LENGTH = 0.1           # Simulation step length in seconds (fallback value)

# 4. PLOT SETTINGS
DPI = 300                   # Resolution for saved plots
FIGURE_SIZE = (12, 8)       # Figure size in inches for 2D plots
FIGURE_SIZE_3D = (15, 12)   # Figure size in inches for 3D plots

# 5. 3D VISUALIZATION SETTINGS (only relevant if ENABLE_3D_PLOTS = True)
VIEW_ELEVATION = 35         # 3D plot elevation angle (degrees)
VIEW_AZIMUTH = 270          # 3D plot azimuth angle (degrees)
Z_AXIS_SCALE_FACTOR = 2.0   # Scale factor for z-axis relative to x/y axes

# 6. VRU VEHICLE TYPES
VRU_VEHICLE_TYPES = ["bicycle", "DEFAULT_BIKETYPE", "floating_bike_observer"]

# 7. OBSERVER VEHICLE TYPES  
OBSERVER_VEHICLE_TYPES = ["floating_car_observer", "floating_bike_observer"]

# =============================
# OPTIONAL MANUAL CONFIGURATION (only needed if SCENARIO_OUTPUT_PATH = None)
# =============================

# Manual configuration (used only if SCENARIO_OUTPUT_PATH is None)
MANUAL_SCENARIO_PATH = "outputs/ETRR_single-FCO"
MANUAL_FILE_TAG = "ETRR_single-FCO"
MANUAL_FCO_SHARE = 100  # FCO penetration percentage
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
            if SCENARIO_OUTPUT_PATH and not kwargs.get('scenario_path'):
                config = self._auto_detect_configuration()
            else:
                config = self._get_manual_configuration(**kwargs)
        
        # Apply any command-line overrides
        for key, value in kwargs.items():
            if value is not None:
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
        print(f"Auto-detecting parameters from: {scenario_name}")
        
        # Extract file tag and FCO/FBO shares from scenario name
        if '_FCO' in scenario_name and '_FBO' in scenario_name:
            # Extract file tag (part before _FCO)
            file_tag = scenario_name.split('_FCO')[0]
            
            # Extract FCO and FBO shares
            fco_match = re.search(r'FCO(\d+)%', scenario_name)
            fbo_match = re.search(r'FBO(\d+)%', scenario_name)
            
            fco_share = int(fco_match.group(1)) if fco_match else 100
            fbo_share = int(fbo_match.group(1)) if fbo_match else 0
            
            print(f"  - Parsed scenario: {file_tag}, FCO: {fco_share}%, FBO: {fbo_share}%")
        else:
            # Fallback parsing
            file_tag = scenario_name
            fco_share = 100
            fbo_share = 0
            print(f"  - Using fallback parsing: {file_tag}")
        
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
            'enable_2d_plots': ENABLE_2D_PLOTS,
            'enable_3d_plots': ENABLE_3D_PLOTS,
            'enable_statistics': ENABLE_STATISTICS,
            'enable_traffic_lights': ENABLE_TRAFFIC_LIGHTS,
            'view_elevation': VIEW_ELEVATION,
            'view_azimuth': VIEW_AZIMUTH,
            'z_axis_scale_factor': Z_AXIS_SCALE_FACTOR
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
            'figure_size': kwargs.get('figure_size', FIGURE_SIZE),
            'figure_size_3d': kwargs.get('figure_size_3d', FIGURE_SIZE_3D),
            'enable_2d_plots': kwargs.get('enable_2d_plots', ENABLE_2D_PLOTS),
            'enable_3d_plots': kwargs.get('enable_3d_plots', ENABLE_3D_PLOTS),
            'enable_statistics': kwargs.get('enable_statistics', ENABLE_STATISTICS),
            'enable_traffic_lights': kwargs.get('enable_traffic_lights', ENABLE_TRAFFIC_LIGHTS),
            'view_elevation': kwargs.get('view_elevation', VIEW_ELEVATION),
            'view_azimuth': kwargs.get('view_azimuth', VIEW_AZIMUTH),
            'z_axis_scale_factor': kwargs.get('z_axis_scale_factor', Z_AXIS_SCALE_FACTOR)
        }
    
    def _detect_step_length(self, scenario_path):
        """Detect simulation step length from log files."""
        # Try to find step length in summary log
        summary_log = scenario_path / 'out_logging' / f'summary_log_{scenario_path.name}.csv'
        
        if summary_log.exists():
            try:
                # Read the header comments to find step length
                with open(summary_log, 'r') as f:
                    for line in f:
                        if '# Step length:' in line:
                            step_length = float(line.split(':')[1].split('seconds')[0].strip())
                            print(f"  - Found step length in summary log: {step_length}s")
                            return step_length
                        if not line.startswith('#'):
                            break
            except Exception as e:
                print(f"  - Warning: Could not parse step length from summary log: {e}")
        
        # Fallback
        print(f"  - Using fallback step length: {STEP_LENGTH}s")
        return STEP_LENGTH
    
    def _ensure_output_directories(self):
        """Create output directory if it doesn't exist."""
        # Since all outputs go to the same directory now, just create once
        output_dir = self.config['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        print(f"✓ VRU-specific detection output directory: {output_dir}")
    
    def load_trajectory_data(self):
        """Load bicycle trajectory data from CSV log file."""
        trajectory_file = Path(self.config['scenario_path']) / 'out_logging' / f'log_bicycle_trajectories_{Path(self.config["scenario_path"]).name}.csv'
        
        if not trajectory_file.exists():
            raise FileNotFoundError(f"Bicycle trajectory log file not found: {trajectory_file}")
        
        print(f"Loading bicycle trajectory data: {trajectory_file}")
        
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
        
        print(f"Loaded {len(df)} trajectory data points for {df['vehicle_id'].nunique()} bicycles")
        
        return df
    
    def load_detection_data(self):
        """Load bicycle detection data from CSV log file."""
        detection_file = Path(self.config['scenario_path']) / 'out_logging' / f'log_detections_{Path(self.config["scenario_path"]).name}.csv'
        
        if not detection_file.exists():
            print("Warning: No detection log file found - trajectories will show as undetected")
            return pd.DataFrame()
        
        print(f"Loading detection data: {detection_file}")
        
        # Read CSV, skipping comment lines
        df = pd.read_csv(detection_file, comment='#')
        
        print(f"Loaded {len(df)} detection events")
        
        return df
    
    def load_observer_trajectories(self):
        """Load observer vehicle trajectory data from CSV log file."""
        trajectory_file = Path(self.config['scenario_path']) / 'out_logging' / f'log_vehicle_trajectories_{Path(self.config["scenario_path"]).name}.csv'
        
        if not trajectory_file.exists():
            print("Warning: No vehicle trajectory log file found - 3D plots will not show observer trajectories")
            return pd.DataFrame()
        
        print(f"Loading observer trajectory data: {trajectory_file}")
        
        # Read CSV, skipping comment lines
        df = pd.read_csv(trajectory_file, comment='#')
        
        # Filter for observer vehicle types only
        df = df[df['vehicle_type'].isin(OBSERVER_VEHICLE_TYPES)]
        
        print(f"Loaded {len(df)} trajectory data points for {df['vehicle_id'].nunique()} observer vehicles")
        
        return df
    
    def load_geometry_data(self):
        """Load comprehensive geometry data for 3D visualization background using OSM data extraction."""
        import osmnx as ox
        
        # Get the bounding box from the main.py configuration
        # For now, use the ETRR bounds as fallback
        north, south, east, west = 48.15050, 48.14905, 11.57100, 11.56790  # ETRR bounds
        bbox = (north, south, east, west)
        
        print(f"Loading comprehensive geometry data using OSM extraction for bbox: {bbox}")
        
        try:
            # Initialize coordinate transformer
            transformer = pyproj.Transformer.from_crs('EPSG:4326', 'EPSG:32632', always_xy=True)
            
            # Load road network (same as in main.py)
            try:
                print("  - Loading road network...")
                G = ox.graph_from_bbox(bbox=bbox, network_type='all', simplify=True, retain_all=True)
                gdf1 = ox.graph_to_gdfs(G, nodes=False)  # road space distribution
                gdf1_proj = gdf1.to_crs("EPSG:32632")
            except Exception as e:
                print(f"  - Warning: Could not load road network: {e}")
                gdf1_proj = None
            
            # Load buildings
            try:
                print("  - Loading buildings...")
                buildings = ox.features_from_bbox(bbox=bbox, tags={'building': True})
                buildings_proj = buildings.to_crs("EPSG:32632")
            except Exception as e:
                print(f"  - No buildings found: {e}")
                buildings_proj = None
                
            # Load parks
            try:
                print("  - Loading parks...")
                parks = ox.features_from_bbox(bbox=bbox, tags={'leisure': 'park'})
                parks_proj = parks.to_crs("EPSG:32632")
            except Exception as e:
                print(f"  - No parks found: {e}")
                parks_proj = None
                
            # Load trees
            try:
                print("  - Loading trees...")
                trees = ox.features_from_bbox(bbox=bbox, tags={'natural': 'tree'})
                trees_proj = trees.to_crs("EPSG:32632")
                # Use same data for leaves (crown representation)
                leaves_proj = trees_proj
            except Exception as e:
                print(f"  - No trees found: {e}")
                trees_proj = None
                leaves_proj = None
                
            # Load barriers
            try:
                print("  - Loading barriers...")
                barriers = ox.features_from_bbox(bbox=bbox, tags={'barrier': 'retaining_wall'})
                barriers_proj = barriers.to_crs("EPSG:32632")
            except Exception as e:
                print(f"  - No barriers found: {e}")
                barriers_proj = None
                
            # Load PT shelters
            try:
                print("  - Loading PT shelters...")
                PT_shelters = ox.features_from_bbox(bbox=bbox, tags={'shelter_type': 'public_transport'})
                PT_shelters_proj = PT_shelters.to_crs("EPSG:32632")
            except Exception as e:
                print(f"  - No PT shelters found: {e}")
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
            
            print(f"  - Loaded comprehensive scene: {element_counts}")
            
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
            print("Traffic light visualization disabled - skipping traffic light data loading")
            return pd.DataFrame()
            
        tl_file = Path(self.config['scenario_path']) / 'out_logging' / f'log_traffic_lights_{Path(self.config["scenario_path"]).name}.csv'
        
        if not tl_file.exists():
            print("Warning: No traffic light log file found - no traffic light visualization")
            return pd.DataFrame()
        
        print(f"Loading traffic light data: {tl_file}")
        
        # Read CSV, skipping comment lines
        df = pd.read_csv(tl_file, comment='#')
        
        print(f"Loaded {len(df)} traffic light state records")
        
        return df
    
    def process_bicycle_trajectories(self, trajectory_df, detection_df, traffic_light_df):
        """Process and plot individual bicycle trajectories."""
        
        print("\n=== Processing Individual Bicycle Trajectories ===")
        
        # Group trajectory data by bicycle
        bicycle_groups = trajectory_df.groupby('vehicle_id')
        
        trajectory_count = 0
        
        for bicycle_id, bicycle_data in bicycle_groups:
            print(f"\nProcessing bicycle: {bicycle_id}")
            
            # Sort by time step
            bicycle_data = bicycle_data.sort_values('time_step')
            
            # Extract trajectory information
            time_steps = bicycle_data['time_step'].values
            distances = bicycle_data['distance'].values
            
            # Convert time steps to elapsed time (relative to first appearance)
            start_time_step = time_steps[0]
            elapsed_times = (time_steps - start_time_step) * self.config['step_length']
            
            # Get detection status for this bicycle
            bicycle_detections = detection_df[detection_df['bicycle_id'] == bicycle_id] if len(detection_df) > 0 else pd.DataFrame()
            
            # Create detection timeline
            detection_timeline = self._create_detection_timeline(time_steps, bicycle_detections, start_time_step)
            
            # Apply detection smoothing
            smoothed_detection = self._smooth_detection_timeline(detection_timeline)
            
            # Split trajectory into detected/undetected segments
            segments = self._split_trajectory_segments(distances, elapsed_times, smoothed_detection)
            
            # Get traffic light information for this bicycle
            tl_info = self._get_bicycle_traffic_lights(bicycle_data, traffic_light_df)
            
            # Generate the plot
            self._plot_individual_trajectory(
                bicycle_id, segments, tl_info,
                start_time_step, elapsed_times[-1] if len(elapsed_times) > 0 else 0
            )
            
            trajectory_count += 1
        
        print(f"\n✓ Generated {trajectory_count} individual trajectory plots")
    
    def process_3d_detection_plots(self, trajectory_df, detection_df, observer_df, geometry_data):
        """Process and generate 3D detection plots for bicycle trajectories."""
        
        print("\n=== Processing 3D Detection Plots ===")
        
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
        
        trajectory_count = 0
        
        for bicycle_id, bicycle_data in bicycle_groups:
            print(f"\nProcessing 3D plot for bicycle: {bicycle_id}")
            
            # Sort by time step
            bicycle_data = bicycle_data.sort_values('time_step')
            
            # Extract trajectory coordinates and times  
            time_steps = bicycle_data['time_step'].values
            x_coords = bicycle_data['x_coord'].values
            y_coords = bicycle_data['y_coord'].values
            
            # Convert time steps to elapsed time (relative to first appearance)
            start_time_step = time_steps[0]
            elapsed_times = (time_steps - start_time_step) * self.config['step_length']
            
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
        
        print(f"\n✓ Generated {trajectory_count} 3D detection plots")
    
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
    
    def _split_trajectory_segments(self, distances, times, detection_status):
        """Split trajectory into detected and undetected segments."""
        segments = {'detected': [], 'undetected': []}
        
        if len(distances) == 0:
            return segments
        
        current_segment = []
        current_status = detection_status[0]
        
        for i in range(len(distances)):
            if detection_status[i] == current_status:
                current_segment.append((distances[i], times[i]))
            else:
                # Status changed, save current segment if long enough
                if len(current_segment) >= self.config['min_segment_length']:
                    segment_key = 'detected' if current_status else 'undetected'
                    segments[segment_key].append(current_segment)
                
                # Start new segment
                current_segment = [(distances[i], times[i])]
                current_status = detection_status[i]
        
        # Add final segment
        if len(current_segment) >= self.config['min_segment_length']:
            segment_key = 'detected' if current_status else 'undetected'
            segments[segment_key].append(current_segment)
        
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
            
            # Convert to elapsed time relative to bicycle start
            observer_data['elapsed_time'] = (observer_data['time_step'] - start_time_step) * self.config['step_length']
            
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
            detection_times = set((det_row['time_step'] - start_time_step) * self.config['step_length'] 
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
                    
                    # Convert time steps to elapsed time
                    start_time_step = bicycle_data['time_step'].min()
                    tl_data['elapsed_time'] = (tl_data['time_step'] - start_time_step) * self.config['step_length']
                    
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
                        
                        print(f"    Found traffic light: {tl_id[:20]}... at ~{tl_info[tl_id]['avg_position']:.1f}m (signal {tl_info[tl_id]['signal_index']})")
                
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
            
            # Convert time steps to elapsed time relative to bicycle start
            tl_data['elapsed_time'] = (tl_data['time_step'] - start_time_step) * self.config['step_length']
            
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
    
    def _plot_individual_trajectory(self, bicycle_id, segments, tl_info, start_time_step, total_time):
        """Generate individual trajectory plot."""
        
        fig, ax = plt.subplots(figsize=self.config['figure_size'])
        
        # Plot undetected segments
        for segment in segments['undetected']:
            if len(segment) > 1:
                distances, times = zip(*segment)
                ax.plot(times, distances, color='black', linewidth=1.5, linestyle='solid', label='bicycle undetected')
        
        # Plot detected segments (swap x and y axes)
        for segment in segments['detected']:
            if len(segment) > 1:
                distances, times = zip(*segment)
                ax.plot(times, distances, color='darkturquoise', linewidth=1.5, linestyle='solid', label='bicycle detected')
        
        # Plot traffic light state changes as vertical lines
        if tl_info:
            print(f"    Plotting {len(tl_info)} traffic lights")
            for tl_id, tl_data in tl_info.items():
                states = tl_data['states']
                avg_position = tl_data['avg_position']
                signal_index = tl_data['signal_index']
                
                print(f"      {tl_id[:20]}... (signal {signal_index}): {len(states)} state changes at ~{avg_position:.1f}m")
                
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
        
        
        # Calculate trajectory statistics
        total_distance = 0
        detected_distance = 0
        detected_time = 0
        
        all_segments = segments['detected'] + segments['undetected']
        if all_segments:
            for segment in all_segments:
                if len(segment) > 1:
                    seg_distance = segment[-1][0] - segment[0][0]
                    total_distance += seg_distance
            
            for segment in segments['detected']:
                if len(segment) > 1:
                    seg_distance = segment[-1][0] - segment[0][0]
                    seg_time = segment[-1][1] - segment[0][1]
                    detected_distance += seg_distance
                    detected_time += seg_time
        
        # Calculate detection rates
        distance_detection_rate = (detected_distance / total_distance * 100) if total_distance > 0 else 0
        time_detection_rate = (detected_time / total_time * 100) if total_time > 0 else 0
        spatiotemporal_detection_rate = (distance_detection_rate + time_detection_rate) / 2
        
        # Add information text box with updated terminology
        info_text = (
            f"Bicycle: {bicycle_id}\n"
            f"Departure time: {start_time_step * self.config['step_length']:.1f} s\n"
            f"Temporal detection rate: {time_detection_rate:.1f}%\n"
            f"Spatial detection rate: {distance_detection_rate:.1f}%\n"
            f"Spatio-temporal detection rate: {spatiotemporal_detection_rate:.1f}%"
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
            Line2D([0], [0], color='black', lw=2, label='bicycle undetected'),
            Line2D([0], [0], color='darkturquoise', lw=2, label='bicycle detected'),
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
        
        # Save plot
        output_filename = f'2D_individual_{self.config["file_tag"]}_FCO{self.config["fco_share"]}%_FBO{self.config["fbo_share"]}%_{bicycle_id}.png'
        output_path = os.path.join(self.config['output_dir'], output_filename)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.config['dpi'], bbox_inches='tight')
        plt.close(fig)
        
        print(f"  ✓ Saved: {output_filename}")
    
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
            
            print(f"    Using simulation bounding box: ({north:.5f}, {south:.5f}, {east:.5f}, {west:.5f})")
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
            
            print(f"    Using trajectory-based bounds (fallback)")
        
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
            
            print(f"    Using full bicycle trajectory time range: {t_max_trajectory:.1f}s")
        else:
            t_min, t_max = 0, max_elapsed_time
        
        print(f"    Z-axis range: {t_min:.1f}s to {t_max:.1f}s (rounded to multiple of 5, from full trajectory)")
        
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
        
        # Calculate dynamic figure size based on data aspect ratio
        dx = x_max - x_min
        dy = y_max - y_min
        dz = top_z - base_z
        
        # Calculate aspect ratio (prioritize spatial dimensions)
        max_spatial = max(dx, dy)
        aspect_x = dx / max_spatial
        aspect_y = dy / max_spatial
        
        # Base figure size (can be adjusted)
        base_size = 10  # inches
        margin_for_legend = 2  # extra inches for legend space
        
        # Calculate figure dimensions
        fig_width = base_size * aspect_x + margin_for_legend
        fig_height = base_size * aspect_y + 1  # small margin for axis labels
        
        # Ensure minimum size for readability
        fig_width = max(fig_width, 8)
        fig_height = max(fig_height, 6)
        
        # Create 3D figure with dynamic size
        fig = plt.figure(figsize=(fig_width, fig_height))
        ax = fig.add_subplot(111, projection='3d')
        
        # Optimize subplot parameters to minimize white space
        fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
        
        print(f"    Dynamic figure size: {fig_width:.1f} x {fig_height:.1f} inches (aspect: {aspect_x:.2f} x {aspect_y:.2f})")
        
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
        
        # Calculate aspect ratios
        dx = x_max - x_min
        dy = y_max - y_min
        dz = top_z - base_z
        max_range = max(dx, dy)
        aspect_ratios = [dx/max_range, dy/max_range, (dz/max_range) * self.config['z_axis_scale_factor']]
        ax.set_box_aspect(aspect_ratios)  # type: ignore # 3D plot aspect ratios work with lists
        
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
        
        # Save plot
        output_filename = f'3D_detection_{self.config["file_tag"]}_FCO{self.config["fco_share"]}%_FBO{self.config["fbo_share"]}%_{bicycle_id}.png'
        output_path = os.path.join(self.config['output_dir_3d'], output_filename)
        
        # Use tight layout and aggressive bbox trimming to minimize white space
        plt.tight_layout(pad=0.5)  # Reduce padding between elements
        plt.savefig(output_path, dpi=self.config['dpi'], bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        
        print(f"  ✓ Saved 3D plot: {output_filename}")
    
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
                                    alpha=0.5, linewidth=0.5, zorder=-1)
        
        # 2. Plot parks (seagreen, alpha=0.5, zorder=2)
        parks_proj = geometry_data.get('parks')
        if parks_proj is not None:
            self._plot_geometry_layer(ax, parks_proj, bbox, base_z, to_rel_x, to_rel_y,
                                    facecolor='seagreen', edgecolor='black', 
                                    alpha=0.5, linewidth=0.5, zorder=0)
        
        # 3. Plot buildings (darkgray, zorder=3)
        buildings_proj = geometry_data.get('buildings')
        if buildings_proj is not None:
            self._plot_geometry_layer(ax, buildings_proj, bbox, base_z, to_rel_x, to_rel_y,
                                    facecolor='darkgray', edgecolor='black', 
                                    alpha=1.0, linewidth=0.5, zorder=0)
        
        # 4. Plot barriers (black, linewidth=1.0, zorder=4)
        barriers_proj = geometry_data.get('barriers')
        if barriers_proj is not None:
            self._plot_geometry_layer(ax, barriers_proj, bbox, base_z, to_rel_x, to_rel_y,
                                    facecolor='none', edgecolor='black', 
                                    alpha=1.0, linewidth=1.0, zorder=0)
        
        # 5. Plot trees (forestgreen circles, zorder=5) - trunk + crown
        trees_proj = geometry_data.get('trees')
        leaves_proj = geometry_data.get('leaves')
        if trees_proj is not None:
            # Plot tree trunks (small circles)
            self._plot_point_features(ax, trees_proj, bbox, base_z, to_rel_x, to_rel_y,
                                    radius=0.5, facecolor='forestgreen', edgecolor='black',
                                    alpha=1.0, linewidth=0.5, zorder=0)
            
            # Plot tree crowns (larger circles, semi-transparent)
            if leaves_proj is not None:
                self._plot_point_features(ax, leaves_proj, bbox, base_z, to_rel_x, to_rel_y,
                                        radius=2.5, facecolor='forestgreen', edgecolor='black',
                                        alpha=0.5, linewidth=0.5, zorder=0)
        
        # 6. Plot PT shelters (lightgray, zorder=6)
        pt_shelters_proj = geometry_data.get('pt_shelters')
        if pt_shelters_proj is not None:
            self._plot_geometry_layer(ax, pt_shelters_proj, bbox, base_z, to_rel_x, to_rel_y,
                                    facecolor='lightgray', edgecolor='black', 
                                    alpha=1.0, linewidth=0.5, zorder=0)
    
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
            
            print(f"    Applying spatial bounding box filter for bicycle trajectories")
        
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
                print(f"    Plotting observer {observer_id}: {len(spatially_filtered_trajectory)}/{len(trajectory)} points (time + spatial filtering)")
            else:
                spatially_filtered_trajectory = time_filtered_trajectory
                print(f"    Plotting observer {observer_id}: {len(spatially_filtered_trajectory)}/{len(trajectory)} points (time filtering only)")
            
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
    
    def analyze_vru_trajectories(self):
        """Main method to perform VRU trajectory analysis."""
        print("=== VRU-Specific Detection Analysis ===")
        print("Configuration:")
        print(f"  - Scenario path: {self.config['scenario_path']}")
        print(f"  - File tag: {self.config['file_tag']}")
        print(f"  - FCO/FBO penetration: {self.config['fco_share']}%/{self.config['fbo_share']}%")
        print("Analysis features:")
        print(f"  - 2D trajectory plots: {'ENABLED' if self.config.get('enable_2d_plots', True) else 'DISABLED'}")
        print(f"  - 3D detection plots: {'ENABLED' if self.config.get('enable_3d_plots', False) else 'DISABLED'}")
        print(f"  - Statistics summary: {'ENABLED' if self.config.get('enable_statistics', True) else 'DISABLED'}")
        print("Output directory:")
        print(f"  - All outputs: {self.config['output_dir']}")
        print("Processing parameters:")
        print(f"  - Step length: {self.config['step_length']}s")
        print(f"  - Min segment length: {self.config['min_segment_length']} points")
        print(f"  - Max gap bridge: {self.config['max_gap_bridge']} points")
        
        try:
            # Load data
            trajectory_df = self.load_trajectory_data()
            detection_df = self.load_detection_data()
            traffic_light_df = self.load_traffic_light_data()
            
            # Load additional data for 3D plots if enabled
            observer_df = pd.DataFrame()
            geometry_data = None
            
            if self.config.get('enable_3d_plots', False):
                print(f"  - 3D plots enabled: Loading additional data...")
                observer_df = self.load_observer_trajectories()
                geometry_data = self.load_geometry_data()
            
            # Process 2D trajectories if enabled
            if self.config.get('enable_2d_plots', True):
                print(f"  - 2D plots enabled: Processing individual bicycle trajectories...")
                self.process_bicycle_trajectories(trajectory_df, detection_df, traffic_light_df)
            
            # Process 3D detection plots if enabled
            if self.config.get('enable_3d_plots', False) and len(trajectory_df) > 0:
                print(f"  - 3D plots enabled: Processing 3D detection plots...")
                self.process_3d_detection_plots(trajectory_df, detection_df, observer_df, geometry_data)
            
            # Process statistics if enabled
            if self.config.get('enable_statistics', True):
                print(f"  - Statistics enabled: Computing detection rate statistics...")
                statistics_results = self.calculate_detection_statistics(trajectory_df, detection_df)
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
    
    def calculate_detection_statistics(self, trajectory_df, detection_df):
        """
        Calculate detection statistics for three layers:
        1. Individual bicycle level
        2. Flow-based level (mean values per flow)
        3. System-wide level (overall mean values)
        """
        from datetime import datetime
        
        print("\n=== Computing Detection Rate Statistics ===")
        
        # Initialize dictionaries for storing metrics
        bicycle_metrics = {}
        flow_metrics = {}
        
        # Process each individual bicycle
        bicycle_groups = trajectory_df.groupby('vehicle_id')
        
        for bicycle_id, bicycle_data in bicycle_groups:
            print(f"Processing bicycle: {bicycle_id}")
            
            # Extract flow ID from bicycle ID
            flow_id = bicycle_id.rsplit('.', 1)[0] if '.' in bicycle_id else 'default_flow'
            
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
            elapsed_times = (time_steps - start_time_step) * self.config['step_length']
            
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
                'important_temporal_rate': important_temporal_rate,
                'important_spatial_rate': important_spatial_rate,
                'important_spatiotemporal_rate': important_spatiotemporal_rate,
                'important_total_steps': important_area_steps,
                'important_detected_steps': important_area_detected_steps,
                'important_total_distance': total_important_distance,
                'important_detected_distance': total_important_detected_distance,
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
        
        # Calculate flow-based detection rates
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
        
        # Calculate system-wide statistics
        system_metrics = self._calculate_system_wide_metrics(bicycle_metrics, flow_metrics)
        
        # Package results
        results = {
            'individual': bicycle_metrics,
            'flow_based': flow_metrics,
            'system_wide': system_metrics,
            'summary': {
                'total_bicycles': len(bicycle_metrics),
                'total_flows': len(flow_metrics),
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
        
        print(f"✓ Processed {len(bicycle_metrics)} bicycles across {len(flow_metrics)} flows")
        
        return results
    
    def _calculate_system_wide_metrics(self, bicycle_metrics, flow_metrics):
        """Calculate system-wide aggregated metrics"""
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
        
        return {
            # Individual bicycle averages
            'avg_individual_temporal_rate': avg_temporal_rate,
            'avg_individual_spatial_rate': avg_spatial_rate,
            'avg_individual_spatiotemporal_rate': avg_spatiotemporal_rate,
            
            # Individual bicycle important area averages
            'avg_individual_important_temporal_rate': avg_important_temporal_rate,
            'avg_individual_important_spatial_rate': avg_important_spatial_rate,
            'avg_individual_important_spatiotemporal_rate': avg_important_spatiotemporal_rate,
            
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
            'total_system_important_detected_distance': total_system_important_detected_distance
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
            csv_data.append({
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
                'important_detected_distance': metrics['important_detected_distance']
            })
        
        # Add flow-based data
        for flow_id, metrics in results['flow_based'].items():
            csv_data.append({
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
            })
        
        # Add system-wide data
        system_wide = results['system_wide']
        csv_data.append({
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
        })
        
        csv_data.append({
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
        })
        
        csv_data.append({
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
        })
        
        # Convert to DataFrame and save
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_file, index=False)
    
    def _export_summary_report(self, results, output_dir, file_prefix):
        """Export concise human-readable summary report"""
        
        report_file = os.path.join(output_dir, f"{file_prefix}_summary.txt")
        
        with open(report_file, 'w') as f:
            # Header
            f.write('=======================================================\n')
            f.write('VRU DETECTION ANALYSIS - SUMMARY STATISTICS\n')
            f.write('=======================================================\n')
            config = results['summary']['configuration']
            f.write(f'Scenario: {config["file_tag"]} (FCO: {config["fco_share"]}%, FBO: {config["fbo_share"]}%)\n')
            f.write(f'Generated: {results["summary"]["analysis_timestamp"][:19]}\n')
            f.write('=======================================================\n\n')
            
            system_wide = results['system_wide']
            
            # Key metrics overview
            f.write('KEY METRICS OVERVIEW:\n')
            f.write(f'  Total Bicycles: {results["summary"]["total_bicycles"]}\n')
            f.write(f'  Total Flows: {results["summary"]["total_flows"]}\n')
            f.write(f'  Avg Bicycles/Flow: {system_wide["avg_bicycles_per_flow"]:.1f}\n')
            f.write(f'  Total Distance: {system_wide["total_system_distance"]:.0f}m\n')
            f.write(f'  Total Time Steps: {system_wide["total_system_steps"]}\n\n')
            
            # Detection rates summary
            f.write('DETECTION RATES SUMMARY:\n')
            f.write('                          Temporal   Spatial   Spatio-temporal\n')
            f.write('  Individual Average:     ' + 
                   f'{system_wide["avg_individual_temporal_rate"]:8.1f}%  ' +
                   f'{system_wide["avg_individual_spatial_rate"]:7.1f}%  ' +
                   f'{system_wide["avg_individual_spatiotemporal_rate"]:13.1f}%\n')
            f.write('  Flow Average:           ' + 
                   f'{system_wide["avg_flow_temporal_rate"]:8.1f}%  ' +
                   f'{system_wide["avg_flow_spatial_rate"]:7.1f}%  ' +
                   f'{system_wide["avg_flow_spatiotemporal_rate"]:13.1f}%\n')
            f.write('  System-wide Cumulative: ' + 
                   f'{system_wide["overall_temporal_rate"]:8.1f}%  ' +
                   f'{system_wide["overall_spatial_rate"]:7.1f}%  ' +
                   f'{system_wide["overall_spatiotemporal_rate"]:13.1f}%\n\n')
            
            # Important area detection rates summary
            if (system_wide["total_system_important_steps"] > 0 or 
                system_wide["total_system_important_distance"] > 0):
                f.write('CRITICAL INTERACTION AREA DETECTION RATES:\n')
                f.write('                          Temporal   Spatial   Spatio-temporal\n')
                f.write('  Individual Average:     ' + 
                       f'{system_wide["avg_individual_important_temporal_rate"]:8.1f}%  ' +
                       f'{system_wide["avg_individual_important_spatial_rate"]:7.1f}%  ' +
                       f'{system_wide["avg_individual_important_spatiotemporal_rate"]:13.1f}%\n')
                f.write('  Flow Average:           ' + 
                       f'{system_wide["avg_flow_important_temporal_rate"]:8.1f}%  ' +
                       f'{system_wide["avg_flow_important_spatial_rate"]:7.1f}%  ' +
                       f'{system_wide["avg_flow_important_spatiotemporal_rate"]:13.1f}%\n')
                f.write('  System-wide Cumulative: ' + 
                       f'{system_wide["overall_important_temporal_rate"]:8.1f}%  ' +
                       f'{system_wide["overall_important_spatial_rate"]:7.1f}%  ' +
                       f'{system_wide["overall_important_spatiotemporal_rate"]:13.1f}%\n')
                f.write(f'  Total Steps in Areas:   {system_wide["total_system_important_steps"]}\n')
                f.write(f'  Total Distance in Areas: {system_wide["total_system_important_distance"]:.0f}m\n\n')
            else:
                f.write('CRITICAL INTERACTION AREA DETECTION RATES:\n')
                f.write('  No critical interaction areas defined in this simulation.\n\n')
            
            # Top/bottom performers
            individual_rates = [(bike_id, metrics['spatiotemporal_rate']) 
                              for bike_id, metrics in results['individual'].items()]
            individual_rates.sort(key=lambda x: x[1], reverse=True)
            
            f.write('PERFORMANCE RANGE:\n')
            if len(individual_rates) >= 3:
                f.write(f'  Best performing bicycle:  {individual_rates[0][0]} ({individual_rates[0][1]:.1f}%)\n')
                f.write(f'  Worst performing bicycle: {individual_rates[-1][0]} ({individual_rates[-1][1]:.1f}%)\n')
            
            if results['flow_based']:
                flow_rates = [(flow_id, metrics['spatiotemporal_rate']) 
                            for flow_id, metrics in results['flow_based'].items()]
                flow_rates.sort(key=lambda x: x[1], reverse=True)
                f.write(f'  Best performing flow:     {flow_rates[0][0]} ({flow_rates[0][1]:.1f}%)\n')
                if len(flow_rates) > 1:
                    f.write(f'  Worst performing flow:    {flow_rates[-1][0]} ({flow_rates[-1][1]:.1f}%)\n')
            
            f.write('\n')
            f.write('For detailed data, see the accompanying CSV file.\n')
    
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
    parser.add_argument('--enable-2d-plots', action='store_true', help='Generate 2D bicycle trajectory plots')
    parser.add_argument('--disable-2d-plots', action='store_true', help='Disable 2D bicycle trajectory plots')
    parser.add_argument('--enable-3d-plots', action='store_true', help='Generate 3D detection plots')
    parser.add_argument('--disable-3d-plots', action='store_true', help='Disable 3D detection plots')
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
        config_kwargs['enable_2d_plots'] = True
    elif args.disable_2d_plots:
        config_kwargs['enable_2d_plots'] = False
        
    if args.enable_3d_plots:
        config_kwargs['enable_3d_plots'] = True
    elif args.disable_3d_plots:
        config_kwargs['enable_3d_plots'] = False
        
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
