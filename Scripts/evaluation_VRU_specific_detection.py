#!/usr/bin/env python3
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
from pathlib import Path
import argparse
from datetime import datetime
import json
import re

# =============================
# CONFIGURATION CONSTANTS
# =============================

# 1. SCENARIO CONFIGURATION
SCENARIO_OUTPUT_PATH = "outputs/small_example_FCO100%_FBO0%"  # Path to scenario output folder (set to None to use manual configuration)

# 2. TRAJECTORY ANALYSIS SETTINGS  
MIN_SEGMENT_LENGTH = 3      # Minimum segment length for bicycle trajectory analysis (data points)
MAX_GAP_BRIDGE = 10         # Maximum number of undetected frames to bridge between detected segments
STEP_LENGTH = 0.1           # Simulation step length in seconds (fallback value)

# 3. PLOT SETTINGS
DPI = 300                   # Resolution for saved plots
FIGURE_SIZE = (12, 8)       # Figure size in inches

# 4. VRU VEHICLE TYPES
VRU_VEHICLE_TYPES = ["bicycle", "DEFAULT_BIKETYPE", "floating_bike_observer"]

# =============================
# OPTIONAL MANUAL CONFIGURATION (only needed if SCENARIO_OUTPUT_PATH = None)
# =============================

# Manual configuration (used only if SCENARIO_OUTPUT_PATH is None)
MANUAL_SCENARIO_PATH = "outputs/example_scenario"
MANUAL_FILE_TAG = "example"
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
        
        return {
            'scenario_path': str(scenario_path),
            'file_tag': file_tag,
            'fco_share': fco_share,
            'fbo_share': fbo_share,
            'step_length': step_length,
            'output_dir': str(scenario_path / 'out_2d_individual_trajectories'),
            'min_segment_length': MIN_SEGMENT_LENGTH,
            'max_gap_bridge': MAX_GAP_BRIDGE,
            'dpi': DPI,
            'figure_size': FIGURE_SIZE
        }
    
    def _get_manual_configuration(self, **kwargs):
        """Get manual configuration with overrides."""
        scenario_dir = kwargs.get('scenario_path', MANUAL_SCENARIO_PATH)
        
        return {
            'scenario_path': scenario_dir,
            'file_tag': kwargs.get('file_tag', MANUAL_FILE_TAG),
            'fco_share': kwargs.get('fco_share', MANUAL_FCO_SHARE),
            'fbo_share': kwargs.get('fbo_share', MANUAL_FBO_SHARE),
            'step_length': kwargs.get('step_length', MANUAL_STEP_LENGTH),
            'output_dir': kwargs.get('output_dir', f"{scenario_dir}/out_2d_individual_trajectories"),
            'min_segment_length': kwargs.get('min_segment_length', MIN_SEGMENT_LENGTH),
            'max_gap_bridge': kwargs.get('max_gap_bridge', MAX_GAP_BRIDGE),
            'dpi': kwargs.get('dpi', DPI),
            'figure_size': kwargs.get('figure_size', FIGURE_SIZE)
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
        """Create output directories if they don't exist."""
        os.makedirs(self.config['output_dir'], exist_ok=True)
        print(f"✓ Output directory: {self.config['output_dir']}")
    
    def load_trajectory_data(self):
        """Load bicycle trajectory data from CSV log file."""
        trajectory_file = Path(self.config['scenario_path']) / 'out_logging' / f'log_bicycle_trajectories_{Path(self.config["scenario_path"]).name}.csv'
        
        if not trajectory_file.exists():
            raise FileNotFoundError(f"Bicycle trajectory log file not found: {trajectory_file}")
        
        print(f"Loading bicycle trajectory data: {trajectory_file}")
        
        # Read CSV, skipping comment lines
        df = pd.read_csv(trajectory_file, comment='#')
        
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
    
    def load_traffic_light_data(self):
        """Load traffic light data from CSV log file."""
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
                bicycle_data['next_tl_distance'].notna() &
                bicycle_data['next_tl_state'].notna() &
                bicycle_data['next_tl_index'].notna()
            ]
            
            if len(tl_rows) > 0:
                # Original method: traffic light data is embedded in bicycle trajectory
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
                        dominant_state = max(signal_counts, key=signal_counts.get)
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
            plt.Line2D([0], [0], color='black', lw=2, label='bicycle undetected'),
            plt.Line2D([0], [0], color='darkturquoise', lw=2, label='bicycle detected'),
        ]
        
        # Add traffic light legend items if any were plotted
        if tl_info:
            handles.extend([
                plt.Line2D([0], [0], color='red', linestyle='--', alpha=0.7, label='Red TL'),
                plt.Line2D([0], [0], color='orange', linestyle='--', alpha=0.7, label='Yellow TL'),
                plt.Line2D([0], [0], color='green', linestyle='--', alpha=0.7, label='Green TL')
            ])
            
        ax.legend(handles=handles, loc='lower right', bbox_to_anchor=(0.99, 0.01))
        
        # Set labels and grid (swap axis labels)
        ax.set_xlabel('Time [s]')
        ax.set_ylabel('Space [m]')
        ax.grid(True)
        
        # Save plot
        output_filename = f'{bicycle_id}_space_time_diagram_{self.config["file_tag"]}_FCO{self.config["fco_share"]}%_FBO{self.config["fbo_share"]}%.png'
        output_path = os.path.join(self.config['output_dir'], output_filename)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.config['dpi'], bbox_inches='tight')
        plt.close(fig)
        
        print(f"  ✓ Saved: {output_filename}")
    
    def analyze_vru_trajectories(self):
        """Main method to perform VRU trajectory analysis."""
        print("=== VRU-Specific Detection Analysis ===")
        print("Configuration:")
        print(f"  - Scenario path: {self.config['scenario_path']}")
        print(f"  - File tag: {self.config['file_tag']}")
        print(f"  - FCO/FBO penetration: {self.config['fco_share']}%/{self.config['fbo_share']}%")
        print(f"  - Output directory: {self.config['output_dir']}")
        print(f"  - Step length: {self.config['step_length']}s")
        print(f"  - Min segment length: {self.config['min_segment_length']} points")
        print(f"  - Max gap bridge: {self.config['max_gap_bridge']} points")
        
        try:
            # Load data
            trajectory_df = self.load_trajectory_data()
            detection_df = self.load_detection_data()
            traffic_light_df = self.load_traffic_light_data()
            
            # Process trajectories
            self.process_bicycle_trajectories(trajectory_df, detection_df, traffic_light_df)
            
            print("\n=== VRU-Specific Detection Analysis completed successfully! ===")
            
        except Exception as e:
            print(f"\nAnalysis failed: {e}")
            raise
    
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
    
    # Scenario parameters
    parser.add_argument('--scenario-path', help='Path to scenario output directory')
    parser.add_argument('--file-tag', help='File tag for output naming')
    parser.add_argument('--fco-share', type=int, help='FCO penetration percentage')
    parser.add_argument('--fbo-share', type=int, help='FBO penetration percentage')
    
    # Analysis parameters
    parser.add_argument('--step-length', type=float, help='Simulation step length in seconds')
    parser.add_argument('--min-segment-length', type=int, help='Minimum segment length for trajectory analysis')
    parser.add_argument('--max-gap-bridge', type=int, help='Maximum gap to bridge in detection timeline')
    
    # Optional parameters
    parser.add_argument('--output-dir', default=None, help='Output directory for trajectory plots')
    parser.add_argument('--save-config', help='Save configuration to JSON file')
    
    args = parser.parse_args()
    
    # Build configuration overrides
    config_kwargs = {}
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
