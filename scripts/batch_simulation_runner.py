#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch Simulation Runner for FTO-Sim

This script runs multiple simulations with different FCO/FBO penetration rates.
It modifies the configuration in main.py and executes it for each scenario.

Usage:
    python Scripts/batch_simulation_runner.py
"""

import os
import sys
import subprocess
import time
from pathlib import Path
import re

# =============================================================================
# BATCH SIMULATION CONFIGURATION
# =============================================================================

# SUMO Random Seed Configuration
SUMO_SEED = 672  # Set different seeds for parallel batch runs (e.g., 153, 154, 155, etc.)

# Define penetration rate scenarios to simulate
# Each tuple is (FCO_share, FBO_share)
PENETRATION_SCENARIOS = [
    (0.1, 0.0),   # 10% FCO
    (0.2, 0.0),   # 20% FCO
    (0.3, 0.0),   # 30% FCO
    (0.4, 0.0),   # 40% FCO
    (0.5, 0.0),   # 50% FCO
    (0.6, 0.0),   # 60% FCO
    (0.7, 0.0),   # 70% FCO
    (0.8, 0.0),   # 80% FCO
    (0.9, 0.0),   # 90% FCO
    (1.0, 0.0),   # 100% FCO
]

# Base configuration file tag (will be appended with seed and FCO/FBO rates)
BASE_FILE_TAG = "TR-A_status-quo"

# SUMO configuration file to use for all simulations
base_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(base_dir)
SUMO_CONFIG_PATH = os.path.join(parent_dir, 'simulation_examples', 'Intersection-Redesign_Ilic-TR-PartA-2026', 'high_demand.sumocfg')

# Performance optimization level for all simulations
PERFORMANCE_LEVEL = "cpu"  # Options: "none", "cpu", "gpu"

# Other simulation parameters (optional - set to None to use main.py defaults)
NUMBER_OF_RAYS = None      # None = use main.py default
RAY_RADIUS = None          # None = use main.py default
GRID_SIZE = None           # None = use main.py default
SINGLE_SENSOR_ACCURACY = None  # None = use main.py default (70)

# Visualization settings (usually disabled for batch runs)
USE_LIVE_VISUALIZATION = False
SAVE_ANIMATION = False

# =============================================================================
# SCRIPT IMPLEMENTATION
# =============================================================================

def modify_main_py(fco_share, fbo_share, file_tag, seed):
    """
    Modifies main.py configuration for the current scenario.
    
    Args:
        fco_share: FCO penetration rate (0.0-1.0)
        fbo_share: FBO penetration rate (0.0-1.0)
        file_tag: Unique identifier for this simulation
        seed: SUMO random seed value
    """
    main_py_path = Path(__file__).parent / "main.py"
    
    # Read current main.py content
    with open(main_py_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Modify configuration lines
    modified_lines = []
    in_load_sumo_function = False
    
    for i, line in enumerate(lines):
        # Track when we enter the load_sumo_simulation function
        if line.strip().startswith("def load_sumo_simulation():"):
            in_load_sumo_function = True
            modified_lines.append(line)
            continue
        
        # Track when we exit the function (next def or class definition)
        if in_load_sumo_function and line.strip() and not line.strip().startswith('#') and not line.strip().startswith('"""'):
            if line.strip().startswith('def ') or line.strip().startswith('class '):
                in_load_sumo_function = False
        
        # Update seed in load_sumo_simulation function
        if in_load_sumo_function and 'sumoCmd = [' in line and '--seed' in line:
            # Find the seed value and replace it
            modified_line = re.sub(r'"--seed",\s*"(\d+)"', f'"--seed", "{seed}"', line)
            modified_lines.append(modified_line)
            continue
        
        # Update file_tag
        if line.strip().startswith("file_tag = "):
            modified_lines.append(f"file_tag = '{file_tag}'\n")
        
        # Update FCO_share
        elif line.strip().startswith("FCO_share = "):
            modified_lines.append(f"FCO_share = {fco_share}\n")
        
        # Update FBO_share
        elif line.strip().startswith("FBO_share = "):
            modified_lines.append(f"FBO_share = {fbo_share}\n")
        
        # Update SUMO config path
        elif line.strip().startswith("sumo_config_path = os.path.join"):
            modified_lines.append(f"sumo_config_path = os.path.join(parent_dir, r'{SUMO_CONFIG_PATH}')\n")
        
        # Update performance level
        elif line.strip().startswith("performance_optimization_level = "):
            modified_lines.append(f'performance_optimization_level = "{PERFORMANCE_LEVEL}"\n')
        
        # Update visualization settings
        elif line.strip().startswith("useLiveVisualization = "):
            modified_lines.append(f"useLiveVisualization = {USE_LIVE_VISUALIZATION}\n")
        elif line.strip().startswith("saveAnimation = "):
            modified_lines.append(f"saveAnimation = {SAVE_ANIMATION}\n")
        
        # Update optional parameters if specified
        elif NUMBER_OF_RAYS is not None and line.strip().startswith("numberOfRays = "):
            modified_lines.append(f"numberOfRays = {NUMBER_OF_RAYS}\n")
        elif RAY_RADIUS is not None and line.strip().startswith("radius = "):
            modified_lines.append(f"radius = {RAY_RADIUS}\n")
        elif GRID_SIZE is not None and line.strip().startswith("grid_size = "):
            modified_lines.append(f"grid_size = {GRID_SIZE}\n")
        elif SINGLE_SENSOR_ACCURACY is not None and line.strip().startswith("single_sensor_accuracy = "):
            modified_lines.append(f"single_sensor_accuracy = {SINGLE_SENSOR_ACCURACY}\n")
        
        else:
            modified_lines.append(line)
    
    # Write modified content back to main.py
    with open(main_py_path, 'w', encoding='utf-8') as f:
        f.writelines(modified_lines)
    
    print(f"  ✓ Modified main.py for scenario: {file_tag} (seed: {seed})")

def run_simulation():
    """
    Executes main.py as a subprocess.
    
    Returns:
        int: Return code from subprocess (0 = success)
    """
    main_py_path = Path(__file__).parent / "main.py"
    python_executable = sys.executable
    
    # Run main.py and capture output
    try:
        result = subprocess.run(
            [python_executable, str(main_py_path)],
            check=True,
            capture_output=False,  # Show output in real-time
            text=True
        )
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"  ✗ Simulation failed with error code {e.returncode}")
        return e.returncode

def main():
    """
    Main function to run batch simulations.
    """
    print("=" * 80)
    print("FTO-Sim Batch Simulation Runner")
    print("=" * 80)
    print(f"\nSUMO Random Seed: {SUMO_SEED}")
    print(f"Total scenarios to simulate: {len(PENETRATION_SCENARIOS)}")
    print(f"Base file tag: {BASE_FILE_TAG}")
    print(f"Performance level: {PERFORMANCE_LEVEL}")
    print(f"SUMO config: {SUMO_CONFIG_PATH}")
    print("\n" + "=" * 80)
    
    # Track simulation statistics
    total_scenarios = len(PENETRATION_SCENARIOS)
    successful_simulations = 0
    failed_simulations = 0
    start_time = time.time()
    
    # Run each scenario
    for idx, (fco, fbo) in enumerate(PENETRATION_SCENARIOS, 1):
        scenario_start = time.time()
        
        print(f"\n{'=' * 80}")
        print(f"Scenario {idx}/{total_scenarios}: FCO={fco*100:.0f}%, FBO={fbo*100:.0f}% (Seed: {SUMO_SEED})")
        print(f"{'=' * 80}")
        
        # Create unique file tag for this scenario including seed
        # Note: main.py's setup_scenario_output_directory() will append _FCO{X}%_FBO{Y}%
        file_tag = f"{BASE_FILE_TAG}_seed{SUMO_SEED}"
        
        # Modify main.py configuration
        try:
            modify_main_py(fco, fbo, file_tag, SUMO_SEED)
        except Exception as e:
            print(f"  ✗ Failed to modify configuration: {e}")
            failed_simulations += 1
            continue
        
        # Run simulation
        print(f"\n  ▶ Starting simulation...")
        return_code = run_simulation()
        
        scenario_duration = time.time() - scenario_start
        
        if return_code == 0:
            print(f"\n  ✓ Scenario completed successfully in {scenario_duration/60:.1f} minutes")
            successful_simulations += 1
        else:
            print(f"\n  ✗ Scenario failed after {scenario_duration/60:.1f} minutes")
            failed_simulations += 1
    
    # Print final summary
    total_duration = time.time() - start_time
    
    print("\n" + "=" * 80)
    print("BATCH SIMULATION SUMMARY")
    print("=" * 80)
    print(f"SUMO Seed: {SUMO_SEED}")
    print(f"Total scenarios: {total_scenarios}")
    print(f"Successful: {successful_simulations}")
    print(f"Failed: {failed_simulations}")
    print(f"Total time: {total_duration/60:.1f} minutes ({total_duration/3600:.2f} hours)")
    print(f"Average time per scenario: {total_duration/total_scenarios/60:.1f} minutes")
    print("=" * 80)
    
    # Return exit code
    sys.exit(0 if failed_simulations == 0 else 1)

if __name__ == "__main__":
    main()
