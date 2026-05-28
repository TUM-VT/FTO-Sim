"""
Aggregate visibility counts across seeds and generate averaged LoV heatmaps.

This script:
1. Groups scenarios by (demand, measure, FCO rate) - excluding seed
2. Averages visibility counts across all seeds for each group
3. Saves averaged visibility count CSVs
4. Generates LoV heatmaps from averaged data
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
import shutil
import subprocess


# ============================================================================
# CONFIGURATION
# ============================================================================

# Source folder with all scenario results
SOURCE_BASE_PATH = r"C:\FTO-Sim\outputs\TR-A_final"

# Destination folder for averaged results
DEST_BASE_PATH = r"C:\FTO-Sim\outputs\TR-A_final_LoV_averaged"

# Which visibility types to process
PROCESS_DISCRETE = True
PROCESS_CONTINUOUS = True


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def parse_folder_name(folder_name: str) -> Optional[Dict[str, str]]:
    """
    Parse a scenario folder name and extract components.
    
    Expected format: TR-A_{demand}_{measure}_{seed}_FCO{fco}%_FBO{fbo}%
    """
    # Pattern without measure (baseline)
    pattern_no_measure = r"TR-A_([^_]+)_(seed\d+)_FCO(\d+)%_FBO(\d+)%"
    # Pattern with measure
    pattern_with_measure = r"TR-A_([^_]+)_([^_]+)_(seed\d+)_FCO(\d+)%_FBO(\d+)%"
    
    match = re.match(pattern_with_measure, folder_name)
    if match:
        demand, measure, seed, fco, fbo = match.groups()
        return {
            'demand': demand,
            'measure': measure,
            'seed': seed,
            'fco_rate': int(fco),
            'fbo_rate': int(fbo)
        }
    
    match = re.match(pattern_no_measure, folder_name)
    if match:
        demand, seed, fco, fbo = match.groups()
        return {
            'demand': demand,
            'measure': 'none',
            'seed': seed,
            'fco_rate': int(fco),
            'fbo_rate': int(fbo)
        }
    
    return None


def get_scenario_key(parsed: Dict) -> Tuple[str, str, int, int]:
    """Get grouping key for scenario (excludes seed)."""
    return (parsed['demand'], parsed['measure'], parsed['fco_rate'], parsed['fbo_rate'])


def get_scenario_name(demand: str, measure: str, fco: int, fbo: int) -> str:
    """Generate scenario folder name without seed."""
    if measure == 'none':
        return f"TR-A_{demand}_FCO{fco}%_FBO{fbo}%"
    else:
        return f"TR-A_{demand}_{measure}_FCO{fco}%_FBO{fbo}%"


def find_all_scenarios(base_path: str) -> Dict[Tuple, List[Tuple[str, Dict]]]:
    """
    Find all scenarios and group them by (demand, measure, fco, fbo).
    
    Returns:
        Dictionary mapping scenario_key -> list of (folder_path, parsed_info)
    """
    base_path = Path(base_path)
    
    if not base_path.exists():
        print(f"Error: Base path does not exist: {base_path}")
        return {}
    
    scenarios_by_key = {}
    
    for folder in base_path.iterdir():
        if not folder.is_dir():
            continue
        
        parsed = parse_folder_name(folder.name)
        if parsed is None:
            continue
        
        # Only process FBO=0 scenarios
        if parsed['fbo_rate'] != 0:
            continue
        
        key = get_scenario_key(parsed)
        
        if key not in scenarios_by_key:
            scenarios_by_key[key] = []
        
        scenarios_by_key[key].append((str(folder), parsed))
    
    return scenarios_by_key


def average_visibility_counts(scenario_folders: List[Tuple[str, Dict]], vis_type: str) -> Optional[pd.DataFrame]:
    """
    Average visibility counts across multiple seed folders.
    
    Args:
        scenario_folders: List of (folder_path, parsed_info) tuples
        vis_type: "discrete" or "continuous"
        
    Returns:
        DataFrame with averaged visibility counts, or None if no data found
    """
    all_data = []
    
    for folder_path, parsed_info in scenario_folders:
        raytracing_folder = Path(folder_path) / "out_raytracing"
        
        if not raytracing_folder.exists():
            continue
        
        # Find visibility count file
        if vis_type == "discrete":
            pattern = "discrete_visibility_counts_*.csv"
            count_col = "discrete_visibility_count"
        else:  # continuous
            pattern = "continuous_visibility_counts_*_SSA*.csv"
            count_col = "continuous_visibility_count"
        
        files = list(raytracing_folder.glob(pattern))
        
        if not files:
            continue
        
        count_file = files[0]
        
        try:
            df = pd.read_csv(count_file, comment='#')
            
            # Verify required columns exist
            if 'x_coord' not in df.columns or 'y_coord' not in df.columns or count_col not in df.columns:
                print(f"  Warning: Missing columns in {count_file.name}")
                continue
            
            all_data.append(df)
            
        except Exception as e:
            print(f"  Error reading {count_file.name}: {e}")
            continue
    
    if not all_data:
        return None
    
    # Merge all dataframes and average counts by coordinate
    # Concatenate all data
    combined = pd.concat(all_data, ignore_index=True)
    
    # Group by coordinates and average the counts
    if vis_type == "discrete":
        count_col = "discrete_visibility_count"
    else:
        count_col = "continuous_visibility_count"
    
    averaged = combined.groupby(['x_coord', 'y_coord']).agg({
        count_col: 'mean'
    }).reset_index()
    
    return averaged


def copy_auxiliary_files(source_folder: str, dest_folder: Path):
    """Copy necessary auxiliary files (summary logs, GeoJSON, etc.) from source to destination."""
    source_path = Path(source_folder)
    
    # Copy out_logging folder if it exists
    source_logging = source_path / "out_logging"
    if source_logging.exists():
        dest_logging = dest_folder / "out_logging"
        dest_logging.mkdir(parents=True, exist_ok=True)
        
        # Copy all log files
        for log_file in source_logging.iterdir():
            if log_file.is_file():
                shutil.copy2(log_file, dest_logging / log_file.name)
    
    # Copy GeoJSON file if it exists in parent directory
    for parent in [source_path.parent, source_path.parent.parent]:
        geojson_files = list(parent.glob("*.geojson"))
        if geojson_files:
            for geojson_file in geojson_files:
                shutil.copy2(geojson_file, dest_folder / geojson_file.name)
            break


def save_averaged_counts(df: pd.DataFrame, dest_folder: Path, scenario_name: str, vis_type: str, source_folders: List[Tuple[str, Dict]]) -> Path:
    """Save averaged visibility counts to CSV file."""
    raytracing_folder = dest_folder / "out_raytracing"
    raytracing_folder.mkdir(parents=True, exist_ok=True)
    
    if vis_type == "discrete":
        filename = f"discrete_visibility_counts_{scenario_name}.csv"
    else:
        # For continuous, extract SSA from first source file
        first_folder = Path(source_folders[0][0]) / "out_raytracing"
        continuous_files = list(first_folder.glob("continuous_visibility_counts_*_SSA*.csv"))
        
        if continuous_files:
            # Extract SSA percentage from filename
            match = re.search(r'SSA(\d+(?:\.\d+)?)%', continuous_files[0].name)
            if match:
                ssa = match.group(1)
                filename = f"continuous_visibility_counts_{scenario_name}_SSA{ssa}%.csv"
            else:
                filename = f"continuous_visibility_counts_{scenario_name}_SSA70%.csv"
        else:
            filename = f"continuous_visibility_counts_{scenario_name}_SSA70%.csv"
    
    output_path = raytracing_folder / filename
    
    # Save with comment header
    with open(output_path, 'w') as f:
        f.write(f"# Averaged visibility counts across {len(source_folders)} seeds\n")
        f.write(f"# Scenario: {scenario_name}\n")
        f.write(f"# Visibility type: {vis_type}\n")
    
    # Append data
    df.to_csv(output_path, mode='a', index=False)
    
    return output_path


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    
    print("=" * 80)
    print("Aggregate LoV Heatmaps - Averaging Across Seeds")
    print("=" * 80)
    print()
    
    print(f"Source: {SOURCE_BASE_PATH}")
    print(f"Destination: {DEST_BASE_PATH}")
    print()
    
    # Step 1: Find all scenarios and group by key
    print("Step 1: Scanning scenarios...")
    scenarios_by_key = find_all_scenarios(SOURCE_BASE_PATH)
    
    print(f"Found {len(scenarios_by_key)} unique scenario configurations")
    total_folders = sum(len(folders) for folders in scenarios_by_key.values())
    print(f"Total seed folders: {total_folders}")
    print()
    
    # Step 2: Process each scenario group
    print("Step 2: Aggregating visibility counts...")
    
    processed_count = 0
    
    for scenario_key, scenario_folders in scenarios_by_key.items():
        demand, measure, fco, fbo = scenario_key
        scenario_name = get_scenario_name(demand, measure, fco, fbo)
        
        print(f"\n{scenario_name} ({len(scenario_folders)} seeds):")
        
        # Create destination folder
        dest_folder = Path(DEST_BASE_PATH) / scenario_name
        dest_folder.mkdir(parents=True, exist_ok=True)
        
        # Copy auxiliary files from first seed folder
        copy_auxiliary_files(scenario_folders[0][0], dest_folder)
        
        # Process discrete visibility counts
        if PROCESS_DISCRETE:
            print(f"  Averaging discrete visibility counts...")
            discrete_df = average_visibility_counts(scenario_folders, "discrete")
            
            if discrete_df is not None:
                output_path = save_averaged_counts(discrete_df, dest_folder, scenario_name, "discrete", scenario_folders)
                print(f"    ✓ Saved: {output_path.name}")
            else:
                print(f"    ✗ No discrete visibility data found")
        
        # Process continuous visibility counts
        if PROCESS_CONTINUOUS:
            print(f"  Averaging continuous visibility counts...")
            continuous_df = average_visibility_counts(scenario_folders, "continuous")
            
            if continuous_df is not None:
                output_path = save_averaged_counts(continuous_df, dest_folder, scenario_name, "continuous", scenario_folders)
                print(f"    ✓ Saved: {output_path.name}")
            else:
                print(f"    ✗ No continuous visibility data found")
        
        processed_count += 1
    
    print()
    print("=" * 80)
    print(f"Processing complete!")
    print(f"Processed {processed_count} scenario configurations")
    print(f"Output saved to: {DEST_BASE_PATH}")
    print()
    print(f"To generate LoV heatmaps, run:")
    print(f"  python scripts/batch_process_spatial_visibility.py")
    print("=" * 80)


if __name__ == "__main__":
    main()