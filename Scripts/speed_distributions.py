import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
import re
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

class SpeedDistributionAnalyzer:
    def __init__(self, logging_dir: str):
        """
        Initialize the speed distribution analyzer.
        
        Args:
            logging_dir: Path to the logging directory containing simulation results
        """
        self.logging_dir = Path(logging_dir)
        
    def _extract_scenario_info(self, filename: str) -> Dict[str, str]:
        """
        Extract scenario information from filename.
        
        Args:
            filename: Name of the log file
            
        Returns:
            Dictionary with scenario information
        """
        # Extract speed limit from filename (30, 50, etc.)
        speed_match = re.search(r'small_(\d+)', filename)
        speed_limit = speed_match.group(1) if speed_match else "unknown"
        
        # Extract FCO percentage
        fco_match = re.search(r'FCO(\d+)', filename)
        fco_percent = fco_match.group(1) if fco_match else "unknown"
        
        # Extract seed
        seed_match = re.search(r'seed(\d+)', filename)
        seed = seed_match.group(1) if seed_match else "unknown"
        
        return {
            'speed_limit': speed_limit,
            'fco_percent': fco_percent,
            'seed': seed,
            'filename': filename
        }
    
    def _load_csv_data(self, file_path: str) -> pd.DataFrame:
        """
        Load CSV data with error handling.
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            DataFrame with the data
        """
        try:
            df = pd.read_csv(file_path, comment='#')
        except Exception as e:
            # Try with python engine if the default engine fails
            try:
                df = pd.read_csv(file_path, comment='#', engine='python')
            except Exception as e2:
                print(f"    Failed to read {file_path}: {e2}")
                return pd.DataFrame()
        return df
    
    def get_speed_distributions(self, speed_limits: List[str] = ['30', '50']) -> Dict[str, List[float]]:
        """
        Get speed distributions for different scenarios.
        
        Args:
            speed_limits: List of speed limits to analyze
            
        Returns:
            Dictionary with speed distributions for each scenario
        """
        speed_distributions = {}
        
        # Find all bicycle trajectory files
        bicycle_files = glob.glob(str(self.logging_dir / "**/log_bicycle_trajectories_*.csv"), recursive=True)
        
        # Filter files by speed limits
        filtered_files = []
        for bicycle_file in bicycle_files:
            filename = Path(bicycle_file).name
            scenario_info = self._extract_scenario_info(filename)
            if scenario_info['speed_limit'] in speed_limits:
                filtered_files.append(bicycle_file)
        
        total_files = len(filtered_files)
        print(f"Found {total_files} files to analyze")
        
        for i, bicycle_file in enumerate(filtered_files, 1):
            filename = Path(bicycle_file).name
            scenario_info = self._extract_scenario_info(filename)
            
            # Only process files with the specified speed limits
            if scenario_info['speed_limit'] not in speed_limits:
                continue
            
            print(f"Processing {filename}... ({i}/{total_files})")
            
            try:
                # Load bicycle trajectory data
                df_bicycle = self._load_csv_data(bicycle_file)
                if df_bicycle.empty:
                    continue
                
                # Find corresponding detections file
                detections_file = bicycle_file.replace('log_bicycle_trajectories_', 'log_detections_')
                if not Path(detections_file).exists():
                    print(f"  No corresponding detections file found for {filename}")
                    continue
                
                # Load detections data
                df_detections = self._load_csv_data(detections_file)
                if df_detections.empty:
                    continue
                
                # Find timesteps where bicycles are in important area
                in_test_area_timesteps = df_bicycle[df_bicycle['in_test_area'] == 1]['time_step'].unique()
                
                if len(in_test_area_timesteps) == 0:
                    print(f"  No bicycles found in important area for {filename}")
                    continue
                
                print(f"  Found {len(in_test_area_timesteps)} timesteps with bicycles in important area")
                
                # Get observer speeds for those timesteps
                observer_speeds = []
                for timestep in in_test_area_timesteps:
                    # Get detections for this timestep
                    timestep_detections = df_detections[df_detections['time_step'] == timestep]
                    
                    if len(timestep_detections) > 0:
                        # Get unique observer speeds for this timestep (avoid duplicates)
                        unique_speeds = timestep_detections['observer_speed'].unique()
                        observer_speeds.extend(unique_speeds.tolist())
                
                if len(observer_speeds) > 0:
                    # Create scenario key
                    scenario_key = f"{scenario_info['speed_limit']}kmh_FCO{scenario_info['fco_percent']}%_seed{scenario_info['seed']}"
                    
                    if scenario_key not in speed_distributions:
                        speed_distributions[scenario_key] = []
                    
                    speed_distributions[scenario_key].extend(observer_speeds)
                    print(f"  Added {len(observer_speeds)} observer speed measurements for {scenario_key}")
                    
                    # Add debugging information for first few files
                    if i <= 3:  # Only for first 3 files to avoid spam
                        print(f"    Speed range: {min(observer_speeds):.2f} - {max(observer_speeds):.2f} m/s")
                        print(f"    Sample speeds: {observer_speeds[:5]}")
                        
                        # Check if we're getting bicycle speeds instead of observer speeds
                        sample_detections = df_detections[df_detections['time_step'] == in_test_area_timesteps[0]].head(3)
                        if not sample_detections.empty:
                            print(f"    Sample detection row - observer_speed: {sample_detections['observer_speed'].iloc[0]:.2f} m/s, bicycle_speed: {sample_detections['bicycle_speed'].iloc[0]:.2f} m/s")
                else:
                    print(f"  No observer speeds found for timesteps with bicycles in important area")
                    
            except Exception as e:
                print(f"  Error processing {filename}: {e}")
        
        print(f"\nCompleted analysis of {total_files} files")
        return speed_distributions
    
    def plot_speed_distributions(self, speed_distributions: Dict[str, List[float]], 
                               output_file: str = "speed_distributions.png"):
        """
        Create speed distribution plots.
        
        Args:
            speed_distributions: Dictionary with speed distributions
            output_file: Output file path for the plot
        """
        # Group by speed limit
        speed_30_data = []
        speed_50_data = []
        
        for scenario, speeds in speed_distributions.items():
            if '30kmh' in scenario:
                speed_30_data.extend(speeds)
            elif '50kmh' in scenario:
                speed_50_data.extend(speeds)
        
        # Keep speeds in m/s (no conversion to km/h)
        speed_30_ms = speed_30_data
        speed_50_ms = speed_50_data
        
        # Create the plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Kernel density estimation
        kde_30 = None
        kde_50 = None
        if speed_30_ms:
            kde_30 = sns.kdeplot(speed_30_ms, ax=ax, label='30 km/h scenario', color='forestgreen', linewidth=2)
        if speed_50_ms:
            kde_50 = sns.kdeplot(speed_50_ms, ax=ax, label='50 km/h scenario', color='darkcyan', linewidth=2)
        
        ax.set_xlabel('Observer Speed (m/s)', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Observer Speed Distribution Comparison\n(when bicycles detected in important area)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Add speed limit lines
        speed_30_limit = 30 / 3.6  # Convert 30 km/h to m/s
        speed_50_limit = 50 / 3.6  # Convert 50 km/h to m/s
        
        ax.axvline(x=speed_30_limit, color='darkseagreen', linestyle='--', alpha=0.3, label=f'30 km/h limit ({speed_30_limit:.1f} m/s)')
        ax.axvline(x=speed_50_limit, color='lightblue', linestyle='--', alpha=0.3, label=f'50 km/h limit ({speed_50_limit:.1f} m/s)')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Plot saved as {output_file}")
        
        # Print summary statistics
        print("\n" + "="*50)
        print("SUMMARY STATISTICS")
        print("="*50)
        print(f"30 km/h scenario speed range: {min(speed_30_ms):.2f} - {max(speed_30_ms):.2f} m/s")
        print(f"50 km/h scenario speed range: {min(speed_50_ms):.2f} - {max(speed_50_ms):.2f} m/s")
        
        # Check for speeds exceeding speed limits
        exceeding_30 = [s for s in speed_30_ms if s > speed_30_limit]
        exceeding_50 = [s for s in speed_50_ms if s > speed_50_limit]
        
        print(f"30 km/h scenario: {len(exceeding_30)} speeds exceed {speed_30_limit:.1f} m/s limit ({len(exceeding_30)/len(speed_30_ms)*100:.1f}%)")
        print(f"50 km/h scenario: {len(exceeding_50)} speeds exceed {speed_50_limit:.1f} m/s limit ({len(exceeding_50)/len(speed_50_ms)*100:.1f}%)")
        
        if exceeding_30:
            print(f"  Max speed in 30 km/h scenario: {max(exceeding_30):.2f} m/s ({max(exceeding_30)*3.6:.1f} km/h)")
        if exceeding_50:
            print(f"  Max speed in 50 km/h scenario: {max(exceeding_50):.2f} m/s ({max(exceeding_50)*3.6:.1f} km/h)")
    
    def create_detailed_analysis(self, speed_distributions: Dict[str, List[float]], 
                               output_file: str = "detailed_speed_analysis.png"):
        """
        Create a detailed analysis with multiple plots.
        
        Args:
            speed_distributions: Dictionary with speed distributions
            output_file: Output file path for the plot
        """
        # Group by scenario type
        scenarios_30 = {k: v for k, v in speed_distributions.items() if '30kmh' in k}
        scenarios_50 = {k: v for k, v in speed_distributions.items() if '50kmh' in k}
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Individual scenario distributions
        ax1 = axes[0, 0]
        for scenario, speeds in scenarios_30.items():
            speeds_ms = speeds  # Keep in m/s
            ax1.hist(speeds_ms, bins=30, alpha=0.6, label=scenario, density=True)
        ax1.set_xlabel('Observer Speed (m/s)')
        ax1.set_ylabel('Density')
        ax1.set_title('30 km/h Scenarios')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[0, 1]
        for scenario, speeds in scenarios_50.items():
            speeds_ms = speeds  # Keep in m/s
            ax2.hist(speeds_ms, bins=30, alpha=0.6, label=scenario, density=True)
        ax2.set_xlabel('Observer Speed (m/s)')
        ax2.set_ylabel('Density')
        ax2.set_title('50 km/h Scenarios')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # 2. Combined distributions with KDE
        ax3 = axes[1, 0]
        all_30_speeds = []
        all_50_speeds = []
        
        for speeds in scenarios_30.values():
            all_30_speeds.extend(speeds)  # Keep in m/s
        for speeds in scenarios_50.values():
            all_50_speeds.extend(speeds)  # Keep in m/s
        
        if all_30_speeds:
            sns.kdeplot(all_30_speeds, ax=ax3, label='30 km/h scenarios', color='blue')
        if all_50_speeds:
            sns.kdeplot(all_50_speeds, ax=ax3, label='50 km/h scenarios', color='red')
        
        ax3.set_xlabel('Observer Speed (m/s)')
        ax3.set_ylabel('Density')
        ax3.set_title('Kernel Density Estimation')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 3. Statistical comparison
        ax4 = axes[1, 1]
        if all_30_speeds and all_50_speeds:
            data_to_plot = [all_30_speeds, all_50_speeds]
            labels = ['30 km/h', '50 km/h']
            bp = ax4.boxplot(data_to_plot, labels=labels, patch_artist=True)
            
            colors = ['lightblue', 'lightcoral']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            
            ax4.set_ylabel('Observer Speed (m/s)')
            ax4.set_title('Statistical Comparison')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Detailed analysis saved as {output_file}")

    def create_statistics_comparison(self, speed_distributions: Dict[str, List[float]], 
                                   output_file: str = "speed_statistics_comparison.png"):
        """
        Create a statistics comparison plot with box plots.
        
        Args:
            speed_distributions: Dictionary with speed distributions
            output_file: Output file path for the plot
        """
        # Group by speed limit
        speed_30_data = []
        speed_50_data = []
        
        for scenario, speeds in speed_distributions.items():
            if '30kmh' in scenario:
                speed_30_data.extend(speeds)
            elif '50kmh' in scenario:
                speed_50_data.extend(speeds)
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Box plot for 30 km/h scenario
        bp1 = ax1.boxplot(speed_30_data, patch_artist=True, 
                         boxprops=dict(facecolor='forestgreen', alpha=0.7),
                         medianprops=dict(color='darkgreen', linewidth=2),
                         flierprops=dict(marker='o', markerfacecolor='forestgreen', alpha=0.5))
        
        # Add mean line
        mean_30 = np.mean(speed_30_data)
        ax1.axhline(y=mean_30, color='darkgreen', linestyle='--', linewidth=2, label=f'Mean: {mean_30:.1f} m/s')
        
        ax1.set_title('30 km/h Scenario Speed Distribution', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Speed (m/s)', fontsize=12)
        ax1.set_xticklabels([f'30 km/h\n(n={len(speed_30_data):,})'])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot for 50 km/h scenario
        bp2 = ax2.boxplot(speed_50_data, patch_artist=True,
                         boxprops=dict(facecolor='darkcyan', alpha=0.7),
                         medianprops=dict(color='darkblue', linewidth=2),
                         flierprops=dict(marker='o', markerfacecolor='darkcyan', alpha=0.5))
        
        # Add mean line
        mean_50 = np.mean(speed_50_data)
        ax2.axhline(y=mean_50, color='darkblue', linestyle='--', linewidth=2, label=f'Mean: {mean_50:.1f} m/s')
        
        ax2.set_title('50 km/h Scenario Speed Distribution', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Speed (m/s)', fontsize=12)
        ax2.set_xticklabels([f'50 km/h\n(n={len(speed_50_data):,})'])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Set same y-axis limits for comparison
        all_speeds = speed_30_data + speed_50_data
        y_min, y_max = min(all_speeds), max(all_speeds)
        ax1.set_ylim(y_min - 0.5, y_max + 0.5)
        ax2.set_ylim(y_min - 0.5, y_max + 0.5)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Statistics comparison saved as {output_file}")

    def plot_relative_speed_distributions(self, speed_distributions: Dict[str, List[float]], 
                                        output_file: str = "relative_speed_distributions.png"):
        """
        Create relative speed distribution plots (speed relative to speed limit).
        
        Args:
            speed_distributions: Dictionary with speed distributions
            output_file: Output file path for the plot
        """
        # Group by speed limit
        speed_30_data = []
        speed_50_data = []
        
        for scenario, speeds in speed_distributions.items():
            if '30kmh' in scenario:
                speed_30_data.extend(speeds)
            elif '50kmh' in scenario:
                speed_50_data.extend(speeds)
        
        # Convert to relative speeds (speed / speed limit)
        speed_30_limit = 30 / 3.6  # Convert 30 km/h to m/s
        speed_50_limit = 50 / 3.6  # Convert 50 km/h to m/s
        
        relative_30 = [s / speed_30_limit for s in speed_30_data]
        relative_50 = [s / speed_50_limit for s in speed_50_data]
        
        # Create the plot
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Kernel density estimation
        if relative_30:
            sns.kdeplot(relative_30, ax=ax, label='30 km/h scenario', color='forestgreen', linewidth=2)
        if relative_50:
            sns.kdeplot(relative_50, ax=ax, label='50 km/h scenario', color='darkcyan', linewidth=2)
        
        ax.set_xlabel('Relative Speed (speed / speed limit)', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title('Relative Observer Speed Distribution Comparison\n(when bicycles detected in important area)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Add speed limit line (relative speed = 1.0)
        ax.axvline(x=1.0, color='black', linestyle='--', alpha=0.7, label='Speed limit (relative = 1.0)')
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Relative speed plot saved as {output_file}")
        
        # Print summary statistics
        print("\n" + "="*50)
        print("RELATIVE SPEED SUMMARY STATISTICS")
        print("="*50)
        stats_text = f"""
        30 km/h scenario (relative to 30 km/h limit):
        - Mean: {np.mean(relative_30):.2f} × speed limit
        - Median: {np.median(relative_30):.2f} × speed limit
        - Count: {len(relative_30)} measurements
        
        50 km/h scenario (relative to 50 km/h limit):
        - Mean: {np.mean(relative_50):.2f} × speed limit
        - Median: {np.median(relative_50):.2f} × speed limit
        - Count: {len(relative_50)} measurements
        """
        print(stats_text)
        
        # Add debugging information
        print("\n" + "="*50)
        print("RELATIVE SPEED DEBUGGING INFORMATION")
        print("="*50)
        print(f"30 km/h scenario relative speed range: {min(relative_30):.2f} - {max(relative_30):.2f} × speed limit")
        print(f"50 km/h scenario relative speed range: {min(relative_50):.2f} - {max(relative_50):.2f} × speed limit")
        
        # Check for speeds exceeding speed limits
        exceeding_30 = [r for r in relative_30 if r > 1.0]
        exceeding_50 = [r for r in relative_50 if r > 1.0]
        
        print(f"30 km/h scenario: {len(exceeding_30)} speeds exceed speed limit ({len(exceeding_30)/len(relative_30)*100:.1f}%)")
        print(f"50 km/h scenario: {len(exceeding_50)} speeds exceed speed limit ({len(exceeding_50)/len(relative_50)*100:.1f}%)")
        
        if exceeding_30:
            print(f"  Max relative speed in 30 km/h scenario: {max(exceeding_30):.2f} × speed limit")
        if exceeding_50:
            print(f"  Max relative speed in 50 km/h scenario: {max(exceeding_50):.2f} × speed limit")

    def create_relative_statistics_comparison(self, speed_distributions: Dict[str, List[float]], 
                                            output_file: str = "relative_speed_statistics_comparison.png"):
        """
        Create a relative statistics comparison plot with box plots.
        
        Args:
            speed_distributions: Dictionary with speed distributions
            output_file: Output file path for the plot
        """
        # Group by speed limit
        speed_30_data = []
        speed_50_data = []
        
        for scenario, speeds in speed_distributions.items():
            if '30kmh' in scenario:
                speed_30_data.extend(speeds)
            elif '50kmh' in scenario:
                speed_50_data.extend(speeds)
        
        # Convert to relative speeds (speed / speed limit)
        speed_30_limit = 30 / 3.6  # Convert 30 km/h to m/s
        speed_50_limit = 50 / 3.6  # Convert 50 km/h to m/s
        
        relative_30 = [s / speed_30_limit for s in speed_30_data]
        relative_50 = [s / speed_50_limit for s in speed_50_data]
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Box plot for 30 km/h scenario
        bp1 = ax1.boxplot(relative_30, patch_artist=True, 
                         boxprops=dict(facecolor='forestgreen', alpha=0.7),
                         medianprops=dict(color='darkgreen', linewidth=2),
                         flierprops=dict(marker='o', markerfacecolor='forestgreen', alpha=0.5))
        
        # Add mean line
        mean_30 = np.mean(relative_30)
        ax1.axhline(y=mean_30, color='darkgreen', linestyle='--', linewidth=2, label=f'Mean: {mean_30:.2f} × speed limit')
        
        # Add speed limit line
        ax1.axhline(y=1.0, color='black', linestyle='-', linewidth=1, alpha=0.5, label='Speed limit (1.0)')
        
        ax1.set_title('30 km/h Scenario Relative Speed Distribution', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Relative Speed (speed / speed limit)', fontsize=12)
        ax1.set_xticklabels([f'30 km/h\n(n={len(relative_30):,})'])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Box plot for 50 km/h scenario
        bp2 = ax2.boxplot(relative_50, patch_artist=True,
                         boxprops=dict(facecolor='darkcyan', alpha=0.7),
                         medianprops=dict(color='darkblue', linewidth=2),
                         flierprops=dict(marker='o', markerfacecolor='darkcyan', alpha=0.5))
        
        # Add mean line
        mean_50 = np.mean(relative_50)
        ax2.axhline(y=mean_50, color='darkblue', linestyle='--', linewidth=2, label=f'Mean: {mean_50:.2f} × speed limit')
        
        # Add speed limit line
        ax2.axhline(y=1.0, color='black', linestyle='-', linewidth=1, alpha=0.5, label='Speed limit (1.0)')
        
        ax2.set_title('50 km/h Scenario Relative Speed Distribution', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Relative Speed (speed / speed limit)', fontsize=12)
        ax2.set_xticklabels([f'50 km/h\n(n={len(relative_50):,})'])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Set same y-axis limits for comparison
        all_relative_speeds = relative_30 + relative_50
        y_min, y_max = min(all_relative_speeds), max(all_relative_speeds)
        ax1.set_ylim(y_min - 0.1, y_max + 0.1)
        ax2.set_ylim(y_min - 0.1, y_max + 0.1)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Relative statistics comparison saved as {output_file}")

def main():
    """
    Main function to run the speed distribution analysis.
    """
    # Initialize the analyzer
    analyzer = SpeedDistributionAnalyzer(logging_dir="out_logging")
    
    # Get speed distributions
    print("Analyzing observer speed distributions...")
    speed_distributions = analyzer.get_speed_distributions(speed_limits=['30', '50'])
    
    if not speed_distributions:
        print("No data found! Please check your file paths and data.")
        return
    
    print(f"\nFound data for {len(speed_distributions)} scenarios:")
    for scenario, speeds in speed_distributions.items():
        print(f"  {scenario}: {len(speeds)} speed measurements")
    
    # Create plots
    print("\nCreating plots...")
    analyzer.plot_speed_distributions(speed_distributions, "observer_speed_distributions.png")
    analyzer.create_statistics_comparison(speed_distributions, "speed_statistics_comparison.png")
    analyzer.plot_relative_speed_distributions(speed_distributions, "relative_speed_distributions.png")
    analyzer.create_relative_statistics_comparison(speed_distributions, "relative_speed_statistics_comparison.png")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
