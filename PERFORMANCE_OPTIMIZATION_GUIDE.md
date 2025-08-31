# FTO-Sim Performance Optimization Guide

This guide explains how to enable and use the multi-threading and GPU acceleration features implemented in FTO-Sim for improved ray tracing performance.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Multi-threading Optimization](#multi-threading-optimization)
3. [GPU Acceleration](#gpu-acceleration)
4. [Installation Requirements](#installation-requirements)
5. [Configuration Options](#configuration-options)
6. [Performance Monitoring](#performance-monitoring)
7. [Troubleshooting](#troubleshooting)
8. [Performance Benchmarks](#performance-benchmarks)

## Quick Start

To enable basic performance optimizations without additional dependencies:

1. **Enable Multi-threading (Default)**:
   ```python
   # In main.py configuration section:
   max_worker_threads = None  # Auto-detect CPU cores (recommended)
   ```

2. **Run the simulation** - multi-threading is enabled by default and will automatically use your CPU cores efficiently.

## Multi-threading Optimization

### How It Works

The optimized ray tracing system processes rays across multiple CPU cores:

- **Static Object Caching**: Buildings, trees, and barriers are cached to avoid reprocessing
- **Parallel Ray Processing**: Each ray is processed independently across CPU threads
- **Optimized Intersection Detection**: Reduced overhead and improved data structures
- **Performance Monitoring**: Built-in timing and statistics collection

### Configuration

```python
# Performance Optimization Settings in main.py:
max_worker_threads = None        # None = auto-detect, or set specific number (e.g., 8)
use_gpu_acceleration = False     # Enable GPU acceleration if CUDA/CuPy available
```

### Automatic Thread Detection

The system automatically detects your CPU capabilities:
- Uses `min(cpu_count, 16)` threads to avoid overhead
- Adapts batch sizes based on workload
- Provides feedback on thread utilization

## GPU Acceleration

### Prerequisites

GPU acceleration requires CUDA-compatible hardware and software:

- **NVIDIA GPU** with CUDA support
- **CUDA Toolkit** (version 11.x or 12.x)
- **CuPy library** (automatically handles GPU array operations)

### Installation

#### Option 1: CUDA 11.x
```bash
# Activate your virtual environment
pip install cupy-cuda11x>=12.0.0
```

#### Option 2: CUDA 12.x
```bash
# Activate your virtual environment
pip install cupy-cuda12x>=12.0.0
```

#### Optional: Numba for JIT Compilation
```bash
pip install numba>=0.58.0
```

### Enable GPU Acceleration

```python
# In main.py configuration section:
use_gpu_acceleration = True  # Enable GPU processing

# The system will automatically:
# 1. Check for CUDA availability
# 2. Fall back to CPU if GPU unavailable
# 3. Provide status messages
```

### GPU Processing Features

- **Automatic Fallback**: If GPU processing fails, automatically switches to optimized CPU processing
- **Memory Management**: Efficient GPU memory allocation and cleanup
- **Batch Processing**: Optimized for GPU parallel processing architecture

## Installation Requirements

### Base Requirements (Already in requirements.txt)
All basic dependencies are already included in the existing `requirements.txt`.

### Optional Performance Dependencies

Uncomment these lines in `requirements.txt` and run `pip install -r requirements.txt`:

```text
# Performance optimization dependencies (optional)
# Uncomment and install these for enhanced performance:
numba>=0.58.0          # JIT compilation for faster numerical operations
cupy-cuda11x>=12.0.0   # GPU acceleration (CUDA 11.x)
# cupy-cuda12x>=12.0.0   # GPU acceleration (CUDA 12.x) - choose based on your CUDA version
```

### Manual Installation
```bash
# For enhanced CPU performance
pip install numba>=0.58.0

# For GPU acceleration (choose one based on your CUDA version)
pip install cupy-cuda11x>=12.0.0  # For CUDA 11.x
# OR
pip install cupy-cuda12x>=12.0.0  # For CUDA 12.x
```

## Configuration Options

### Basic Settings

```python
# Performance Optimization Settings:
use_gpu_acceleration = False     # GPU: Enable/disable GPU acceleration
max_worker_threads = None        # Multi-threading: Auto-detect CPU cores (recommended)
```

### Advanced Tuning

For different scenarios, you can optimize these settings:

**High Observer Count (>10 observers)**:
```python
max_worker_threads = 16          # Multi-threading: Use more CPU threads
use_gpu_acceleration = True      # GPU: Enable if available
```

**Low Observer Count (<5 observers)**:
```python
max_worker_threads = 4           # Multi-threading: Fewer threads for less overhead
use_gpu_acceleration = False     # GPU: CPU may be faster for small workloads
```

**High Ray Count (>360 rays per observer)**:
```python
max_worker_threads = 12          # Multi-threading: Balance threads and memory
use_gpu_acceleration = True      # GPU: Recommended for high ray counts
```

## Performance Monitoring

The system automatically tracks performance metrics:

### Built-in Performance Profiler

```python
# Automatic timing of operations:
- Ray tracing total time
- GPU/CPU processing time
- Static object caching
- Intersection detection

# Frame statistics:
- Average rays per frame
- Average objects per frame
- Intersection rates
- Processing efficiency
```

### Performance Summary

At the end of each simulation, you'll see a detailed performance report:

```
==================================================
RAY TRACING PERFORMANCE SUMMARY
==================================================
ray_tracing_total:
  Average: 45.23ms
  Total: 12.45s
  Count: 275

cpu_ray_processing:
  Average: 42.15ms
  Total: 11.89s
  Count: 275

Frame Statistics (over 275 frames):
  Average rays per frame: 360.0
  Average objects per frame: 87.3
  Average intersections per frame: 156.2
  Intersection rate: 43.4%
```

### Benchmarking Tool

Use the included benchmark tool:

```python
# Run from Scripts directory:
python performance_optimizer.py
```

This will test different configurations and suggest optimal settings for your hardware.

## Troubleshooting

### Common Issues and Solutions

#### 1. "CUDA/CuPy not available"
**Solution**: Install CuPy matching your CUDA version:
```bash
# Check CUDA version
nvcc --version

# Install appropriate CuPy version
pip install cupy-cuda11x  # For CUDA 11.x
pip install cupy-cuda12x  # For CUDA 12.x
```

#### 2. "Performance optimizer not available"
**Solution**: The `performance_optimizer.py` file should be in your Scripts directory. If missing, the system will use basic optimization automatically.

#### 3. Low Performance Improvement
**Possible causes and solutions**:
- **Low observer count**: Multi-threading overhead may exceed benefits with <3 observers
- **Small ray count**: Consider increasing `numberOfRays` if detection quality allows
- **Hardware limitations**: Check CPU usage and memory availability

#### 4. GPU Processing Errors
**Solution**: The system automatically falls back to CPU processing. Check:
```bash
# Verify GPU availability
python -c "import cupy; print('GPU available:', cupy.cuda.is_available())"
```

### Debugging Performance

Enable detailed performance tracking:

```python
# Add to main.py configuration:
import logging
logging.basicConfig(level=logging.INFO)  # Shows thread activity
```

## Performance Benchmarks

### Expected Performance Improvements

| Configuration | Baseline | Multi-threading | Multi-threading + GPU |
|---------------|----------|-----------------|------------------------|
| 5 observers, 360 rays | 100% | 60-70% | 40-50% |
| 10 observers, 360 rays | 100% | 40-50% | 25-35% |
| 20 observers, 720 rays | 100% | 30-40% | 20-30% |

*Times shown as percentage of original processing time (lower is better)*

### Hardware Recommendations

**Optimal CPU Configuration**:
- 8+ CPU cores for best multi-threading performance
- 16GB+ RAM for large simulations
- SSD storage for faster data I/O

**GPU Requirements**:
- NVIDIA GPU with 4GB+ VRAM
- CUDA Compute Capability 6.0+
- PCIe 3.0 x16 slot for optimal data transfer

### Optimization Suggestions Tool

The system provides automatic optimization suggestions:

```python
from performance_optimizer import suggest_optimal_settings

suggestions = suggest_optimal_settings(
    num_observers=10,
    rays_per_observer=360,
    num_objects=50
)
print(suggestions)
```

## Advanced Usage

### Custom Performance Profiler

```python
from performance_optimizer import PerformanceProfiler

# Create custom profiler
profiler = PerformanceProfiler()

# Time custom operations
profiler.start_timer("custom_operation")
# ... your code here ...
profiler.end_timer()

# Print results
profiler.print_summary()
```

### Batch Processing Optimization

```python
# For very large simulations, customize batch processing:
batch_size = max(1, total_rays // (max_worker_threads * 4))
```

## Best Practices

1. **Start with Default Settings**: The auto-detection works well for most scenarios
2. **Monitor Performance**: Use the built-in profiler to identify bottlenecks
3. **Scale Gradually**: Test with small simulations before running large scenarios
4. **Hardware-Specific Tuning**: Use the benchmark tool to find optimal settings for your system
5. **GPU Memory Management**: For very large simulations, monitor GPU memory usage

## Support and Updates

For issues or feature requests related to performance optimization:
1. Check the troubleshooting section above
2. Run the benchmark tool to verify your configuration
3. Review the performance summary output for insights
4. Consider hardware upgrades for consistently high-demand simulations

The performance optimization features are designed to work automatically while providing detailed feedback and fallback options to ensure your simulations always complete successfully.
