# Performance Configuration Guide for FTO-Sim

## Overview
FTO-Sim offers three different performance optimization levels to ensure compatibility across different systems while maximizing performance when possible. **All configuration is done directly in the main script** - no external configuration files needed.

## Quick Start

Edit the configuration section in `Scripts/main.py`:

```python
# ═══════════════════════════════════════════════════════════════════════════════════
# SIMULATION CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════════

# Performance Optimization Level:
performance_optimization_level = "cpu"  # Options: "none", "cpu", "gpu"

# Observer Penetration Rates:
FCO_share = 1.0  # Floating Car Observers (0.0 to 1.0)
FBO_share = 0.0  # Floating Bike Observers (0.0 to 1.0)

# Ray Tracing Parameters:
numberOfRays = 360  # Number of rays per observer
radius = 30         # Ray radius in meters

# ... (see script for all options)
```

## Performance Optimization Levels

### 1. "none" - Maximum Compatibility Mode
```python
performance_optimization_level = "none"
```
- **Processing**: Single-threaded ray tracing
- **Compatibility**: Works on all systems
- **Performance**: Slowest (baseline)
- **Use case**: Systems with limited resources, troubleshooting, or when maximum compatibility is required

### 2. "cpu" - Balanced Mode (Recommended Default)
```python
performance_optimization_level = "cpu"
```
- **Processing**: Multi-threaded CPU ray tracing
- **Compatibility**: Works on most modern systems
- **Performance**: 8-20% faster than single-threaded
- **Use case**: Standard deployment for most users
- **Thread count**: Auto-detected, capped at 8 threads for stability

### 3. "gpu" - Maximum Performance Mode
```python
performance_optimization_level = "gpu"
```
- **Processing**: Multi-threaded CPU + GPU acceleration
- **Compatibility**: Requires NVIDIA GPU with CUDA support
- **Performance**: 20-50% faster than CPU-only (when properly configured)
- **Use case**: High-performance workstations with NVIDIA GPUs
- **Requirements**: 
  - NVIDIA GPU with CUDA Compute Capability 3.5+
  - CUDA Toolkit 11.0+ or 12.0+
  - CuPy library installed
  - Visual C++ Build Tools (Windows)

## Advanced Configuration

### Custom Thread Count
```python
max_worker_threads = 4  # Override auto-detection
```

### Simulation Parameters
```python
FCO_share = 0.5        # 50% of cars become observers
FBO_share = 0.1        # 10% of bikes become observers
numberOfRays = 180     # Reduce rays for faster processing
radius = 20            # Reduce radius for smaller visibility area
```

### Visualization Options
```python
useLiveVisualization = False  # Disable for faster processing
visualizeRays = False         # Hide rays to improve performance
saveAnimation = True          # Save result as video file
```

## System Requirements by Level

### All Levels
- Python 3.8+
- NumPy, Matplotlib, Shapely, GeoPandas
- SUMO traffic simulator

### CPU Level (Additional)
- Multi-core processor (4+ cores recommended)
- 8GB+ RAM recommended for large simulations

### GPU Level (Additional)
- NVIDIA GPU with CUDA support
- CUDA Toolkit installed
- CuPy library (`pip install cupy-cuda12x` or `cupy-cuda11x`)
- On Windows: Visual Studio Build Tools with C++ support

## Getting Help

Call the built-in help function in your script:
```python
print_configuration_help()
```

## Performance Benchmarks

Based on a simulation with 360 rays, 30m radius, typical urban scenario:

| Level | Processing Time | Relative Performance | System Requirements |
|-------|-----------------|---------------------|-------------------|
| none  | 351s           | 1.0x (baseline)     | Any system        |
| cpu   | 321s           | 1.09x faster        | Multi-core CPU    |
| gpu   | ~245s*         | 1.43x faster        | NVIDIA GPU + CUDA |

*Estimated based on typical GPU acceleration gains

## Troubleshooting

### Performance Level Falls Back
If a higher performance level is selected but requirements aren't met, the system automatically falls back:
- `gpu` → `cpu` if CUDA/CuPy not available
- System prints warning messages explaining the fallback

### Common Issues

1. **GPU mode not working**: Check CUDA installation and CuPy
   ```bash
   python -c "import cupy; print('CuPy available')"
   ```

2. **CPU mode slower than expected**: Check system load and available CPU cores

3. **Memory issues**: Reduce `numberOfRays` or `radius` values

## Recommendations for Different Use Cases

### Research/Academic Use
- Development/testing: `performance_optimization_level = "cpu"`
- Production runs: `performance_optimization_level = "gpu"` (if available)

### Public Distribution
- Default: `performance_optimization_level = "cpu"`
- Include instructions for users to adjust based on their hardware

### CI/CD Pipelines
- Use: `performance_optimization_level = "none"` for maximum compatibility

## Configuration Best Practices

1. **Keep it simple**: Start with default "cpu" level
2. **Document changes**: Add comments when modifying values
3. **Test performance**: Measure actual improvement on your hardware
4. **Consider users**: Choose conservative defaults for shared code
