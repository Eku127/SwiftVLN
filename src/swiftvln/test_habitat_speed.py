#!/usr/bin/env python
"""
Habitat Rendering Speed Diagnostic Tool

This script tests Habitat-sim rendering speed at the low level,
bypassing VLN task registration issues.

Usage:
    conda activate swift-vln-eval
    cd /mnt/data1/home/jiangjiajun/workspace/SwiftVLN/src/swiftvln
    python test_habitat_speed.py
"""

import os
import sys
import time
import numpy as np
import torch

# Suppress verbose logging
os.environ['MAGNUM_LOG'] = 'quiet'
os.environ['HABITAT_SIM_LOG'] = 'quiet'

# Add project paths
current_dir = os.path.dirname(os.path.abspath(__file__))
msswift_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, msswift_root)
sys.path.insert(0, current_dir)


def print_system_info():
    """Print system and CUDA information."""
    print("=" * 60)
    print("System Information")
    print("=" * 60)
    
    # CUDA info
    print(f"\nPyTorch CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f} GB)")
    
    # Environment variables
    print(f"\nCUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    print(f"DISPLAY: {os.environ.get('DISPLAY', 'not set')}")
    
    # Check habitat-sim
    try:
        import habitat_sim
        print(f"\nhabitat-sim Version: {habitat_sim.__version__}")
    except ImportError:
        print("\nhabitat-sim not found!")
    except Exception as e:
        print(f"\nError checking habitat-sim: {e}")


def test_habitat_sim_direct(scene_path: str, num_steps: int = 100):
    """
    Test habitat-sim rendering speed directly without using habitat Env.
    This bypasses VLN task initialization issues.
    """
    print("\n" + "=" * 60)
    print("Direct habitat-sim Rendering Test")
    print("=" * 60)
    
    import habitat_sim
    from habitat_sim import Simulator, SimulatorConfiguration, CameraSensorSpec, SensorType, AgentConfiguration
    
    # Configure simulator
    sim_cfg = SimulatorConfiguration()
    sim_cfg.scene_id = scene_path
    sim_cfg.enable_physics = False
    sim_cfg.gpu_device_id = 0
    
    # Configure RGB camera sensor
    sensor_spec = CameraSensorSpec()
    sensor_spec.uuid = "color_sensor"
    sensor_spec.sensor_type = SensorType.COLOR
    sensor_spec.resolution = [480, 640]  # Height, Width
    sensor_spec.position = [0.0, 1.5, 0.0]  # Eye height
    sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    
    # Configure depth sensor
    depth_spec = CameraSensorSpec()
    depth_spec.uuid = "depth_sensor"
    depth_spec.sensor_type = SensorType.DEPTH
    depth_spec.resolution = [480, 640]
    depth_spec.position = [0.0, 1.5, 0.0]
    depth_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    
    # Configure agent
    agent_cfg = AgentConfiguration()
    agent_cfg.sensor_specifications = [sensor_spec, depth_spec]
    agent_cfg.action_space = {
        "move_forward": habitat_sim.ActionSpec("move_forward", habitat_sim.ActuationSpec(amount=0.25)),
        "turn_left": habitat_sim.ActionSpec("turn_left", habitat_sim.ActuationSpec(amount=15.0)),
        "turn_right": habitat_sim.ActionSpec("turn_right", habitat_sim.ActuationSpec(amount=15.0)),
    }
    
    # Create configuration
    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    
    print(f"\nCreating simulator...")
    print(f"  Scene: {scene_path}")
    print(f"  GPU Device: {sim_cfg.gpu_device_id}")
    print(f"  Resolution: {sensor_spec.resolution}")
    
    create_start = time.time()
    sim = Simulator(cfg)
    create_time = time.time() - create_start
    print(f"  Simulator creation time: {create_time:.2f}s")
    
    # Initialize agent
    agent = sim.initialize_agent(0)
    agent_state = habitat_sim.AgentState()
    
    # Set initial position (from navmesh if available)
    if sim.pathfinder.is_loaded:
        agent_state.position = sim.pathfinder.get_random_navigable_point()
    else:
        agent_state.position = [0.0, 0.0, 0.0]
    agent.set_state(agent_state)
    
    # Get initial observation (warm-up)
    print(f"\nWarming up (first observation)...")
    warmup_start = time.time()
    obs = sim.get_sensor_observations()
    warmup_time = time.time() - warmup_start
    print(f"  First observation time: {warmup_time*1000:.1f}ms")
    print(f"  RGB shape: {obs['color_sensor'].shape}, dtype: {obs['color_sensor'].dtype}")
    print(f"  Depth shape: {obs['depth_sensor'].shape}, dtype: {obs['depth_sensor'].dtype}")
    
    # Test step speed
    print(f"\nTesting step speed ({num_steps} steps)...")
    actions = ["move_forward", "turn_left", "turn_right"]
    
    step_times = []
    obs_times = []
    
    for i in range(num_steps):
        action = actions[i % 3]
        
        # Time the step (agent action)
        step_start = time.time()
        sim.step(action)
        step_time = time.time() - step_start
        step_times.append(step_time)
        
        # Time the observation retrieval (rendering)
        obs_start = time.time()
        obs = sim.get_sensor_observations()
        obs_time = time.time() - obs_start
        obs_times.append(obs_time)
    
    # Statistics
    print("\n" + "-" * 40)
    print("Step (Action Only) Statistics:")
    print(f"  Mean: {np.mean(step_times)*1000:.2f}ms")
    print(f"  Std: {np.std(step_times)*1000:.2f}ms")
    print(f"  Min: {np.min(step_times)*1000:.2f}ms")
    print(f"  Max: {np.max(step_times)*1000:.2f}ms")
    
    print("\nObservation (Rendering) Statistics:")
    print(f"  Mean: {np.mean(obs_times)*1000:.2f}ms")
    print(f"  Std: {np.std(obs_times)*1000:.2f}ms")
    print(f"  Min: {np.min(obs_times)*1000:.2f}ms")
    print(f"  Max: {np.max(obs_times)*1000:.2f}ms")
    
    total_times = [s + o for s, o in zip(step_times, obs_times)]
    print("\nTotal (Step + Observation) Statistics:")
    print(f"  Mean: {np.mean(total_times)*1000:.2f}ms")
    print(f"  Theoretical FPS: {1.0/np.mean(total_times):.1f}")
    
    sim.close()
    return step_times, obs_times


def test_resolution_impact(scene_path: str, num_steps: int = 50):
    """Test rendering speed with different resolutions."""
    print("\n" + "=" * 60)
    print("Resolution Impact Test")
    print("=" * 60)
    
    import habitat_sim
    from habitat_sim import Simulator, SimulatorConfiguration, CameraSensorSpec, SensorType, AgentConfiguration
    
    resolutions = [
        (240, 320),   # Small
        (480, 640),   # Standard VLN
        (720, 1280),  # HD
        (1080, 1920), # Full HD
    ]
    
    results = {}
    
    for height, width in resolutions:
        print(f"\nTesting {width}x{height}...")
        
        sim_cfg = SimulatorConfiguration()
        sim_cfg.scene_id = scene_path
        sim_cfg.enable_physics = False
        sim_cfg.gpu_device_id = 0
        
        sensor_spec = CameraSensorSpec()
        sensor_spec.uuid = "color_sensor"
        sensor_spec.sensor_type = SensorType.COLOR
        sensor_spec.resolution = [height, width]
        sensor_spec.position = [0.0, 1.5, 0.0]
        sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
        
        agent_cfg = AgentConfiguration()
        agent_cfg.sensor_specifications = [sensor_spec]
        agent_cfg.action_space = {
            "move_forward": habitat_sim.ActionSpec("move_forward", habitat_sim.ActuationSpec(amount=0.25)),
            "turn_left": habitat_sim.ActionSpec("turn_left", habitat_sim.ActuationSpec(amount=15.0)),
            "turn_right": habitat_sim.ActionSpec("turn_right", habitat_sim.ActuationSpec(amount=15.0)),
        }
        
        cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
        
        try:
            sim = Simulator(cfg)
            agent = sim.initialize_agent(0)
            agent_state = habitat_sim.AgentState()
            if sim.pathfinder.is_loaded:
                agent_state.position = sim.pathfinder.get_random_navigable_point()
            agent.set_state(agent_state)
            
            # Warm-up
            obs = sim.get_sensor_observations()
            
            # Test
            times = []
            actions = ["move_forward", "turn_left", "turn_right"]
            for i in range(num_steps):
                start = time.time()
                sim.step(actions[i % 3])
                obs = sim.get_sensor_observations()
                times.append(time.time() - start)
            
            avg_ms = np.mean(times) * 1000
            fps = 1.0 / np.mean(times)
            results[(width, height)] = (avg_ms, fps)
            print(f"  Avg step+render: {avg_ms:.1f}ms ({fps:.1f} FPS)")
            
            sim.close()
        except Exception as e:
            print(f"  Error: {e}")
            results[(width, height)] = None
    
    print("\n" + "-" * 40)
    print("Summary (Resolution vs Performance):")
    for (w, h), result in results.items():
        if result:
            ms, fps = result
            print(f"  {w}x{h}: {ms:.1f}ms ({fps:.1f} FPS)")


def test_multiple_sensors(scene_path: str, num_steps: int = 50):
    """Test how multiple sensors affect performance."""
    print("\n" + "=" * 60)
    print("Multiple Sensors Impact Test")
    print("=" * 60)
    
    import habitat_sim
    from habitat_sim import Simulator, SimulatorConfiguration, CameraSensorSpec, SensorType, AgentConfiguration
    
    configurations = [
        ("RGB only", [SensorType.COLOR]),
        ("Depth only", [SensorType.DEPTH]),
        ("RGB + Depth", [SensorType.COLOR, SensorType.DEPTH]),
        ("RGB + Depth + Semantic", [SensorType.COLOR, SensorType.DEPTH, SensorType.SEMANTIC]),
    ]
    
    for name, sensor_types in configurations:
        print(f"\nTesting: {name}")
        
        sim_cfg = SimulatorConfiguration()
        sim_cfg.scene_id = scene_path
        sim_cfg.enable_physics = False
        sim_cfg.gpu_device_id = 0
        
        sensors = []
        for i, sensor_type in enumerate(sensor_types):
            sensor_spec = CameraSensorSpec()
            if sensor_type == SensorType.COLOR:
                sensor_spec.uuid = "color_sensor"
            elif sensor_type == SensorType.DEPTH:
                sensor_spec.uuid = "depth_sensor"
            elif sensor_type == SensorType.SEMANTIC:
                sensor_spec.uuid = "semantic_sensor"
            sensor_spec.sensor_type = sensor_type
            sensor_spec.resolution = [480, 640]
            sensor_spec.position = [0.0, 1.5, 0.0]
            sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
            sensors.append(sensor_spec)
        
        agent_cfg = AgentConfiguration()
        agent_cfg.sensor_specifications = sensors
        agent_cfg.action_space = {
            "move_forward": habitat_sim.ActionSpec("move_forward", habitat_sim.ActuationSpec(amount=0.25)),
            "turn_left": habitat_sim.ActionSpec("turn_left", habitat_sim.ActuationSpec(amount=15.0)),
            "turn_right": habitat_sim.ActionSpec("turn_right", habitat_sim.ActuationSpec(amount=15.0)),
        }
        
        cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
        
        try:
            sim = Simulator(cfg)
            agent = sim.initialize_agent(0)
            agent_state = habitat_sim.AgentState()
            if sim.pathfinder.is_loaded:
                agent_state.position = sim.pathfinder.get_random_navigable_point()
            agent.set_state(agent_state)
            
            # Warm-up
            obs = sim.get_sensor_observations()
            
            # Test
            times = []
            actions = ["move_forward", "turn_left", "turn_right"]
            for i in range(num_steps):
                start = time.time()
                sim.step(actions[i % 3])
                obs = sim.get_sensor_observations()
                times.append(time.time() - start)
            
            avg_ms = np.mean(times) * 1000
            fps = 1.0 / np.mean(times)
            print(f"  Avg: {avg_ms:.1f}ms ({fps:.1f} FPS)")
            
            sim.close()
        except Exception as e:
            print(f"  Error: {e}")


def find_scene_file():
    """Find an available scene file for testing."""
    # Common dataset locations
    dataset_paths = [
        "/mnt/data3/jiangjiajun/dataset/mp3d",
        "/data/datasets/mp3d",
        os.path.expanduser("~/data/mp3d"),
    ]
    
    for base_path in dataset_paths:
        if os.path.exists(base_path):
            # Find first .glb file
            for scene_dir in os.listdir(base_path):
                scene_path = os.path.join(base_path, scene_dir, f"{scene_dir}.glb")
                if os.path.exists(scene_path):
                    return scene_path
    
    return None


def diagnose_slow_rendering(avg_step_ms: float):
    """Print diagnosis based on observed performance."""
    print("\n" + "=" * 60)
    print("Performance Diagnosis")
    print("=" * 60)
    
    if avg_step_ms > 100:
        print("⚠️  VERY SLOW: Step time > 100ms")
        print("\nPossible causes and solutions:")
        print("  1. GPU rendering not working properly")
        print("     - Check: nvidia-smi while running to see GPU utilization")
        print("     - Solution: Ensure habitat-sim was built with CUDA support")
        print()
        print("  2. Scene mesh is very large")
        print("     - Check: The scene file size")
        print("     - Solution: Use simpler scenes for testing")
        print()
        print("  3. Running on CPU instead of GPU")
        print("     - Check: Look for 'Renderer: NVIDIA' in startup logs")
        print("     - Solution: Set gpu_device_id correctly")
        print()
        print("  4. EGL/display issues")
        print("     - Check: echo $DISPLAY")
        print("     - Solution: For headless, unset DISPLAY or use EGL")
        
    elif avg_step_ms > 50:
        print("⚡ MODERATE: Step time 50-100ms")
        print("\nThis is acceptable but could be improved:")
        print("  - Consider reducing sensor resolution")
        print("  - Reduce number of sensors")
        print("  - Check if depth sensor is needed")
        
    elif avg_step_ms > 20:
        print("✅ GOOD: Step time 20-50ms")
        print("\nPerformance is reasonable for VLN tasks.")
        
    else:
        print("🚀 EXCELLENT: Step time < 20ms")
        print("\nRendering performance is very good!")


def main():
    """Main entry point."""
    print("\n" + "=" * 60)
    print("Habitat-sim Rendering Speed Diagnostic")
    print("=" * 60)
    
    # Print system info
    print_system_info()
    
    # Find a scene file
    scene_path = find_scene_file()
    if scene_path is None:
        print("\nError: No scene file found!")
        print("Please specify a scene path manually.")
        return
    
    print(f"\nUsing scene: {scene_path}")
    
    # Run tests
    try:
        step_times, obs_times = test_habitat_sim_direct(scene_path, num_steps=100)
        
        # Calculate average for diagnosis
        total_times = [s + o for s, o in zip(step_times, obs_times)]
        avg_ms = np.mean(total_times) * 1000
        
        # Resolution impact
        test_resolution_impact(scene_path, num_steps=50)
        
        # Multiple sensors impact
        test_multiple_sensors(scene_path, num_steps=50)
        
        # Diagnosis
        diagnose_slow_rendering(avg_ms)
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("Diagnostic Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
