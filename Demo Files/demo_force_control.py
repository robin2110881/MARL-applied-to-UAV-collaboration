import os
import time
import numpy as np
import torch as T
from PPO_Learning import Agent
from gym_pybullet_drones.envs.ParallelAviary import ParallelAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

if __name__ == '__main__':
    
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    checkpoint_path = os.path.join(SCRIPT_DIR, 'tmp', 'ppo')
    output_dir = os.path.join(SCRIPT_DIR, 'ppo_training_outputs')
    print(f"Checkpoint path for loading weights: {checkpoint_path}")
    
    ######################## ENVIRONMENT SETUP ########################
    sim_freq_hz = 240
    ctrl_freq_hz = 48
    init_z = 1.0

    # 1 Batch layout of 4 drones
    init_xyzs = np.array([
        [-0.5,  0.5, init_z],
        [ 0.5,  0.5, init_z],
        [ 0.5, -0.5, init_z],
        [-0.5, -0.5, init_z]
    ])

    # Just 1 batch center point at the origin
    init_batch_center_xyzs = np.array([[0.0, 0.0, 0.0]])

    num_drones_per_batch = 4
    num_batches_per_world = 1
    num_worlds = 1  # Only 1 CPU core/world for testing

    target_pos = np.array([
            [-1,  1, 4],
            [ 1,  1, 4],
            [ 1, -1, 4],
            [-1, -1, 4]
        ])
    
    # Test Environment configuration
    env = ParallelAviary(
        drone_model=DroneModel.CF2X,
        num_drones_per_batch=num_drones_per_batch,
        num_batches_per_world=num_batches_per_world,
        num_worlds=num_worlds,
        initial_xyzs_batch=init_xyzs,
        initial_batch_center_xyzs=init_batch_center_xyzs,
        initial_rpys_batch=np.zeros((num_drones_per_batch, 3)),
        cable_lenght = 2.0, 
        target_pos=target_pos,
        physics=Physics.PYB,
        pyb_freq=sim_freq_hz,
        ctrl_freq=ctrl_freq_hz,
        gui=True,   
        obstacles=False,
        user_debug_gui=False,
        episode_len_sec=20
    )


    ######################## EVALUATION LOOP ########################
    steps_per_episode = ctrl_freq_hz * 30  # 30 seconds

    for step in range(steps_per_episode):
        start_time = time.time()
        
        # Feeding deterministic normalized 'obs' into the model
        action = np.zeros((num_batches_per_world * num_worlds, 12))
        for i in range(num_drones_per_batch):
            action[:, i*3] += 0.057 * 9.81 / 4
            action[:, i*3 + 1] +=  0.2 * (i > 1) - 0.2 * (i <= 1)
            action[:, i*3 + 2] += - 0.2 * (i == 0 or i == 3) + 0.2 * (i == 1 or i == 2)
        # Step the environment
        raw_obs_, reward, terminated, truncated, info = env.step(action)

        # Maintain control frequency loop timing
        elapsed = time.time() - start_time
        if elapsed < (1.0 / ctrl_freq_hz):
            time.sleep((1.0 / ctrl_freq_hz) - elapsed)

        if np.any(terminated) or np.any(truncated):
            print(f"Episode finished early at step {step} due to crash or out-of-bounds boundary criteria.")
            break
            
    env.close()