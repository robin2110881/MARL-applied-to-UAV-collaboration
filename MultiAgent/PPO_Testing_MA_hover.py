import os
import time
import numpy as np
import torch as T
from PPO_Learning_MA import Agent
from gym_pybullet_drones.envs.Parallel_MA_Aviary import Parallel_MA_Aviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
import matplotlib.pyplot as plt
if __name__ == '__main__':
    plotting = False


    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    
    checkpoint_path = os.path.join(SCRIPT_DIR, 'ma_ppo')
    output_dir = os.path.join(SCRIPT_DIR, 'ma_ppo_training_outputs')

    

    if plotting:
            
            score_history = np.load(os.path.join(output_dir, "score_history_2026-06-09_15-50-37.npy"))
            length_history = np.load(os.path.join(output_dir, "length_history_2026-06-09_15-50-37.npy"))
            value_loss_history = np.load(os.path.join(output_dir, "value_loss_history_2026-06-09_15-50-37.npy"))
            entropy_loss_history = np.load(os.path.join(output_dir, "entropy_loss_history_2026-06-09_15-50-37.npy"))

            if len(score_history) > 0:
                print("Generating plots...")
                plt.figure(figsize=(10, 5))
                plt.plot(score_history, alpha=0.3, label='Raw Iteration Score', color='blue')
                
                if len(score_history) >= 50:
                    avg_score_history = np.convolve(score_history, np.ones(50)/50, mode='valid')
                    plt.plot(range(49, len(score_history)), avg_score_history, label='50-Iter Moving Avg', color='darkblue', linewidth=2)
                
                plt.xlabel('Training Iteration')
                plt.ylabel('True Episodic Mean Reward')
                plt.title('PPO Reward Training Progress')
                plt.legend()
                plt.grid(True)

                plt.figure(figsize=(10, 5))
                plt.plot(length_history, alpha=0.3, label='Raw Iteration Length', color='orange')
                
                if len(length_history) >= 50:
                    avg_length_history = np.convolve(length_history, np.ones(50)/50, mode='valid')
                    plt.plot(range(49, len(length_history)), avg_length_history, label='50-Iter Moving Avg', color='darkorange', linewidth=2)
                    
                plt.xlabel('Training Iteration')
                plt.ylabel('Episodic Mean Length')
                plt.title('PPO Episode Length Training Progress')
                plt.legend()
                plt.grid(True)
                
                plt.figure(figsize=(10, 5))
                plt.plot(value_loss_history, label='Value Loss', color='green')
                plt.xlabel('Training Iteration')
                plt.ylabel('Value Loss')
                plt.title('PPO Value Loss During Training')
                plt.legend()
                plt.grid(True)

                plt.figure(figsize=(10, 5))
                plt.plot(entropy_loss_history, label='Entropy Loss', color='red')
                plt.xlabel('Training Iteration')
                plt.ylabel('Entropy Loss')
                plt.title('PPO Entropy Loss During Training')
                plt.legend()
                plt.grid(True)
                plt.show()
    

    def euler_to_quaternion(roll, pitch, yaw):
        cy = np.cos(yaw * 0.5)
        sy = np.sin(yaw * 0.5)
        cp = np.cos(pitch * 0.5)
        sp = np.sin(pitch * 0.5)
        cr = np.cos(roll * 0.5)
        sr = np.sin(roll * 0.5)

        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy

        return np.array([qx, qy, qz, qw])
    

    target = [
            np.array([0,  0, 1]),
            euler_to_quaternion(0, 0, 0),
        ]
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
    init_batch_center_xyzs = np.array([[0.0, 0.0, 10.0]])

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
    env = Parallel_MA_Aviary(
        drone_model=DroneModel.CF2X,
        num_drones_per_batch=num_drones_per_batch,
        num_batches_per_world=num_batches_per_world,
        num_worlds=num_worlds,
        initial_xyzs_batch=init_xyzs,
        initial_batch_center_xyzs=init_batch_center_xyzs,
        initial_rpys_batch=np.zeros((num_drones_per_batch, 3)),
        cable_lenght = 2.0, 
        target_pos=target,
        physics=Physics.PYB,
        pyb_freq=sim_freq_hz,
        ctrl_freq=ctrl_freq_hz,
        gui=True,   
        user_debug_gui=False,
        episode_len_sec=20
    )

    ######################## LOAD TRAINED AGENT ########################
    agent = Agent(
        input_dims=env.observation_space.shape,
        n_actions=3,

        actor_fc1_dims=512, actor_fc2_dims=256, actor_fc3_dims=128, actor_fc4_dims=64,
        critic_fc1_dims=1024, critic_fc2_dims=512, critic_fc3_dims=256, critic_fc4_dims=128,

        checkpoint_dir=checkpoint_path
    )
    
    print("Loading pre-trained policy weights and scaler state...")
    try:
        agent.load_models()  
        agent.actor.eval()
        agent.critic.eval()
        print("Weights and Scaler successfully loaded!")
    except FileNotFoundError:
        print("Warning: Checkpoint files not found. Running with random initialization.")

    low_action = np.array([-env.GRAVITY, -1.0, -1.0])
    high_action = np.array([env.MAX_THRUST - env.GRAVITY, 1.0, 1.0])
    
    low_t = T.tensor(low_action, dtype=T.float32).to(agent.actor.device)
    high_t = T.tensor(high_action, dtype=T.float32).to(agent.actor.device)

    ######################## EVALUATION LOOP ########################
    num_test_episodes = 5
    steps_per_episode = ctrl_freq_hz * 20  # 20  seconds

    for episode in range(num_test_episodes):
        multiplier = 1.0
        env.set_reset_std(
                stdxy=0.2*multiplier, stdz=0.2*multiplier, stdrp=0.1*multiplier, stdy=0.1*multiplier,
                stddxy=0.1*multiplier, stddz=0.1*multiplier, stddrp=0.1*multiplier, stddy=0.1*multiplier,
                stddvxy=0.6*multiplier, stddvz=0.6*multiplier, stddvrp=0.08*multiplier, stddvy=0.08*multiplier,
                stdpvxy=0.4*multiplier, stdpvz=0.4*multiplier, stdpvrp=0.08*multiplier, stdpvy=0.08*multiplier,
                min_mass_prc = 1.0 - 0.3 * multiplier, max_mass_prc = 1.0 + 0.3 * multiplier,
                stdspxy = 5, stdspz = 0.5, stdsprp = 1, stdspy = 3.14 #no multiplier on target 
            )
        raw_obs, info = env.reset()
        obs = agent.actor_scaler.transform(raw_obs)
        episode_reward = 0
        print(f"\n--- Starting Evaluation Episode {episode + 1} ---")
        
        for step in range(steps_per_episode):
            start_time = time.time()
            
            num_envs, num_agents, obs_dim = obs.shape
            flat_scaled_obs = obs.reshape(num_envs * num_agents, obs_dim)
            state_t = T.tensor(flat_scaled_obs, dtype=T.float32).to(agent.actor.device)
            
            with T.no_grad():
                dist = agent.actor(state_t)
                u = dist.mean
                a = T.tanh(u)

                action = low_t + (high_t - low_t) * (a + 1) / 2.0
                action = action.cpu().numpy().reshape(num_envs, -1)

            # Step the environment
            raw_obs_, reward, terminated, truncated, info = env.step(action)
            episode_reward += np.mean(reward)

            obs = agent.actor_scaler.transform(raw_obs_)

            # Maintain control frequency loop timing
            elapsed = time.time() - start_time
            if elapsed < (1.0 / ctrl_freq_hz):
                time.sleep((1.0 / ctrl_freq_hz) - elapsed)

            if np.any(terminated) or np.any(truncated):
                print(f"Episode finished early at step {step} due to crash or out-of-bounds boundary criteria.")
                break
                
        print(f"Episode {episode + 1} Finished | Cumulative Reward: {episode_reward:.2f}")

    env.close()