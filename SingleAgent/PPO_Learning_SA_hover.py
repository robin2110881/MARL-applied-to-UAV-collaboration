import os
import datetime
import numpy as np
from collections import deque
import torch as T
from PPO_Learning_SA import Agent # No longer need to import RunningStandardScaler here
from gym_pybullet_drones.envs.Parallel_SA_Aviary import Parallel_SA_Aviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics
import matplotlib.pyplot as plt
from time import time

if __name__ == '__main__':


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

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(SCRIPT_DIR, 'sa_ppo')
    output_dir = os.path.join(SCRIPT_DIR, 'sa_ppo_training_outputs')
    date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(output_dir, exist_ok=True)

    sim_freq_hz = 240
    ctrl_freq_hz = 48
    init_z = 1.0

    init_xyzs = np.array([
        [-0.5,  0.5, init_z],
        [ 0.5,  0.5, init_z],
        [ 0.5, -0.5, init_z],
        [-0.5, -0.5, init_z]
    ])

    target = [
            np.array([0,  0, 1]),
            euler_to_quaternion(0, 0, 0),
        ]

    grid_dim = 9
    grid_spacing = 25.0
    max_extension = (grid_dim - 1) / 2 * grid_spacing
    ticks = np.linspace(-max_extension, max_extension, grid_dim)
    X, Y = np.meshgrid(ticks, ticks)
    init_batch_center_xyzs = np.column_stack((X.ravel(), Y.ravel(), 10 * np.ones(X.size)))

    num_drones_per_batch = 4
    num_batches_per_world = len(init_batch_center_xyzs)
    num_worlds = 4

    total_batches = num_batches_per_world * num_worlds
    max_episode_len_sec = 20

    env = Parallel_SA_Aviary(
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
        gui=False,   
        user_debug_gui=False,
        episode_len_sec=max_episode_len_sec
    )

    
    rollout = int(ctrl_freq_hz * 2)
    num_minibatches = 4
    n_epochs = 5
    alpha_actor = 5e-4
    alpha_critic = 1e-4 # critic lr is typically larger than actor for stable learning in PPO, especially with continuous action spaces
    policy_clip = 0.1 #policy clip also named ratio clip
    value_loss_clip = 0.1 # same as disabling
    gradient_norm_clip = 1.0
    value_loss_scale = 1.0
    entropy_loss_scale = 0.001
    gae_lambda = 0.95
    gamma = 0.99

    total_buffer_size = rollout * total_batches
    batch_size = total_buffer_size // num_minibatches
    agent = Agent(input_dims=env.observation_space.shape, n_actions=12, num_envs=total_batches,
                  policy_clip=policy_clip, value_loss_clip=value_loss_clip, gradient_norm_clip=gradient_norm_clip,
                  gae_lambda=gae_lambda, gamma=gamma, 
                  value_loss_scale=value_loss_scale, entropy_loss_scale=entropy_loss_scale,
                  alpha_actor=alpha_actor, alpha_critic=alpha_critic, 
                  actor_fc1_dims=512, actor_fc2_dims=256, actor_fc3_dims=128, actor_fc4_dims=64,
                  critic_fc1_dims=1024, critic_fc2_dims=512, critic_fc3_dims=256, critic_fc4_dims=128,
                  batch_size=batch_size, N=rollout, n_epochs=n_epochs, 
                  checkpoint_dir=checkpoint_path)
    
    low_action = np.array([-env.GRAVITY, -1.0, -1.0] * 4)
    high_action = np.array([env.MAX_THRUST - env.GRAVITY, 1.0, 1.0] * 4)

    n_games = 4385
    n_noise = 500
    best_score = -np.inf
    n_steps = 0

    running_rewards = np.zeros(total_batches)
    running_lengths = np.zeros(total_batches)
    completed_scores_window = deque(maxlen=100)
    completed_lengths_window = deque(maxlen=100)
    
    score_history = []
    length_history = []
    value_loss_history = []
    entropy_loss_history = []

    t = time()

    try:
        raw_obs, info = env.reset() 

        for i in range(n_games):
            multiplier = 0.3 + 0.7 * np.tanh(3 * (i-n_noise) / (n_games - n_noise) * (i > n_noise))
            env.set_reset_std(
                stdxy=0.2*multiplier, stdz=0.2*multiplier, stdrp=0.1*multiplier, stdy=0.1*multiplier,
                stddxy=0.1*multiplier, stddz=0.1*multiplier, stddrp=0.1*multiplier, stddy=0.1*multiplier,
                stddvxy=0.6*multiplier, stddvz=0.6*multiplier, stddvrp=0.08*multiplier, stddvy=0.08*multiplier,
                stdpvxy=0.4*multiplier, stdpvz=0.4*multiplier, stdpvrp=0.08*multiplier, stdpvy=0.08*multiplier,
                min_mass_prc = 1.0 - 0.25 * multiplier, max_mass_prc = 1.0 + 0.25 * multiplier,
                stdspxy = 5, stdspz = 0.5, stdsprp = np.pi/4, stdspy = np.pi #no multiplier on target 
            )
            if i == 0:
                raw_obs, info = env.reset() 
            
            for step in range(rollout): 
                action, u, prob, val, scaled_obs = agent.choose_action(raw_obs, low_action, high_action, update_scaler=True)
                
                # Unpack terminated and truncated
                raw_obs_, reward, terminated, truncated, info = env.step(action)

                # FIX: Combine them into a single done array for local metric tracking
                done = np.logical_or(terminated, truncated)

                # FIX: Pass terminated and truncated separately to the agent's memory
                agent.remember(scaled_obs, u, prob, val, reward, terminated, truncated)
                
                for b in range(total_batches):
                    running_rewards[b] += reward[b]
                    running_lengths[b] += 1
                    if done[b]: # Tracks if an episode ended for either reason
                        completed_scores_window.append(running_rewards[b])
                        completed_lengths_window.append(running_lengths[b])
                        running_rewards[b] = 0.0 
                        running_lengths[b] = 0

                n_steps += 1
                raw_obs = raw_obs_ 

            # We copy raw_obs to create a pristine bootstrap array. 
            # Without overwriting raw_obs directly, or the drones will start the 
            # next iteration step loop with old terminal positions instead of their new resets
            bootstrap_obs = raw_obs.copy()
            
            if info is not None:
                for b in range(total_batches):
                    if terminated[b] or truncated[b]:
                        if b < len(info) and "terminal_observation" in info[b]:
                            bootstrap_obs[b] = info[b]["terminal_observation"]

            # Pass the modified bootstrap_obs array to your learning function
            value_loss, entropy_loss = agent.learn(bootstrap_obs, terminated, truncated, low_action, high_action)

            if len(completed_scores_window) > 0:
                recent_score = np.mean(completed_scores_window)
                recent_length = np.mean(completed_lengths_window)
            else:
                # Fallback purely for the very first few iterations before any episodes finish
                recent_score = np.mean(running_rewards)
                recent_length = np.mean(running_lengths)

            score_history.append(recent_score)
            length_history.append(recent_length)

            avg_score = np.mean(score_history[-50:])
            avg_length = np.mean(length_history[-50:])

            if avg_score > best_score and len(score_history) > 1:
                best_score = avg_score
                agent.save_models() 

            

            print(f'Iteration {i:3d} | Recent Ep Score: {recent_score:6.3f} | 50-Iter Avg: {avg_score:6.3f} | Length: {recent_length:6.0f} | Total Steps : {n_steps * total_batches} | Value Loss: {value_loss:.4f} | Entropy Loss: {entropy_loss:.2f} | Time for Iteration: {time() - t:.2f} sec')

            value_loss_history.append(float(value_loss))
            entropy_loss_history.append(float(entropy_loss))

            t = time()

    except KeyboardInterrupt:
        print("\n[-] Training interrupted by user. Gathering data and plotting...")
        np.save(f'{output_dir}/score_history_{date}.npy', np.array(score_history))
        np.save(f'{output_dir}/length_history_{date}.npy', np.array(length_history))
        np.save(f'{output_dir}/value_loss_history_{date}.npy', np.array(value_loss_history))
        np.save(f'{output_dir}/entropy_loss_history_{date}.npy', np.array(entropy_loss_history))
        print("Training terminated. Saved training history arrays for later analysis.")
        
        

    except Exception as e:
        print(f"An error occurred during training: {e}")
        import traceback
        traceback.print_exc()

    finally:
        np.save(f'{output_dir}/score_history_{date}.npy', np.array(score_history))
        np.save(f'{output_dir}/length_history_{date}.npy', np.array(length_history))
        np.save(f'{output_dir}/value_loss_history_{date}.npy', np.array(value_loss_history))
        np.save(f'{output_dir}/entropy_loss_history_{date}.npy', np.array(entropy_loss_history))
        print("Training terminated. Saved training history arrays for later analysis.")
        if len(score_history) > 0:
            print("Generating plots...")
            fig, axs = plt.subplots(2, 2, figsize=(15, 10))
            
            axs[0, 0].plot(score_history, alpha=0.3, color='blue', label='Raw')
            if len(score_history) >= 50:
                axs[0, 0].plot(range(49, len(score_history)), np.convolve(score_history, np.ones(50)/50, mode='valid'), color='darkblue', linewidth=2, label='50-Iter Avg')
            axs[0, 0].set_title('PPO Reward Training Progress')
            axs[0, 0].set_xlabel('Iteration')
            axs[0, 0].set_ylabel('Mean Drone Reward')
            axs[0, 0].grid(True)
            axs[0, 0].legend()

            axs[0, 1].plot(length_history, alpha=0.3, color='orange', label='Raw')
            if len(length_history) >= 50:
                axs[0, 1].plot(range(49, len(length_history)), np.convolve(length_history, np.ones(50)/50, mode='valid'), color='darkorange', linewidth=2, label='50-Iter Avg')
            axs[0, 1].set_title('PPO Episode Length Training Progress')
            axs[0, 1].set_xlabel('Iteration')
            axs[0, 1].set_ylabel('Mean Length')
            axs[0, 1].grid(True)
            axs[0, 1].legend()

            axs[1, 0].plot(value_loss_history, color='green')
            axs[1, 0].set_title('PPO Value Loss During Training')
            axs[1, 0].set_xlabel('Iteration')
            axs[1, 0].set_ylabel('Value Loss')
            axs[1, 0].grid(True)

            axs[1, 1].plot(entropy_loss_history, color='red')
            axs[1, 1].set_title('PPO Entropy Loss During Training')
            axs[1, 1].set_xlabel('Iteration')
            axs[1, 1].set_ylabel('Entropy Loss')
            axs[1, 1].grid(True)

            plt.tight_layout()
            plt.show()
        else:
            print("Training terminated before any history was recorded. Skipping plots.")