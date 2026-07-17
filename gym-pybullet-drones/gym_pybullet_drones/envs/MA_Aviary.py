from gymnasium import spaces
import numpy as np
import pybullet as p
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.BaseAviary import BaseAviary

default_target = [
    np.array([-1,  1, 4]),
    np.array([0, 0, 0, 1]),
]

class MA_Aviary(BaseAviary):
    """Multi-Agent RL Aviary Class with Parameter Sharing Support"""

    def __init__(self,
                 drone_model: DroneModel = DroneModel.CF2X,
                 num_drones_per_batch: int = 4,
                 num_batches_per_world: int = 1,
                 neighbourhood_radius: float = np.inf,
                 initial_xyzs_batch=None,
                 target_pos=default_target,
                 cable_lenght = 2.0,
                 initial_batch_center_xyzs=None,
                 initial_rpys_batch=None,
                 physics: Physics = Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 48,
                 gui=False,
                 user_debug_gui=False,
                 output_folder='results',
                 episode_len_sec=10
                 ):
        
        super().__init__(drone_model=drone_model,
                        num_drones_per_batch=num_drones_per_batch,
                        num_batches_per_world=num_batches_per_world,
                        neighbourhood_radius=neighbourhood_radius,
                        initial_xyzs_batch=initial_xyzs_batch,
                        target_pos=target_pos,
                        cable_lenght=cable_lenght,
                        initial_batch_center_xyzs=initial_batch_center_xyzs,
                        initial_rpys_batch=initial_rpys_batch,
                        physics=physics,    
                        pyb_freq=pyb_freq,
                        ctrl_freq=ctrl_freq,
                        gui=gui,
                        user_debug_gui=user_debug_gui,
                        output_folder=output_folder,
                        episode_len_sec=episode_len_sec
                        )

    def _actionSpace(self):
        """Returns the action space for a SINGLE agent."""
        low = np.array([-self.GRAVITY, -0.4, -0.4], dtype=np.float32)
        high = np.array([self.MAX_THRUST - self.GRAVITY, 0.4, 0.4], dtype=np.float32)
        
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def _observationSpace(self):
        """Returns the restricted observation space for a SINGLE agent."""
        obs_dim = 3 + 9  # Payload target position and orientation matrix
        obs_dim += 3 + 9 + 3 + 3  # Payload relative pos, rot mat, vel, angular vel
        obs_dim += 1 * (3 + 9 + 3 + 3)  # ONLY SELF drone state (1 drone instead of 4)
        obs_dim += 1 * 3  # ONLY SELF last action (3 elements instead of 12)
        obs_dim += self.NUM_DRONES_PER_BATCH  # One-hot identifier vector

        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
    
    def _globalStateSpace(self):
        """Returns the global observation space for the Centralized Critic."""
        state_dim = 3 + 9  # Payload target position and orientation matrix
        state_dim += 3 + 9 + 3 + 3  # Payload relative pos, rot mat, vel, angular vel
        state_dim += self.NUM_DRONES_PER_BATCH * (3 + 9 + 3 + 3) # ALL drone states
        state_dim += self.NUM_DRONES_PER_BATCH * 3 # ALL last actions
        
        # Note: We do NOT need the one-hot agent IDs here because this is a 
        # single global state representing the entire team environment.

        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32
        )
    
    def _computeObs(self):
        """Computes and returns a restricted multi-agent observations tensor."""
        obs_all_batches = []

        for batch_id in range(self.NUM_BATCHES):
            # 1. Base shared observations (Target + Payload)
            base_obs = [
                self.PAYLOAD_TARGET_POS[batch_id],
                p.getMatrixFromQuaternion(self.PAYLOAD_TARGET_ORN[batch_id])
            ]

            payload_pos, payload_quat = p.getBasePositionAndOrientation(
                self.PAYLOAD_ID[batch_id],
                physicsClientId=self.CLIENT
            )
            payload_pos = payload_pos - self.INIT_BATCH_CENTER_XYZS[batch_id, :]

            base_obs.extend([
                payload_pos,
                p.getMatrixFromQuaternion(payload_quat),
                *p.getBaseVelocity(self.PAYLOAD_ID[batch_id], physicsClientId=self.CLIENT)
            ])
            
            flat_shared_obs = np.concatenate(base_obs).astype(np.float32)

            # 2. Construct individual agent observations containing ONLY self data
            batch_agent_obs = []
            for agent_id in range(self.NUM_DRONES_PER_BATCH):
                # Get ONLY this specific drone's state
                drone_idx = batch_id * self.NUM_DRONES_PER_BATCH + agent_id
                state = self._getDroneStateVector(drone_idx)

                pos = state[0:3] - self.INIT_BATCH_CENTER_XYZS[batch_id, :]
                quat = state[3:7]
                vel = state[10:13]
                ang_vel = state[13:16]

                self_drone_state = np.concatenate([
                    pos,
                    p.getMatrixFromQuaternion(quat),
                    vel,
                    ang_vel
                ])

                # Get ONLY this specific drone's last action (3 elements)
                # self.last_actions shape is typically (NUM_BATCHES, NUM_DRONES_PER_BATCH * 3)
                start_act_idx = agent_id * 3
                self_last_action = self.last_actions[batch_id, start_act_idx:start_act_idx+3]

                # Create the individual one-hot ID vector
                one_hot_id = np.zeros(self.NUM_DRONES_PER_BATCH, dtype=np.float32)
                one_hot_id[agent_id] = 1.0
                
                # Combine shared target/payload data + self state + self action + ID
                agent_specific_obs = np.concatenate([
                    flat_shared_obs, 
                    self_drone_state, 
                    self_last_action, 
                    one_hot_id
                ])
                batch_agent_obs.append(agent_specific_obs)
                
            obs_all_batches.append(batch_agent_obs)

        return np.asarray(obs_all_batches, dtype=np.float32)
    
    def _computeGlobalState(self):
        """Computes and returns a global state tensor of shape:
           (NUM_BATCHES, state_dim)
        """
        state_all_batches = []

        for batch_id in range(self.NUM_BATCHES):
            # 1. Target & Payload
            batch_state = [
                self.PAYLOAD_TARGET_POS[batch_id],
                p.getMatrixFromQuaternion(self.PAYLOAD_TARGET_ORN[batch_id])
            ]

            payload_pos, payload_quat = p.getBasePositionAndOrientation(
                self.PAYLOAD_ID[batch_id],
                physicsClientId=self.CLIENT
            )
            payload_pos = payload_pos - self.INIT_BATCH_CENTER_XYZS[batch_id, :]

            batch_state.extend([
                payload_pos,
                p.getMatrixFromQuaternion(payload_quat),
                *p.getBaseVelocity(self.PAYLOAD_ID[batch_id], physicsClientId=self.CLIENT)
            ])

            # 2. Append ALL drone states
            for j in range(self.NUM_DRONES_PER_BATCH):
                drone_idx = batch_id * self.NUM_DRONES_PER_BATCH + j
                state = self._getDroneStateVector(drone_idx)

                pos = state[0:3] - self.INIT_BATCH_CENTER_XYZS[batch_id, :]
                quat = state[3:7]
                vel = state[10:13]
                ang_vel = state[13:16]

                batch_state.extend([pos, p.getMatrixFromQuaternion(quat), vel, ang_vel])

            # 3. Append ALL last actions
            batch_state.extend([self.last_actions[batch_id, :].flatten()])
            
            state_all_batches.append(np.concatenate(batch_state).astype(np.float32))

        return np.asarray(state_all_batches, dtype=np.float32)
    

    def _computeReward(self, terminated, action, clipped_action, lambda1=1.5, lambda2=0.5, lambda3=1.0, lambda4=1.5, lambda5=0.25, lambda6=0.5, lambda7=2.0):
        """Computes and returns a multi-agent reward array of shape:
           (NUM_BATCHES, NUM_DRONES_PER_BATCH)
        """
        rewards = np.zeros((self.NUM_BATCHES, self.NUM_DRONES_PER_BATCH), dtype=np.float32)

        for batch_id in range(self.NUM_BATCHES):
            # Check if any agent in the batch caused a termination step
            if np.any(terminated[batch_id]):
                step_penalty = -0.5 * self.CTRL_FREQ * (1 + lambda1 + lambda3 + lambda5 + lambda6)
                rewards[batch_id, :] = step_penalty / self.CTRL_FREQ / (1 + lambda1 + lambda3 + lambda5 + lambda6)
                continue 

            payload_pos, payload_quat = p.getBasePositionAndOrientation(
                self.PAYLOAD_ID[batch_id],
                physicsClientId=self.CLIENT
            )
            payload_pos = payload_pos - self.INIT_BATCH_CENTER_XYZS[batch_id, :]
            
            # Action array tracking parameters
            range_action = np.array([self.MAX_THRUST, 2, 2] * self.NUM_DRONES_PER_BATCH)
            flat_last_actions = self.last_actions[batch_id, :].flatten()
            flat_current_actions = action[batch_id, :].flatten()

            # Shared team metrics 
            pos_reward = lambda1 * np.exp(-lambda2 * np.linalg.norm(np.array(payload_pos) - self.PAYLOAD_TARGET_POS[batch_id]))
            orient_reward = lambda3 * np.exp(-lambda4 * self.quaternion_error_magnitude(payload_quat, self.PAYLOAD_TARGET_ORN[batch_id])) * np.exp(-lambda2 * np.linalg.norm(np.array(payload_pos) - self.PAYLOAD_TARGET_POS[batch_id]))
            smooth_reward = lambda5 * np.exp(- (np.linalg.norm((flat_current_actions - flat_last_actions) / range_action) / self.NUM_DRONES_PER_BATCH) ** 2)

            for j in range(self.NUM_DRONES_PER_BATCH):
                drone_idx = batch_id * self.NUM_DRONES_PER_BATCH + j
                drone_clipped_thrust = clipped_action[drone_idx, 0] # Extract unique individual scalar metric
                thrust_from_rpm = 4 * self.KF * (drone_clipped_thrust ** 2)
                
                individual_thrust_penalty = lambda6 * np.exp(-lambda7 * thrust_from_rpm / self.MAX_THRUST)

                # Aggregate shared reward signals with local drone constraints
                step_reward = pos_reward + orient_reward + smooth_reward + individual_thrust_penalty
                
                # Normalize step metric bounds to [-1, 1] range mapping
                rewards[batch_id, j] = step_reward / self.CTRL_FREQ / (1 + lambda1 + lambda3 + lambda5 + lambda6)

        return rewards

    def _computeTerminated(self):
        """Computes and returns a multi-agent termination flag array of shape:
           (NUM_BATCHES, NUM_DRONES_PER_BATCH) 
        """
        done = np.zeros((self.NUM_BATCHES, self.NUM_DRONES_PER_BATCH), dtype=bool)

        for batch_id in range(self.NUM_BATCHES):
            batch_failed = False
            
            # Evaluate constraints for each drone within the environment batch
            for j in range(self.NUM_DRONES_PER_BATCH):
                drone_idx = batch_id * self.NUM_DRONES_PER_BATCH + j
                state = self._getDroneStateVector(drone_idx)

                pos = state[0:3]
                
                # Boundary Conditions: Vertical Altitude Check
                if pos[2] < 0.0 or pos[2] > 20.0 + self.REST_LENGTH:
                    batch_failed = True
                    break
                    
                # Boundary Conditions: Horizontal Arena Radius Check
                spawn_pos = self.INIT_BATCH_CENTER_XYZS[batch_id, :]
                if abs(pos[0] - spawn_pos[0]) > 12.5 or abs(pos[1] - spawn_pos[1]) > 12.5:
                    batch_failed = True
                    break

                # Boundary Conditions: Pitch and Roll Limits
                roll, pitch = state[7:9]
                if abs(roll) > np.pi or abs(pitch) > np.pi:
                    batch_failed = True
                    break

            # In cooperative multi-agent tasks, if one agent fails, the whole team terminates
            if batch_failed:
                done[batch_id, :] = True

        return done

    def _computeTruncated(self):
        """Returns an environment truncation flag array of shape:
           (NUM_BATCHES, NUM_DRONES_PER_BATCH)
        """
        time_expired = (
            self.step_counter >=
            self.EPISODE_LEN_SEC *
            self.PYB_STEPS_PER_CTRL *
            self.CTRL_FREQ
        )
        return np.broadcast_to(
                time_expired[:, np.newaxis], 
                (self.NUM_BATCHES, self.NUM_DRONES_PER_BATCH)
            )

    def _computeInfo(self):
        return [{"terminal_observation": None} for _ in range(self.NUM_BATCHES)]