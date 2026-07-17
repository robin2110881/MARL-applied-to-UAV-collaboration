from gymnasium import spaces
import numpy as np
import pybullet as p
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.BaseAviary import BaseAviary

default_target = [
            np.array([-1,  1, 4]),
            np.array([0,0,0,1]),
        ]
class SA_Aviary(BaseAviary):
    """SA RL class"""

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones_per_batch: int=4,
                 num_batches_per_world: int=1,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs_batch=None,
                 target_pos=default_target,
                 cable_lenght = 2.0,
                 initial_batch_center_xyzs=None,
                 initial_rpys_batch=None,
                 physics: Physics=Physics.PYB,
                 pyb_freq: int = 240,
                 ctrl_freq: int = 48,
                 gui=False,
                 user_debug_gui=False,
                 output_folder='results',
                 episode_len_sec=10
                 ):
        """Initialization of a generic aviary environment.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        num_drones_per_batch : int, optional
            The desired number of drones in each batch.
        num_batches_per_world : int, optional
            The desired number of batches, i.e., independent drone groups in the same simulation.
        neighbourhood_radius : float, optional
            Radius used to compute the drones' adjacency matrix, in meters.
        initial_xyzs_batch: ndarray | None, optional
            (NUM_DRONES_PER_BATCH, 3)-shaped array containing the initial XYZ positions of the drones.
        initial_batch_center_xyzs: ndarray | None, optional
            (NUM_BATCHES, 3)-shaped array containing the initial XYZ positions of each batch center.
        initial_rpys_batch: ndarray | None, optional
            (NUM_DRONES_PER_BATCH, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
        ctrl_freq : int, optional
            The frequency at which the environment steps.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        user_debug_gui : bool, optional
            Whether to draw the drones' axes and the GUI RPMs sliders.
        episode_len_sec : int, optional
            The desired episode length in seconds, used to compute the maximum number of steps per episode.

        """
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

        low = np.tile(
            np.array([
                -self.GRAVITY,
                -0.4,
                -0.4
            ], dtype=np.float32),
            self.NUM_DRONES_PER_BATCH
        )

        high = np.tile(
            np.array([
                self.MAX_THRUST - self.GRAVITY,
                0.4,
                0.4
            ], dtype=np.float32),
            self.NUM_DRONES_PER_BATCH
        )

        return spaces.Box(
            low=low,
            high=high,
            dtype=np.float32
        )
    
    
    def _observationSpace(self):

        obs_dim = 3 + 9  # Payload target position and orientation
        obs_dim += 3 + 9 + 3 + 3  # Payload relative pos, rot mat , vel, angular velocity
        obs_dim += self.NUM_DRONES_PER_BATCH * (3 + 9 + 3 + 3) # Relative position, rot mat, velocity, angular velocity for each drone
        obs_dim += self.NUM_DRONES_PER_BATCH * 3 # Last action for each drone

        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )


    def _computeObs(self):

        obs_all = []

        for batch_id in range(self.NUM_BATCHES):
            # target observations: relative position, orientation
            obs = [
                self.PAYLOAD_TARGET_POS[batch_id],
                p.getMatrixFromQuaternion(self.PAYLOAD_TARGET_ORN[batch_id])
            ]

            payload_pos, payload_quat = p.getBasePositionAndOrientation(
                self.PAYLOAD_ID[batch_id],
                physicsClientId=self.CLIENT
            )

            payload_pos = payload_pos - self.INIT_BATCH_CENTER_XYZS[batch_id, :]

            #payload observations: relative position, orientation, velocity, angular velocity
            obs.extend([
                payload_pos,
                p.getMatrixFromQuaternion(payload_quat),
                *p.getBaseVelocity(self.PAYLOAD_ID[batch_id], physicsClientId=self.CLIENT) # vel and angular vel
            ])

            # drone observations: relative position, orientation, velocity, angular velocity
            for j in range(self.NUM_DRONES_PER_BATCH):

                drone_idx = batch_id * self.NUM_DRONES_PER_BATCH + j

                state = self._getDroneStateVector(drone_idx)

                pos = state[0:3] - self.INIT_BATCH_CENTER_XYZS[batch_id, :]
                quat = state[3:7]
                vel = state[10:13]
                ang_vel = state[13:16]

                obs.extend([
                    pos,
                    p.getMatrixFromQuaternion(quat),
                    vel,
                    ang_vel
                ])

            obs.extend([self.last_actions[batch_id, :]])
            obs_all.append(
                np.concatenate(obs).astype(np.float32)
            )

        return np.asarray(obs_all, dtype=np.float32)
    
    def _computeReward(self, terminated, action, clipped_action, lambda1 = 1.5, lambda2 = 0.5, lambda3 = 1.0, lambda4 = 1.5, lambda5 = 0.25, lambda6 = 0.5, lambda7 = 2.0):

        rewards = np.zeros(self.NUM_BATCHES, dtype=np.float32)

        for batch_id in range(self.NUM_BATCHES):
            if terminated[batch_id]:
                rewards[batch_id] = - 0.5 * self.CTRL_FREQ * (1 + lambda1 + lambda3 + lambda5 + lambda6) #loss of 0.5sec of max reward
                continue 

            payload_pos, payload_quat = p.getBasePositionAndOrientation(
                self.PAYLOAD_ID[batch_id],
                physicsClientId=self.CLIENT
            )
            payload_pos = payload_pos - self.INIT_BATCH_CENTER_XYZS[batch_id, :]
            thrust_from_rpm = np.array([4 *self.KF * clipped_action[batch_id * self.NUM_DRONES_PER_BATCH + k]**2 for k in range(self.NUM_DRONES_PER_BATCH)])
            range_action = np.array([self.MAX_THRUST, 2, 2,self.MAX_THRUST, 2, 2, self.MAX_THRUST, 2, 2,self.MAX_THRUST, 2, 2])

            reward = 0.0 # reward each timestep
            reward += lambda1 * np.exp(-lambda2 * np.linalg.norm(np.array(payload_pos) - self.PAYLOAD_TARGET_POS[batch_id])) #POS
            reward += lambda3 * np.exp(-lambda4 * self.quaternion_error_magnitude(payload_quat, self.PAYLOAD_TARGET_ORN[batch_id])) * np.exp(-lambda2 * np.linalg.norm(np.array(payload_pos) - self.PAYLOAD_TARGET_POS[batch_id])) #ORN but onl when pos is good
            reward += lambda5 * np.exp(- ( np.linalg.norm((action[batch_id] - self.last_actions[batch_id])/range_action)/self.NUM_DRONES_PER_BATCH )** 2 ) #ACTION SMOOTHNESS
            reward += lambda6 * np.exp(-lambda7 * np.max(thrust_from_rpm)/self.MAX_THRUST) #THRUST PENALTY

            rewards[batch_id] = reward

        return rewards / self.CTRL_FREQ / (1 + lambda1 + lambda3 + lambda5 + lambda6)  # Normalize by max possible reward per step
    
    
    def _computeTerminated(self):
        # Initialize all environments as not done
        done = np.zeros(self.NUM_BATCHES, dtype=bool)

        for batch_id in range(self.NUM_BATCHES):
            for j in range(self.NUM_DRONES_PER_BATCH):
                drone_idx = batch_id * self.NUM_DRONES_PER_BATCH + j
                state = self._getDroneStateVector(drone_idx)

                # 1. Position checks (XYZ)
                pos = state[0:3]
                
                # Crash check (Altitude too low)
                if pos[2] < 0.0 or pos[2] > 20.0 + self.REST_LENGTH:
                    done[batch_id] = True
                    break  # Stop checking this batch, it's already done
                    
                spawn_pos = self.INIT_BATCH_CENTER_XYZS[batch_id, :]
                # Out of bounds check (Flew too far away horizontally)
                if abs(pos[0] - spawn_pos[0]) > 12.5 or abs(pos[1] - spawn_pos[1]) > 12.5:
                    done[batch_id] = True
                    break

                # 2. Orientation checks (Roll/Pitch in radians)
                roll, pitch = state[7:9]
                if abs(roll) > np.pi or abs(pitch) > np.pi:
                    done[batch_id] = True
                    break


        return done
    
    
    def _computeTruncated(self):

        return (
            self.step_counter >=
            self.EPISODE_LEN_SEC *
            self.PYB_STEPS_PER_CTRL *
            self.CTRL_FREQ
        )

    
    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused as this subclass is not meant for reinforcement learning.

        Returns
        -------
        list[dict[str, int]]
            Dummy value.

        """
        return [{"terminal_observation": None} for _ in range(self.NUM_BATCHES)]
