from sys import platform
import time
from collections import deque
import xml.etree.ElementTree as etxml
import pkg_resources
import numpy as np
import pybullet as p
import pybullet_data
from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.control.ForceControl import ForceControl

default_target = [
            np.array([-1,  1, 4]),
            np.array([0,0,0,1]),
        ]
class BaseAviary():
    """Base class"""

    # metadata = {'render.modes': ['human']}
    
    ################################################################################

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
        
        self.EPISODE_LEN_SEC = episode_len_sec
        self.rope_vis_ids = [-1] * (num_drones_per_batch * num_batches_per_world)
        self.REST_LENGTH = cable_lenght
        self.STIFFNESS = 100.0
        self.DAMPING = 1.5
        self.MAX_TENSION = 20.0

        self.drone_attach_points = np.array([
            [0.0, 0.0, -0.0],
            [0.0, 0.0, -0.0],
            [0.0, 0.0, -0.0],
            [0.0, 0.0, -0.0],
        ])

        URDF_TREE = etxml.parse(pkg_resources.resource_filename('gym_pybullet_drones', 'assets/'+"small_box.urdf")).getroot()
        self.PAYLOAD_MASS = float(URDF_TREE[1][0][1].attrib['value'])
        self.PAYLOAD_L1 = float(URDF_TREE[1][1][1][0].attrib['size'].split(' ')[0])
        self.PAYLOAD_L2 = float(URDF_TREE[1][1][1][0].attrib['size'].split(' ')[1])
        self.PAYLOAD_L3 = float(URDF_TREE[1][1][1][0].attrib['size'].split(' ')[2])
        #print("[INFO] BaseAviary.__init__() loaded the payload URDF and extracted the following parameters:\n[INFO] mass {:f}kg, length {:f}m".format(self.PAYLOAD_MASS, self.PAYLOAD_LENGTH))


        self.payload_corners = np.array([
            [-self.PAYLOAD_L1/2,  self.PAYLOAD_L2/2, self.PAYLOAD_L3/2],
            [ self.PAYLOAD_L1/2,  self.PAYLOAD_L2/2, self.PAYLOAD_L3/2],
            [ self.PAYLOAD_L1/2, -self.PAYLOAD_L2/2, self.PAYLOAD_L3/2],
            [-self.PAYLOAD_L1/2, -self.PAYLOAD_L2/2, self.PAYLOAD_L3/2],
        ])
                
        #### Constants #############################################
        self.G = 9.8
        
        self.RAD2DEG = 180/np.pi
        self.DEG2RAD = np.pi/180
        self.CTRL_FREQ = ctrl_freq
        self.PYB_FREQ = pyb_freq
        if self.PYB_FREQ % self.CTRL_FREQ != 0:
            raise ValueError('[ERROR] in BaseAviary.__init__(), pyb_freq is not divisible by env_freq.')
        self.PYB_STEPS_PER_CTRL = int(self.PYB_FREQ / self.CTRL_FREQ)
        self.CTRL_TIMESTEP = 1. / self.CTRL_FREQ
        self.PYB_TIMESTEP = 1. / self.PYB_FREQ
        self.max_steps = int(self.EPISODE_LEN_SEC * self.CTRL_FREQ * self.PYB_STEPS_PER_CTRL)
        #### Parameters ############################################
        self.NUM_DRONES_PER_BATCH = num_drones_per_batch
        self.NUM_BATCHES = num_batches_per_world
        if self.NUM_BATCHES > 0:
            self.NUM_DRONES = num_drones_per_batch * num_batches_per_world
        else:
            self.NUM_DRONES = num_drones_per_batch    
        self.NEIGHBOURHOOD_RADIUS = neighbourhood_radius
        #### Options ###############################################
        self.DRONE_MODEL = drone_model
        self.ctrls = [ForceControl(drone_model=self.DRONE_MODEL) for _ in range(self.NUM_DRONES)]
        self.GUI = gui
        self.PHYSICS = physics
        self.USER_DEBUG = user_debug_gui
        self.URDF = self.DRONE_MODEL.value + ".urdf"
        self.OUTPUT_FOLDER = output_folder
        #### Load the drone properties from the .urdf file #########
        self.M, \
        self.L, \
        self.THRUST2WEIGHT_RATIO, \
        self.J, \
        self.J_INV, \
        self.KF, \
        self.KM, \
        self.COLLISION_H,\
        self.COLLISION_R, \
        self.COLLISION_Z_OFFSET, \
        self.MAX_SPEED_KMH, \
        self.GND_EFF_COEFF, \
        self.PROP_RADIUS, \
        self.DRAG_COEFF, \
        self.DW_COEFF_1, \
        self.DW_COEFF_2, \
        self.DW_COEFF_3 = self._parseURDFParameters()
        #print("[INFO] BaseAviary.__init__() loaded parameters from the drone's .urdf:\n[INFO] m {:f}, L {:f},\n[INFO] ixx {:f}, iyy {:f}, izz {:f},\n[INFO] kf {:e}, km {:e},\n[INFO] t2w {:f}, max_speed_kmh {:f},\n[INFO] gnd_eff_coeff {:f}, prop_radius {:f},\n[INFO] drag_xy_coeff {:f}, drag_z_coeff {:f},\n[INFO] dw_coeff_1 {:f}, dw_coeff_2 {:f}, dw_coeff_3 {:f}".format(
        #   self.M, self.L, self.J[0,0], self.J[1,1], self.J[2,2], self.KF, self.KM, self.THRUST2WEIGHT_RATIO, self.MAX_SPEED_KMH, self.GND_EFF_COEFF, self.PROP_RADIUS, self.DRAG_COEFF[0], self.DRAG_COEFF[2], self.DW_COEFF_1, self.DW_COEFF_2, self.DW_COEFF_3))
        #### Compute constants #####################################
        self.GRAVITY = self.G*self.M
        self.HOVER_RPM = np.sqrt(self.GRAVITY / (4*self.KF))
        self.MAX_RPM = np.sqrt((self.THRUST2WEIGHT_RATIO*self.GRAVITY) / (4*self.KF))
        self.MAX_THRUST = (4*self.KF*self.MAX_RPM**2)
        #print("[INFO] BaseAviary.__init__() computed the following constants:\n[INFO] gravity {:f}, hover_rpm {:f}, max_rpm {:f}, max_thrust {:f}".format(
        #    self.GRAVITY, self.HOVER_RPM, self.MAX_RPM, self.MAX_THRUST))
        if self.DRONE_MODEL == DroneModel.CF2X:
            self.MAX_XY_TORQUE = (2*self.L*self.KF*self.MAX_RPM**2)/np.sqrt(2)
        elif self.DRONE_MODEL == DroneModel.CF2P:
            self.MAX_XY_TORQUE = (self.L*self.KF*self.MAX_RPM**2)
        elif self.DRONE_MODEL == DroneModel.RACE:
            self.MAX_XY_TORQUE = (2*self.L*self.KF*self.MAX_RPM**2)/np.sqrt(2)
        self.MAX_Z_TORQUE = (2*self.KM*self.MAX_RPM**2)
        self.GND_EFF_H_CLIP = 0.25 * self.PROP_RADIUS * np.sqrt((15 * self.MAX_RPM**2 * self.KF * self.GND_EFF_COEFF) / self.MAX_THRUST)
        
        #### Connect to PyBullet ###################################
        if self.GUI:
            #### With debug GUI ########################################
            self.CLIENT = p.connect(p.GUI) # p.connect(p.GUI, options="--opengl2")
            for i in [p.COV_ENABLE_RGB_BUFFER_PREVIEW, p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, p.COV_ENABLE_GUI]:
                p.configureDebugVisualizer(i, 0, physicsClientId=self.CLIENT)
            p.resetDebugVisualizerCamera(cameraDistance=3,
                                         cameraYaw=-30,
                                         cameraPitch=-30,
                                         cameraTargetPosition=[0, 0, 10],
                                         physicsClientId=self.CLIENT
                                         )
            ret = p.getDebugVisualizerCamera(physicsClientId=self.CLIENT)
            #print("viewMatrix", ret[2])
            #print("projectionMatrix", ret[3])
        else:
            #### Without debug GUI #####################################
            self.CLIENT = p.connect(p.DIRECT)
            #### Uncomment the following line to use EGL Render Plugin #
            #### Instead of TinyRender (CPU-based) in PYB's Direct mode
            # if platform == "linux": p.setAdditionalSearchPath(pybullet_data.getDataPath()); plugin = p.loadPlugin(egl.get_filename(), "_eglRendererPlugin"); print("plugin=", plugin)
        
        #### Set initial poses #####################################
        if initial_xyzs_batch is None:
            self.INIT_XYZS_BATCH = np.vstack([np.array([x*4*self.L for x in range(self.NUM_DRONES)]), \
                                        np.array([y*4*self.L for y in range(self.NUM_DRONES)]), \
                                        np.ones(self.NUM_DRONES) * (self.COLLISION_H/2-self.COLLISION_Z_OFFSET+.1)]).transpose().reshape(self.NUM_DRONES, 3)
        elif np.array(initial_xyzs_batch).shape == (self.NUM_DRONES_PER_BATCH, 3):
            self.INIT_XYZS_BATCH = initial_xyzs_batch
        else:
            print("[ERROR] invalid initial_xyzs_batch in BaseAviary.__init__(), try initial_xyzs_batch.reshape(NUM_DRONES_PER_BATCH,3)")

        if num_batches_per_world > 0:
            if initial_batch_center_xyzs is None:
                self.INIT_BATCH_CENTER_XYZS = np.zeros((self.NUM_BATCHES, 3))
            elif np.array(initial_batch_center_xyzs).shape == (self.NUM_BATCHES, 3):
                self.INIT_BATCH_CENTER_XYZS = initial_batch_center_xyzs
            else:
                print("[ERROR] invalid initial_batch_center_xyzs in BaseAviary.__init__(), try initial_batch_center_xyzs.reshape(NUM_BATCHES,3)")
        else:
            self.INIT_BATCH_CENTER_XYZS = np.array([[0.0, 0.0, 0.0]])

        if initial_rpys_batch is None:
            self.INIT_RPYS_BATCH = np.zeros((self.NUM_BATCHES, 3))
        elif np.array(initial_rpys_batch).shape == (self.NUM_DRONES_PER_BATCH, 3):
            self.INIT_RPYS_BATCH = initial_rpys_batch
        else:
            print("[ERROR] invalid initial_rpys_batch in BaseAviary.__init__(), try initial_rpys_batch.reshape(NUM_DRONES_PER_BATCH,3)")
        #### Create action and observation spaces ##################
        self.action_space = self._actionSpace()
        self.observation_space = self._observationSpace()
        #### Housekeeping ##########################################
        self._housekeeping()
        #### Update and store the drones kinematic information #####
        self._updateAndStoreKinematicInformation()

        self.STATE_DIM = 69
        self.HISTORY_LEN = 1

        self.history = [
            deque(maxlen=self.HISTORY_LEN)
            for _ in range(self.NUM_DRONES)
        ]

        self.PAYLOAD_TARGET_POS = [target_pos[0] for _ in range(self.NUM_BATCHES)]
        self.PAYLOAD_TARGET_ORN = [target_pos[1] for _ in range(self.NUM_BATCHES)]
        self.DEFAULT_TARGET_POS = target_pos[0]
        self.DEFAULT_TARGET_ORN = target_pos[1]
        self.last_actions = np.zeros((self.NUM_BATCHES, self.NUM_DRONES_PER_BATCH*3))
        self.set_reset_std()

        if self.GUI:
            self.TGT_X_AX = [-1 for _ in range(self.NUM_BATCHES)]
            self.TGT_Y_AX = [-1 for _ in range(self.NUM_BATCHES)]
            self.TGT_Z_AX = [-1 for _ in range(self.NUM_BATCHES)]

            #draw a marker at the target position
            for b in range(self.NUM_BATCHES):
                self._draw_target_axes(b)
           
    
    ################################################################################
    def _draw_target_axes(self, b):
                AXIS_LENGTH = 2 * self.PAYLOAD_L1  # length of the axes lines, adjust as needed
                
                pos = self.PAYLOAD_TARGET_POS[b] + self.INIT_BATCH_CENTER_XYZS[b]  # target position in world coordinates
                orn = self.PAYLOAD_TARGET_ORN[b]
                
                # Get 3x3 rotation matrix (flat list of 9 elements)
                rot_matrix = p.getMatrixFromQuaternion(orn, physicsClientId=self.CLIENT)
                
                # Map out the X, Y, Z vector directions explicitly 
                x_dir = np.array([rot_matrix[0], rot_matrix[3], rot_matrix[6]])
                y_dir = np.array([rot_matrix[1], rot_matrix[4], rot_matrix[7]])
                z_dir = np.array([rot_matrix[2], rot_matrix[5], rot_matrix[8]])
                
                line_to_x = (np.array(pos) + AXIS_LENGTH * x_dir).tolist()
                line_to_y = (np.array(pos) + AXIS_LENGTH * y_dir).tolist()
                line_to_z = (np.array(pos) + AXIS_LENGTH * z_dir).tolist()
                
                # --- Draw X Axis (Red) ---
                self.TGT_X_AX[b] = p.addUserDebugLine(
                    lineFromXYZ=pos,
                    lineToXYZ=line_to_x,
                    lineColorRGB=[1, 0, 0],
                    replaceItemUniqueId=int(self.TGT_X_AX[b]),
                    physicsClientId=self.CLIENT
                )
                
                # --- Draw Y Axis (Green) ---
                self.TGT_Y_AX[b] = p.addUserDebugLine(
                    lineFromXYZ=pos,
                    lineToXYZ=line_to_y,
                    lineColorRGB=[0, 1, 0],
                    replaceItemUniqueId=int(self.TGT_Y_AX[b]),
                    physicsClientId=self.CLIENT
                )
                
                # --- Draw Z Axis (Blue) ---
                self.TGT_Z_AX[b] = p.addUserDebugLine(
                    lineFromXYZ=pos,
                    lineToXYZ=line_to_z,
                    lineColorRGB=[0, 0, 1],
                    replaceItemUniqueId=int(self.TGT_Z_AX[b]),
                    physicsClientId=self.CLIENT
                )

    def set_reset_std(self, stdxy = 0.0, stdz = 0.0, stdrp = 0.0, stdy = 0.0, 
                      stddxy = 0.0, stddz = 0.0, stddrp = 0.0, stddy = 0.0, 
                      stddvxy = 0.0, stddvz = 0.0, stddvrp = 0.0, stddvy = 0.0, 
                      stdpvxy = 0.0, stdpvz = 0.0, stdpvrp = 0.0, stdpvy = 0.0, 
                      min_mass_prc = 1, max_mass_prc = 1, 
                      stdspxy = 0.0, stdspz = 0.0, stdsprp = 0.0, stdspy = 0.0):
        #payload pos
        self.stdxy = stdxy
        self.stdz = stdz

        self.stdrp = stdrp
        self.stdy = stdy

        #drone pos
        self.stddxy = stddxy
        self.stddz = stddz

        self.stddrp = stddrp
        self.stddy = stddy

        #drone vel
        self.stddvxy = stddvxy
        self.stddvz = stddvz

        self.stddvrp = stddvrp
        self.stddvy = stddvy

        #payload vel
        self.stdpvxy = stdpvxy
        self.stdpvz = stdpvz

        self.stdpvrp = stdpvrp
        self.stdpvy = stdpvy

        #payload mass
        self.min_mass_prc = min_mass_prc
        self.max_mass_prc = max_mass_prc

        self.stdspxy = stdspxy
        self.stdspz = stdspz

        self.stdsprp = stdsprp
        self.stdspy = stdspy

    def _reset_batch(self, batch_idx):
        # episode bookkeeping
        self.RESET_TIME = time.time()
        self.step_counter[batch_idx] = 0
        # clear runtime buffers
        self.last_clipped_action[batch_idx * self.NUM_DRONES_PER_BATCH:(batch_idx + 1) * self.NUM_DRONES_PER_BATCH] = 0.0
        self.last_actions[batch_idx, :] = 0.0
    
        if self.stdxy > 0.0 and self.stdz > 0.0:
            payload_offset = np.random.normal(0.0, [self.stdxy, self.stdxy, self.stdz])
        else:
            payload_offset = np.zeros(3)
        if self.stdrp > 0.0 and self.stdy > 0.0:
            rpy_p_jitter = np.random.normal(0.0, [self.stdrp, self.stdrp, self.stdy])
        else:
            rpy_p_jitter = np.zeros(3)
        payload_rpy_jitter = p.getQuaternionFromEuler(rpy_p_jitter)
        payload_offset[2] = abs(payload_offset[2])
        randomized_center = self.INIT_BATCH_CENTER_XYZS[batch_idx] + payload_offset

        # reset drones
        for i in range(self.NUM_DRONES_PER_BATCH * batch_idx, self.NUM_DRONES_PER_BATCH * (batch_idx+1)):


            drone_idx = i % self.NUM_DRONES_PER_BATCH

            if self.stdxy > 0.0 and self.stdz > 0.0:
                drone_jitter = np.random.normal(0.0, [self.stddxy, self.stddxy, self.stddz])
            else:
                drone_jitter = np.zeros(3)
            if self.stddrp > 0.0 and self.stddy > 0.0:          
                rpy_jitter = np.random.normal(0.0, [self.stddrp, self.stddrp, self.stddy])
            else:
                rpy_jitter = np.zeros(3)
            if self.stddvxy > 0.0 and self.stddvz > 0.0:
                vel_jitter = np.random.normal(0.0, [self.stddvxy, self.stddvxy, self.stddvz])
            else:
                vel_jitter = np.zeros(3)
            if self.stddvrp > 0.0 and self.stddvy > 0.0:
                angl_vel_jitter = np.random.normal(0.0, [self.stddvrp, self.stddvrp, self.stddvy])
            else:
                angl_vel_jitter = np.zeros(3)

            init_pos = (
                self.INIT_XYZS_BATCH[drone_idx]
                + self.INIT_BATCH_CENTER_XYZS[batch_idx]
                + drone_jitter
            )

            init_quat = p.getQuaternionFromEuler(
                self.INIT_RPYS_BATCH[drone_idx] + rpy_jitter
            )

            p.resetBasePositionAndOrientation(
                self.DRONE_IDS[i],
                init_pos,
                init_quat,
                physicsClientId=self.CLIENT
            )

            p.resetBaseVelocity(
                self.DRONE_IDS[i],
                vel_jitter,
                angl_vel_jitter,
                physicsClientId=self.CLIENT
            )

            #reset controllers
            self.ctrls[i].reset()

        payload_pos = [
            randomized_center[0],
            randomized_center[1],
            randomized_center[2] + 0.036
        ]

        p.resetBasePositionAndOrientation(
            self.PAYLOAD_ID[batch_idx],
            payload_pos,
            payload_rpy_jitter,
            physicsClientId=self.CLIENT
        )

        if self.min_mass_prc != self.max_mass_prc:
            random_mass = np.random.uniform(
                self.min_mass_prc * self.PAYLOAD_MASS,
                self.max_mass_prc * self.PAYLOAD_MASS
            )
        else:
            random_mass = self.min_mass_prc * self.PAYLOAD_MASS

        p.changeDynamics(
            self.PAYLOAD_ID[batch_idx],
            linkIndex=-1,  # -1 represents the base link
            mass=random_mass,
            physicsClientId=self.CLIENT
        )

        if self.stdpvxy > 0.0 and self.stdpvz > 0.0:
            payload_vel_jitter = np.random.normal(0.0, [self.stdpvxy, self.stdpvz, self.stdpvrp])
        else:
            payload_vel_jitter = np.zeros(3)
        if self.stdpvrp > 0.0 and self.stdpvy > 0.0:
            payload_angl_vel_jitter = np.random.normal(0.0, [self.stdpvrp, self.stdpvrp, self.stdpvy])
        else:
            payload_angl_vel_jitter = np.zeros(3)
        p.resetBaseVelocity(
            self.PAYLOAD_ID[batch_idx],
            payload_vel_jitter,
            payload_angl_vel_jitter,
            physicsClientId=self.CLIENT
        )

        #randomize target position
        if self.stdspxy > 0.0 and self.stdspz > 0.0: #uniform
            target_offset = np.random.uniform(
                low=[-self.stdspxy, -self.stdspxy, -self.stdspz],
                high=[self.stdspxy, self.stdspxy, self.stdspz]
            )
        else:
            target_offset = np.zeros(3)
        if self.stdsprp > 0.0 and self.stdspy > 0.0: #uniform
            target_rpy_jitter = np.random.uniform(
                low=[-self.stdsprp, -self.stdsprp, -self.stdspy],
                high=[self.stdsprp, self.stdsprp, self.stdspy]
            )
        else:
            target_rpy_jitter = np.zeros(3)

        self.PAYLOAD_TARGET_POS[batch_idx] = self.DEFAULT_TARGET_POS + target_offset
        self.PAYLOAD_TARGET_ORN[batch_idx] = p.getQuaternionFromEuler(
            np.array(p.getEulerFromQuaternion(self.DEFAULT_TARGET_ORN)) + target_rpy_jitter
        )

        #update target axes visualization
        if self.GUI:
            self._draw_target_axes(batch_idx)

        # reset DYN physics state
        if self.PHYSICS == Physics.DYN:
            self.rpy_rates[batch_idx * self.NUM_DRONES_PER_BATCH:(batch_idx + 1) * self.NUM_DRONES_PER_BATCH] = 0.0

        # refresh cached state
        self._updateAndStoreKinematicInformation()
        

    def reset(self,
              seed : int = None,
              options : dict = None):
        """Resets the environment.

        Parameters
        ----------
        seed : int, optional
            Random seed.
        options : dict[..], optional
            Additional options, unused

        Returns
        -------
        ndarray | dict[..]
            The initial observation, check the specific implementation of `_computeObs()`
            in each subclass for its format.
        dict[..]
            Additional information as a dictionary, check the specific implementation of `_computeInfo()`
            in each subclass for its format.

        """
        for batch_idx in range(self.NUM_BATCHES):
            self._reset_batch(batch_idx)
        #one settle step
        p.stepSimulation(physicsClientId=self.CLIENT)
        #### Return the initial observation ########################
        initial_obs = self._computeObs()
        initial_info = self._computeInfo()
        return initial_obs, initial_info
    
    ################################################################################

    def _world_point(self, body_pos, body_orn, local_point):
        rot = np.array(p.getMatrixFromQuaternion(body_orn)).reshape(3, 3)
        return np.array(body_pos) + rot @ np.array(local_point)


    def _tension_to_color(self, tension, max_tension):
        t = np.clip(tension/self.GRAVITY/2 , 0.0, 1.0)
    
        # blue → red gradient
        r = t
        g = 0.0
        b = 1.0 - t

        return [r, g, b]

    def _drawCable(self, drone_id, batch_id):
        if self.GUI:
            drone_pos, drone_orn = p.getBasePositionAndOrientation(
                self.DRONE_IDS[drone_id],
                physicsClientId=self.CLIENT
            )

            payload_pos, payload_orn = p.getBasePositionAndOrientation(
                self.PAYLOAD_ID[batch_id],
                physicsClientId=self.CLIENT
            )

            def world_point(pos, orn, local):
                rot = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
                return np.array(pos) + rot @ np.array(local)

            pA = self._world_point(drone_pos, drone_orn, self.drone_attach_points[drone_id%self.NUM_DRONES_PER_BATCH])
            pB = self._world_point(payload_pos, payload_orn, self.payload_corners[drone_id%self.NUM_DRONES_PER_BATCH])

            color = self._tension_to_color(
                self.cable_tensions[drone_id],
                self.MAX_TENSION
            )

            self.rope_vis_ids[drone_id] = p.addUserDebugLine(
                pA,
                pB,
                lineColorRGB=color,
                lineWidth=2,
                lifeTime=0,
                replaceItemUniqueId=self.rope_vis_ids[drone_id],
                physicsClientId=self.CLIENT
            )

    def _applyCableForces(self, drone_id, batch_id):
        # Drone state
        drone_pos, drone_orn = p.getBasePositionAndOrientation(
            self.DRONE_IDS[drone_id],
            physicsClientId=self.CLIENT
        )

        drone_vel, _ = p.getBaseVelocity(
            self.DRONE_IDS[drone_id],
            physicsClientId=self.CLIENT
        )

        # Payload state
        payload_pos, payload_orn = p.getBasePositionAndOrientation(
            self.PAYLOAD_ID[batch_id],
            physicsClientId=self.CLIENT
        )

        payload_vel, _ = p.getBaseVelocity(
            self.PAYLOAD_ID[batch_id],
            physicsClientId=self.CLIENT
        )

        pA = self._world_point(drone_pos, drone_orn, self.drone_attach_points[drone_id%self.NUM_DRONES_PER_BATCH])
        pB = self._world_point(payload_pos, payload_orn, self.payload_corners[drone_id%self.NUM_DRONES_PER_BATCH])

        # Cable force model
        delta = pB - pA
        dist = np.linalg.norm(delta)

        if dist < 1e-8:
            return

        direction = delta / dist
        extension = dist - self.REST_LENGTH

        if extension <= 0:
            return

        rel_vel = np.dot(np.array(payload_vel) - np.array(drone_vel), direction)

        tension = self.STIFFNESS * extension + self.DAMPING * rel_vel

        tension = max(0.0, min(self.MAX_TENSION, tension))
        self.cable_tensions[drone_id] = tension
        force = tension * direction

        p.applyExternalForce(
            objectUniqueId=self.DRONE_IDS[drone_id],
            linkIndex=-1,
            forceObj=force.tolist(),
            posObj=pA.tolist(),
            flags=p.WORLD_FRAME,
            physicsClientId=self.CLIENT
        )

        p.applyExternalForce(
            objectUniqueId=self.PAYLOAD_ID[batch_id],
            linkIndex=-1,
            forceObj=(-force).tolist(),
            posObj=pB.tolist(),
            flags=p.WORLD_FRAME,
            physicsClientId=self.CLIENT
        )



    def step(self, action):
        # PROCESS ACTION
        clipped_action = np.reshape(
            self._preprocessAction(action),
            (self.NUM_DRONES, 4)
        )

        # MULTI SUBSTEP LOOP
        self.cable_tensions = np.zeros(self.NUM_DRONES)
        for _ in range(self.PYB_STEPS_PER_CTRL):
            # DRONE PHYSICS
            for i in range(self.NUM_DRONES):
                self._physics(clipped_action[i, :], i)

            if self.NUM_BATCHES > 0:
                for i in range(self.NUM_DRONES):
                    self._applyCableForces(i, i // self.NUM_DRONES_PER_BATCH)

            # STEP SIMULATION
            p.stepSimulation(physicsClientId=self.CLIENT)

            # VISUALIZATION (rope line)
            if self.NUM_BATCHES > 0:
                for i in range(self.NUM_DRONES):
                    self._drawCable(i, i // self.NUM_DRONES_PER_BATCH)
                
            # STORE KINEMATICS
            self._updateAndStoreKinematicInformation()
            self.last_clipped_action = clipped_action

        # OUTPUTS
        terminated = self._computeTerminated()
        truncated = self._computeTruncated()
        info = self._computeInfo()
        obs = self._computeObs()
        reward = self._computeReward(terminated, action, clipped_action)

        self.step_counter += self.PYB_STEPS_PER_CTRL

        for batch_idx in range(self.NUM_BATCHES):
            if np.any(terminated[batch_idx]) or np.any(truncated[batch_idx]):
                info[batch_idx]["terminal_observation"] = obs[batch_idx]
                self._reset_batch(batch_idx)

        self.last_actions = action
        return obs, reward, terminated, truncated, info
    

    ################################################################################

    def close(self):
        """Terminates the environment.
        """
        p.disconnect(physicsClientId=self.CLIENT)
    
    ################################################################################

    def getPyBulletClient(self):
        """Returns the PyBullet Client Id.

        Returns
        -------
        int:
            The PyBullet Client Id.

        """
        return self.CLIENT
    
    ################################################################################

    def getDroneIds(self):
        """Return the Drone Ids.

        Returns
        -------
        ndarray:
            (NUM_DRONES,)-shaped array of ints containing the drones' ids.

        """
        return self.DRONE_IDS
    
    ################################################################################

    def _housekeeping(self):
        """Housekeeping function.

        Allocation and zero-ing of the variables and PyBullet's parameters/objects
        in the `reset()` function.

        """
        #### Initialize/reset counters and zero-valued variables ###
        self.RESET_TIME = time.time()
        self.step_counter = np.array([0]*self.NUM_BATCHES)
        self.first_render_call = True
        self.X_AX = -1*np.ones(self.NUM_DRONES)
        self.Y_AX = -1*np.ones(self.NUM_DRONES)
        self.Z_AX = -1*np.ones(self.NUM_DRONES)
        self.X_AX_P = -1*np.ones(self.NUM_BATCHES)
        self.Y_AX_P = -1*np.ones(self.NUM_BATCHES)
        self.Z_AX_P = -1*np.ones(self.NUM_BATCHES)
        self.GUI_INPUT_TEXT = -1*np.ones(self.NUM_DRONES)
        self.USE_GUI_RPM=False
        self.last_input_switch = 0
        self.last_clipped_action = np.zeros((self.NUM_DRONES, 4))
        self.gui_input = np.zeros(4)
        #### Initialize the drones kinemaatic information ##########
        self.pos = np.zeros((self.NUM_DRONES, 3))
        self.quat = np.zeros((self.NUM_DRONES, 4))
        self.rpy = np.zeros((self.NUM_DRONES, 3))
        self.vel = np.zeros((self.NUM_DRONES, 3))
        self.ang_v = np.zeros((self.NUM_DRONES, 3))
        if self.PHYSICS == Physics.DYN:
            self.rpy_rates = np.zeros((self.NUM_DRONES, 3))
        #### Set PyBullet's parameters #############################
        p.setGravity(0, 0, -self.G, physicsClientId=self.CLIENT)
        p.setRealTimeSimulation(0, physicsClientId=self.CLIENT)
        p.setTimeStep(self.PYB_TIMESTEP, physicsClientId=self.CLIENT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.CLIENT)

        self.DRONE_IDS = np.array([p.loadURDF(pkg_resources.resource_filename('gym_pybullet_drones', 'assets/'+self.URDF),
                                              self.INIT_XYZS_BATCH[i%self.NUM_DRONES_PER_BATCH,:] + self.INIT_BATCH_CENTER_XYZS[i//self.NUM_DRONES_PER_BATCH,:],
                                              p.getQuaternionFromEuler(self.INIT_RPYS_BATCH[i%self.NUM_DRONES_PER_BATCH,:]),
                                              flags = p.URDF_USE_INERTIA_FROM_FILE,
                                              physicsClientId=self.CLIENT
                                              ) for i in range(self.NUM_DRONES)])

        if self.NUM_BATCHES > 0:
            self.PAYLOAD_ID = [self._addPayload(i) for i in range(self.NUM_BATCHES)]

        if self.GUI and self.USER_DEBUG:
            for i in range(self.NUM_DRONES):
                self._showDroneLocalAxes(i)
            for i in range(self.NUM_BATCHES):
                self._showPayloadLocalAxes(i)

    ################################################################################

    def _updateAndStoreKinematicInformation(self):
        """Updates and stores the drones kinemaatic information.

        This method is meant to limit the number of calls to PyBullet in each step
        and improve performance (at the expense of memory).

        """
        for i in range (self.NUM_DRONES):
            self.pos[i], self.quat[i] = p.getBasePositionAndOrientation(self.DRONE_IDS[i], physicsClientId=self.CLIENT)
            self.rpy[i] = p.getEulerFromQuaternion(self.quat[i])
            self.vel[i], self.ang_v[i] = p.getBaseVelocity(self.DRONE_IDS[i], physicsClientId=self.CLIENT)
    
    ################################################################################

    def quaternion_conjugate(self, q):
        """Returns the conjugate of a quaternion q = [w, x, y, z]"""
        return np.array([q[0], -q[1], -q[2], -q[3]])


    def quaternion_multiply(self, q1, q2):
        """Multiplies two quaternions (q1 Hamilton product q2)"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return np.array([w, x, y, z])


    def quaternion_error_magnitude(self, qG, qL):
        """Calculates theta(qG, qL), the norm of the axis-angle representation

        of the quaternion difference qG (*) qL_conjugate.
        """
        # 1. Compute the conjugate of qL
        qL_conj = self.quaternion_conjugate(qL)

        # 2. Compute the quaternion difference (qG otimes qL*)
        q_diff = self.quaternion_multiply(qG, qL_conj)

        # Ensure the scalar part is within valid clipping range for arccos due to float precision
        w = np.clip(q_diff[0], -1.0, 1.0)

        # 3. Calculate the angle
        # The angle of rotation alpha = 2 * arccos(w)
        angle = 2 * np.arccos(w)

        # Normalize the angle to stay within [-pi, pi]
        angle = (angle + np.pi) % (2 * np.pi) - np.pi

        return np.abs(angle)

    def _getDroneStateVector(self,
                             nth_drone
                             ):
        """Returns the state vector of the n-th drone.

        Parameters
        ----------
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        Returns
        -------
        ndarray 
            (20,)-shaped array of floats containing the state vector of the n-th drone.
            Check the only line in this method and `_updateAndStoreKinematicInformation()`
            to understand its format.

        """
        state = np.hstack([self.pos[nth_drone, :], self.quat[nth_drone, :], self.rpy[nth_drone, :],
                           self.vel[nth_drone, :], self.ang_v[nth_drone, :], self.last_clipped_action[nth_drone, :]])
        return state.reshape(20,)

    ################################################################################

    

    ################################################################################

    def _getAdjacencyMatrix(self):
        """Computes the adjacency matrix of a multi-drone system.

        Attribute NEIGHBOURHOOD_RADIUS is used to determine neighboring relationships.

        Returns
        -------
        ndarray
            (NUM_DRONES, NUM_DRONES)-shaped array of 0's and 1's representing the adjacency matrix 
            of the system: adj_mat[i,j] == 1 if (i, j) are neighbors; == 0 otherwise.

        """
        adjacency_mat = np.identity(self.NUM_DRONES)
        for i in range(self.NUM_DRONES-1):
            for j in range(self.NUM_DRONES-i-1):
                if np.linalg.norm(self.pos[i, :]-self.pos[j+i+1, :]) < self.NEIGHBOURHOOD_RADIUS:
                    adjacency_mat[i, j+i+1] = adjacency_mat[j+i+1, i] = 1
        return adjacency_mat
    
    ################################################################################
    
    def _physics(self,
                 rpm,
                 nth_drone
                 ):
        """Base PyBullet physics implementation.

        Parameters
        ----------
        rpm : ndarray
            (4)-shaped array of ints containing the RPMs values of the 4 motors.
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        """
        forces = np.array(rpm**2)*self.KF
        torques = np.array(rpm**2)*self.KM
        if self.DRONE_MODEL == DroneModel.RACE:
            torques = -torques
        z_torque = (-torques[0] + torques[1] - torques[2] + torques[3])
        for i in range(4):
            p.applyExternalForce(self.DRONE_IDS[nth_drone],
                                 i,
                                 forceObj=[0, 0, forces[i]],
                                 posObj=[0, 0, 0],
                                 flags=p.LINK_FRAME,
                                 physicsClientId=self.CLIENT
                                 )
        p.applyExternalTorque(self.DRONE_IDS[nth_drone],
                              4,
                              torqueObj=[0, 0, z_torque],
                              flags=p.LINK_FRAME,
                              physicsClientId=self.CLIENT
                              )
    

    ################################################################################

    def _groundEffect(self,
                      rpm,
                      nth_drone
                      ):
        """PyBullet implementation of a ground effect model.

        Inspired by the analytical model used for comparison in (Shi et al., 2019).

        Parameters
        ----------
        rpm : ndarray
            (4)-shaped array of ints containing the RPMs values of the 4 motors.
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        """
        #### Kin. info of all links (propellers and center of mass)
        link_states = p.getLinkStates(self.DRONE_IDS[nth_drone],
                                        linkIndices=[0, 1, 2, 3, 4],
                                        computeLinkVelocity=1,
                                        computeForwardKinematics=1,
                                        physicsClientId=self.CLIENT
                                        )
        #### Simple, per-propeller ground effects ##################
        prop_heights = np.array([link_states[0][0][2], link_states[1][0][2], link_states[2][0][2], link_states[3][0][2]])
        prop_heights = np.clip(prop_heights, self.GND_EFF_H_CLIP, np.inf)
        gnd_effects = np.array(rpm**2) * self.KF * self.GND_EFF_COEFF * (self.PROP_RADIUS/(4 * prop_heights))**2
        if np.abs(self.rpy[nth_drone,0]) < np.pi/2 and np.abs(self.rpy[nth_drone,1]) < np.pi/2:
            for i in range(4):
                p.applyExternalForce(self.DRONE_IDS[nth_drone],
                                     i,
                                     forceObj=[0, 0, gnd_effects[i]],
                                     posObj=[0, 0, 0],
                                     flags=p.LINK_FRAME,
                                     physicsClientId=self.CLIENT
                                     )
    
    ################################################################################

    def _drag(self,
              rpm,
              nth_drone
              ):
        """PyBullet implementation of a drag model.

        Based on the the system identification in (Forster, 2015).

        
        Parameters
        ----------
        rpm : ndarray
            (4)-shaped array of ints containing the RPMs values of the 4 motors.
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        """
        #### Rotation matrix of the base ###########################
        base_rot = np.array(p.getMatrixFromQuaternion(self.quat[nth_drone, :])).reshape(3, 3)
        #### Simple draft model applied to the base/center of mass #
        drag_factors = -1 * self.DRAG_COEFF * np.sum(np.array(2*np.pi*rpm/60))
        drag = np.dot(base_rot.T, drag_factors*np.array(self.vel[nth_drone, :]))
        p.applyExternalForce(self.DRONE_IDS[nth_drone],
                             4,
                             forceObj=drag,
                             posObj=[0, 0, 0],
                             flags=p.LINK_FRAME,
                             physicsClientId=self.CLIENT
                             )
    
    ################################################################################

    def _downwash(self,
                  nth_drone
                  ):
        """PyBullet implementation of a ground effect model.

        Based on experiments conducted at the Dynamic Systems Lab by SiQi Zhou.

        Parameters
        ----------
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        """
        for i in range(self.NUM_DRONES):
            delta_z = self.pos[i, 2] - self.pos[nth_drone, 2]
            delta_xy = np.linalg.norm(np.array(self.pos[i, 0:2]) - np.array(self.pos[nth_drone, 0:2]))
            if delta_z > 0 and delta_xy < 10: # Ignore drones more than 10 meters away
                alpha = self.DW_COEFF_1 * (self.PROP_RADIUS/(4*delta_z))**2
                beta = self.DW_COEFF_2 * delta_z + self.DW_COEFF_3
                downwash = [0, 0, -alpha * np.exp(-.5*(delta_xy/beta)**2)]
                p.applyExternalForce(self.DRONE_IDS[nth_drone],
                                     4,
                                     forceObj=downwash,
                                     posObj=[0, 0, 0],
                                     flags=p.LINK_FRAME,
                                     physicsClientId=self.CLIENT
                                     )

    ################################################################################

    def _dynamics(self,
                  rpm,
                  nth_drone
                  ):
        """Explicit dynamics implementation.

        Based on code written at the Dynamic Systems Lab by James Xu.

        Parameters
        ----------
        rpm : ndarray
            (4)-shaped array of ints containing the RPMs values of the 4 motors.
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        """
        #### Current state #########################################
        pos = self.pos[nth_drone,:]
        quat = self.quat[nth_drone,:]
        vel = self.vel[nth_drone,:]
        rpy_rates = self.rpy_rates[nth_drone,:]
        rotation = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
        #### Compute forces and torques ############################
        forces = np.array(rpm**2) * self.KF
        thrust = np.array([0, 0, np.sum(forces)])
        thrust_world_frame = np.dot(rotation, thrust)
        force_world_frame = thrust_world_frame - np.array([0, 0, self.GRAVITY])
        z_torques = np.array(rpm**2)*self.KM
        if self.DRONE_MODEL == DroneModel.RACE:
            z_torques = -z_torques
        z_torque = (-z_torques[0] + z_torques[1] - z_torques[2] + z_torques[3])
        if self.DRONE_MODEL==DroneModel.RACE:
            x_torque = (forces[0] + forces[1] - forces[2] - forces[3]) * (self.L/np.sqrt(2))
            y_torque = (- forces[0] + forces[1] + forces[2] - forces[3]) * (self.L/np.sqrt(2))
        elif self.DRONE_MODEL==DroneModel.CF2X:
            x_torque = - (forces[0] + forces[1] - forces[2] - forces[3]) * (self.L/np.sqrt(2))
            y_torque = (- forces[0] + forces[1] + forces[2] - forces[3]) * (self.L/np.sqrt(2))
        elif self.DRONE_MODEL==DroneModel.CF2P:
            x_torque = (forces[1] - forces[3]) * self.L
            y_torque = (-forces[0] + forces[2]) * self.L
        torques = np.array([x_torque, y_torque, z_torque])
        torques = torques - np.cross(rpy_rates, np.dot(self.J, rpy_rates))
        rpy_rates_deriv = np.dot(self.J_INV, torques)
        no_pybullet_dyn_accs = force_world_frame / self.M
        #### Update state ##########################################
        vel = vel + self.PYB_TIMESTEP * no_pybullet_dyn_accs
        rpy_rates = rpy_rates + self.PYB_TIMESTEP * rpy_rates_deriv
        pos = pos + self.PYB_TIMESTEP * vel
        quat = self._integrateQ(quat, rpy_rates, self.PYB_TIMESTEP)
        #### Set PyBullet's state ##################################
        p.resetBasePositionAndOrientation(self.DRONE_IDS[nth_drone],
                                          pos,
                                          quat,
                                          physicsClientId=self.CLIENT
                                          )
        #### Note: the base's velocity only stored and not used ####
        p.resetBaseVelocity(self.DRONE_IDS[nth_drone],
                            vel,
                            np.dot(rotation, rpy_rates),
                            physicsClientId=self.CLIENT
                            )
        #### Store the roll, pitch, yaw rates for the next step ####
        self.rpy_rates[nth_drone,:] = rpy_rates

    def _integrateQ(self, quat, omega, dt):
        omega_norm = np.linalg.norm(omega)
        p, q, r = omega
        if np.isclose(omega_norm, 0):
            return quat
        lambda_ = np.array([
            [ 0,  r, -q, p],
            [-r,  0,  p, q],
            [ q, -p,  0, r],
            [-p, -q, -r, 0]
        ]) * .5
        theta = omega_norm * dt / 2
        quat = np.dot(np.eye(4) * np.cos(theta) + 2 / omega_norm * lambda_ * np.sin(theta), quat)
        return quat

    ################################################################################

    def _showDroneLocalAxes(self,
                            nth_drone
                            ):
        """Draws the local frame of the n-th drone in PyBullet's GUI.

        Parameters
        ----------
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        """
        if self.GUI:
            AXIS_LENGTH = 2*self.L
            self.X_AX[nth_drone] = p.addUserDebugLine(lineFromXYZ=[0, 0, 0],
                                                      lineToXYZ=[AXIS_LENGTH, 0, 0],
                                                      lineColorRGB=[1, 0, 0],
                                                      parentObjectUniqueId=self.DRONE_IDS[nth_drone],
                                                      parentLinkIndex=-1,
                                                      replaceItemUniqueId=int(self.X_AX[nth_drone]),
                                                      physicsClientId=self.CLIENT
                                                      )
            self.Y_AX[nth_drone] = p.addUserDebugLine(lineFromXYZ=[0, 0, 0],
                                                      lineToXYZ=[0, AXIS_LENGTH, 0],
                                                      lineColorRGB=[0, 1, 0],
                                                      parentObjectUniqueId=self.DRONE_IDS[nth_drone],
                                                      parentLinkIndex=-1,
                                                      replaceItemUniqueId=int(self.Y_AX[nth_drone]),
                                                      physicsClientId=self.CLIENT
                                                      )
            self.Z_AX[nth_drone] = p.addUserDebugLine(lineFromXYZ=[0, 0, 0],
                                                      lineToXYZ=[0, 0, AXIS_LENGTH],
                                                      lineColorRGB=[0, 0, 1],
                                                      parentObjectUniqueId=self.DRONE_IDS[nth_drone],
                                                      parentLinkIndex=-1,
                                                      replaceItemUniqueId=int(self.Z_AX[nth_drone]),
                                                      physicsClientId=self.CLIENT
                                                      )

    def _showPayloadLocalAxes(self,
                            batch_id,
                            ):
        """Draws the local frame of the n-th drone in PyBullet's GUI.

        Parameters
        ----------
        nth_drone : int
            The ordinal number/position of the desired drone in list self.DRONE_IDS.

        """
        if self.GUI:
            AXIS_LENGTH = 2*self.PAYLOAD_L1
            self.X_AX_P[batch_id] = p.addUserDebugLine(lineFromXYZ=[0, 0, 0],
                                                      lineToXYZ=[AXIS_LENGTH, 0, 0],
                                                      lineColorRGB=[1, 0, 0],
                                                      parentObjectUniqueId=self.PAYLOAD_ID[batch_id],
                                                      parentLinkIndex=-1,
                                                      replaceItemUniqueId=int(self.X_AX_P[batch_id]),
                                                      physicsClientId=self.CLIENT
                                                      )
            self.Y_AX_P[batch_id] = p.addUserDebugLine(lineFromXYZ=[0, 0, 0],
                                                      lineToXYZ=[0, AXIS_LENGTH, 0],
                                                      lineColorRGB=[0, 1, 0],
                                                      parentObjectUniqueId=self.PAYLOAD_ID[batch_id],
                                                      parentLinkIndex=-1,
                                                      replaceItemUniqueId=int(self.Y_AX_P[batch_id]),
                                                      physicsClientId=self.CLIENT
                                                      )
            self.Z_AX_P[batch_id] = p.addUserDebugLine(lineFromXYZ=[0, 0, 0],
                                                      lineToXYZ=[0, 0, AXIS_LENGTH],
                                                      lineColorRGB=[0, 0, 1],
                                                      parentObjectUniqueId=self.PAYLOAD_ID[batch_id],
                                                      parentLinkIndex=-1,
                                                      replaceItemUniqueId=int(self.Z_AX_P[batch_id]),
                                                      physicsClientId=self.CLIENT
                                                      )
    
    
    def _addPayload(self, batch_id):
        """Add payload to the environment.

        This payload is loaded from a standard URDF file included in Bullet.

        """
        return p.loadURDF(pkg_resources.resource_filename('gym_pybullet_drones', 'assets/small_box.urdf'),
                                              [self.INIT_BATCH_CENTER_XYZS[batch_id, 0], self.INIT_BATCH_CENTER_XYZS[batch_id, 1], self.INIT_BATCH_CENTER_XYZS[batch_id, 2] + 0.036],
                                              p.getQuaternionFromEuler([0,0,0]),
                                              flags = p.URDF_USE_INERTIA_FROM_FILE,
                                              physicsClientId=self.CLIENT
                                              ) 
    
    def _parseURDFParameters(self):
        """Loads parameters from an URDF file.

        This method is nothing more than a custom XML parser for the .urdf
        files in folder `assets/`.

        """
        URDF_TREE = etxml.parse(pkg_resources.resource_filename('gym_pybullet_drones', 'assets/'+self.URDF)).getroot()
        M = float(URDF_TREE[1][0][1].attrib['value'])
        L = float(URDF_TREE[0].attrib['arm'])
        THRUST2WEIGHT_RATIO = float(URDF_TREE[0].attrib['thrust2weight'])
        IXX = float(URDF_TREE[1][0][2].attrib['ixx'])
        IYY = float(URDF_TREE[1][0][2].attrib['iyy'])
        IZZ = float(URDF_TREE[1][0][2].attrib['izz'])
        J = np.diag([IXX, IYY, IZZ])
        J_INV = np.linalg.inv(J)
        KF = float(URDF_TREE[0].attrib['kf'])
        KM = float(URDF_TREE[0].attrib['km'])
        COLLISION_H = float(URDF_TREE[1][2][1][0].attrib['length'])
        COLLISION_R = float(URDF_TREE[1][2][1][0].attrib['radius'])
        COLLISION_SHAPE_OFFSETS = [float(s) for s in URDF_TREE[1][2][0].attrib['xyz'].split(' ')]
        COLLISION_Z_OFFSET = COLLISION_SHAPE_OFFSETS[2]
        MAX_SPEED_KMH = float(URDF_TREE[0].attrib['max_speed_kmh'])
        GND_EFF_COEFF = float(URDF_TREE[0].attrib['gnd_eff_coeff'])
        PROP_RADIUS = float(URDF_TREE[0].attrib['prop_radius'])
        DRAG_COEFF_XY = float(URDF_TREE[0].attrib['drag_coeff_xy'])
        DRAG_COEFF_Z = float(URDF_TREE[0].attrib['drag_coeff_z'])
        DRAG_COEFF = np.array([DRAG_COEFF_XY, DRAG_COEFF_XY, DRAG_COEFF_Z])
        DW_COEFF_1 = float(URDF_TREE[0].attrib['dw_coeff_1'])
        DW_COEFF_2 = float(URDF_TREE[0].attrib['dw_coeff_2'])
        DW_COEFF_3 = float(URDF_TREE[0].attrib['dw_coeff_3'])
        return M, L, THRUST2WEIGHT_RATIO, J, J_INV, KF, KM, COLLISION_H, COLLISION_R, COLLISION_Z_OFFSET, MAX_SPEED_KMH, \
               GND_EFF_COEFF, PROP_RADIUS, DRAG_COEFF, DW_COEFF_1, DW_COEFF_2, DW_COEFF_3
    
    def _preprocessAction(self, action):
        """Translates the network's [12] batch command values into [4] motor RPMs for each drone."""
        rpm_actions = np.zeros((self.NUM_DRONES, 4))

        for i in range(self.NUM_BATCHES):
            for j in range(self.NUM_DRONES_PER_BATCH):
                state = self._getDroneStateVector(i * self.NUM_DRONES_PER_BATCH + j)

                target_thrust = self.GRAVITY + action[i, 0 + j*3]
                target_rpy = np.array([action[i, 1 + j*3], action[i, 2 + j*3], 0.0])

                rpm_actions[i * self.NUM_DRONES_PER_BATCH + j, :] = self.ctrls[i * self.NUM_DRONES_PER_BATCH + j].computeControlFromState(
                    control_timestep=self.CTRL_TIMESTEP,
                    state=state,
                    target_thrust=target_thrust,
                    target_rpy=target_rpy,
                )

        return rpm_actions
    
    ################################################################################

    def _actionSpace(self):
        raise NotImplementedError("The action space is defined by the network's output and is not used for sampling random actions. The method `_preprocessAction()` translates the network's output into motor RPMs.")
    
    
    def _observationSpace(self):
        raise NotImplementedError("The observation space is defined by the network's input and is not used for sampling random observations. The method `_computeObs()` translates the environment's state into the network's input format.")

    def _computeObs(self):
        raise NotImplementedError("The observation space is defined by the network's input and is not used for sampling random observations. The method `_computeObs()` translates the environment's state into the network's input format.")
    
    def _computeReward(self, terminated, action, clipped_action, lambda1 = 1.5, lambda2 = 0.5, lambda3 = 1.0, lambda4 = 1.5, lambda5 = 0.25, lambda6 = 0.5, lambda7 = 2.0):
        raise NotImplementedError("The reward function is not defined in this base class. This method should be implemented in a subclass according to the task at hand. The parameters lambda1-lambda7 are just suggestions for weighting different components of the reward and can be modified or ignored as needed.")
    
    
    def _computeTerminated(self):
        raise NotImplementedError("The termination condition is not defined in this base class. This method should be implemented in a subclass according to the task at hand.")
    
    def _computeTruncated(self):
        raise NotImplementedError("The truncation condition is not defined in this base class. This method should be implemented in a subclass according to the task at hand.")

    
    def _computeInfo(self):
        raise NotImplementedError("The info dictionary is not defined in this base class. This method should be implemented in a subclass according to the task at hand.")

