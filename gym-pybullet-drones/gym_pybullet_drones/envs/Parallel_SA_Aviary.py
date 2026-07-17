import multiprocessing as mp
import numpy as np
import time
from gym_pybullet_drones.envs.SA_Aviary import SA_Aviary


def _worker(remote, env_kwargs):
    env = None
    try:
        env = SA_Aviary(**env_kwargs)
        while True:
            try:
                cmd, data = remote.recv()
            except EOFError:
                break 

            if cmd == "reset":
                remote.send(env.reset())

            elif cmd == "step":
                remote.send(env.step(data))

            elif cmd == "soft_reset":
                remote.send(env.soft_reset(data))
            
            elif cmd == "set_reset_std":
                env.set_reset_std(*data)
                remote.send(None)

            elif cmd == "get_spaces":
                remote.send((env.action_space, env.observation_space, env.MAX_THRUST, env.GRAVITY))

            elif cmd == "close":
                break
    except KeyboardInterrupt:
        pass
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        remote.close()
    

class Parallel_SA_Aviary():

    def __init__(
        self,
        num_worlds=4,
        num_batches_per_world=9,
        num_drones_per_batch=4,
        **env_kwargs
    ):

        self.num_worlds = num_worlds
        self.num_batches_per_world = num_batches_per_world
        self.num_drones_per_batch = num_drones_per_batch

        self.total_batches = num_worlds * num_batches_per_world
        self.num_envs = self.total_batches
        self.total_drones = self.total_batches * num_drones_per_batch

        self.use_mp = num_worlds > 1
        
        self.reward_range = (0, +np.inf)
        self.remotes = []
        self.processes = []

        if not self.use_mp:
            self.env = SA_Aviary(
                num_batches_per_world=num_batches_per_world,
                num_drones_per_batch=num_drones_per_batch,
                **env_kwargs
            )

            self.action_space = self.env.action_space
            self.observation_space = self.env.observation_space
            self.MAX_THRUST = self.env.MAX_THRUST
            self.GRAVITY = self.env.GRAVITY
            return


        for world_id in range(num_worlds):

            parent_remote, child_remote = mp.Pipe()

            kwargs = dict(env_kwargs)
            kwargs["num_batches_per_world"] = num_batches_per_world
            kwargs["num_drones_per_batch"] = num_drones_per_batch

            if world_id != 0:
                kwargs["gui"] = False

            p = mp.Process(
                target=_worker,
                args=(child_remote, kwargs)
            )

            p.daemon = True
            p.start()

            child_remote.close()

            self.remotes.append(parent_remote)
            self.processes.append(p)

        self.remotes[0].send(("get_spaces", None))
        actionspace, obsspace, max_thrust, gravity = self.remotes[0].recv()

        self.action_space = actionspace
        self.observation_space = obsspace
        self.MAX_THRUST = max_thrust
        self.GRAVITY = gravity

    def step(self, actions):

        actions = np.asarray(actions, dtype=np.float32)
        if not self.use_mp:
            return self.env.step(actions)

        expected = self.total_batches

        if actions.shape[0] != expected:
            raise ValueError(
                f"Expected {expected} actions but got {actions.shape[0]}"
            )

        split_actions = []

        for world_id in range(self.num_worlds):
            start = world_id * self.num_batches_per_world
            end = (world_id + 1) * self.num_batches_per_world
            split_actions.append(actions[start:end])

        for remote, act in zip(self.remotes, split_actions):
            remote.send(("step", act))

        results = [r.recv() for r in self.remotes]

        obs = np.concatenate([r[0] for r in results], axis=0)
        reward = np.concatenate([r[1] for r in results], axis=0)
        terminated = np.concatenate([r[2] for r in results], axis=0)
        truncated = np.concatenate([r[3] for r in results], axis=0)

        infos = []
        for r in results:
            infos.extend(r[4])

        return obs, reward, terminated, truncated, infos

    ###########################################################################
    # SOFT RESET
    def set_reset_std(self, stdxy = 0.0, stdz = 0.0, stdrp = 0.0, stdy = 0.0, 
                      stddxy = 0.0, stddz = 0.0, stddrp = 0.0, stddy = 0.0, 
                      stddvxy = 0.0, stddvz = 0.0, stddvrp = 0.0, stddvy = 0.0, 
                      stdpvxy = 0.0, stdpvz = 0.0, stdpvrp = 0.0, stdpvy = 0.0, 
                      min_mass_prc = 0.0, max_mass_prc = 0.0, 
                      stdspxy = 0.0, stdspz = 0.0, stdsprp = 0.0, stdspy = 0.0):

        if not self.use_mp:
            self.env.set_reset_std(stdxy, stdz, stdrp, stdy, stddxy, stddz, stddrp, stddy, stddvxy, stddvz, stddvrp, stddvy, stdpvxy, stdpvz, stdpvrp, stdpvy, min_mass_prc, max_mass_prc, stdspxy, stdspz, stdsprp, stdspy)
            return

        for remote in self.remotes:
            remote.send(("set_reset_std", (stdxy, stdz, stdrp, stdy, stddxy, stddz, stddrp, stddy, stddvxy, stddvz, stddvrp, stddvy, stdpvxy, stdpvz, stdpvrp, stdpvy, min_mass_prc, max_mass_prc, stdspxy, stdspz, stdsprp, stdspy)))
        for remote in self.remotes:
            remote.recv()

    def reset(self):

        if not self.use_mp:
            return self.env.reset()

        for remote in self.remotes:
            remote.send(("reset", None))

        
        results = [r.recv() for r in self.remotes]

        obs = np.concatenate([r[0] for r in results], axis=0)
        
        infos = []
        for r in results:
            infos.extend(r[1])

        return obs, infos


    def close(self):
        print("Closing ParallelAviary...")
        if not self.use_mp:
            self.env.close()
            return

        for remote in self.remotes:
            remote.send(("close", None))

        time.sleep(1)

        for p in self.processes:
            p.join()
