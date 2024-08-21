import os
import imageio
import numpy as np
import torch
from IPython.display import HTML
from base64 import b64encode

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv, DummyVectorEnv
from libero.lifelong.metric import raw_obs_to_tensor_obs


class Visualiser:
    def __init__(self, cfg, env_num, action_dim, algo, benchmark):
        """
        Initialize the Visualiser with the given configuration, environment number, and action dimension.

        :param cfg: Configuration object
        :param env_num: Number of environments
        :param action_dim: Dimension of the action space
        """
        self.cfg = cfg
        self.env_num = env_num
        self.action_dim = action_dim
        self.algo = algo
        self.benchmark = benchmark

    def visualize(self):
        """
        Visualize the task by running a trained model and rendering the environment.
        """
        task_id = 0

        task = self.benchmark.get_task(task_id)
        task_emb = self.benchmark.get_task_emb(task_id)

        if self.cfg.lifelong.algo == "PackNet":
            algo = self.algo.get_eval_algo(task_id)

        self.algo.eval()
        env_args = {
            "bddl_file_name": os.path.join(
                self.cfg.bddl_folder, task.problem_folder, task.bddl_file
            ),
            "camera_heights": self.cfg.data.img_h,
            "camera_widths": self.cfg.data.img_w,
        }

        env = DummyVectorEnv(
            [lambda: OffScreenRenderEnv(**env_args) for _ in range(self.env_num)]
        )

        # TODO This probably needs the same change as the other code

        init_states_path = os.path.join(
            self.cfg.init_states_folder, task.problem_folder, task.init_states_file
        )
        init_states = torch.load(init_states_path)

        env.reset()
        init_state = init_states[0:1]
        dones = [False] * self.env_num

        self.algo.reset()
        obs = env.set_init_state(init_state)

        dummy_actions = np.zeros((self.env_num, self.action_dim))
        for _ in range(5):
            obs, _, _, _ = env.step(dummy_actions)

        steps = 0
        obs_tensors = [[] for _ in range(self.env_num)]
        while steps < self.cfg.eval.max_steps:
            steps += 1
            data = raw_obs_to_tensor_obs(obs, task_emb, self.cfg)
            action = self.algo.policy.get_action(data)
            obs, reward, done, info = env.step(action)

            for k in range(self.env_num):
                dones[k] = dones[k] or done[k]
                obs_tensors[k].append(obs[k]["agentview_image"])
            if all(dones):
                break

        images = [img[::-1] for img in obs_tensors[0]]
        fps = 30
        writer = imageio.get_writer('tmp_video.mp4', fps=fps)
        for image in images:
            writer.append_data(image)
        writer.close()

        video_data = open("tmp_video.mp4", "rb").read()
        video_tag = f'<video controls alt="test" src="data:video/mp4;base64,{b64encode(video_data).decode()}">'
        HTML(data=video_tag)

        video_path = os.path.abspath("tmp_video.mp4")
        print(video_path)
