import os
import numpy as np
import torch
from tqdm import trange

from MyCode.MyLifelong import MyLifelong
from libero.lifelong.utils import create_experiment_dir, safe_device
from libero.lifelong.metric import evaluate_loss, evaluate_success

class MyTrainer:
    def __init__(self, cfg, shape_meta, datasets, benchmark, checkpoint_dir, my_encdec):
        self.cfg = cfg
        self.shape_meta = shape_meta
        self.datasets = datasets
        self.benchmark = benchmark
        self.result_summary = self._initialize_result_summary()

        # These steps permit loading without yet calling train()
        self.checkpoint_dir = checkpoint_dir
        self.cfg.shape_meta = self.shape_meta
        self.algo = safe_device(MyLifelong(self.benchmark.n_tasks, self.cfg, my_encdec), self.cfg.device)

        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _initialize_result_summary(self):
        n_tasks = self.benchmark.n_tasks
        return {
            'L_conf_mat': np.zeros((n_tasks, n_tasks)),
            'S_conf_mat': np.zeros((n_tasks, n_tasks)),
            'L_fwd': np.zeros((n_tasks,)),
            'S_fwd': np.zeros((n_tasks,)),
        }

    def train(self, run_num, checkpoint_dir, my_encdec=None, resume_args=None, n_epochs=5, robots=None):

        self.checkpoint_dir = os.path.join(self.cfg.experiment_dir, "checkpoints")
        self.cfg.shape_meta = self.shape_meta

        # Initialize MyLifelong with the updated cfg
        self.algo = safe_device(MyLifelong(self.benchmark.n_tasks, self.cfg, my_encdec=my_encdec), self.cfg.device)

        if checkpoint_dir is not None:
            print("Loading model from " + str(checkpoint_dir))
            self.algo.load_checkpoint(checkpoint_dir)
            self.cfg.train.n_epochs = n_epochs
            # TODO load my_encdec here also

        gsz = self.cfg.data.task_group_size

        for i in trange(self.benchmark.n_tasks):
            print("Training for task " + str(i))
            print("self.cfg = " + str(self.cfg))

            # print("Pre_eval")
            # S = evaluate_success(self.cfg, self.algo, self.benchmark, list(range((i + 1) * gsz)))
            # self.result_summary["S_conf_mat"][i][:i + 1] = S
            #
            # torch.save(self.result_summary, os.path.join(self.cfg.experiment_dir, 'result.pt'))
            # print("done")
            ## NOTE: At this point of the first run it's fine. Not sure why not after

            self.algo.train()
            s_fwd, l_fwd, resume_args = self.algo.learn_one_task(self.datasets[i], i, self.benchmark, self.result_summary, resume_args, robots=robots)
            self.result_summary["S_fwd"][i] = s_fwd
            self.result_summary["L_fwd"][i] = l_fwd

            if self.cfg.eval.eval:
                self.algo.eval()
                L = evaluate_loss(self.cfg, self.algo, self.benchmark, self.datasets[:i + 1])
                S = evaluate_success(self.cfg, self.algo, self.benchmark, list(range((i + 1) * gsz)))
                self.result_summary["L_conf_mat"][i][:i + 1] = L
                self.result_summary["S_conf_mat"][i][:i + 1] = S

                torch.save(self.result_summary, os.path.join(self.cfg.experiment_dir, 'result.pt'))

        # Save final model
        final_model_path = os.path.join(self.cfg.experiment_dir, f"final_model_{run_num}.pth")
        self.algo.save_checkpoint(final_model_path, resume_args)
