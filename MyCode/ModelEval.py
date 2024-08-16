import os

import numpy as np
import torch
from ConfigLoader import ConfigLoader
from DatasetCreator import DatasetCreator
from MyLifelong import MyLifelong
from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.lifelong.utils import safe_device
from libero.lifelong.metric import evaluate_loss, evaluate_success

class ModelEval:
    def __init__(self, cfg, shape_meta, datasets, benchmark):
        self.cfg = cfg
        self.shape_meta = shape_meta
        self.datasets = datasets
        self.benchmark = benchmark
        self.algo = safe_device(MyLifelong(self.benchmark.n_tasks, self.cfg), self.cfg.device)

    def load_model(self, checkpoint_path):
        """
        Load the model from the specified checkpoint.
        """
        if os.path.exists(checkpoint_path):
            self.algo.load_checkpoint(checkpoint_path)
            print(f"Checkpoint loaded from {checkpoint_path}")
        else:
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    def evaluate(self):
        """
        Perform evaluation on the loaded model.
        """
        self.algo.eval()
        gsz = self.cfg.data.task_group_size
        result_summary = {
            'L_conf_mat': np.zeros((self.benchmark.n_tasks, self.benchmark.n_tasks)),
            'S_conf_mat': np.zeros((self.benchmark.n_tasks, self.benchmark.n_tasks)),
            'L_fwd': np.zeros((self.benchmark.n_tasks,)),
            'S_fwd': np.zeros((self.benchmark.n_tasks,)),
        }

        for i in range(self.benchmark.n_tasks):
            # Evaluate loss and success
            L = evaluate_loss(self.cfg, self.algo, self.benchmark, self.datasets[:i + 1])
            S = evaluate_success(self.cfg, self.algo, self.benchmark, list(range((i + 1) * gsz)))

            result_summary["L_conf_mat"][i][:i + 1] = L
            result_summary["S_conf_mat"][i][:i + 1] = S

            print(f"Task {i}: Loss AUC = {L.mean()}, Success Rate = {S.mean()}")

        # Save the result summary if needed
        torch.save(result_summary, os.path.join(self.cfg.experiment_dir, 'evaluation_result.pt'))
        print("Evaluation results saved to 'evaluation_result.pt'")

def main():
    # Step 1: Load configuration
    config_loader = ConfigLoader()
    cfg = config_loader.get_config()

    # Step 2: Setup benchmark and prepare datasets
    cfg.folder = get_libero_path("datasets")
    cfg.bddl_folder = get_libero_path("bddl_files")
    cfg.init_states_folder = get_libero_path("init_states")
    cfg.eval.num_procs = 1
    cfg.eval.n_eval = 1

    benchmark = get_benchmark(cfg.benchmark_name)(cfg.data.task_order_index)
    dataset_preparation = DatasetCreator(cfg, benchmark)
    dataset_preparation.prepare_datasets()
    datasets, shape_meta = dataset_preparation.get_datasets()

    # Step 3: Initialize evaluator
    evaluator = ModelEval(cfg, shape_meta, datasets, benchmark)

    # Step 4: Load the model checkpoint
    checkpoint_path = os.path.join(cfg.experiment_dir, "final_model.pth")
    evaluator.load_model(checkpoint_path)

    # Step 5: Perform evaluation
    evaluator.evaluate()

if __name__ == "__main__":
    main()
