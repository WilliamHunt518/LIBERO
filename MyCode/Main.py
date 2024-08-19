import os

import numpy as np
import torch

from ConfigLoader import ConfigLoader
from DatasetCreator import DatasetCreator
from MyCode.MyLifelong import MyLifelong
#from MyCode.ModelEval import ModelEval
from MyCode.Visualiser import Visualiser
from MyTrainer import MyTrainer
from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.lifelong.metric import evaluate_loss, evaluate_success
from libero.lifelong.utils import create_experiment_dir, safe_device


class Main:
    def __init__(self):
        self.cfg = None
        self.benchmark = None
        self.algo = None

    def main(self):
        torch.cuda.empty_cache()
    
        # Step 1: Load configuration
        config_loader = ConfigLoader()
        self.cfg = config_loader.get_config()

        # Step 2: Setup benchmark and prepare datasets
        self.cfg.folder = get_libero_path("datasets")
        self.cfg.bddl_folder = get_libero_path("bddl_files")
        self.cfg.init_states_folder = get_libero_path("init_states")
        self.cfg.eval.num_procs = 1
        self.cfg.eval.n_eval = 1
        self.cfg.train.n_epochs = 50
        # Set to 25 in real scenarios

        # Checkpoint saved to ./experiments/miniset_3/Sequential/MyBCTransformerPolicy_seed10000/run_027/final_model.pth
        # /home/will/Documents/Libero/GIT_LOCAL_ROOT/LIBERO/MyCode/tmp_video_orig.mp4

        # You can turn on subprocess
        env_num = 1
        action_dim = 7

        self.cfg.policy.policy_type = "MyBCTransformerPolicy"
        # Registering the policy
        #register_policy(MyTransformerPolicy)

        self.benchmark = get_benchmark(self.cfg.benchmark_name)(self.cfg.data.task_order_index)

        dataset_preparation = DatasetCreator(self.cfg, self.benchmark)
        dataset_preparation.prepare_datasets()
        self.datasets, self.shape_meta = dataset_preparation.get_datasets()

        create_experiment_dir(self.cfg)
        checkpoint_dir = os.path.join(self.cfg.experiment_dir, "checkpoints")

        # Step 3: Initialize and train or load checkpoint
        trainer = MyTrainer(self.cfg, self.shape_meta, self.datasets, self.benchmark, checkpoint_dir)

        load = True
        if load:
            #checkpoint_path = os.path.join(self.cfg.experiment_dir, "../run_014", "checkpoints", "checkpoint_task_2.pth")
            checkpoint_path = "./TrainedModels/run_027/final_model.pth"
            if os.path.exists(checkpoint_path):
                trainer.algo.load_checkpoint(checkpoint_path)
                print("Checkpoint loaded, resuming training...")
            else:
                print("No checkpoint found, exiting")
                quit()
        else:
            print("Training new model from scratch")
            trainer.train()

    # ------ Perform Evaluation ------- #
        # Step 1: Load configuration
        #config_loader = ConfigLoader()
        #cfg = config_loader.get_config()

        # Step 2: Setup benchmark and prepare datasets
        #cfg.folder = get_libero_path("datasets")
        #cfg.bddl_folder = get_libero_path("bddl_files")
        #cfg.init_states_folder = get_libero_path("init_states")
        #cfg.eval.num_procs = 1
        #cfg.eval.n_eval = 1

        #benchmark = get_benchmark(cfg.benchmark_name)(cfg.data.task_order_index)
        #dataset_preparation = DatasetCreator(cfg, benchmark)
        #dataset_preparation.prepare_datasets()
        #datasets, shape_meta = dataset_preparation.get_datasets()

        # Step 5: Perform evaluation
        self.evaluate()

    # --------------------------------- #



        my_visualiser = Visualiser(self.cfg, env_num, action_dim, trainer.algo, self.benchmark)
        my_visualiser.visualize()

    def evaluate(self):
        """
        Perform evaluation on the loaded model.
        """
        print("Evaluating")
        self.algo = safe_device(MyLifelong(self.benchmark.n_tasks, self.cfg), self.cfg.device)

        self.algo.eval()
        gsz = self.cfg.data.task_group_size
        result_summary = {
            'L_conf_mat': np.zeros((self.benchmark.n_tasks, self.benchmark.n_tasks)),
            'S_conf_mat': np.zeros((self.benchmark.n_tasks, self.benchmark.n_tasks)),
            'L_fwd': np.zeros((self.benchmark.n_tasks,)),
            'S_fwd': np.zeros((self.benchmark.n_tasks,)),
        }

        for i in range(self.benchmark.n_tasks):
            print("    -> Evaluating task " + str(i))
            # Evaluate loss and success
            L = evaluate_loss(self.cfg, self.algo, self.benchmark, self.datasets[:i + 1])
            S = evaluate_success(self.cfg, self.algo, self.benchmark, list(range((i + 1) * gsz)))

            result_summary["L_conf_mat"][i][:i + 1] = L
            result_summary["S_conf_mat"][i][:i + 1] = S

            print(f"Task {i}: Loss AUC = {L.mean()}, Success Rate = {S.mean()}")

        # Save the result summary if needed
        torch.save(result_summary, os.path.join(self.cfg.experiment_dir, 'evaluation_result.pt'))
        print("Evaluation results saved to 'evaluation_result.pt'")

if __name__ == "__main__":
    m = Main()
    m.main()

    print("Done")
    m.evaluate()
