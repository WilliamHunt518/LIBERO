import os
import time

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
        #self.cfg.eval.n_eval = 1
        # Set to 25 in real scenarios

        # Checkpoint saved to ./experiments/miniset_1/Sequential/MyBCTransformerPolicy_seed10000/run_027/final_model.pth
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
        self.trainer = MyTrainer(self.cfg, self.shape_meta, self.datasets, self.benchmark, checkpoint_dir)



        NUM_TRAINING_LOOPS = 1 #(We did 60 in sequence for run 010)
        self.cfg.train.n_epochs = 5 # We alter the training code so this is # epochs for a task within the higher loop
        # ~3.5min per epoch
        # 8-10hr target. n_epoch * NUM_TRAINING_LOOPS * 3.5/60 = 10
        # 8-10hr target. n_epoch * NUM_TRAINING_LOOPS * 0.05 = 10
        # 8-10hr target. n_epoch * NUM_TRAINING_LOOPS = 200

        print("With chkpt dir = " + checkpoint_dir)

        load = False
        if load:
            checkpoint_path = os.path.join(self.cfg.experiment_dir, "../run_016", "final_model_1.pth")
            #checkpoint_path = "./TrainedModels/run_027/checkpoints/checkpoint_task_0.pth"
            if os.path.exists(checkpoint_path):
                self.trainer.algo.load_checkpoint(checkpoint_path)
                print("Checkpoint loaded, resuming training...")
            else:
                print("No checkpoint found, exiting")
                quit()
        else:
            print("Training new model from scratch")
            print(f"Training loops: {NUM_TRAINING_LOOPS}")
            print(f"Epochs: {self.cfg.train.n_epochs}")
            start = time.time()

            for i in range(1, NUM_TRAINING_LOOPS + 1):
                print(f"Run num {i}")
                elapsed = time.time() - start
                print(f"Elapsed time: {elapsed}, and we are {i}/{NUM_TRAINING_LOOPS}")

                avg_time_per_iteration = elapsed / i
                eta = avg_time_per_iteration * (NUM_TRAINING_LOOPS + 1 - i)#( elapsed * (NUM_TRAINING_LOOPS / i) )
                print(f"This means we have {eta} seconds left")

                self.trainer.train(i)

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
        #UR5e: Task 0: Loss AUC = 5.2222746775700495, Success Rate = 0.0
        #Panda:Task 0: Loss AUC = 5.296631717681885,  Success Rate = 0.0


    # --------------------------------- #



        #my_visualiser = Visualiser(self.cfg, env_num, action_dim, trainer.algo, self.benchmark)
        #my_visualiser.visualize()

    def evaluate(self):
        """
        Perform evaluation on the loaded model.
        """
        print("Evaluating")
        #self.algo = safe_device(MyLifelong(self.benchmark.n_tasks, self.cfg), self.cfg.device)

        self.trainer.algo.eval()
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
            L = evaluate_loss(self.cfg, self.trainer.algo, self.benchmark, self.datasets[:i + 1])
            S = evaluate_success(self.cfg, self.trainer.algo, self.benchmark, list(range((i + 1) * gsz)))
            #S = evaluate_success(self.cfg, self.algo, self.benchmark, [0])

            result_summary["L_conf_mat"][i][:i + 1] = L
            result_summary["S_conf_mat"][i][:i + 1] = S

            #print(f"Task {i}: Loss AUC = {L.mean()}, Success Rate = {S.mean()}")
            print(f"Task {i}: Success Rate = {S.mean()}")

        # Save the result summary if needed
        torch.save(result_summary, os.path.join(self.cfg.experiment_dir, 'evaluation_result.pt'))
        print("Evaluation results saved to 'evaluation_result.pt'")

if __name__ == "__main__":
    m = Main()
    m.main()

    print("Done")
