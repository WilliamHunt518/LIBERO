import os

import torch

from ConfigLoader import ConfigLoader
from DatasetCreator import DatasetCreator
from MyCode.Visualiser import Visualiser
from MyTrainer import MyTrainer
from libero.libero import get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.lifelong.utils import create_experiment_dir


def main():
    torch.cuda.empty_cache()

    # Step 1: Load configuration
    config_loader = ConfigLoader()
    cfg = config_loader.get_config()

    # Step 2: Setup benchmark and prepare datasets
    cfg.folder = get_libero_path("datasets")
    cfg.bddl_folder = get_libero_path("bddl_files")
    cfg.init_states_folder = get_libero_path("init_states")
    cfg.eval.num_procs = 1
    cfg.eval.n_eval = 1
    cfg.train.n_epochs = 50
    # Set to 25 in real scenarios

    # Checkpoint saved to ./experiments/miniset_3/Sequential/MyBCTransformerPolicy_seed10000/run_027/final_model.pth
    # /home/will/Documents/Libero/GIT_LOCAL_ROOT/LIBERO/MyCode/tmp_video.mp4

    # You can turn on subprocess
    env_num = 1
    action_dim = 7

    cfg.policy.policy_type = "MyBCTransformerPolicy"
    # Registering the policy
    #register_policy(MyTransformerPolicy)

    benchmark = get_benchmark(cfg.benchmark_name)(cfg.data.task_order_index)
    dataset_preparation = DatasetCreator(cfg, benchmark)
    dataset_preparation.prepare_datasets()
    datasets, shape_meta = dataset_preparation.get_datasets()

    create_experiment_dir(cfg)
    checkpoint_dir = os.path.join(cfg.experiment_dir, "checkpoints")

    # Step 3: Initialize and train or load checkpoint
    trainer = MyTrainer(cfg, shape_meta, datasets, benchmark, checkpoint_dir)

    load = False
    if load:
        checkpoint_path = os.path.join(cfg.experiment_dir, "../run_014", "checkpoints", "checkpoint_task_2.pth")
        if os.path.exists(checkpoint_path):
            trainer.algo.load_checkpoint(checkpoint_path)
            print("Checkpoint loaded, resuming training...")
        else:
            print("No checkpoint found, exiting")
            quit()
    else:
        print("Training new model from scratch")
        trainer.train()

    my_visualiser = Visualiser(cfg, env_num, action_dim, trainer.algo, benchmark)
    my_visualiser.visualize()



if __name__ == "__main__":
    main()
