import torch
from libero.lifelong.algos.base import Sequential
from my_bc_transformer_policy import MyBCTransformerPolicy

class MyLifelong(Sequential):
    def __init__(self, n_tasks, cfg, **policy_kwargs):
        super().__init__(n_tasks=n_tasks, cfg=cfg, **policy_kwargs)
        self.datasets = []
        self.policy = eval(cfg.policy.policy_type)(cfg, cfg.shape_meta)

    def start_task(self, task):
        super().start_task(task)

    def end_task(self, dataset, task_id, benchmark):
        self.datasets.append(dataset)

    def observe(self, data):
        loss = super().observe(data)
        return loss

    def save_checkpoint(self, filepath):
        """
        Save the current model state to a file.
        """
        torch.save({
            'model_state_dict': self.policy.state_dict(),
            #'optimizer_state_dict': self.optimizer.state_dict(),
            'cfg': self.cfg,
            #'task_idx': self.task_idx,
        }, filepath)
        print(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath):
        """
        Load the model state from a file.
        """
        checkpoint = torch.load(filepath)
        #self.policy.load_state_dict(checkpoint['model_state_dict'])
        #self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.cfg = checkpoint['cfg']
        #self.task_idx = checkpoint['task_idx']
        print(f"Checkpoint loaded from {filepath}")
