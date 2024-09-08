import torch
from libero.lifelong.algos.base import Sequential
from my_bc_transformer_policy import MyBCTransformerPolicy

class MyLifelong(Sequential):
    def __init__(self, n_tasks, cfg, my_encdec, **policy_kwargs):
        super().__init__(n_tasks=n_tasks, cfg=cfg, my_encdec=my_encdec, **policy_kwargs)
        self.datasets = []
        self.policy = eval(cfg.policy.policy_type)(cfg, cfg.shape_meta, my_encdec)
        self.my_encdec = my_encdec



    def start_task(self, task):
        super().start_task(task)

    def end_task(self, dataset, task_id, benchmark):
        self.datasets.append(dataset)

    def observe(self, data):
        loss = super().observe(data)
        return loss

    def save_checkpoint(self, filepath, resume_args):
        """
        Save the current model state to a file.
        """

        torch.save({
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'cfg': self.cfg,
            'resume_args': resume_args
        }, filepath)
        print(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath):
        """
        Load the model state from a file.
        """
        checkpoint = torch.load(filepath)
        self.cfg = checkpoint['cfg']
        self.policy.load_state_dict(checkpoint['model_state_dict'])

        self.optimizer = eval(self.cfg.train.optimizer.name)(
            self.policy.parameters(), **self.cfg.train.optimizer.kwargs
        )

        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        resume_args = checkpoint['resume_args']
        self.policy.eval()
        print(f"Checkpoint loaded from {filepath}")
        return self.cfg, resume_args


