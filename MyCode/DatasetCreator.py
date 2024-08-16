import os
from libero.libero import get_libero_path
from libero.lifelong.datasets import SequenceVLDataset, get_dataset
from libero.lifelong.utils import get_task_embs

class DatasetCreator:
    def __init__(self, cfg, benchmark):
        self.cfg = cfg
        self.benchmark = benchmark
        self.datasets = []
        self.descriptions = []
        self.shape_meta = None

    def prepare_datasets(self):
        n_tasks = self.benchmark.n_tasks
        for i in range(n_tasks):
            task_i_dataset, shape_meta = get_dataset(
                dataset_path=os.path.join(self.cfg.folder, self.benchmark.get_task_demonstration(i)),
                obs_modality=self.cfg.data.obs.modality,
                initialize_obs_utils=(i == 0),
                seq_len=self.cfg.data.seq_len,
            )
            self.descriptions.append(self.benchmark.get_task(i).language)
            self.datasets.append(task_i_dataset)

        self.shape_meta = shape_meta
        task_embs = get_task_embs(self.cfg, self.descriptions)
        self.benchmark.set_task_embs(task_embs)
        self.datasets = [SequenceVLDataset(ds, emb) for (ds, emb) in zip(self.datasets, task_embs)]

    def get_datasets(self):
        return self.datasets, self.shape_meta