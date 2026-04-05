from torch.utils.data.dataloader import DataLoader
from monoscene.data.kitti_360.kitti_360_dataset import Kitti360Dataset
import pytorch_lightning as pl
from monoscene.data.kitti_360.collate import collate_fn
from monoscene.data.utils.torch_util import worker_init_fn


class Kitti360DataModule(pl.LightningDataModule):
    def __init__(self, root, sequences, n_scans, batch_size=4, num_workers=3):
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sequences = sequences
        self.n_scans = n_scans

    def _dataloader_kwargs(self):
        if self.num_workers == 0:
            return {}
        return {
            "worker_init_fn": worker_init_fn,
            "persistent_workers": True,
            "multiprocessing_context": "spawn",
        }

    def setup(self, stage=None):
        self.ds = Kitti360Dataset(
            root=self.root, sequences=self.sequences, n_scans=self.n_scans
        )

    def dataloader(self):
        return DataLoader(
            self.ds,
            batch_size=self.batch_size,
            drop_last=False,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            collate_fn=collate_fn,
            **self._dataloader_kwargs(),
        )
