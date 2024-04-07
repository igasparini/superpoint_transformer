import logging
from src.datamodules.base import BaseDataModule
from src.datasets import CustomData, MiniCustomData


log = logging.getLogger(__name__)


class CustomDataModule(BaseDataModule):
    """LightningDataModule for a custom dataset.

    A DataModule implements 5 key methods:

        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html
    """
    _DATASET_CLASS = CustomData
    _MINIDATASET_CLASS = MiniCustomData

    # def prepare_data(self):
    #     """Download and heavy preprocessing of data should be triggered here."""
    #     self.dataset_class(
    #         self.hparams.data_dir, stage=self.train_stage,
    #         transform=self.train_transform, pre_transform=self.pre_transform,
    #         on_device_transform=self.on_device_train_transform, **self.kwargs)

    #     self.dataset_class(
    #         self.hparams.data_dir, stage=self.val_stage,
    #         transform=self.val_transform, pre_transform=self.pre_transform,
    #         on_device_transform=self.on_device_val_transform, **self.kwargs)

    #     self.dataset_class(
    #         self.hparams.data_dir, stage='test',
    #         transform=self.test_transform, pre_transform=self.pre_transform,
    #         on_device_transform=self.on_device_test_transform, **self.kwargs)

    # def setup(self, stage=None):
    #     """Load data. Set variables: `self.train_dataset`, `self.val_dataset`, `self.test_dataset`."""
        
    #     self.train_dataset = self.dataset_class(
    #         self.hparams.data_dir, stage=self.train_stage,
    #         transform=self.train_transform, pre_transform=self.pre_transform,
    #         on_device_transform=self.on_device_train_transform, **self.kwargs)

    #     self.val_dataset = self.dataset_class(
    #         self.hparams.data_dir, stage=self.val_stage,
    #         transform=self.val_transform, pre_transform=self.pre_transform,
    #         on_device_transform=self.on_device_val_transform, **self.kwargs)

    #     self.test_dataset = self.dataset_class(
    #         self.hparams.data_dir, stage='test',
    #         transform=self.test_transform, pre_transform=self.pre_transform,
    #         on_device_transform=self.on_device_test_transform, **self.kwargs)

if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = str(pyrootutils.setup_root(__file__, pythonpath=True))
    cfg = omegaconf.OmegaConf.load(root + "/configs/datamodule/semantic/custom_data.yaml")
    cfg.data_dir = root + "/data"
    _ = hydra.utils.instantiate(cfg)
