import torch
from torch.utils.data import DataLoader
from torch.nn import NLLLoss
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler, RandomSampler

from data_utils.utils import collate_fn
from utils.logging_utils import setup_logger
from builders.model_builder import build_model
from torch import distributed as dist

import os
import numpy as np
import pickle
import random

logger = setup_logger()


class BaseTrainer:
    def __init__(self, config):

        self.checkpoint_path = os.path.join(
            config.TRAINING.CHECKPOINT_PATH, config.MODEL.NAME
        )
        if not os.path.isdir(self.checkpoint_path):
            logger.info("Creating checkpoint path")
            os.makedirs(self.checkpoint_path)

        if not os.path.isfile(os.path.join(self.checkpoint_path, "vocab.bin")):
            logger.info("Creating vocab")
            self.vocab = self.load_vocab(config)
            logger.info(
                "Saving vocab to %s" % os.path.join(self.checkpoint_path, "vocab.bin")
            )
            pickle.dump(
                self.vocab, open(os.path.join(self.checkpoint_path, "vocab.bin"), "wb")
            )
        else:
            logger.info(
                "Loading vocab from %s"
                % os.path.join(self.checkpoint_path, "vocab.bin")
            )
            self.vocab = pickle.load(
                open(os.path.join(self.checkpoint_path, "vocab.bin"), "rb")
            )
        
        logger.info("Building model")
        self.model = build_model(config.MODEL, self.vocab)
        self.config = config
        self.device = torch.device(config.MODEL.DEVICE)
        
        self.ddp = config.TRAINING.DDP
    
        logger.info("Loading data")
        self.train_dataset, self.dev_dataset, self.test_dataset = (
            self.load_feature_datasets(config.DATASET)
        )
        self.train_dict_dataset, self.dev_dict_dataset, self.test_dict_dataset = (
            self.load_dict_datasets(config.DATASET)
        )
        gpus = self.setup_ddp()
        if self.ddp:
            self.model = DDP(self.model, device_ids=[gpus])
            self.train_dataset = DistributedSampler(self.train_dataset)
            self.dev_dataset = DistributedSampler(self.dev_dataset)
        else:
            self.train_dataset = RandomSampler(self.train_dataset)
            self.dev_dataset = RandomSampler(self.dev_dataset)
        
        # creating iterable-dataset data loader
        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            batch_size=config.DATASET.FEATURE_BATCH_SIZE,
            shuffle=True,
            num_workers=config.DATASET.WORKERS,
            collate_fn=collate_fn,
        )
        self.val_dataloader = DataLoader(
            dataset=self.dev_dataset,
            batch_size=config.DATASET.FEATURE_BATCH_SIZE,
            shuffle=True,
            num_workers=config.DATASET.WORKERS,
            collate_fn=collate_fn,
        )
        self.test_dataloader = DataLoader(
            dataset=self.test_dataset,
            batch_size=config.DATASET.FEATURE_BATCH_SIZE,
            shuffle=True,
            num_workers=config.DATASET.WORKERS,
            collate_fn=collate_fn,
        )

        # creating dictionary iterable-dataset data loader
        self.train_dict_dataloader = DataLoader(
            dataset=self.train_dict_dataset,
            batch_size=config.DATASET.DICT_BATCH_SIZE
            // config.TRAINING.TRAINING_BEAM_SIZE,
            shuffle=True,
            collate_fn=collate_fn,
        )
        self.val_dict_dataloader = DataLoader(
            dataset=self.dev_dict_dataset,
            batch_size=config.DATASET.DICT_BATCH_SIZE
            // config.TRAINING.EVALUATING_BEAM_SIZE,
            shuffle=True,
            collate_fn=collate_fn,
        )
        self.test_dict_dataloader = DataLoader(
            dataset=self.test_dict_dataset,
            batch_size=1,
            shuffle=True,
            collate_fn=collate_fn,
        )

        logger.info("Defining optimizer and objective function")
        self.configuring_hyperparameters(config)
        self.optim = Adam(
            self.model.parameters(), lr=config.TRAINING.LEARNING_RATE, betas=(0.9, 0.98)
        )
        self.scheduler = LambdaLR(self.optim, self.lambda_lr)
        self.loss_fn = NLLLoss(ignore_index=self.vocab.padding_idx)

    def configuring_hyperparameters(self, config):
        raise NotImplementedError

    def load_vocab(self, config):
        raise NotImplementedError

    def load_feature_datasets(self, config):
        raise NotImplementedError

    def load_dict_datasets(self, config):
        raise NotImplementedError

    def evaluate_loss(self, dataloader: DataLoader):
        raise NotImplementedError

    def evaluate_metrics(self, dataloader: DataLoader):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def lambda_lr(self, step):
        warm_up = self.warmup
        step += 1
        return (self.model.encoder.d_model**-0.5) * min(
            step**-0.5, step * warm_up**-1.5
        )

    def load_checkpoint(self, fname) -> dict:
        if not os.path.exists(fname):
            return None

        logger.info("Loading checkpoint from %s", fname)

        checkpoint = torch.load(fname, map_location=torch.device("cpu"))

        torch.set_rng_state(checkpoint["torch_rng_state"])
        torch.cuda.set_rng_state(checkpoint["cuda_rng_state"])
        np.random.set_state(checkpoint["numpy_rng_state"])
        random.setstate(checkpoint["random_rng_state"])

        self.model.load_state_dict(checkpoint["state_dict"], strict=False)

        logger.info(f"Resuming from epoch %s", checkpoint["epoch"])

        return checkpoint

    def save_checkpoint(self, dict_for_updating: dict) -> None:
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        dict_for_saving = {
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state(),
            "numpy_rng_state": np.random.get_state(),
            "random_rng_state": random.getstate(),
            "epoch": self.epoch,
            "state_dict": model_to_save.state_dict(),
            "optimizer": self.optim.state_dict(),
            "scheduler": self.scheduler.state_dict(),
        }

        for key, value in dict_for_updating.items():
            dict_for_saving[key] = value

        torch.save(
            dict_for_saving, os.path.join(self.checkpoint_path, "last_model.pth")
        )

    def start(self):
        raise NotImplementedError

    def get_predictions(self, dataset, get_scores=True):
        raise NotImplementedError

    def setup_ddp(self) -> int:
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            rank = int(os.environ['RANK'])
            world_size = int(os.environ['WORLD_SIZE'])
            gpu = int(os.environ(['LOCAL_RANK']))
            torch.cuda.set_device(gpu)
            dist.init_process_group('nccl', init_method="env://",world_size=world_size, rank=rank)
            dist.barrier()
        else:
            gpu = 0
        return gpu