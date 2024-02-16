import os.path as osp
import torch, torchvision
from data.dataset import H5Dataset, ImageDataset, Subset, CIFAR10, CIFAR100


class ImageDatasetLoader():
    def __init__(self, cfg, rank):
        self.cfg = cfg 
        self.rank = rank
        self.num_replicas = cfg.WORLD_SIZE
        self.distributed = cfg.DISTRIBUTED
        self.workers = cfg.WORKERS
        self.data_dir = cfg.DATA.PATH_TO_DATA_DIR

    def get_loader(self, stage, batch_size, transforms):
        dataset = self.get_dataset(stage, transforms)

        if self.distributed and stage in ('TRAIN', 'FT'):
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, num_replicas=self.num_replicas, rank=self.rank)
        else:
            self.train_sampler = None

        # drop_last = True if stage.lower() in ('train', 'ft') else False
        drop_last = False
        shuffle = (self.train_sampler is None and stage.lower() not in ('val', 'test'))
        data_loader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.workers,
            pin_memory=True,
            sampler=self.train_sampler,
            drop_last=drop_last
        )

        if self.rank == 0:
            print(f"Data loaded: there are {len(dataset)} images.")

        return data_loader, len(dataset)

    def get_dataset(self, stage, transforms):
        file_name = self.cfg.TRAIN.FILE_NAME if stage.lower() in ('train', 'ft') else self.cfg.VAL.FILE_NAME
        file_path = osp.join(self.data_dir, file_name)
        if 'cifar' in self.cfg.TRAIN.DATASET.lower():
            dataset = globals()[self.cfg.TRAIN.DATASET.upper()](self.data_dir, train=(stage.lower() in ('train', 'ft')), transform=transforms, download=False)
        elif 'h5' in file_name:
            dataset = H5Dataset(file_path, transform=transforms)
        else:
            dataset = ImageDataset(file_path, transform=transforms)
        subset_path = self.cfg.TRAIN.SUBSET_FILE_PATH if stage.lower() in ('train', 'ft') else self.cfg.VAL.SUBSET_FILE_PATH
        if subset_path != '':
            with open(subset_path) as f:
                lines = f.read().splitlines()
            sample_inds = [int(line.split(',')[0]) for line in lines]
            dataset = Subset(dataset, indices=sample_inds)
        return dataset

    def set_epoch(self, epoch):
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(epoch)
