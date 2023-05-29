import os.path as osp
from PIL import Image
import torch, io, h5py, time


class Subset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx):
        _, x, y = self.dataset[self.indices[idx]]
        return idx, x, y
        
    def __len__(self):
        return len(self.indices)


def safe_record_loader(raw_frame, attempts=10, retry_delay=1):
    for j in range(attempts):
        try:
            img = Image.open(io.BytesIO(raw_frame)).convert('RGB')
            return img
        except OSError as e:
            print(f'Attempt {j}/{attempts}: failed to load\n{e}', flush=True)
            if j == attempts - 1:
                raise
        time.sleep(retry_delay)


class H5Dataset(torch.utils.data.Dataset):
    def __init__(self, h5_path, transform=None):
        self.h5_fp = h5_path
        assert osp.isfile(self.h5_fp), "File not found: {}".format(self.h5_fp)
        self.h5_file = None
        h5_file = h5py.File(self.h5_fp, 'r')
        self.dataset = []
        labels = list(h5_file.keys())
        for key, value in h5_file.items():
            target = labels.index(key)
            for img_name in value.keys():
                self.dataset.append({'image_name': img_name, 'class_name': key, 'label': target})

        self.length = len(self.dataset)
        self.transform = transform

    def __getitem__(self, index):
        if self.h5_file is None:
            self.h5_file = h5py.File(self.h5_fp, 'r')
        record = self.dataset[index]
        raw_frame = self.h5_file[record['class_name']][record['image_name']][()]
        img = safe_record_loader(raw_frame)
        if self.transform != None:
            img = self.transform(img)
        return index, img, int(record['label'])

    def __len__(self):
        return self.length