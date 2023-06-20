from torch.utils import data
import matplotlib.pyplot as plt 
from PIL import Image
import numpy as np

def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)

def as_png(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    x = (x + 1.0) * 127.5
    x = x.clip(0, 255).astype(np.uint8)
    return Image.fromarray(x)


data_path = '/home/aiiacvmllab/Documents/datasets/LookOut_UE4'

# Testing loading test data
from src.data.custom.custom_abs import VideoDataset
dataset = VideoDataset(root_path=data_path, length = 3, low = 10, high = 10, split = "test")
test_loader_abs = data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last = True
    )
batch = next(iter(test_loader_abs))

# Testing loading train data
from src.data.custom.custom_cview import VideoDataset
dataset = VideoDataset(root_path = data_path, length = 3, low = 3, high = 20, split = "train")
train_loader = data.DataLoader(
        dataset,
        batch_size=2,
        sampler=data_sampler(dataset, shuffle=True, distributed=False),
        drop_last=True
)
train_loader = sample_data(train_loader)
batch = next(train_loader)


print(batch)
# print(batch.get('t_s'))
# print(batch.get('R_s'))

plt.show()
