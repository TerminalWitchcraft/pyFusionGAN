import os
import glob
from PIL import Image
import torch
from toch.utils.data import Dataset
from torchvision import transforms


class ImageData(Dataset):
    "Class to load image data"

    def __init__(self, dataset, load_size, channels, augment_flag, transforms=None):
        "Initialize variables"
        self.load_size = load_size
        self.channels = channels
        self.augment_flag = augment_flag
        if transforms:
            self.transforms = transforms.Compose(transforms)
        else:
            self.transforms = None

        #  Check if dataset exists
        check_exists(dataset)
        self.dataset_name = dataset
        self.train = glob("./dataset/{}/*.*".format(self.dataset_name))

    def __getitem__(self, index):
        "Process each image"
        img_data = Image.open(self.train[index])
        if self.transforms:
            self.transforms(img_data)
        return (img, label)

    def __len__(self):
        "Get length"
        return len(self.train)

def check_exists(folder_name):
    "Function to check if folder exists"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name


if __name__ == '__main__':
    data = ImageData()
