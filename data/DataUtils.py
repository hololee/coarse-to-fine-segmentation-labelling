import torch
from torch.utils.data.dataset import Dataset
import os
from imageio import imread
import numpy as np
import random


# TODO: choose ratio(choose dataset), divide ratio(divide ratio).
class InitializeData:
    def __init__(self, image_path, coarse_label_path, fine_label_path, ratio_choose=1, ratio_divide=0.9):
        self.all_items = []

        # image list on directory.
        image_list = os.listdir(image_path)
        label_list = os.listdir(coarse_label_path)

        # image path.
        self.image_path = image_path
        self.coarse_label_path = coarse_label_path
        self.fine_label_path = fine_label_path

        # image and label matching check.
        image_list.sort()
        label_list.sort()
        for i in range(len(image_list)):
            assert image_list[i] == label_list[i], "image and label should be same : {} and {}".format(
                image_list[i], label_list[i]
            )
            self.all_items.append(image_list[i])

        # choose data some ratio.
        self.items = random.sample(self.all_items, int(len(self.all_items) * ratio_choose))
        print("selected data N : {}".format(int(len(self.all_items) * ratio_choose)))

        # select data which have coarse label from selected data using divide ratio.
        self.targets = self.items
        self.contrasts = self.items[int(len(self.items) * ratio_divide) :]

        print("target size: {}\r\ncontrast size: {}".format(len(self.targets), len(self.contrasts)))


class SegDataset(Dataset):
    def __init__(self, mode, data: InitializeData, transform=None):
        self.transform = transform
        self.init_data = data

        if mode == 'train':
            self.items = self.init_data.targets
        elif mode == 'test':
            self.items = self.init_data.contrasts
        else:
            raise Exception('mode should be train or test. now : {}'.format(mode))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        # load image here because of memory load.

        if torch.is_tensor(index):
            index = index.tolist()

        image = np.array(imread(os.path.join(self.init_data.image_path, self.items[index]), pilmode="RGB"))
        label_coarse = np.expand_dims(
            imread(os.path.join(self.init_data.coarse_label_path, self.items[index]), pilmode="L"), -1
        )
        label_fine = np.expand_dims(
            imread(os.path.join(self.init_data.fine_label_path, self.items[index]), pilmode="L"), -1
        )

        sample = {
            'file_name': self.items[index],
            'image': image,
            'label_coarse': label_coarse,
            'label_fine': label_fine,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


# NOTICE : general


class InitializeDataGeneral:
    def __init__(
        self, image_path, coarse_label_path, fine_label_path, generate_label_path, ratio_choose=1, ratio_divide=0.9
    ):
        self.all_items = []

        # image list on directory.
        image_list = os.listdir(image_path)
        label_list = os.listdir(coarse_label_path)
        gen_list = os.listdir(generate_label_path)

        # image path.
        self.image_path = image_path
        self.coarse_label_path = coarse_label_path
        self.fine_label_path = fine_label_path
        self.generate_label_path = generate_label_path

        # image and label matching check.
        image_list.sort()
        label_list.sort()
        gen_list.sort()
        for i in range(len(image_list)):
            assert image_list[i] == label_list[i], "image and label should be same : {} and {}".format(
                image_list[i], label_list[i]
            )
            self.all_items.append(image_list[i])

        # choose data some ratio.
        self.items = random.sample(self.all_items, int(len(self.all_items) * ratio_choose))
        print("selected data N : {}".format(int(len(self.all_items) * ratio_choose)))

        # select data which have coarse label from selected data using divide ratio.
        self.targets = self.items
        self.contrasts = self.items[int(len(self.items) * ratio_divide) :]

        print("target size: {}\r\ncontrast size: {}".format(len(self.targets), len(self.contrasts)))


class SegDatasetGeneral(Dataset):
    def __init__(self, mode, data: InitializeDataGeneral, transform=None):
        self.transform = transform
        self.init_data = data

        if mode == 'train':
            self.items = self.init_data.targets
        elif mode == 'test':
            self.items = self.init_data.contrasts
        else:
            raise Exception('mode should be train or test. now : {}'.format(mode))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        # load image here because of memory load.

        if torch.is_tensor(index):
            index = index.tolist()

        image = np.array(imread(os.path.join(self.init_data.image_path, self.items[index]), pilmode="RGB"))
        label_coarse = np.expand_dims(
            imread(os.path.join(self.init_data.coarse_label_path, self.items[index]), pilmode="L"), -1
        )
        label_fine = np.expand_dims(
            imread(os.path.join(self.init_data.fine_label_path, self.items[index]), pilmode="L"), -1
        )
        label_generate = np.expand_dims(
            imread(os.path.join(self.init_data.generate_label_path, self.items[index]), pilmode="L"), -1
        )

        sample = {
            'file_name': self.items[index],
            'image': image,
            'label_coarse': label_coarse,
            'label_fine': label_fine,
            'label_gen': label_generate,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


# NOTICE : general multi class


class InitializeDataGeneralMulti:
    def __init__(
        self, image_path, coarse_label_path, fine_label_path, generate_label_path, ratio_choose=1, ratio_divide=0.9
    ):
        self.all_items = []

        # image list on directory.
        image_list = os.listdir(image_path)
        label_list = os.listdir(coarse_label_path)
        gen_list = os.listdir(generate_label_path)

        # image path.
        self.image_path = image_path
        self.coarse_label_path = coarse_label_path
        self.fine_label_path = fine_label_path
        self.generate_label_path = generate_label_path

        # image and label matching check.
        image_list.sort()
        label_list.sort()
        gen_list.sort()
        for i in range(len(image_list)):
            assert image_list[i] == label_list[i], "image and label should be same : {} and {}".format(
                image_list[i], label_list[i]
            )
            self.all_items.append(image_list[i])

        # choose data some ratio.
        self.items = random.sample(self.all_items, int(len(self.all_items) * ratio_choose))
        print("selected data N : {}".format(int(len(self.all_items) * ratio_choose)))

        # select data which have coarse label from selected data using divide ratio.
        self.targets = self.items
        self.contrasts = self.items[int(len(self.items) * ratio_divide) :]

        print("target size: {}\r\ncontrast size: {}".format(len(self.targets), len(self.contrasts)))


class SegDatasetGeneralMulti(Dataset):
    def __init__(self, mode, data: InitializeDataGeneralMulti, transform=None):
        self.transform = transform
        self.init_data = data

        if mode == 'train':
            self.items = self.init_data.targets
        elif mode == 'test':
            self.items = self.init_data.contrasts
        else:
            raise Exception('mode should be train or test. now : {}'.format(mode))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        # load image here because of memory load.

        if torch.is_tensor(index):
            index = index.tolist()

        image = np.array(imread(os.path.join(self.init_data.image_path, self.items[index]), pilmode="RGB"))
        label_coarse = np.expand_dims(
            imread(os.path.join(self.init_data.coarse_label_path, self.items[index]), pilmode="L"), -1
        )
        label_fine = np.expand_dims(
            imread(os.path.join(self.init_data.fine_label_path, self.items[index]), pilmode="L"), -1
        )
        label_generate = np.expand_dims(
            imread(os.path.join(self.init_data.generate_label_path, self.items[index]), pilmode="L"), -1
        )

        sample = {
            'file_name': self.items[index],
            'image': image,
            'label_coarse': label_coarse,
            'label_fine': label_fine,
            'label_gen': label_generate,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
