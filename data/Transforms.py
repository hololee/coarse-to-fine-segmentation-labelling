import numpy as np
import torch


class ToTensor(object):
    def __call__(self, sample):

        for key in sample.keys():
            if key != 'file_name':
                # change channel format and type to float32 and transpose to torch array.
                sample[key] = torch.from_numpy(np.transpose(sample[key], (2, 0, 1)).astype(np.float32))

        return sample


class Normalize(object):
    def __call__(self, sample):

        for key in sample.keys():
            if key != 'file_name':
                # normalize.
                sample[key] = (sample[key] - np.min(sample[key])) / (np.max(sample[key]) - np.min(sample[key]))
        return sample
