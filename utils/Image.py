import numpy as np
import cv2
import torch


#
# def hierarchical_color_grouping():
#

def nonlinear_threshold(image, coarse_label, mid=None):
    N, C, H, W = image.shape

    assert C == 3, "Channel should be 3."

    G = image[:, 1, :, :].view([-1, H * W])
    R = image[:, 0, :, :].view([-1, H * W])

    highlight = (torch.div(G, torch.max(G, dim=1).values.view([-1, 1])).view([-1, H, W])
                 - torch.div(R, torch.max(R, dim=1).values.view([-1, 1])).view([-1, H, W]))

    # remove not important part.
    for idx, hi in enumerate(highlight):

        # NOTICE : mid of bean leaf is 0.
        if mid is None:
            mid = ((torch.max(hi) + torch.min(hi)) / 2).item()
        # threshold.
        highlight[idx][torch.where(hi <= mid)] = 0
        highlight[idx][torch.where(hi > mid)] = 1

        highlight[idx][torch.where(torch.squeeze(coarse_label[idx]) < 1)] = 0

    return highlight.view(N, 1, H, W)


# TODO: Check immutable of params in function.
def post_processing(output_one, label_coarse_one):
    '''
    :param output_one: (512, 512) index : 0,1,2,3
    :param label_coarse_one: (512, 512) index : 0,1,2,3
    :return: output_one: (512, 512) index : 0,1,2,3
    '''

    for coarse in np.unique(label_coarse_one):
        if coarse not in output_one:
            # set inference to background if that class is not annotated on coarse label.
            output_one[np.where(output_one == coarse)] = 0

    return output_one


def torch_threshold(image, ratio):
    assert len(image.shape) == 4 and image.shape[
        1] == 1, "Input must be a binary image. and shape should be (N, 1, H, W)"
    assert 0 < ratio <= 1, "'cut' must be in 0 to 1"

    (N, C, H, W) = image.shape

    image = image.view(image.shape[0], -1)

    max_val = torch.max(image, dim=1).values
    cut_val = ratio * max_val

    for idx, i in enumerate(cut_val):
        image[idx][torch.where(image[idx] < i)] = 0
        image[idx][torch.where(image[idx] >= i)] = 1

    return image.view(N, C, H, W)


def threshold(image, ratio):
    assert len(image.shape) == 2, "Input must be a binary image."
    assert 0 < ratio <= 1, "'cut' must be in 0 to 1"

    max_val = np.max(image)
    cut_val = ratio * max_val

    image[np.where(image < cut_val)] = 0
    image[np.where(image >= cut_val)] = 1

    return image


def blend(image1, image2, ratio, color=(1., 0.7, 0.)):
    assert 0 < ratio <= 1, "'cut' must be in 0 to 1"

    alpha = ratio
    beta = 1 - alpha

    # coloring yellow.
    image2 *= list(color)
    image = image1 * alpha + image2 * beta

    # image = image1 * image2
    return image


def blur(image):
    return cv2.GaussianBlur(image, (5, 5), 0)


def resize(image):
    """

    :param image: numpy array.
    :return:
    """
    resized = cv2.resize(src=image, dsize=(512, 512))

    return resized
