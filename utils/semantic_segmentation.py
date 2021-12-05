import numpy as np


def binary_cal_jaccard_index_expend(c1, c2, c3):
    c1_region = set(zip(np.where(c1 > 0)[0], np.where(c1 > 0)[1]))
    c2_region = set(zip(np.where(c2 > 0)[0], np.where(c2 > 0)[1]))
    c3_region = set(zip(np.where(c3 > 0)[0], np.where(c3 > 0)[1]))

    intersection_all = len(set.intersection(c1_region, c2_region, c3_region))
    intersection_c1_c2 = len(set.intersection(c1_region, c2_region))
    intersection_c1_c3 = len(set.intersection(c1_region, c3_region))
    intersection_c2_c3 = len(set.intersection(c2_region, c3_region))
    union = len(c1_region) \
            + len(c2_region) \
            + len(c3_region) \
            - intersection_c1_c2 \
            - intersection_c1_c3 \
            - intersection_c2_c3 \
            + intersection_all
    iou = intersection_all / union

    return iou


# ============================== <binary semantic segmentation> ============================= #

def binary_cal_iou(img, target):
    """
    :param img: binary image (ingredients are 0 or 1)
    :param target: binary image (ingredients are 0 or 1)
    :return: iou
    """
    img_region = list(zip(np.where(img > 0)[0], np.where(img > 0)[1]))
    target_region = list(zip(np.where(target > 0)[0], np.where(target > 0)[1]))

    intersection = len(set(img_region).intersection(target_region))
    union = len(img_region) + len(target_region) - intersection

    iou = intersection / union + 1e-5

    return iou


def binary_cal_precision_and_recall(img, target):
    """
    :param img: binary image (ingredients are 0 or 1)
    :param target: binary image (ingredients are 0 or 1)
    :return:
    """
    # area interesting regions.
    img_region = list(zip(np.where(img > 0)[0], np.where(img > 0)[1]))
    target_region = list(zip(np.where(target > 0)[0], np.where(target > 0)[1]))

    TP = list(set(img_region).intersection(target_region))
    FN = set(img_region).difference(TP)
    FP = set(target_region).difference(TP)

    intersection = len(set(img_region).intersection(target_region))
    union = len(img_region) + len(target_region) - intersection

    TN_val = (img.shape[0] * img.shape[1]) - union

    precision = len(TP) / (len(TP) + len(FP) + 1e-5)
    recall = len(TP) / (len(TP) + len(FN) + 1e-5)

    return precision, recall


# ============================== <semantic segmentation> ============================= #
def semantic_cal_iou(img, target, n_classes):
    '''
    :param img:  (H, W):numpy.ndarray,  idexing by 0,1,2 .. C+1 (0 : background)
    :param target: (H, W):numpy.ndarray,  idexing by 0,1,2 .. C+1 (0 : background)
    :return: iou
    '''
    iou_c = list()

    # print(np.unique(target))

    for c in range(n_classes):
        if c != 0:  # except background.
            temp_img = np.zeros_like(img)
            temp_img[np.where(img == c)] = 1

            temp_target = np.zeros_like(target)
            temp_target[np.where(target == c)] = 1

            if np.sum(temp_target) == 0:
                iou = np.nan
            else:
                iou = binary_cal_iou(temp_img, temp_target)
            iou_c.append(iou)

    # NOTICE: np.nanmean([np.nan, 0,1,2,3,4])
    return iou_c


def semantic_cal_precision_and_recall(img, target, n_classes):
    """
    :param img: (H, W):numpy.ndarray,  idexing by 0,1,2 .. C+1 (0 : background)
    :param target: (H, W):numpy.ndarray,  idexing by 0,1,2 .. C+1 (0 : background)
    :return:
    """

    precision_c = list()
    recall_c = list()

    for c in range(n_classes):
        if c != 0:  # except background.

            temp_img = np.zeros_like(img)
            temp_img[np.where(img == c)] = 1

            temp_target = np.zeros_like(target)
            temp_target[np.where(target == c)] = 1

            if np.sum(temp_target) == 0:
                precision = np.nan
                recall = np.nan
            else:
                precision, recall = binary_cal_precision_and_recall(temp_img, temp_target)

            precision_c.append(precision)
            recall_c.append(recall)

    return precision_c, recall_c


# ============================== <scores> ============================= #

def cal_F1_score(precision, recall):
    return 2 * (precision * recall / (precision + recall))
