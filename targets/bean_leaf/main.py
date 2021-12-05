import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader

import cv2
from scipy import ndimage

from imageio import imwrite, imread
import matplotlib.pyplot as plt

import numpy as np

from data.DataUtils import SegDataset, InitializeData
from data.Transforms import ToTensor, Normalize
from models.UNet_weight_connection import UNet

from utils.Image import nonlinear_threshold as twh, threshold as th, blend as bd

import targets.bean_leaf.configrun as cfg

import os

from utils.semantic_segmentation import binary_cal_iou, binary_cal_precision_and_recall, cal_F1_score


def cal_evaluation(fine_label_path, img_out, file_name):
    # compare output(fine predict from coarse label.) and fine label.
    img_target = imread(os.path.join(fine_label_path, "{}".format(file_name)), pilmode="L")
    iou = binary_cal_iou(img_out, img_target)
    precision, recall = binary_cal_precision_and_recall(img_out, img_target)

    return iou, precision, recall


'''
cpu number.
cuda:0 : 3
cuda:1 : 0
cuda:2 : 2
cuda:3 : 1
'''

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1'

# check all directory.
if not os.path.isdir(cfg.PATH_TEST_OUTPUT_MAIN_SOLID):
    os.makedirs(cfg.PATH_TEST_OUTPUT_MAIN_SOLID)
if not os.path.isdir(cfg.PATH_TEST_OUTPUT_PLOT_SOLID):
    os.makedirs(cfg.PATH_TEST_OUTPUT_PLOT_SOLID)
if not os.path.isdir(cfg.PATH_TEST_OUTPUT_SEG_SOLID):
    os.makedirs(cfg.PATH_TEST_OUTPUT_SEG_SOLID)

# set device.
device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')

# load data.
init_data = InitializeData(image_path=cfg.PATH_DATA_IMAGE_ORIGIN_SOLID,
                           coarse_label_path=cfg.PATH_DATA_LABEL_COARSE_SOLID,
                           fine_label_path=cfg.PATH_DATA_LABEL_FINE_SOLID,
                           ratio_choose=1.0,
                           ratio_divide=0.9)
dataset_train = SegDataset(mode='train', data=init_data,
                           transform=transforms.Compose([Normalize(), ToTensor()]))
dataset_test = SegDataset(mode='test', data=init_data,
                          transform=transforms.Compose([Normalize(), ToTensor()]))
data_loader_train = DataLoader(dataset=dataset_train, batch_size=12, shuffle=True,
                               num_workers=0)
data_loader_test = DataLoader(dataset=dataset_test, batch_size=1, shuffle=True, num_workers=0)

# set models.
model = UNet([0.2, 0.2, 0.2, 0.2])
model = torch.nn.DataParallel(model)
model.to(device)

# set optimizer and loss.
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
loss_func = nn.MSELoss(reduction='mean')
# loss_func = nn.BCEWithLogitsLoss(reduction='mean')

# write log.
f = open(cfg.PATH_TEST_LOG_SOLID, 'w')

val_iou_list = []
val_precision_list = []
val_recall_list = []

lowpass_iou_list = []
lowpass_precision_list = []
lowpass_recall_list = []

median_iou_list = []
median_precision_list = []
median_recall_list = []

# check saved models
if not os.path.isfile(cfg.PATH_TEST_MODEL_WEIGHT_SOLID):
    print("MODE : TRAIN MODEL.")

    # set train mode.
    model.train()

    # set epoch.
    epoch = cfg.epoch

    loss_train = []
    iou_test = []

    import time

    start_time = time.time()

    for idx_epoch in range(epoch):
        for idx_batch, batch in enumerate(data_loader_train):

            image, label, file_names = batch['image'].to(device), batch['label_coarse'].to(
                device), batch['file_name']
            pseudo = twh(image, label, mid=0)
            # pseudo = tth(pseudo, cfg.TH_VAL_2)
            out = model(image)

            # Notice :  loss select part.
            loss = loss_func(torch.squeeze(out), torch.squeeze(label)) + loss_func(
                torch.squeeze(out), torch.squeeze(pseudo))
            # loss = loss_func(torch.squeeze(out), torch.squeeze(label))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(idx_epoch, idx_batch, loss.detach().cpu().item())

            loss_train.append(loss.detach().cpu().item())

            # each batch avg test iou
            temp_test_iou = []
            for idx_test, test in enumerate(data_loader_test):
                t_image, t_label = test['image'].to(device), test['label_fine']
                t_out = np.squeeze(model(t_image).detach().cpu().numpy())
                t_label = np.squeeze(t_label)

                # calculate child-batch iou
                th_test = th(t_out, cfg.TH_VAL)
                t_iou = binary_cal_iou(th_test, t_label)

                temp_test_iou.append(t_iou)

            iou_test.append(np.mean(temp_test_iou))

            # final epoch, train result.
            if idx_epoch + 1 == epoch:
                for i in range(len(image)):
                    th_img = th(out.detach().cpu().numpy()[i].transpose((1, 2, 0)).squeeze(),
                                cfg.TH_VAL)
                    output = np.stack([th_img, th_img, th_img]).transpose([1, 2, 0])

                    iou_val, precision_val, recall_val = cal_evaluation(
                        cfg.PATH_DATA_LABEL_FINE_SOLID, th_img, batch['file_name'][i])
                    val_iou_list.append(iou_val)
                    val_precision_list.append(precision_val)
                    val_recall_list.append(recall_val)

                    plt.rcParams['figure.figsize'] = (30, 30)
                    plt.rcParams["axes.prop_cycle"] = plt.cycler('color',
                                                                 ['#ff0000', '#00ff00',
                                                                  '#0000ff'])

                    figure = plt.figure()
                    fig, ax = plt.subplots(4, 4, gridspec_kw={'width_ratios': [1, 1, 1, 1],
                                                              'height_ratios': [1, 1, 1, 1]})

                    fig.suptitle("".join([file_names[i], ": ", str(iou_val)]))

                    # image
                    ax[0][0].title.set_text('Image')
                    ax[0][0].imshow(
                        image.detach().cpu().detach().numpy()[i].transpose((1, 2, 0)))

                    # coarse label
                    ax[1][0].title.set_text('Coarse_label')
                    ax[1][0].imshow(
                        label.detach().cpu().numpy()[i].transpose((1, 2, 0)).squeeze())

                    # pseudo_label
                    ax[2][0].title.set_text('Pseudo_label')
                    ax[2][0].imshow(
                        pseudo.detach().cpu().numpy()[i].transpose((1, 2, 0)).squeeze())

                    # fine label
                    ax[3][0].title.set_text('Fine Label')
                    img_target = imread(os.path.join(cfg.PATH_DATA_LABEL_FINE_SOLID,
                                                     "{}".format(batch['file_name'][i])),
                                        pilmode="L")
                    ax[3][0].imshow(img_target)

                    # output
                    ax[0][1].title.set_text('Output')
                    ax[0][1].imshow(
                        out.detach().cpu().numpy()[i].transpose((1, 2, 0)).squeeze())

                    # output(TH)
                    ax[1][1].title.set_text('Output(TH)')
                    ax[1][1].imshow(th_img.squeeze())

                    # overlay(TH)
                    ax[2][1].title.set_text('Overlay output(TH)')
                    mixed_img = bd(
                        image.detach().cpu().detach().numpy()[i].transpose((1, 2, 0)), output,
                        0.5)
                    ax[2][1].imshow(mixed_img)

                    # low pass filter (size = 3)
                    ax[0][2].title.set_text('low pass filter(pseudo label)')
                    pseudo_put = pseudo.detach().cpu().numpy()[i].transpose(
                        (1, 2, 0)).squeeze()
                    blur = cv2.blur(pseudo_put, (3, 3))
                    ax[0][2].imshow(blur)

                    # low pass filter (TH)
                    ax[1][2].title.set_text('low pass filter(TH)')
                    pseudo_put = pseudo.detach().cpu().numpy()[i].transpose(
                        (1, 2, 0)).squeeze()
                    blur = cv2.blur(pseudo_put, (3, 3))
                    low_pass = th(blur, cfg.TH_VAL)
                    ax[1][2].imshow(low_pass)

                    low_iou_val, low_precision_val, low_recall_val = cal_evaluation(
                        cfg.PATH_DATA_LABEL_FINE_SOLID, low_pass, batch['file_name'][i])
                    lowpass_iou_list.append(low_iou_val)
                    lowpass_precision_list.append(low_precision_val)
                    lowpass_recall_list.append(low_recall_val)

                    # median filter (size = 3)
                    ax[2][2].title.set_text('median filter(pseudo label)')
                    pseudo_put = pseudo.detach().cpu().numpy()[i].transpose(
                        (1, 2, 0)).squeeze()
                    median = ndimage.median_filter(pseudo_put, size=(3, 3))
                    ax[2][2].imshow(median)

                    median_iou_val, median_precision_val, median_recall_val = cal_evaluation(
                        cfg.PATH_DATA_LABEL_FINE_SOLID, median, batch['file_name'][i])
                    median_iou_list.append(median_iou_val)
                    median_precision_list.append(median_precision_val)
                    median_recall_list.append(median_recall_val)

                    # origin image histogram
                    ax[0][3].title.set_text('image histogram')
                    spectrums_image = np.zeros([3, 256])
                    origin = image.detach().cpu().detach().numpy()[i].transpose((1, 2, 0))
                    origin *= 255
                    origin = origin.astype(np.int)
                    for j in range(origin.shape[-1]):
                        for k in range(1, 255):
                            spectrums_image[j, k] = len(np.where(origin[:, :, j] == k)[0])
                    ax[0][3].plot(range(256), spectrums_image[0], range(256),
                                  spectrums_image[1], range(256), spectrums_image[2])

                    # output histogram
                    ax[1][3].title.set_text('output histogram')
                    out_row = out.detach().cpu().numpy()[i].transpose((1, 2, 0)).squeeze()
                    out_row[np.where(out_row < 0)] = 0
                    out_row *= 255
                    out_row = out_row.astype(np.int)
                    histogram = np.zeros([256])
                    for j in range(1, 255):
                        histogram[j] = len(np.where(out_row == j)[0])
                    ax[1][3].plot(range(256), histogram, c='k')

                    # low-pass histogram
                    ax[2][3].title.set_text('low-pass histogram')
                    pseudo_put = pseudo.detach().cpu().numpy()[i].transpose(
                        (1, 2, 0)).squeeze()
                    blur = cv2.blur(pseudo_put, (3, 3))
                    blur[np.where(blur < 0)] = 0
                    blur *= 255
                    blur = blur.astype(np.int)
                    histogram = np.zeros([256])
                    for j in range(1, 255):
                        histogram[j] = len(np.where(blur == j)[0])
                    ax[2][3].plot(range(256), histogram, c='k')

                    # save fig.
                    plt.savefig("{}".format(
                        os.path.join(cfg.PATH_TEST_OUTPUT_PLOT_SOLID, batch['file_name'][i])))
                    # plt.show()
                    plt.clf()

                    # print
                    print("IOU img({})  : {}".format(batch['file_name'][i], iou_val))
                    print("Precision, Recall img({})  : {}, {}".format(batch['file_name'][i],
                                                                       precision_val,
                                                                       recall_val))
                    f.write("IOU img({})  : {}\r\n".format(batch['file_name'][i], iou_val))
                    f.write(
                        "Precision, Recall img({})  : {}, {}\r\n".format(batch['file_name'][i],
                                                                         precision_val,
                                                                         recall_val))

                    # save output seg result.
                    imwrite(
                        os.path.join(cfg.PATH_TEST_OUTPUT_SEG_SOLID, batch['file_name'][i]),
                        im=th_img)

        plt.rcParams['figure.figsize'] = (8, 6)
        plt.plot(range(len(loss_train)), loss_train)
        plt.plot(range(len(iou_test)), iou_test)
        plt.ylim([.0, 1.0])
        plt.savefig(cfg.PATH_TEST_OUTPUT_CURVE_SOLID)
        # plt.show()
        plt.clf()

    spent_time = time.time() - start_time
    print(f'spent time : {spent_time}')
    f.write("\r\nspent time : {}\r\n".format(spent_time))

    # save models weight.
    state_dict = {"state_dict": model.module.state_dict(), "total_epoch": epoch,
                  "loss_train": loss_train, "iou_test": iou_test}
    torch.save(state_dict, cfg.PATH_TEST_MODEL_WEIGHT_SOLID)

    # net result
    iou = np.mean(val_iou_list)
    precision = np.mean(val_precision_list)
    recall = np.mean(val_recall_list)

    # low pass result
    low_pass_iou = np.mean(lowpass_iou_list)
    low_pass_precision = np.mean(lowpass_precision_list)
    low_pass_recall = np.mean(lowpass_recall_list)

    # median filter output
    median_iou = np.mean(median_iou_list)
    median_precision = np.mean(median_precision_list)
    median_recall = np.mean(median_recall_list)

    print("IOU all : {}".format(iou))
    print("Precision all : {}".format(precision))
    print("Recall all : {}".format(recall))
    print("F1 score : {}".format(cal_F1_score(precision, recall)))

    print("low_pass IOU all : {}".format(low_pass_iou))
    print("low_pass Precision all : {}".format(low_pass_precision))
    print("low_pass Recall all : {}".format(low_pass_recall))
    print("low_pass F1 score : {}".format(cal_F1_score(low_pass_precision, low_pass_recall)))

    print("median IOU all : {}".format(median_iou))
    print("median Precision all : {}".format(median_precision))
    print("median Recall all : {}".format(median_recall))
    print("median F1 score : {}".format(cal_F1_score(median_precision, median_recall)))

    f.write("\r\n\r\nIOU all : {}\r\n".format(iou))
    f.write("Precision all : {}\r\n".format(precision))
    f.write("Recall all : {}\r\n".format(recall))
    f.write("F1 score : {}\r\n".format(cal_F1_score(precision, recall)))

    f.write("\r\n\r\nlow_pass IOU all : {}\r\n".format(low_pass_iou))
    f.write("low_pass Precision all : {}\r\n".format(low_pass_precision))
    f.write("low_pass Recall all : {}\r\n".format(low_pass_recall))
    f.write(
        "low_pass F1 score : {}\r\n".format(cal_F1_score(low_pass_precision, low_pass_recall)))

    f.write("\r\n\r\nmedian IOU all : {}\r\n".format(median_iou))
    f.write("median Precision all : {}\r\n".format(median_precision))
    f.write("median Recall all : {}\r\n".format(median_recall))
    f.write("median F1 score : {}\r\n".format(cal_F1_score(median_precision, median_recall)))

    f.close()
