import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):

    def __init__(self, weights, n_class=1):
        super(UNet, self).__init__()

        self.weight1 = nn.Parameter(torch.tensor(weights[0]))
        self.weight2 = nn.Parameter(torch.tensor(weights[1]))
        self.weight3 = nn.Parameter(torch.tensor(weights[2]))
        self.weight4 = nn.Parameter(torch.tensor(weights[3]))

        self.level1_conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.level1_conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.level1_maxpool = nn.MaxPool2d(kernel_size=2)
        self.level1_bat1 = nn.BatchNorm2d(num_features=64)
        self.level1_bat2 = nn.BatchNorm2d(num_features=64)

        self.level2_conv1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.level2_conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.level2_maxpool = nn.MaxPool2d(kernel_size=2)
        self.level2_bat1 = nn.BatchNorm2d(num_features=128)
        self.level2_bat2 = nn.BatchNorm2d(num_features=128)

        self.level3_conv1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.level3_conv2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.level3_maxpool = nn.MaxPool2d(kernel_size=2)
        self.level3_bat1 = nn.BatchNorm2d(num_features=256)
        self.level3_bat2 = nn.BatchNorm2d(num_features=256)

        self.level4_conv1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.level4_conv2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.level4_maxpool = nn.MaxPool2d(kernel_size=2)
        self.level4_bat1 = nn.BatchNorm2d(num_features=512)
        self.level4_bat2 = nn.BatchNorm2d(num_features=512)

        self.level5_conv1 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1)
        self.level5_conv2 = nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1)
        self.level5_bat1 = nn.BatchNorm2d(num_features=1024)
        self.level5_bat2 = nn.BatchNorm2d(num_features=1024)

        self.level5_upsample = nn.ConvTranspose2d(in_channels=1024, out_channels=512,
                                                  kernel_size=4, stride=2,
                                                  padding=1)
        self.level4_conv3 = nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1)
        self.level4_conv4 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.level4_bat3 = nn.BatchNorm2d(num_features=512)
        self.level4_bat4 = nn.BatchNorm2d(num_features=512)

        self.level4_upsample = nn.ConvTranspose2d(in_channels=512, out_channels=256,
                                                  kernel_size=4, stride=2, padding=1)
        self.level3_conv3 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1)
        self.level3_conv4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.level3_bat3 = nn.BatchNorm2d(num_features=256)
        self.level3_bat4 = nn.BatchNorm2d(num_features=256)

        self.level3_upsample = nn.ConvTranspose2d(in_channels=256, out_channels=128,
                                                  kernel_size=4, stride=2, padding=1)
        self.level2_conv3 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.level2_conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.level2_bat3 = nn.BatchNorm2d(num_features=128)
        self.level2_bat4 = nn.BatchNorm2d(num_features=128)

        self.level2_upsample = nn.ConvTranspose2d(in_channels=128, out_channels=64,
                                                  kernel_size=4, stride=2, padding=1)
        self.level1_conv3 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.level1_conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.level1_conv5 = nn.Conv2d(64, n_class, kernel_size=3, stride=1, padding=1)
        self.level1_bat3 = nn.BatchNorm2d(num_features=64)
        self.level1_bat4 = nn.BatchNorm2d(num_features=64)
        # self.level1_bat5 = nn.BatchNorm2d(num_features=1)

    def forward(self, x):
        self.feature1 = F.relu(self.level1_bat2(
            self.level1_conv2(F.relu(self.level1_bat1(self.level1_conv1(x))))))
        self.feature2 = F.relu(self.level2_bat2(
            self.level2_conv2(F.relu(
                self.level2_bat1(self.level2_conv1(self.level1_maxpool(self.feature1)))))))
        self.feature3 = F.relu(self.level3_bat2(
            self.level3_conv2(F.relu(
                self.level3_bat1(self.level3_conv1(self.level2_maxpool(self.feature2)))))))
        self.feature4 = F.relu(self.level4_bat2(
            self.level4_conv2(F.relu(
                self.level4_bat1(self.level4_conv1(self.level3_maxpool(self.feature3)))))))

        self.feature5 = self.level5_bat2(
            self.level5_conv2(
                self.level5_bat1(self.level5_conv1(self.level4_maxpool(self.feature4)))))
        self.feature4_up = torch.cat(
            (self.weight1 * self.level5_upsample(self.feature5),
             (1 - self.weight1) * self.feature4), dim=1)
        self.feature4_2 = F.relu(
            self.level4_bat4(self.level4_conv4(
                F.relu(self.level4_bat3(self.level4_conv3(self.feature4_up))))))
        self.feature3_up = torch.cat(
            (self.weight2 * self.level4_upsample(self.feature4_2),
             (1 - self.weight2) * self.feature3), dim=1)
        self.feature3_2 = F.relu(
            self.level3_bat4(self.level3_conv4(
                F.relu(self.level3_bat3(self.level3_conv3(self.feature3_up))))))
        self.feature2_up = torch.cat(
            (self.weight3 * self.level3_upsample(self.feature3_2),
             (1 - self.weight3) * self.feature2), dim=1)
        self.feature2_2 = F.relu(
            self.level2_bat4(self.level2_conv4(
                F.relu(self.level2_bat3(self.level2_conv3(self.feature2_up))))))
        self.feature1_up = torch.cat(
            (self.weight4 * self.level2_upsample(self.feature2_2),
             (1 - self.weight4) * self.feature1), dim=1)
        self.feature1_2 = F.relu(
            self.level1_bat4(self.level1_conv4(
                F.relu(self.level1_bat3(self.level1_conv3(self.feature1_up))))))

        self.out = self.level1_conv5(self.feature1_2)

        # print(self.weight1, self.weight2, self.weight3, self.weight4)

        return self.out
