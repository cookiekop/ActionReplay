import os
from torchvision import transforms
import cv2
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from skimage.feature import hog
from torchvision.datasets import CIFAR10, ImageFolder, MNIST
from torch.utils.data import DataLoader, random_split
import json

video_extensions = ['.avi', '.mpg']
image_extensions = ['.jpg']

transform = transforms.Compose([transforms.Resize([224, 224]),
                                transforms.ToTensor()])

gray_scale_transform = transforms.Compose([transforms.Resize([224, 224]),
                                           transforms.ToTensor(),
                                           transforms.Lambda(lambda x: x.repeat(3, 1, 1))])

class MPIIDataSet(Dataset):
    def __init__(self, dir, mode='train', transform=transform):
        self.dir = dir
        with open(dir+'/anno/'+mode+'.json') as f:
            self.anno = json.load(f)
        self.transform = transform

    def __len__(self):
        return len(self.anno)

    def __getitem__(self, idx):
        item_anno = self.anno[idx]
        img_loc = self.dir + '/images/' + item_anno['image']
        image = Image.open(img_loc).convert("RGB")

        box_ = torch.FloatTensor([item_anno['scale'] * 200.0, item_anno['scale'] * 200.0])
        center_ = torch.FloatTensor(item_anno['center'])
        img_size = torch.FloatTensor(image.size)
        box_ = box_ * torch.FloatTensor([224, 224]) / img_size
        center_ = center_ * torch.FloatTensor([224, 224]) / img_size
        tl_ = torch.ceil(center_ - box_ / 2).clamp(0, 224).type(torch.IntTensor)
        br_ = torch.floor(center_ + box_ / 2).clamp(0, 224).type(torch.IntTensor)

        tensor_image = self.transform(image)
        mask = torch.ones_like(tensor_image)
        mask[:, tl_[0]:br_[0], tl_[1]:br_[1]] = 2.0
        hog_feat, hog_image = hog(tensor_image.squeeze(0).permute(1, 2, 0), visualize=True)
        return {'image': tensor_image,
                'mask': mask/mask.mean(),
                'hog': torch.tensor(hog_image, dtype=torch.float32).repeat(3, 1, 1)}

class GeneralVideoDataset(Dataset):
    """Dataset Class for Loading Video"""

    def __init__(
        self,
        dir,
        channels=3,
        transform=transform,
    ):

        self.clips_list = []
        for root, dirs, files in os.walk(dir):
            for file in files:
                file_name, file_extension = os.path.splitext(file)
                if file_extension in video_extensions:
                    self.clips_list.append(root + "/" + file)
        self.root_dir = dir
        self.channels = channels
        self.transform = transform

    def __len__(self):
        return len(self.clips_list)

    def read_video(self, video_file):
        # Open the video file
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            print('Error when opening video file!')
        frames = None
        while (cap.isOpened()):
            ret, frame = cap.read()

            if ret:
                if self.channels == 3:
                    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                else:
                    pil_img = Image.fromarray(frame)
                frame = self.transform(pil_img)
                if frames == None:
                    frames = frame.unsqueeze(0)
                else:
                    frames = torch.cat((frames, frame.unsqueeze(0)), 0)
            else:
                break
        return frames

    def __getitem__(self, idx):

        video_file = self.clips_list[idx]
        clip = self.read_video(video_file)
        sample = {
            "clip": clip
        }

        return sample

def get_data(dataset_used, batch_size, get_mean_std=False):
    if dataset_used == 'CIFAR10':
        data = CIFAR10('datasets/',
                            train=True,
                            transform=transform,
                            download=True)
    elif dataset_used == 'MNIST':
        data = MNIST('datasets/',
                     train=True,
                     transform=gray_scale_transform,
                     download=True)
    elif dataset_used == 'MPII':
        data = MPIIDataSet('datasets/mpii_human_pose_v1')

    if get_mean_std:
        mean = torch.zeros(3)
        std = torch.zeros(3)
        print('Computing mean and std...')
        full_data_loader = DataLoader(data,
                                      batch_size=batch_size,
                                      shuffle=False,
                                      num_workers=os.cpu_count())
        for idx, batch in enumerate(full_data_loader):
            if dataset_used == 'MNIST':
                img, mask = batch[0], None
            elif dataset_used == 'MPII':
                img, mask = batch['image'], batch['mask']
            for i in range(3):
                mean[i] += img[:, i, :, :].mean()
                std[i] += img[:, i, :, :].std()
        mean.div_(idx)
        std.div_(idx)
        print(mean, std)

    data_size = len(data)

    train_size = data_size * 9 // 10
    print("Train size: ", train_size)
    val_size = data_size - train_size

    train_set, val_set = random_split(data, [train_size, val_size])

    train_data_loader = DataLoader(train_set,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=os.cpu_count())
    val_data_loader = DataLoader(val_set,
                                 batch_size=1,
                                 shuffle=True,
                                 num_workers=0)
    return train_data_loader, val_data_loader, batch_size, train_size
