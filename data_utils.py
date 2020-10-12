import os
from torchvision import transforms
import cv2
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat
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
        mask[:, tl_[0]:br_[0], tl_[1]:br_[1]] = 5.0
        return {'image': tensor_image, 'mask': mask/mask.mean()}

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

def get_data(dataset_used, batch_size):
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

    data_size = len(data)

    train_size = data_size * 9 // 10
    print("Train size: ", train_size)
    val_size = data_size - train_size

    train_set, val_set = random_split(data, [train_size, val_size])

    train_data_loader = DataLoader(train_set,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=4)
    val_data_loader = DataLoader(val_set,
                                 batch_size=1,
                                 shuffle=True,
                                 num_workers=0)
    return train_data_loader, val_data_loader, batch_size, train_size