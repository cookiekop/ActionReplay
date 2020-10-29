import os
from torchvision import transforms
import cv2
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, ImageFolder, MNIST, FashionMNIST
from torch.utils.data import DataLoader, random_split
import json
from scipy.io import loadmat
from matplotlib.image import imsave

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

        return {'image': tensor_image,
                'mask': mask/mask.mean()}

class UTDVideo(Dataset):
    def __init__(self, dir):
        self.dir = dir
        self.files = []
        self.max_frame = 40
        for root, dirs, files in os.walk(self.dir):
            for name in files:
                self.files.append(name)
        self.transform = gray_scale_transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        action = file.split('_')[0]
        action_class = int(action[1:])
        mat = loadmat(self.dir + file)['d_depth'].transpose(2, 0, 1)
        frame_num = mat.shape[0]
        clip = mat[frame_num//2 - self.max_frame//2: frame_num//2 + self.max_frame//2]
        frame_num = clip.shape[0]
        frames = torch.zeros([frame_num, 3, 224, 224])
        for i in range(frame_num):
            frame = clip[i]
            pil_img = Image.fromarray(frame).convert('L')
            frame = self.transform(pil_img)
            frames[i] = frame

        return {'clip': frames,
                'label': action_class}

class GeneralVideoDataset(Dataset):
    """Dataset Class for Loading Video"""

    def __init__(
        self,
        dir,
        channels=3,
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

session_num = 5 # 0 for whole dataset learning, 1-5 for 5 class-inc learning
def train_collate(batch):
    data = None
    target = []
    for item in batch:
        if item[1] < (session_num-1)*2:
            continue
        if item[1] >= session_num*2:
            continue
        if data is None:
            data = item[0].unsqueeze(0)
        else:
            data = torch.cat((data, item[0].unsqueeze(0)), 0)
        target.append(item[1])
    return [data, torch.LongTensor(target)]

def val_collate(batch):
    data = None
    target = []
    for item in batch:
        if item[1] >= session_num*2:
            continue
        if data is None:
            data = item[0].unsqueeze(0)
        else:
            data = torch.cat((data, item[0].unsqueeze(0)), 0)
        target.append(item[1])
    return [data, torch.LongTensor(target)]

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
    elif dataset_used == 'FashionMNIST':
        data = FashionMNIST('datasets/',
                            train=True,
                            transform=gray_scale_transform,
                            download=True)
    elif dataset_used == 'MPII':
        data = MPIIDataSet('datasets/mpii_human_pose_v1')
    elif dataset_used == 'UTD':
        data = ImageFolder('datasets/UTD-MHAD/Image/',
                           transform=transform)
    elif dataset_used == 'UTDVideo':
        data = UTDVideo('datasets/UTD-MHAD/Depth/')

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

    train_collate_func = None if session_num == 0 else train_collate
    val_collate_func = None if session_num == 0 else val_collate
    train_data_loader = DataLoader(train_set,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   num_workers=os.cpu_count(),
                                   collate_fn=train_collate_func)
    val_data_loader = DataLoader(val_set,
                                 batch_size=1,
                                 shuffle=True,
                                 num_workers=0,
                                 collate_fn=val_collate_func)
    return train_data_loader, val_data_loader, train_size

def utd2image(rootdir):
    frame_interval = 2
    count = 0
    if not os.path.exists(rootdir+'/Image/'):
        os.makedirs(rootdir+'/Image/')
    for root, dirs, files in os.walk(rootdir+'/Depth/'):
        for name in files:
            action = name.split('_')[0]
            if not os.path.exists(rootdir + '/Image/'+action):
                os.makedirs(rootdir + '/Image/'+action)
            mat = loadmat(root+name)['d_depth'].transpose(2, 0, 1)
            for i in range(0, mat.shape[0], frame_interval):
                frame = mat[i]
                imsave(rootdir+'/Image/'+action+'/'+str(count)+'.png', frame, cmap='gray')
                count += 1
    print('Total Image Num:{}'.format(count))

# utd2image('datasets/UTD-MHAD')