import os
from torchvision import transforms
import cv2
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.io import loadmat

video_extensions = ['.avi', '.mpg']
image_extensions = ['.jpg']

transform = transforms.Compose([transforms.Resize([224, 224]),
                                transforms.ToTensor()])

class MPIIDataSet(Dataset):
    def __init__(self, dir, transform=transform):
        self.dir = dir
        self.transform = transform
        self.images_list = []
        for root, dirs, files in os.walk(dir):
            for file in files:
                file_name, file_extension = os.path.splitext(file)
                if file_extension in image_extensions:
                    self.images_list.append(root + "/" + file)

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        img_loc = self.images_list[idx]
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image

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