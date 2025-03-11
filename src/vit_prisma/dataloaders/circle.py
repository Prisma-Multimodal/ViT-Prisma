import torch
import torch.nn as nn

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from itertools import combinations
import torchvision.transforms as transforms

import numpy as np

from PIL import Image, ImageDraw
import math

import random

import os

from torchvision.transforms.transforms import Grayscale, ToTensor

def check_if_valid_angle(angle, angle_range):
    if angle < 0 or angle > 360:
        raise ValueError('Angle must be between 0 and 360')
    
    # Check if in range
    if not np.isin(angle, angle_range):
        raise ValueError('Angle must at correct interval of angle_range')
    
# def get_datasets(split_ratio, model_type):
#     circle_metadata = get_circle_metadata()
#     # Create train and test data once
#     train_data_raw, test_data_raw = get_train_test_data(circle_metadata, split_ratio) 
#     train_dataset = CircleDataset(circle_metadata, train_data_raw, train_or_test='train', model_type = model_type)
#     test_dataset = CircleDataset(circle_metadata, test_data_raw, train_or_test='test', model_type = model_type)
#     return train_dataset, test_dataset

def get_train_test_data(circle_metadata, split_ratio=0.5):
    data = list(combinations(range(0, circle_metadata['mod_arith']), 2))
    random.shuffle(data)
    split_idx = int(len(data) * split_ratio)
    return data[:split_idx], data[split_idx:]
    
def get_circle_metadata():
    circle_metadata = {
    "mod_arith": 60,
    "image_size": 32,
    "center": (16, 16),
    "radius": 15.5,
    "multiplier": 6,
    "angle_range": np.arange(0, 60, 1)
    } 
    return circle_metadata

def draw_circle_with_points(angle1=None, angle2=None, metadata=None, model_type=None):

    # Create a new image with white background
    image_size = metadata['image_size']
    img = Image.new('L', (image_size, image_size), color=255) # white image

    pixels = img.load()

    center = metadata['center']
    radius = metadata['radius']

    # Transform angle to get point in circle
    MULTIPLIER = metadata['multiplier']
    angle_range = metadata['angle_range']
    transformed_angle_range = angle_range * MULTIPLIER

    # Draw circle in black.
    for i in transformed_angle_range:
        x = center[0] + radius * math.cos(math.radians(i))
        y = center[1] + radius * math.sin(math.radians(i))
        pixels[x, y] = 0

    # Specify the angle
    def _draw_point_(angle, color=128):
        angle_rad = math.radians(angle) # x3 b/c representing the circle in intervals of 3
        x = center[0] + radius * math.cos(angle_rad)
        y = center[1] + radius * math.sin(angle_rad)
        pixels[x, y] = color

    if angle1 is not None:
        check_if_valid_angle(angle1, angle_range)
        _draw_point_(angle1*MULTIPLIER)
    if angle2 is not None:
        check_if_valid_angle(angle2, angle_range)
        _draw_point_(angle2*MULTIPLIER)


    if model_type == 'pretrained_transformer':

        # Define padding dimensions
        left_padding = (224 - 32) // 2
        top_padding = (224 - 32) // 2
        right_padding = 224 - 32 - left_padding
        bottom_padding = 224 - 32 - top_padding

        transform = transforms.Compose([
        transforms.Pad((left_padding, top_padding, right_padding, bottom_padding), fill=1),  # Assuming white padding,  # Resize the image to 224x224
        transforms.Grayscale(num_output_channels=3),  
        transforms.ToTensor()           # Convert the PIL Image to a tensor
    ])
    else:
        transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1]
    ])

    img = transform(img)
    
    return img 

class CircleDataset(Dataset):
    def __init__(self, train_or_test='test', 
    cache_path = '/home/institution/s/firstname.joseph/ViT-Planetarium/data/circle',
    transform=None):

        self.cache_path = cache_path
        self.train_or_test = train_or_test

        # self.circle_metadata = circle_metadata
        # self.mod_arith = circle_metadata['mod_arith']
        # self.data = [{'data': item, 'metadata': i} for i, item in enumerate(data)]
        # self.model_type = model_type

        self.transform=transform

        if os.path.exists(f'{cache_path}/{train_or_test}.npz'):
            print("Loading circle dataset from cache...", f'{cache_path}/{train_or_test}.npz')
            self._load_from_cache()
        else:
            print("Generating and saving new circle dataset...")
            self._generate_and_cache()

    def _load_from_cache(self):
        loaded = np.load(f'{self.cache_path}/{self.train_or_test}.npz', allow_pickle=True)
        self.imgs = loaded['imgs']
        self.labels = loaded['labels']
        self.data_points = loaded['data_points']

    def __len__(self):
        return len(self.imgs)

    def _generate_and_cache(self):
        imgs = []
        labels = []
        data_points = []
        
        for idx in range(len(self.data)):
            data_point = self.data[idx]['data']
            label = sum(data_point) % self.mod_arith
            img = draw_circle_with_points(data_point[0], data_point[1], self.circle_metadata, model_type=self.model_type)
            
            imgs.append(img)
            labels.append(label)
            data_points.append(data_point)

        self.imgs = np.array(imgs)
        self.labels = np.array(labels)
        self.data_points = np.array(data_points)

        # Cache the dataset
        np.savez(f'{self.cache_path}/{self.train_or_test}.npz', imgs=self.imgs, labels=self.labels, data_points=self.data_points)

    def __getitem__(self, idx):
        image = self.imgs[idx]
        label = self.labels[idx]
        # meta_data = self.metadata[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

