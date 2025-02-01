import cv2
import numpy as np
from torchvision import transforms
import torch
import torchio as tio
import random
from time import time
import matplotlib.pyplot as plt

class RescaleTo01(object):
    def __call__(self, image):
        return(image - np.min(image)) / (np.max(image) - np.min(image))

class SimCLRTrainDataTransform(object):
    """
    Transforms for SimCLR
    Transform::
        RandomResizedCrop(size=self.input_height)
        RandomHorizontalFlip()
        RandomApply([color_jitter], p=0.8)
        RandomGrayscale(p=0.2)
        GaussianBlur(kernel_size=int(0.1 * self.input_height))
        transforms.ToTensor()
    Example::
        from pl_bolts.models.self_supervised.simclr.transforms import SimCLRTrainDataTransform
        transform = SimCLRTrainDataTransform(input_height=32)
        x = sample()
        (xi, xj) = transform(x)
    """

    def __init__(
        self, input_height: int = 224, gaussian_blur: bool = False, jitter_strength: float = 1., normalize=None
    ) -> None:

        self.jitter_strength = jitter_strength
        self.input_height = input_height
        self.gaussian_blur = gaussian_blur
        self.normalize = normalize

        self.color_jitter = transforms.ColorJitter(
            0.6 * self.jitter_strength, 0.6 * self.jitter_strength
        )

        data_transforms = [
            transforms.ToTensor(),
            transforms.RandomResizedCrop(size=(self.input_height,self.input_height), scale=(0.2, 1)),
            #transforms.ColorJitter(brightness=0.5 * self.jitter_strength, contrast= 0.5 * self.jitter_strength),
        ]

        if self.gaussian_blur:
            kernel_size = int(0.1 * self.input_height)
            if kernel_size % 2 == 0:
                kernel_size += 1

            data_transforms.append(GaussianBlur(kernel_size=kernel_size, p=0.5))

        data_transforms = transforms.Compose(data_transforms)

        if normalize is None:
            self.final_transform = transforms.ToTensor()
        else:
            self.final_transform = transforms.Compose([transforms.ToTensor(), normalize])

        self.train_transform = transforms.Compose([data_transforms])
        
        # add online train transform of the size of global view
        self.online_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomResizedCrop((self.input_height, self.input_height)),
        ])

    def __call__(self, sample):
        transform = self.train_transform
        xi = transform(sample)
        xj = transform(sample)
        return xi, xj, self.online_transform(sample)


class SimCLREvalDataTransform(SimCLRTrainDataTransform):
    """
    Transforms for SimCLR
    Transform::
        Resize(input_height + 10, interpolation=3)
        transforms.CenterCrop(input_height),
        transforms.ToTensor()
    Example::
        from pl_bolts.models.self_supervised.simclr.transforms import SimCLREvalDataTransform
        transform = SimCLREvalDataTransform(input_height=32)
        x = sample()
        (xi, xj) = transform(x)
    """

    def __init__(
        self, input_height: int = 224, gaussian_blur: bool = False, jitter_strength: float = 1., normalize=None
    ):
        super().__init__(
            normalize=normalize,
            input_height=input_height,
            gaussian_blur=gaussian_blur,
            jitter_strength=jitter_strength
        )

        # replace online transform with eval time transform
        self.online_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(int(self.input_height + 0.1 * self.input_height)),
            transforms.CenterCrop(self.input_height)
        ])


class GaussianBlur(object):
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, p=0.5, min=0.1, max=2.0):

        
        self.min = min
        self.max = max

        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size
        self.p = p

    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < self.p:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return torch.from_numpy(sample)


class SimCLR3SlicesTransform(object):
    def __init__(self, input_height: int = 224):
        self.volume_motion_list = []
        self.volume_motion_indices = []

        #self.motion = tio.transforms.RandomMotion(num_transforms=2, degrees=(-7.5,7.5), translation=(-7.5,7.5))
        self.motion_1 = tio.Motion(degrees=np.array([[0.5, 0.5, 0.5]]), translation=np.array([[0.5, 0.5, 0.5]]),
                          times=np.array([0.5]), image_interpolation='linear')
        self.motion_2 = tio.Motion(degrees=np.array([[1.0, 1.0, 1.0]]), translation=np.array([[1.0, 1.0, 1.0]]),
                          times=np.array([0.5]), image_interpolation='linear')
        self.motion_3 = tio.Motion(degrees=np.array([[1.5, 1.5, 1.5]]), translation=np.array([[1.5, 1.5, 1.5]]),
                          times=np.array([0.5]), image_interpolation='linear')
        self.motion_4 = tio.Motion(degrees=np.array([[2.0, 2.0, 2.0]]), translation=np.array([[2.0, 2.0, 2.0]]),
                          times=np.array([0.5]), image_interpolation='linear')
        self.motion_5 = tio.Motion(degrees=np.array([[2.5, 2.5, 2.5]]), translation=np.array([[2.5, 2.5, 2.5]]),
                          times=np.array([0.5]), image_interpolation='linear')
        self.motion_6 = tio.Motion(degrees=np.array([[3.0, 3.0, 3.0]]), translation=np.array([[3.0, 3.0, 3.0]]),
                          times=np.array([0.5]), image_interpolation='linear')

        self.motion_list = [self.motion_1, self.motion_2, self.motion_3, self.motion_4, self.motion_5]#, self.motion_6]

        self.input_height = input_height
        rescale = RescaleTo01()

        data_transforms = [
            rescale,
            transforms.ToTensor(),
            transforms.RandomResizedCrop(size=(self.input_height,self.input_height), scale=(0.2, 1), ratio=(1.0, 1.0))
        ]

        data_transforms = transforms.Compose(data_transforms)
        self.train_transform = transforms.Compose([data_transforms])

        transform3D = tio.transforms.Compose([
            tio.transforms.ZNormalization(masking_method=None),
        ])

    
    def __call__(self, volume, idx):
        #decider = random.choice([0,1])
        decider = 0
        # if idx in self.volume_motion_indices:
        #     volume_motion_index = self.volume_motion_indices.index(idx)
        #     volume_motion = self.volume_motion_list[volume_motion_index]
        # else: 
        #     self.volume_motion_indices.append(idx)
        #     motion = random.choice(self.motion_list)
        #     #motion = tio.Compose([tio.transforms.RandomMotion(num_transforms=1, )])
        #     volume_motion = motion(np.expand_dims(volume, axis=0))
        #     volume_motion = np.squeeze(volume_motion, axis=0)
        #     self.volume_motion_list.append(volume_motion)

        
        slice_idx = np.random.randint(volume.shape[2])
        slice_img = volume[:,:,slice_idx]
        #slice_motion = volume_motion[:,:,slice_idx]

        #slice_img = np.pad(slice_img, ((0,0), (28,28)))
        #slice_motion = np.pad(slice_motion, ((0,0), (28,28)))
        
        if decider == 0:
            x_i = self.train_transform(slice_img)
            x_j = self.train_transform(slice_img)
            #x_z = self.train_transform(slice_motion)

        elif decider == 1:
            #x_i = self.train_transform(slice_motion)
            #x_j = self.train_transform(slice_motion)
            x_z = self.train_transform(slice_img)

        return x_i, x_j#, x_z