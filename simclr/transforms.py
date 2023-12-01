import cv2
import numpy as np
from torchvision import transforms
import torch

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
            0.8 * self.jitter_strength, 0.8 * self.jitter_strength, 0.8 * self.jitter_strength,
            0.2 * self.jitter_strength
        )

        data_transforms = [
            transforms.ToTensor(),
            transforms.RandomResizedCrop(size=(self.input_height,self.input_height), scale=(0.7, 1)),
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
