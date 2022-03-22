import cv2

import torch
from torchvision.transforms import transforms

from .image_transform_net import ImageTransformNet


class ImageTransform:
    def __init__(self, weights_path, device='cpu'):
        self.device = device
        self.net = ImageTransformNet()
        self.net.load_state_dict(torch.load(weights_path, map_location=torch.device(device)))

    def transform(self, image):
        orig_shape = image.shape
        image = cv2.resize(image, (320, 240))
        with torch.no_grad():
            image_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x * 255)])

            image_tensor = image_transform(image).unsqueeze(0)
            if self.device != 'cpu':
                image_tensor = image_tensor.cuda()

            transformed = self.net(image_tensor).squeeze(0).numpy().clip(0, 255).transpose(1, 2, 0)
            transformed = cv2.resize(transformed, (orig_shape[1], orig_shape[0]))

        return transformed

    def __call__(self, frame):
        return self.transform(frame)