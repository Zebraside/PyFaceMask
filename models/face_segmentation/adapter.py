import time

import numpy as np
import cv2

import torch
import torchvision.transforms as transforms

from .model import BiSeNet


class FaceSegmentation:
    def __init__(self, weights_path, n_classes=19, device='cpu'):
        self.device = 'cpu'
        self.net = BiSeNet(n_classes=n_classes)
        self.net.load_state_dict(torch.load(weights_path, map_location=torch.device(device)))
        self.net.eval()

        if device != 'cpu':
            self.net.cuda()

    def segment(self, img: np.array):
        to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        orig_shape = img.shape
        with torch.no_grad():
            image = cv2.resize(img, (512, 512))
            img = to_tensor(image)
            img = torch.unsqueeze(img, 0)
            if self.device != 'cpu':
                img = img.cuda()

            t1 = time.time()
            out = self.net(img)[0]
            t2 = time.time()
            print("Inference time", t2 - t1)

            parsing = out.squeeze(0).cpu().numpy().argmax(0)
            parsing = np.where(parsing < 16, parsing, 0)
        parsing = cv2.resize(parsing, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_NEAREST)
        return parsing
