import timeit
import time

import numpy as np
import cv2
from PIL import Image
import torchvision.transforms as transforms

import torch
import face_recognition
import onnx

from PIL import Image

from models.face_segmentation.adapter import FaceSegmentation
from models.image_transform.adapter import ImageTransform


def find_faces(image):
    faces = face_recognition.face_locations(image)
    return faces


def get_face_segmentation(img, model: FaceSegmentation):
    mask = model.segment(img)
    return mask


def transform_image(img, model: ImageTransform):
    img = model.transform(img)
    return img


def run(mirror=False):
    cam = cv2.VideoCapture(0)
    seg_model = FaceSegmentation("C:\\Dev\\PyFaceMask\\models\\79999_iter.pth")
    transform_model = ImageTransform("C:\\Dev\\PyFaceMask\\models\\manga-face.pth")

    while True:
        t1 = time.time()
        ret_val, img = cam.read()
        if mirror:
            img = cv2.flip(img, 1)
        im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        faces = find_faces(im_rgb)
        mask = img

        if (len(faces) > 0):
            face = faces[0] # top right bottom left
            face = (face[3] - 20, face[0] - 50, face[1] + 20, face[2])  # left top right bottom

            print(faces)
            mask = get_face_segmentation(im_rgb, seg_model)

            face_img = im_rgb[face[1]:face[3], face[0]:face[2], ...]
            transformed = transform_image(face_img, transform_model)
            #
            mask = mask[face[1]:face[3], face[0]:face[2]]
            face_img[mask.astype(np.bool)] = transformed[mask.astype(np.bool)]
            # face_img = transformed

            transformed_bgr = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
            img[face[1]:face[3], face[0]:face[2], ...] = transformed_bgr
            
        cv2.imshow('my webcam', img)
        if cv2.waitKey(1) == 27:
            break  # esc to quit

        t2 = time.time()
        print("Frame time", t2 - t1)
    cv2.destroyAllWindows()


def profile():
    img = face_recognition.load_image_file('data/test_img.png')
    seg_model = FaceSegmentation('C:\\Dev\\PyFaceMask\\models\\79999_iter.pth')

    transform_model = torch.jit.load('models/traced_manga_80.pt')

    test_number = 10
    face_det_time = timeit.timeit(lambda: find_faces(img), number=test_number)
    print("Face det time", face_det_time / test_number)
    mask_time = timeit.timeit(lambda: get_face_segmentation(img, seg_model), number=test_number)
    print("Mask det time", mask_time / test_number)
    transformed_time = timeit.timeit(lambda: transform_image(img, transform_model), number=test_number)
    print("Transform time", transformed_time / test_number)


def main():
    # profile()
    run(mirror=False)


if __name__ == "__main__":
    main()