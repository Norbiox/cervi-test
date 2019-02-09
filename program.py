#!/usr/bin/env python3
import argparse
import cv2
import imutils
import numpy as np
import requests
from datetime import datetime

import get_coordinates


DATETIME_FORMAT = "%Y-%m-%d_%H-%M-%S"


class Image(np.ndarray):
    suffix = '_o'

    def __new__(cls, array: np.ndarray):
        obj = np.asarray(array).view(cls)
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

    @property
    def height(self):
        return self.shape[0]

    @property
    def width(self):
        return self.shape[1]

    @classmethod
    def combine(cls, image1, image2):
        new_img_height = max(image1.height, image2.height)
        new_img_width = image1.width + image2.width
        array = np.zeros((new_img_height, new_img_width, image1.shape[2]),
                         np.uint8)
        array[:image1.height, :image1.width, :] = image1[:, :, :]
        array[:image2.height, image1.width:new_img_width, :] = image2[:, :, :]
        return Image(array)

    @classmethod
    def download(cls, url):
        r = requests.get(url, stream=True)
        if r.status_code == 200:
            image = np.asarray(bytearray(r.content), dtype='uint8')
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        else:
            return cls(np.array([]))
        return cls(image)

    def save(self):
        filename = ''.join([
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            self.suffix,
            '.png'
        ])
        cv2.imwrite(filename, self)
        return filename


class ProcessedImage(Image):
    suffix = '_p'


class CombinedImage(Image):
    suffix = '_b'

    def __new__(cls, image1, image2):
        array = cls.combine(image1, image2)
        obj = np.asarray(array).view(cls)
        obj.left_image = image1
        obj.right_image = image2
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.left_image = getattr(obj, 'left_image', None)
        self.right_image = getattr(obj, 'right_image', None)


class App:
    options = ["grayscale", "binarize", "inverse", "rotate", "clip"]

    def __init__(self, image_url: str, option=None, parameters=[]):
        self.image_url = image_url
        self.option = option
        self.parameters = parameters

    @property
    def option(self):
        return self._option

    @option.setter
    def option(self, option):
        if option in self.options and callable(getattr(self, option)):
            self._option = option
        else:
            raise ValueError(f"option {option} is not available")

    def process_image(self, image):
        function = getattr(self, self.option)
        return ProcessedImage(function(image, self.parameters))

    def run(self):
        image = Image.download(self.image_url)
        cv2.imshow('image', image)
        pressed_key = cv2.waitKey(0)
        while pressed_key != ord("q"):
            pressed_key = cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Image processing methods

    @staticmethod
    def grayscale(image, parameters=[]):
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def binarize(image, parameters=[]):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if not parameters:
            return cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        elif parameters[0] == 'mean':
            return cv2.adaptiveThreshold(image, 255,
                                         cv2.ADAPTIVE_THRESH_MEAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
        elif parameters[0] == 'gauss':
            return cv2.adaptiveThreshold(image, 255,
                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, 11, 2)
        elif parameters[0] == 'otsu':
            return cv2.threshold(image, 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            raise ValueError(
                f"{parameters[0]} binarization method is not known"
            )

    @staticmethod
    def invert(image, parameters=[]):
        return cv2.bitwise_not(image)

    @staticmethod
    def rotate(image, parameters=[]):
        if not parameters:
            angle = 90
        elif isinstance(parameters[0], int):
            angle = parameters[0]
        else:
            raise ValueError("rotation angle must be integer")
        return imutils.rotate_bound(image, angle)

    @staticmethod
    def clip(image, parameters=[]):
        if not parameters:
            x0, y0, x1, y1 = get_coordinates.main(image)
        elif len(parameters) == 4:
            x0, y0, x1, y1 = parameters
        else:
            raise TypeError("clip method takes 4 arguments: x0, y0, x1, y1")
        return image[y0:y1, x0:x1, :]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cervi Robotics test program")
    parser.add_argument('URL', type=str, help="URL of image to process")
    args = parser.parse_args()

    app = App(args.URL)
    app.run()
