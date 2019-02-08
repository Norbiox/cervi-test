#!/usr/bin/env python3
import argparse
import cv2
import numpy as np
import requests
from datetime import datetime


DATETIME_FORMAT = "%Y-%m-%d_%H-%M-%S"


class Image(np.ndarray):
    suffix = '_o'

    def __new__(cls, array: np.ndarray):
        obj = np.asarray(array).view(cls)
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return

    @property
    def height(self):
        return self.shape[0]

    @property
    def width(self):
        return self.shape[1]

    @classmethod
    def combine(cls, image1, image2):
        new_image_height = max(image1.height, image2.height)
        new_image_width = image1.width + image2.width
        array = np.zeros((new_image_height, new_image_width), np.uint8)
        array[:image1.height, :image1.width] = image1[:, :]
        array[:image2.height, image1.width:new_image_width] = image2[:, :]
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
        print(np.asarray(self))
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
        if obj is None: return
        self.left_image = getattr(obj, 'left_image', None)
        self.right_image = getattr(obj, 'right_image', None)


class App:

    def __init__(self, image_url, option=None, parameters=[]):
        self.image_url = image_url
        self.option = option
        self.parameters = parameters

    def run(self):
        image = Image.download(self.image_url)
        cv2.imshow('image', image)
        pressed_key = cv2.waitKey(0)
        while pressed_key != ord("q"):
            pressed_key = cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cervi Robotics test program")
    parser.add_argument('URL', type=str, help="URL of image to process")
    args = parser.parse_args()

    app = App(args.URL)
    app.run()
