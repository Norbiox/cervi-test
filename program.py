#!/usr/bin/env python3
import argparse
import cv2
import numpy as np
import requests


class Image:
    suffix = '_o'

    def __init__(self, array: np.ndarray):
        self.array = array

    @property
    def height(self):
        return self.array.shape[0]

    @property
    def width(self):
        return self.array.shape[1]

    @classmethod
    def combine(cls, image1, image2):
        new_image_height = max(image1.height, image2.height)
        new_image_width = image1.width + image2.width
        array = np.zeros((new_image_height, new_image_width), np.uint8)
        array[:image1.height, :image1.width] = image1.array
        array[:image2.height, image1.width:new_image_width] = image2.array
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


class ProcessedImage(Image):
    suffix = '_p'

    def __init__(self, array, origin_image=None):
        self.array = array
        self.origin = origin_image


class CombinedImage(Image):
    suffix = '_b'

    def __init__(self, image1, image2):
        self.left_image = image1
        self.right_image = image2
        self.array = self.combine(image2, image2).array


class App:

    def __init__(self, image_url, option=None, parameters=[]):
        self.image_url = image_url
        self.option = option
        self.parameters = parameters

    def run(self):
        image = Image.download(self.image_url)
        cv2.imshow('image', image.array)
        pressed_key = cv2.waitKey(0)
        while pressed_key != ord("q"):
            pressed_key = cv2.waitKey(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cervi Robotics test program")
    parser.add_argument('URL', type=str, help="URL of image to process")
    args = parser.parse_args()

    app = App(args.URL)
    app.run()
