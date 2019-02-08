import cv2
import numpy as np
import requests


class Image:
    suffix = '_o'

    def __init__(self, array=None):
        self.array = array

    @classmethod
    def download(cls, url):
        r = requests.get(url, stream=True)
        if r.status_code == 200:
            image = np.asarray(bytearray(r.content), dtype='uint8')
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        else:
            return cls(None)
        return cls(image)
