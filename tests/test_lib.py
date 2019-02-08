import numpy as np
from cervi import lib


URL = 'https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png'


def test_image_download():
    img = lib.Image.download(URL)
    assert isinstance(img, lib.Image)
    assert isinstance(img.array, np.ndarray)
    assert img.array is not None


def test_image_download_from_non_existing_url():
    img = lib.Image.download(URL+'s')
    assert isinstance(img, lib.Image)
    assert img.array is None
