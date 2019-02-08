import numpy as np
from program import Image, CombinedImage


URL = 'https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png'

IMAGE1 = Image(np.array([[1, 2, 3],
                         [4, 5, 6],
                         [7, 8, 9]]))
IMAGE2 = Image(np.array([[10, 11],
                         [12, 13]]))


def test_image_download():
    img = Image.download(URL)
    assert isinstance(img, Image)
    assert isinstance(img.array, np.ndarray)
    assert img.array is not None


def test_image_download_from_non_existing_url():
    img = Image.download(URL+'s')
    assert isinstance(img, Image)
    assert img.array.shape == (0, )


def test_combining_images():
    combined_image = Image.combine(IMAGE1, IMAGE2)
    assert isinstance(combined_image, Image)
    assert combined_image.array.shape == (3, 5)
    assert combined_image.array[2, 1] == 8
    assert combined_image.array[1, 3] == 12
    assert combined_image.array[2, 3] == 0
