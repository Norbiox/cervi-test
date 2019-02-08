import numpy as np
from program import Image, CombinedImage


URL = 'https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png'

ARRAY1 = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9],
                   [10, 11, 12]])
ARRAY2 = np.array([[10, 11],
                   [12, 13]])


def test_image_shape():
    image = Image(ARRAY1)
    assert image.width == 3
    assert image.height == 4


def test_image_download():
    img = Image.download(URL)
    assert isinstance(img, Image)
    assert isinstance(img, np.ndarray)
    assert img.size > 0


def test_image_download_from_non_existing_url():
    img = Image.download(URL+'s')
    assert isinstance(img, Image)
    assert img.shape == (0, )


def test_combining_images():
    combined_image = Image.combine(Image(ARRAY1), Image(ARRAY2))
    assert isinstance(combined_image, Image)
    assert combined_image.shape == (4, 5)
    assert combined_image[2, 1] == 8
    assert combined_image[1, 3] == 12
    assert combined_image[2, 3] == 0


def test_combined_image():
    image1, image2 = Image(ARRAY1), Image(ARRAY2)
    combined_image = CombinedImage(image1, image2)
    assert isinstance(combined_image, CombinedImage)
    assert combined_image.left_image.all() == image1.all()
    assert combined_image.right_image.all() == image2.all()
    assert combined_image.shape == (4, 5)
    assert combined_image[2, 1] == 8
    assert combined_image[1, 3] == 12
    assert combined_image[2, 3] == 0
