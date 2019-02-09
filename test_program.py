import numpy as np
import os
import pytest
from program import (App, Image, CombinedImage)


URL = 'https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png'

ARRAY1 = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9],
                   [10, 11, 12]], dtype=np.uint8)
ARRAY2 = np.array([[10, 11],
                   [12, 13]], dtype=np.uint8)

IMAGE1 = Image(np.dstack((ARRAY1, ARRAY1, ARRAY1)))
IMAGE2 = Image(np.dstack((ARRAY2, ARRAY2, ARRAY2)))


def test_image_shape():
    image = IMAGE1
    assert image.width == 3
    assert image.height == 4


def test_image_download():
    img = Image.download(URL)
    assert isinstance(img, Image)
    assert img.size > 0


def test_image_download_from_non_existing_url():
    with pytest.raises(Exception):
        img = Image.download(URL+'s')


def test_combining_images():
    combined_image = Image.combine(IMAGE1, IMAGE2)
    assert isinstance(combined_image, Image)
    assert combined_image.shape == (4, 5, 3)
    assert combined_image[2, 1, 0] == 8
    assert combined_image[1, 3, 0] == 12
    assert combined_image[2, 3, 0] == 0


def test_combining_colorful_and_grayscale_images():
    combined_image = Image.combine(IMAGE1, Image(ARRAY1))
    assert isinstance(combined_image, Image)
    assert combined_image.shape == (4, 6, 3)


def test_combined_image():
    combined_image = CombinedImage(IMAGE1, IMAGE2)
    assert isinstance(combined_image, CombinedImage)
    assert combined_image.left_image.all() == IMAGE1.all()
    assert combined_image.right_image.all() == IMAGE2.all()
    assert combined_image.shape == (4, 5, 3)
    assert combined_image[2, 1, 0] == 8
    assert combined_image[1, 3, 0] == 12
    assert combined_image[2, 3, 0] == 0


def test_saving_image():
    image = IMAGE1
    fname = image.save()
    assert os.path.exists(fname)
    os.remove(fname)


def test_app_options():
    existing_option = "grayscale"
    non_existing_option = "play_sound"
    App(URL, existing_option)
    App(URL, '')
    with pytest.raises(ValueError):
        App(URL, non_existing_option)


def test_grayscale_method():
    image = IMAGE1
    App.grayscale(image)


def test_binarize_method():
    image = IMAGE1
    App.binarize(image)
    App.binarize(image, ['mean'])
    App.binarize(image, ['gauss'])
    App.binarize(image, ['otsu'])
    with pytest.raises(ValueError):
        App.binarize(image, ['non_existing_method'])


def test_inversion():
    image = IMAGE1
    App.invert(image, [])


def test_rotation():
    image = IMAGE1
    img_rot90 = App.rotate(image, ['90'])
    assert img_rot90.shape == (3, 4, 3)
    img_rot180 = App.rotate(image, ['180'])
    assert img_rot180.shape == (4, 3, 3)
    img_rot270 = App.rotate(image, ['270'])
    assert img_rot270.shape == (3, 4, 3)
    with pytest.raises(ValueError):
        App.rotate(image, ['sdf'])


def test_clipping_with_parameters():
    image = IMAGE1
    clipped_img = App.clip(image, ['0', '1', '2', '3'])
    assert clipped_img.shape == (2, 2, 3)
    assert clipped_img[0, 0, 0] == image[1, 0, 0]
    with pytest.raises(TypeError):
        App.clip(image, ['1', '4'])


# def test_clipping_without_parameters():
#     image = IMAGE1
#     App.clip(image, [])
