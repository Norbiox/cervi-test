# cervi-test
A recruitment practical test from Cervi Robotics.


## Goal

The goal of this test was to develop a simple command line program able to:

* load a image from web recources
* perform one of the following operations on downloaded image:
    - grayscale conversion
    - binarization
    - color inversion
    - rotation
    - cropping
* display both original and modified images
* save one/both of the images.

## Installation

    git clone https://github.com/Norbiox/cervi-test.git
    

## Usage

Run program with command below:

    $ python program.py URL [option] [parameters [parameters ...]]

where: 

    - 'URL' is the url of picture you want to download and process
    - 'option' is the name of processing method that'll be performed on picture
    - 'parameters' can be any length set of parameters, depended on given option
    
After that program will download a picture from recource, perform a operation on it and display both, original and processed images. Then you can use your keyboard to save pictures or close a program, according to the keymap below:

| key  | action                        |
| ---- | ----------------------------- |
| 'o'  | save original image           |
| 'p'  | save processed image          |
| 'b'  | save both images in one file  |
| 'q'  | quit                          |

NOTE: if you'll run program without giving any option, program will show you only original picture and you'll be able to save it.

### Available processing options

| option name | parameters | description |
| ----------- | ---------- | ----------- |
| grayscale   | -          | convert image to grayscale |
| binarize    | - / 'mean' / 'gauss' / 'otsu' | creates binary image by simple thresholding (by default) or using one of available adaptive thresholding methods |
| invert      | -          | performs colors inversion on image |
| rotate      | angle: int | rotates image clockwise by angle given in degrees, default 90 |
| clip        | - / [x0 y0 x1 y1] | clip image, takes (x,y) coordinates of upper-left and lower-right corners of clipped image, if not provided program displays image in interactive clipping mode |
