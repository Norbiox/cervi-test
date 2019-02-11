#!/usr/bin/env python3
import autopy
import cv2

image = None
clipping = False
coordinates = []


def normalize(c):
    x0, x1 = sorted([c[0], c[2]])
    y0, y1 = sorted([c[1], c[3]])
    return [x0, y0, x1, y1]


def crop(event, x, y, flags, param):
    global clipping, coordinates
    if event == cv2.EVENT_LBUTTONDOWN:
        coordinates = [x, y]
        clipping = True
    elif event == cv2.EVENT_LBUTTONUP:
        coordinates += [x, y]
        clipping = False
        autopy.key.type_string('x')
    elif clipping:
        img = image.copy()
        cv2.rectangle(img, (coordinates[0], coordinates[1]),
                      (x,y), (0, 255, 0), 2)
        cv2.imshow('click&drag to clip image', img)


def main(img):
    global image
    image = img.copy()
    cv2.namedWindow('click&drag to clip image')
    cv2.setMouseCallback('click&drag to clip image', crop)
    cv2.imshow('click&drag to clip image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return normalize(coordinates)

