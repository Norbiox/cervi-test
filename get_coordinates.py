#!/usr/bin/env python3
import autopy
import cv2


coordinates = []


def normalize(c):
    x0, x1 = sorted([c[0], c[2]])
    y0, y1 = sorted([c[1], c[3]])
    return [x0, y0, x1, y1]


def crop(event, x, y, flags, param):
    global coordinates
    if event == cv2.EVENT_LBUTTONDOWN:
        coordinates = [x, y]
    elif event == cv2.EVENT_LBUTTONUP:
        coordinates += [x, y]
        autopy.key.type_string('x')


def main(image):
    cv2.namedWindow('click&drag')
    cv2.setMouseCallback('click&drag', crop)
    cv2.imshow('click&drag', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return normalize(coordinates)
