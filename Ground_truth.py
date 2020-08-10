import os
import cv2
import numpy as np
import csv
import re



os.chdir('E:/Stero Rectification/Motorcycle-perfect')
print(os.listdir())
file = open("disp1.pfm","r")
def read_calib(calib_file_path):
    with open(calib_file_path, 'r') as calib_file:
        calib = {}
        csv_reader = csv.reader(calib_file, delimiter='=')
        for attr, value in csv_reader:
            calib.setdefault(attr, value)

    return calib


def read_pfm(pfm_file_path):
    with open(pfm_file_path, 'rb') as pfm_file:
        header = pfm_file.readline().decode().rstrip()
        channels = 3 if header == 'PF' else 1

        dim_match = re.match(r'^(\d+)\s(\d+)\s$', pfm_file.readline().decode())
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception("Malformed PFM header.")

        scale = float(pfm_file.readline().decode().rstrip())
        if scale < 0:
            endian = '<' # littel endian
            scale = -scale
        else:
            endian = '>' # big endian

        dispariy = np.fromfile(pfm_file, endian + 'f') / scale

    return dispariy, [(height, width, channels), scale]


def create_depth_map(pfm_file_path, calib=None):

    dispariy, [shape,scale] = read_pfm(pfm_file_path)

    if calib is None:
        raise Exception("Loss calibration information.")
    else:
        fx = float(calib['cam0'].split(' ')[0].lstrip('['))

        base_line = float(calib['baseline'])
        doffs = float(calib['doffs'])

        depth_map = fx*base_line / (dispariy + doffs)

        depth_map = np.reshape(depth_map, newshape=shape)

        depth_map = np.flipud(depth_map).astype('uint8')

        return depth_map

calib = read_calib("calib.txt")
print(calib)
depth_map = create_depth_map("disp1.pfm",calib)
depth_map = cv2.resize(depth_map,(800,800),interpolation=cv2.INTER_CUBIC)

cv2.imshow("depth",depth_map)
cv2.imwrite("Ground_truth.png",depth_map)
cv2.waitKey(0)
cv2.destroyAllWindows()