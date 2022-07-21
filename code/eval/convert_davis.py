import os
import numpy as np
import cv2
from os.path import expanduser
home = expanduser("~")

from PIL import Image

jpglist = []

import palette
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-o', '--out_folder', default='/scratch/ajabri/davis_results/', type=str)
parser.add_argument('-i', '--in_folder', default='/scratch/ajabri/davis_results_masks/', type=str)
parser.add_argument('-d', '--dataset', default='/scratch/ajabri/data/davis/', type=str)

args = parser.parse_args()
args.out_folder = args.out_folder.replace('~', home)
args.in_folder = args.in_folder.replace('~', home)
args.dataset = args.dataset.replace('~', home)

annotations_folder = os.path.join(args.dataset, 'Annotations/480p/')
f1 = open(os.path.join(args.dataset, 'ImageSets/2017/val.txt'), 'r')
for line in f1:
    line = line[:-1]
    jpglist.append(line)
f1.close()

out_folder = args.out_folder
current_folder = args.in_folder

if not os.path.exists(out_folder):
    os.makedirs(out_folder)

palette = palette.tensor.astype(np.uint8)


def color2id(c):
    return np.arange(0, palette.shape[0])[np.all(palette == c, axis=-1)]


def convert_dir(i):
    fname = jpglist[i]

    gtfolder = os.path.join(annotations_folder, fname)
    outfolder = os.path.join(out_folder, fname)

    if not os.path.exists(outfolder):
        os.mkdir(outfolder)

    files = [_ for _ in os.listdir(gtfolder) if _[-4:] == '.png']

    img_name = os.path.join(gtfolder, "{:05d}.png".format(0))
    lblimg = cv2.imread(img_name)
    height = lblimg.shape[0]
    width = lblimg.shape[1]

    for j in range(len(files)):
        outname = os.path.join(outfolder, "{:05d}.png".format(j))
        inname = os.path.join(current_folder, str(i) + '_' + str(j) + '_mask.png')

        lblimg = cv2.imread(inname)
        flat_lblimg = lblimg.reshape(-1, 3)
        lblidx = np.zeros((lblimg.shape[0], lblimg.shape[1]))
        lblidx2 = np.zeros((lblimg.shape[0], lblimg.shape[1]))

        colors = np.unique(flat_lblimg, axis=0)

        for c in colors:
            cid = color2id(c)
            if len(cid) > 0:
                lblidx2[np.all(lblimg == c, axis=-1)] = cid

        lblidx = lblidx2

        lblidx = lblidx.astype(np.uint8)
        lblidx = cv2.resize(lblidx, (width, height), interpolation=cv2.INTER_NEAREST)
        lblidx = lblidx.astype(np.uint8)

        im = Image.fromarray(lblidx)
        im.putpalette(palette.ravel())
        im.save(outname, format='PNG')


import multiprocessing as mp
pool = mp.Pool(10)
results = pool.map(convert_dir, range(len(jpglist)))
