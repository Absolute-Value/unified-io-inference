import matplotlib.pylab as plt
import numpy as np
from torchvision.io import read_image
import urllib.request
from PIL import Image

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from absl import logging
logging.set_verbosity(logging.INFO)

%cd /home/jikuya/unified-io-inference/uio
import utils
import runner

%cd /home/jikuya/unified-io-inference
uio = runner.ModelRunner("base", "base.bin")
# uio = runner.ModelRunner("large", "large.bin")
# uio = runner.ModelRunner("xl", "xl.bin")

import matplotlib.patches as patches

def show_images(images, nrow=3, scale=6):
    if len(images.shape) == 4 and images.shape[0] == 1:
        images = images[0]
    if len(images.shape) == 3:
        fig, ax = plt.subplots()
        ax.set_xticklabels([])
        ax.set_yticklabels([])    
        ax.imshow(images)
        plt.show()
    else:
        n = images.shape[0]
        ncol = (n + nrow - 1) // nrow
        if ncol == 1:
            nrow = min(nrow, n)
        fig, axes = plt.subplots(ncol, nrow, sharex=True, sharey=True, figsize=(scale*nrow, scale*ncol))
        if ncol > 1:
            axes = [item for rows in axes for item in rows]
        i = 0
        for ax, im in zip(axes, images):
            ax.imshow(im)
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        fig.subplots_adjust(wspace=0, hspace=0)
        fig.show()

def show_location(input_image, boxes, labels=None):
    fig, ax = plt.subplots()
    ax.set_xticklabels([])
    ax.set_yticklabels([])    
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i].tolist()
        w = x2 - x1
        h = y2 - y1
        rect = patches.Rectangle((x1, y1), w, h, linewidth=1, edgecolor='r', facecolor='none')
        if labels is not None:
            plt.text(x1, y1, labels[i], color='r')
        ax.add_patch(rect)
    
    ax.imshow(input_image)
    plt.show()
    
    
def show_pose(boxes, labels, image):
    fig, ax = plt.subplots()
    ax.set_xticklabels([])
    ax.set_yticklabels([])    
    for x1, y1 in boxes[labels > 0]:
        rect = patches.Rectangle((x1, y1), 4, 4, linewidth=1, edgecolor='r', facecolor='r')
        ax.add_patch(rect)    
    ax.imshow(image)
    plt.show()
    
out = uio.image_generation("A horse in a grassy field.", num_decodes=4)
show_images(out["image"])