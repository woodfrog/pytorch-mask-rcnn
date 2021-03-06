import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

import coco
import utils
import model as modellib
import visualize

import torch
import pdb

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to trained weights file
# Download this file and place in the root of your
# project (See README file for details)
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.pth")

# Directory of images to run detection on
DATASET_BASE_DIR = '/media/nelson/Workspace1/Projects/building_reconstruction/data/la_dataset'
IMAGE_DIR = os.path.join(DATASET_BASE_DIR, 'rgb')

class InferenceConfig(coco.BuildingsConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    # GPU_COUNT = 0 for CPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Create model object.
model = modellib.MaskRCNN(model_dir=MODEL_DIR, config=config)
if config.GPU_COUNT:
    model = model.cuda()

# Load weights trained on MS-COCO
_, last_saved = model.find_last()
model.load_state_dict(torch.load(last_saved))
print('loaded weights from {}'.format(last_saved))

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'edge']

# Load a random image from the images folder
im_path = '/media/nelson/Workspace1/Projects/building_reconstruction/la_dataset/valid_list.txt'
with open(im_path) as f:
    im_list = [x.strip()+'.jpg' for x in f.readlines()]
file_names = im_list

for fname in file_names:
    image = skimage.io.imread(os.path.join(IMAGE_DIR, fname))

    # Run detection
    results = model.detect([image])

    # Visualize results
    r = results[0]
    print(len(r['rois']))
    visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                class_names, r['scores'])
    plt.show()