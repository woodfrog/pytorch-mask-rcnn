"""
Mask R-CNN
Configurations and data loading code for MS COCO.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 coco.py train --dataset=/path/to/coco/ --model=coco

    # Train a new model starting from ImageNet weights
    python3 coco.py train --dataset=/path/to/coco/ --model=imagenet

    # Continue training a model that you had trained earlier
    python3 coco.py train --dataset=/path/to/coco/ --model=/path/to/weights.h5

    # Continue training the last model you trained
    python3 coco.py train --dataset=/path/to/coco/ --model=last

    # Run COCO evaluatoin on the last model you trained
    python3 coco.py evaluate --dataset=/path/to/coco/ --model=last
"""

import os
import time
import numpy as np
from PIL import Image, ImageDraw
# Download and install the Python COCO tools from https://github.com/waleedka/coco
# That's a fork from the original https://github.com/pdollar/coco with a bug
# fix for Python 3.
# I submitted a pull request https://github.com/cocodataset/cocoapi/pull/50
# If the PR is merged then use the original repo.
# Note: Edit PythonAPI/Makefile and replace "python" with "python3".
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

import zipfile
import urllib.request
import shutil

from config import Config
import utils
import model as modellib

import torch

# Root directory of the project
ROOT_DIR = os.getcwd()

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.pth")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################

class BuildingsConfig(Config):

    # Give the configuration a recognizable name
    NAME = "la_dataset"

    # We use one GPU with 8GB memory, which can fit one image.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 16

    # Uncomment to train on 8 GPUs (default is 1)
    # GPU_COUNT = 8

    # Number of classes (including background)
    NUM_CLASSES = 2


############################################################
#  Dataset
############################################################

class BuildingsDataset(utils.Dataset):
    def load_buildings(self, dataset_dir):
        # Add classes
        self.add_class("buildings", 1, "edge")

        # Add images
        rgb_prefix = '/media/nelson/Workspace1/Projects/building_reconstruction/la_dataset/rgb'
        train_path = '/media/nelson/Workspace1/Projects/building_reconstruction/la_dataset/train_list.txt'
        with open(train_path) as f:
            train_list = f.readlines()

        for k, im_id in enumerate(train_list):
            im_path = os.path.join(rgb_prefix, im_id.strip()+'.jpg')
            for i in range(4):
                self.add_image("buildings", False, image_id=8*k+2*i, path=im_path)
                self.add_image("buildings", True, image_id=8*k+2*i+1, path=im_path)     


    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """

        info = self.image_info[image_id]
        im_path = info['path']
        rot = (image_id % 4)*90
        masks = []
        masks_v = []
        class_ids = []

        # load annotations
        p_path = im_path.replace('.jpg', '.npy').replace('rgb', 'annots')
        v_set = np.load(open(p_path, 'rb'),  encoding='bytes')
        graph = dict(v_set[()])

        # draw mask
        masks = []
        class_ids = []
        for v1 in graph:
            for v2 in graph[v1]:
                x1, y1 = v1
                x2, y2 = v2

                # create mask
                mask_im = Image.fromarray(np.zeros((256, 256)))               
                
                # draw lines
                draw = ImageDraw.Draw(mask_im)
                draw.line((x1, y1, x2, y2), fill='white', width=3)
                
                # apply augmentation            
                mask_im = mask_im.rotate(rot)
                if info['flip']:
                    mask_im = mask_im.transpose(Image.FLIP_LEFT_RIGHT) 

                # accumul
                masks.append(np.array(mask_im))
                class_ids.append(1)
        masks = np.stack(masks).astype('float').transpose(1, 2, 0)

        # Map class names to class IDs.
        class_ids = np.array(class_ids).astype('int32')
        return masks, class_ids

    def image_reference(self, image_id):
        info = self.image_info[image_id]
        if info["source"] == "buildings":
            return info["buildings"]
        else:
            super(BuildingsDataset, self).image_reference(image_id)

    def load_image(self, image_id):
        info = self.image_info[image_id]
        rot = (image_id % 4)*90 
        im = Image.open(info['path'])
        im = im.rotate(rot)
        if info['flip'] == True:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        return np.array(im)
             
############################################################
#  Training
############################################################


if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate'")
    parser.add_argument('--model', required=False,
                        metavar="/path/to/weights.pth",
                        help="Path to weights .pth file")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--limit', required=False,
                        default=500,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=500)')

    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = BuildingsConfig()
    else:
        class InferenceConfig(BuildingsConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(config=config,
                                  model_dir=args.logs)
    if config.GPU_COUNT:
        model = model.cuda()

    # Select weights file to load
    if args.model:
        if args.model.lower() == "coco":
            model_path = COCO_MODEL_PATH
        elif args.model.lower() == "last":
            # Find last trained weights
            model_path = model.find_last()[1]
        elif args.model.lower() == "imagenet":
            # Start from ImageNet trained weights
            model_path = config.IMAGENET_MODEL_PATH
        else:
            model_path = args.model
    else:
        model_path = ""

    # Load weights
    print("Loading weights ", model_path)
    model.load_weights(model_path)

    # Train or evaluate
    if args.command == "train":
        # Training dataset. Use the training set and 35K from the
        # validation set, as as in the Mask RCNN paper.
        dataset_train = BuildingsDataset()
        dataset_train.load_buildings("train")
        dataset_train.prepare()

        # # Validation dataset
        # dataset_val = BuildingsDataset()
        # dataset_val.load_buildings(args.dataset, "minival")
        # dataset_val.prepare()

        # *** This training schedule is an example. Update to your needs ***

        # Training - Stage 1
        print("Training network heads")
        model.train_model(dataset_train, dataset_train,
                    learning_rate=config.LEARNING_RATE,
                    epochs=20,
                    layers='heads')

        # Training - Stage 2
        # Finetune layers from ResNet stage 4 and up
        print("Fine tune Resnet stage 4 and up")
        model.train_model(dataset_train, dataset_train,
                    learning_rate=config.LEARNING_RATE,
                    epochs=40,
                    layers='4+')

        # Training - Stage 3
        # Fine tune all layers
        print("Fine tune all layers")
        model.train_model(dataset_train, dataset_train,
                    learning_rate=config.LEARNING_RATE / 10,
                    epochs=100,
                    layers='all')

    # elif args.command == "evaluate":
    #     # Validation dataset
    #     dataset_val = CocoDataset()
    #     coco = dataset_val.load_coco(args.dataset, "minival", year=args.year, return_coco=True, auto_download=args.download)
    #     dataset_val.prepare()
    #     print("Running COCO evaluation on {} images.".format(args.limit))
    #     evaluate_coco(model, dataset_val, coco, "bbox", limit=int(args.limit))
    #     evaluate_coco(model, dataset_val, coco, "segm", limit=int(args.limit))
    # else:
    #     print("'{}' is not recognized. "
    #           "Use 'train' or 'evaluate'".format(args.command))
