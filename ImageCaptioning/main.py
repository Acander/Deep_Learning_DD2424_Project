import tensorflow as tf

# You'll generate plots of attention in order to see which parts of an image
# our model focuses on during captioning
import matplotlib.pyplot as plt

# Scikit-learn includes many helpful utilities
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import re
import numpy as np
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import time
import json
from glob import glob
from PIL import Image
import pickle

from ImageCaptioning import preProcessing


def run():
    inceptionNet, dataset = preProcessing.preProcessData()


if __name__ == '__main__':
    run()
