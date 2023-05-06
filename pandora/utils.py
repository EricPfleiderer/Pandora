import re
import os
import sys
import logging
import json
from datetime import datetime

import cv2


def sorted_nicely(unsorted_strings: list):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text

    def alphanum_key(key):
        return [convert(c) for c in re.split('([0-9]+)', key)]
    # alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(unsorted_strings, key=alphanum_key)


def stitch_images(path_to_imgs, video_name):
    images = [img for img in os.listdir(path_to_imgs) if img.endswith(".png")]
    sorted_imgs = sorted_nicely(images)
    frame = cv2.imread(os.path.join(path_to_imgs, sorted_imgs[0]))
    height, width, layers = frame.shape
    video = cv2.VideoWriter(path_to_imgs+ video_name, 0, 1, (width, height))

    for image in sorted_imgs:
        video.write(cv2.imread(os.path.join(path_to_imgs, image)))

    cv2.destroyAllWindows()
    video.release()


def create_experiment_dir(save_path: str, params: dict):
    now = datetime.now() # Current timestamp
    full_path = save_path + 'Experiment_' + str(now) + '/'  # Full path to experiment directory
    os.makedirs(full_path)  # Create the experiment directory
    json.dump(params, open(full_path + 'params.json', 'w'))  # Dump experiment params to json file
    return full_path


def set_logging(level):
    logger = logging.getLogger()
    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Add file handler
    debug_handler = logging.FileHandler('logs/debug.log')
    debug_handler.setLevel(logging.DEBUG)
    debug_handler.setFormatter(formatter)
    logger.addHandler(debug_handler)

    info_handler = logging.FileHandler('logs/info.log')
    info_handler.setLevel(logging.INFO)
    info_handler.setFormatter(formatter)
    logger.addHandler(info_handler)

    # Add stream handler
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)