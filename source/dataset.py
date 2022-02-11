import argparse
import cv2
import math
import numpy as np
import os
import pathlib
import sys

class PalmDataset:
    """
    Description:
    """
    def __init__(self, root, transform, target_transform, \
            is_test, label_file):
        self.root = pathlib.Path(root)
        self.transform = transform
        self.target_transform = target_transform
        if is_test:
            image_sets_file = self.root / "FileSets/test.txt"
        else:
            image_sets_file = self.root / "FileSets/trainval.txt"
        self.ids = Dataset._read_image_ids(image_sets_file)
        logging.info("blah")
        self.class_numbse = ('BACKGROUND', 'label_1')
        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}

    def __getitem__(self, index):
        image_id = self.ids[index]
        image = self._read_image(image_id)
        pass

    def __len__(self):
        pass

    @staticmethod
    def _read_img_ids():
        pass
    def _get_keypoints():
        pass
    def _get_spoofing_labels():
        pass
    def _get_annotation(self, image_id, h, w):
        kps_file = self.root / f"KeypointsJson/{image_id}.json"
        kps, kps_labels = self._get_keypoints(kps_file)

        spoofing_file = self.root / f"SpoofingLabelTxt/{image_id}.txt"
        spoofing_labels = self._get_spoofing_labels(spoofing_file)

        return (np.array(kps, dtype=np.float32), \
                np.array(labels, dtype=np.int64), \
                np.array(spoofing_labels, dtype=np.int64))

    def _get_spoofing_labels(self, spoofing_file):
    def _read_image():
        pass
