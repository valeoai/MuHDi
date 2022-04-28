import numpy as np
import re

from muhdi.utils import project_root
from muhdi.utils.serialization import json_load
from muhdi.dataset.base_dataset import BaseDataset

DEFAULT_INFO_PATH = project_root / 'muhdi/dataset/acdc_list/info.json'
INFO_PATH_7 = project_root / 'muhdi/dataset/acdc_list/info7class.json'

class ACDCDataSet(BaseDataset):
    def __init__(self, root, list_path, set='val',
                 max_iters=None,
                 crop_size=(321, 321), mean=(128, 128, 128),
                 load_labels=True,
                 info_path=DEFAULT_INFO_PATH, labels_size=None,
                 num_classes=19):
        super().__init__(root, list_path, set, max_iters, crop_size, labels_size, mean)
        set = re.split('_',self.set)
        self.num_classes = num_classes
        self.set = set[1]
        self.weather = set[0]

        self.load_labels = load_labels
        if num_classes==7:
            self.info = json_load(INFO_PATH_7)
        elif num_classes==19:
            self.info = json_load(DEFAULT_INFO_PATH)
        else:
            self.info = json_load(info_path)
        self.mapping = np.array(self.info['label2train'], dtype=np.int)
        self.map_vector = np.zeros((self.mapping.shape[0],), dtype=np.int64)
        for source_label, target_label in self.mapping:
            self.map_vector[source_label] = target_label

    def get_metadata(self, name):
        img_file = self.root / 'rgb_anon' / name
        if self.label_folder is not '':
            label_file = self.label_folder + '/' + (name.split('/')[1])
        else:
            label_name = name.replace("rgb_anon", "gt")
            label_name = label_name.replace("gt.png","gt_labelIds.png")
            label_file = self.root / 'gt' / label_name
        return img_file, label_file

    def map_labels(self, input_):
        return self.map_vector[input_.astype(np.int64, copy=False)]

    def __getitem__(self, index):
        img_file, label_file, name = self.files[index]
        label = self.get_labels(label_file)
        label = self.map_labels(label).copy()
        image = self.get_image(img_file)
        image = self.preprocess(image)
        return image.copy(), label, np.array(image.shape), name
