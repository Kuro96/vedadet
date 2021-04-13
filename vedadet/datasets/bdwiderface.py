import numpy as np
import os.path as osp
import xml.etree.ElementTree as ET

import vedacore.fileio as fileio
from vedacore.misc import registry
from .custom import CustomDataset


@registry.register_module('dataset')
class BDWIDERFaceDataset(CustomDataset):
    """Baidu SDK result for WIDER Face dataset in PASCAL VOC format.

    Args:
        min_size (int | float, optional): The minimum size of bounding
            boxes in the images. If the size of a bounding box is less than
            ``min_size``, it would be add to ignored field.
    """
    CLASSES = ('face', )
    VALID_TARGET_TYPE = ['binary', 'hard', 'soft']

    def __init__(
        self,
        min_size=None,
        offset: int = 0,
        target_type: str = 'binary',
        score_thr: float = 0.,
        **kwargs,
    ):
        super(BDWIDERFaceDataset, self).__init__(**kwargs)

        assert target_type in self.VALID_TARGET_TYPE, (
            f'Expect `target_type` in {self.VALID_TARGET_TYPE}, '
            f'but got {target_type}.')

        self.target_type = target_type
        self.score_thr = score_thr
        self.cat2label = {cat: i for i, cat in enumerate(self.CLASSES)}
        self.min_size = min_size
        self.offset = offset

    def load_annotations(self, ann_file):
        """Load annotation from WIDERFace XML style annotation file.

        Args:
            ann_file (str): Path of XML file.

        Returns:
            list[dict]: Annotation info from XML file.
        """

        data_infos = []
        img_ids = fileio.list_from_file(ann_file)
        self.img_ids = img_ids
        for img_id in img_ids:
            filename = f'{img_id}.jpg'
            xml_path = osp.join(self.img_prefix, 'Annotations',
                                f'{img_id}.xml')
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            folder = root.find('folder').text
            data_infos.append(
                dict(
                    id=img_id,
                    filename=osp.join(folder, filename),
                    width=width,
                    height=height))

        return data_infos

    def get_subset_by_classes(self):
        """Filter imgs by user-defined categories."""
        subset_data_infos = []
        for data_info in self.data_infos:
            img_id = data_info['id']
            xml_path = osp.join(self.img_prefix, 'Annotations',
                                f'{img_id}.xml')
            tree = ET.parse(xml_path)
            root = tree.getroot()
            for obj in root.findall('object'):
                name = obj.find('name').text
                if name in self.CLASSES:
                    subset_data_infos.append(data_info)
                    break

        return subset_data_infos

    def get_ann_info(self, idx):
        """Get annotation from XML file by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """
        def parse_bbox(obj):
            bnd_box = obj.find('bndbox')
            return [
                float(bnd_box.find('xmin').text),
                float(bnd_box.find('ymin').text),
                float(bnd_box.find('xmax').text),
                float(bnd_box.find('ymax').text),
            ]

        img_id = self.data_infos[idx]['id']
        xml_path = osp.join(self.img_prefix, 'Annotations', f'{img_id}.xml')
        tree = ET.parse(xml_path)
        root = tree.getroot()
        bboxes = []
        labels = []
        scores = []
        bboxes_ignore = []
        labels_ignore = []
        scores_ignore = []
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in self.CLASSES:
                continue

            label = self.cat2label[name]
            difficult = int(obj.find('difficult').text)
            bbox = parse_bbox(obj)
            score_str = obj.find('score')
            if score_str is not None:
                score = float(score_str.text)
            else:
                score = 1.0

            ignore = False
            if self.target_type == 'binary':
                if score < self.score_thr:
                    ignore = True
                    score = 0.0
                else:
                    score = 1.0
            if self.target_type == 'hard':
                if score < self.score_thr:
                    ignore = True

            if self.min_size:
                # assert not self.test_mode
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                if h < self.min_size or w < self.min_size:
                    ignore = True

            if difficult or ignore:
                bboxes_ignore.append(bbox)
                labels_ignore.append(label)
                scores_ignore.append(score)
            else:
                bboxes.append(bbox)
                labels.append(label)
                scores.append(score)
        if not bboxes:
            bboxes = np.zeros((0, 4))
            labels = np.zeros((0, ))
            scores = np.zeros((0, ))
        else:
            bboxes = np.array(bboxes, ndmin=2) - self.offset
            labels = np.array(labels)
            scores = np.array(scores)
        if not bboxes_ignore:
            bboxes_ignore = np.zeros((0, 4))
            labels_ignore = np.zeros((0, ))
            scores_ignore = np.zeros((0, ))
        else:
            bboxes_ignore = np.array(bboxes_ignore, ndmin=2) - self.offset
            labels_ignore = np.array(labels_ignore)
            scores_ignore = np.array(scores_ignore)
        ann = dict(
            bboxes=bboxes.astype(np.float32),
            labels=labels.astype(np.int64),
            scores=scores.astype(np.float32),
            bboxes_ignore=bboxes_ignore.astype(np.float32),
            labels_ignore=labels_ignore.astype(np.int64),
            scores_ignore=scores_ignore.astype(np.float32),
        )
        return ann

    def get_cat_ids(self, idx):
        """Get category ids in XML file by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        cat_ids = []
        img_id = self.data_infos[idx]['id']
        xml_path = osp.join(self.img_prefix, 'Annotations', f'{img_id}.xml')
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall('object'):
            name = obj.find('name').text
            if name not in self.CLASSES:
                continue
            label = self.cat2label[name]
            cat_ids.append(label)

        return cat_ids
