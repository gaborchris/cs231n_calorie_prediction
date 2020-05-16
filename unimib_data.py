import numpy as np
import scipy.io as sio
import tables
import h5py
import os
import torch
from PIL import Image
import matplotlib.pyplot as plt


def get_annotations():
    image_path = "datasets/UNIMIB2016-images/original"
    annotation_path = "datasets/UNIMIB2016-annotations/annotations.csv"
    imgs = list(sorted(os.listdir(image_path)))

    annotations = {}
    classes = {"background": 0}
    class_lookup = {0: "background"}

    class_idx = 1

    with open(annotation_path) as f:
        f.readline()
        for line in f.readlines():
            fields = line.strip('\n').split(",")
            img = fields[0] + ".jpg"
            label = fields[1]
            if label not in classes:
                classes[label] = class_idx
                class_lookup[class_idx] = label
                class_idx += 1
            x1 = int(fields[2])
            y1 = int(fields[3])
            x2 = int(fields[4])
            y2 = int(fields[5])
            entry = {"label": classes[label], "box": [x1, y1, x1+x2, y1+y2]}

            if img in annotations:
                annotations[img].append(entry)
            else:
                annotations[img] = [entry]

    return annotations, classes


class UNIMIBDataset(object):
    def __init__(self, transforms=None):
        self.root = "datasets/UNIMIB2016-images/original/"
        self.tranforms = transforms
        self.imgs = list(sorted(os.listdir(self.root)))
        self.annotations, self.classes = get_annotations()
        for img in self.imgs:
            if img not in self.annotations:
                self.imgs.remove(img)


    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        width, height = img.size
        resize = (28, 28)
        scale_x = resize[0]*1.0/width
        scale_y = resize[1]*1.0/height
        img = img.resize(resize)

        objects = self.annotations[self.imgs[idx]]
        boxes = []
        labels = []
        for object_ in objects:
            x1, y1, x2, y2 = object_['box']
            x1 = int(x1 * scale_x)
            x2 = int(x2 * scale_x)
            y1 = int(y1 * scale_y)
            y2 = int(y2 * scale_y)
            boxes.append([x1, y1, x2, y2])
            labels.append(object_['label'])

        mask = np.array(img)[:, :, :]*0
        for box, label in zip(boxes, labels):
            mask[box[1]:box[3], box[0]: box[2]] = 1

        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[1:]

        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]

        boxes = np.array(boxes)

        labels = np.array(labels)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor((labels), dtype=torch.int64)
        masks = torch.as_tensor(mask, dtype=torch.uint8)
        img_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(objects), ), dtype=torch.int64)


        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = img_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.tranforms is not None:
            img, target = self.tranforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


if __name__ == "__main__":
    dataset = UNIMIBDataset()
    # print(dataset.annotations)
    # print(dataset.classes)
    # print(len(dataset.classes))
    for i in range(dataset.__len__()):
        print(i)
        dataset.__getitem__(i)
    # print(dataset.__len__())



