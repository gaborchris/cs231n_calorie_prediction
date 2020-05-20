import numpy as np
import scipy.io as sio
import tables
import h5py
import os
import torch
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

TRAY_SIZE = 44*36


def get_annotations(trays=True):
    image_path = "datasets/UNIMIB2016-images/original"
    annotation_path = "datasets/UNIMIB2016-annotations/annotations.csv"
    if trays:
        image_path = "datasets/unimib_tray/images"
        annotation_path = "datasets/unimib_tray/annotations.csv"

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

    return annotations, classes, class_lookup


class UNIMIBDataset(object):
    def __init__(self, transforms=None, getmasks=False, trays=True, resize=(512,512)):
        if trays:
            self.root = "datasets/unimib_tray/images"
        else:
            self.root = "datasets/UNIMIB2016-images/original/"
        self.tranforms = transforms
        self.imgs = list(sorted(os.listdir(self.root)))
        self.annotations, self.classes, self.lookup = get_annotations(trays)
        for img in self.imgs:
            if img not in self.annotations:
                self.imgs.remove(img)
        self.getmasks = getmasks
        self.resize = resize

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")

        objects = self.annotations[self.imgs[idx]]
        boxes = []
        labels = []

        resize = self.resize
        if resize is not None:
            width, height = img.size
            scale_x = resize[0]*1.0/width
            scale_y = resize[1]*1.0/height
            img = img.resize(resize)

        for object_ in objects:
            x1, y1, x2, y2 = object_['box']
            if resize:
                x1 = int(x1 * scale_x)
                x2 = int(x2 * scale_x)
                y1 = int(y1 * scale_y)
                y2 = int(y2 * scale_y)
            boxes.append([x1, y1, x2, y2])
            labels.append(object_['label'])


        if self.getmasks:
            mask = np.array(img)[:, :, :]*0
            for box, label in zip(boxes, labels):
                mask[box[1]:box[3], box[0]: box[2]] = 1
            # obj_ids = np.unique(mask)
            # obj_ids = obj_ids[1:]
            # masks = mask == obj_ids[:, None, None]
            masks = torch.as_tensor(mask, dtype=torch.uint8)

        boxes = np.array(boxes)

        labels = np.array(labels)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor((labels), dtype=torch.int64)
        img_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(objects), ), dtype=torch.int64)


        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = img_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        if self.getmasks:
            target["masks"] = masks

        if self.tranforms is not None:
            img, target = self.tranforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)


def get_box_area(box_area, pixel_size, image_size=TRAY_SIZE):
    area_ratio = box_area / (pixel_size[0] * pixel_size[1])
    area = int(image_size * area_ratio)
    return area


def save_bbox_ground_truth(image_size_cm=44*36):
    _, classes, lookup = get_annotations()
    dataset = UNIMIBDataset(resize=None)
    for idx in range(len(dataset)):
        img, gt_label = dataset[idx]

        # img = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
        img1 = ImageDraw.Draw(img)
        # GT box

        # Put GT label over boxes
        for box, label, area in zip(gt_label['boxes'], gt_label['labels'], gt_label['area']):
            area = get_box_area(area.numpy(), img.size)
            shape = [(box[2], box[3]), (box[0], box[1])]
            img1.rectangle(shape, outline='green')
            name = lookup[int(label.numpy())]
            img1.text((box[0], box[1]), "GT: " + name, fill=(0, 255, 0))
            img1.text((box[0], box[3]-10), "area: " + str(area) + " cm^2", fill=(0, 255, 0))
        path = "./datasets/UNIMIB2016-images/overlays/"+str(idx)+".png"
        print(idx)
        img.save(path)


if __name__ == "__main__":
    # dataset = UNIMIBDataset()
    save_bbox_ground_truth()



