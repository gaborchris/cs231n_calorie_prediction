import torchvision
import torch
from engine import train_one_epoch, evaluate
import matplotlib.pyplot as plt
import utils
import transforms as T
from PIL import Image, ImageDraw

import unimib_data


def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def train_fasterRCNN():
    _, classes, lookup = unimib_data.get_annotations()
    num_classes = len(classes)

    dataset = unimib_data.UNIMIBDataset(get_transform(train=True), resize=None)
    dataset_test = unimib_data.UNIMIBDataset(get_transform(train=False), resize=None)
    indices = torch.randperm(len(dataset)).tolist()
    indices = range(len(dataset))
    num_train = 10
    num_test = 5
    dataset = torch.utils.data.Subset(dataset, indices[:num_train])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[num_train:num_train+num_test])

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, collate_fn=utils.collate_fn
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, collate_fn=utils.collate_fn
    )

    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    device = 'cpu'
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.to(device)
    params = [p for p in model.parameters()]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)
    num_epochs = 2
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        lr_scheduler.step()

    # test_model(model, lookup, dataset, num_images=4)
    test_model(model, lookup, dataset_test, num_images=1)


def test_model(model, lookup, dataset, num_images=1, threshold=0.3):
    device = 'cpu'

    for idx in range(num_images):
        img, gt_label = dataset[idx]
        model.eval()
        with torch.no_grad():
            prediction = model([img.to(device)])[0]

        img = Image.fromarray(img.mul(255).permute(1,2,0).byte().numpy())
        img1 = ImageDraw.Draw(img)

        # GT box
        for box in gt_label['boxes']:
            shape = [(box[2], box[3]), (box[0], box[1])]
            img1.rectangle(shape, outline='green')

        # Pred box
        for preds in zip(prediction['boxes'], prediction['labels'], prediction['scores']):
            box, label, score = preds
            if score > threshold:
                area = (box[3] - box[1]) * (box[2] - box[0])
                area = area.numpy()
                area = unimib_data.get_box_area(area, img.size)
                shape = [(box[2], box[3]), (box[0], box[1])]
                img1.rectangle(shape, outline='red')
                label = int(label.numpy())
                string_label = "Pred: "+lookup[label] + ", Area: " + str(area) + "cm^2"
                if label in lookup:
                    img1.text((box[0], box[3]), string_label, fill=(255, 0, 0))

        # Put GT label over boxes
        for box, label, area in zip(gt_label['boxes'], gt_label['labels'], gt_label['area']):
            name = lookup[int(label.numpy())]
            area = unimib_data.get_box_area(area.numpy(), img.size)
            string_label = "GT: " + name + ", Area: " + str(area) + "cm^2"
            img1.text((box[0], box[1]), string_label, fill=(0, 255, 0))

        img.save("bbox_img.png")
        img.show()

def test_random():
    device = 'cpu'
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.to(device)
    images, boxes = torch.rand(4, 3, 600, 1200), torch.rand(4, 11, 4)
    labels = torch.randint(1, 91, (4, 11))
    images = list(image for image in images)
    targets = []
    for i in range(len(images)):
        d = {}
        d['boxes'] = boxes[i]
        d['labels'] = labels[i]
        targets.append(d)
    images = list(image.to(device) for image in images)
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    output = model(images, targets)


if __name__ == "__main__":
    train_fasterRCNN()
