import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, models
import os
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
from collections import OrderedDict
import time


def load_data(path):
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

    train_data = datasets.ImageFolder(data_dir, transform=train_transforms)
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    return trainloader



def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)))
    plt.show()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3,6,5, padding=2)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,3, padding=1)
        self.pool2 = nn.MaxPool2d(2,2)
        self.conv3 = nn.Conv2d(16,32,3, padding=1)
        self.pool3 = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(32*28*28, 120)
        self.fc2 = nn.Linear(120, 101)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(-1, 32*28*28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def create_embedding(model, data, path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)

    X = np.array([])
    Y = np.array([])
    with torch.no_grad():
        batches = 0
        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)
            output = model(inputs)
            if batches == 0:
                X = output.cpu().numpy()
                Y = labels.cpu().numpy().reshape(-1)
            else:
                X = np.vstack((X, output.cpu().numpy()))
                Y = np.hstack((Y, labels.cpu().numpy().reshape(-1)))
            print(X.shape)
            print(Y.shape)
            batches += 1
    np.save("datasets/food-101/densenet_embed_transformed_data", X)
    np.save("datasets/food-101/densenet_embed_transformed_labels", Y)


def train_embeddings():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    X = np.load("datasets/food-101/densenet_embed_transformed_data.npy")
    Y = np.load("datasets/food-101/densenet_embed_transformed_labels.npy")
    split = int(0.8*X.shape[0])
    X_train = X[:split]
    Y_train = Y[:split]
    X_val = X[split:]
    Y_val = Y[split:]
    X_tensor_train = torch.from_numpy(X_train)
    Y_tensor_train = torch.from_numpy(Y_train)
    X_tensor_val = torch.from_numpy(X_val)
    Y_tensor_val = torch.from_numpy(Y_val)

    train_dataset = torch.utils.data.TensorDataset(X_tensor_train, Y_tensor_train)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=128)
    val_dataset = torch.utils.data.TensorDataset(X_tensor_val, Y_tensor_val)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=128)
    classifier = nn.Sequential(OrderedDict([
        # ('drop', nn.Dropout(0.2)),
        ('fc1', nn.Linear(1024, 1000)),
        ('relu1', nn.ReLU()),
        ('fc2', nn.Linear(1000, 500)),
        ('relu2', nn.ReLU()),
        ('fc3', nn.Linear(500, 101))
    ]))
    classifier.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=0.0001)
    start = time.time()
    for epoch in range(100):
        for i, data in enumerate(train_dataloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = classifier(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print("Batch loss", loss.item())

        accuracy = 0.0
        with torch.no_grad():
            batches = 0
            for inputs, labels in val_dataloader:
                inputs, labels = inputs.cuda(), labels.cuda()
                output = classifier(inputs)
                top_p, top_class = output.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                batches += 1
        print("Accuracy", accuracy / batches)


def train_full(trainloader):
    model = models.densenet121(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(1024, 500)),
        ('relu', nn.ReLU()),
        ('fc2', nn.Linear(500, 101))
        # ('output', nn.LogSoftmax(dim=1))
    ]))

    model.classifier = classifier
    print(len(trainloader))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.0001)

    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)
            print(inputs.shape, labels.shape)
            start = time.time()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            print(loss.item())
            optimizer.step()
            running_loss += loss.item()
            print(i*32)
            if i % 100 == 99:
                print(f"Device = {device}; Time per batch: {(time.time() - start) / 3:.3f} seconds")
                print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
                model.eval()
                valid_loss = 0
                accuracy = 0
        with torch.no_grad():
            batches = 0
            for inputs, labels in trainloader:
                inputs, labels = inputs.cuda(), labels.cuda()
                output = model(inputs)
                top_p, top_class = output.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                batches += 1
                print(accuracy/batches)
        print("Accuracy", accuracy/batches)




if __name__ == "__main__":
    data_dir = os.path.join("datasets", "food-101", "images")
    trainloader = load_data(data_dir)
    # model = models.densenet121(pretrained=True)
    # model.classifier = Identity()
    # create_embedding(model, trainloader, None)
    train_embeddings()


