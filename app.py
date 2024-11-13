import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable

from itertools import chain 
import sklearn.metrics as metrics


import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix

# Set device to CPU
device = torch.device("cpu")

# Load the Fashion MNIST dataset from CSV files
train_csv  = pd.read_csv("Fashion MNIST Archive/fashion-mnist_test.csv") # Training data
test_csv = pd.read_csv("Fashion MNIST Archive/fashion-mnist_train.csv") # Test data

# Define a custom Dataset class for Fashion MNIST
class FashionDataset(Dataset):
    def __init__(self, data, transform=None):
        super().__init__()
        self.fashion_MNIST = list(data.values)
        self.transform = transform

        labels = []
        images = []

        # Separate labels and images from the dataset
        for i in self.fashion_MNIST:
            labels.append(i[0])
            images.append(i[1:])

        # Convert labels and images to numpy arrays
        self.labels = np.asarray(labels)
        self.images = np.asarray(images).reshape(-1, 28, 28).astype("float32")

    def __getitem__(self, index):
        # Get the label and image at the specified index
        label = self.labels[index]
        image = self.images[index]

        # Apply transformations to the image if any
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        # Return the total number of images
        return len(self.images)

# Create dataset objects for training and testing
train_set = FashionDataset(train_csv, transform=transforms.Compose([transforms.ToTensor()]))
test_set = FashionDataset(test_csv, transform=transforms.Compose([transforms.ToTensor()]))

# Create data loaders for training and testing
train_loader = DataLoader(train_set, batch_size=100)
test_loader = DataLoader(test_set, batch_size=100)

def output_label(label):
    output_mapping = {
                 0: "T-shirt/Top",
                 1: "Trouser",
                 2: "Pullover",
                 3: "Dress",
                 4: "Coat", 
                 5: "Sandal", 
                 6: "Shirt",
                 7: "Sneaker",
                 8: "Bag",
                 9: "Ankle Boot"
                 }
    input = (label.item() if type(label) == torch.Tensor else label)
    return output_mapping[input]



def test_stuff():
    a = next(iter(train_loader))
    print(a[0].size())

    print(len(train_set))

    # Get a single image and label from the training set
    image, label = next(iter(train_set))

    # Plot the image
    plt.imshow(image.squeeze(), cmap="gray")

    demo_loader = torch.utils.data.DataLoader(train_set, batch_size=10)

    batch = next(iter(demo_loader))
    images, labels = batch

    print(type(images), type(labels))
    print(images.shape, labels.shape)

    grid = torchvision.utils.make_grid(images, nrow=10)

    plt.figure(figsize=(15, 20))
    plt.imshow(np.transpose(grid, (1, 2, 0)))
    plt.show()


    print("labels: ", end=" ")
    for i, label in enumerate(labels):
        print(output_label(label), end=", ")



class FashionCNN(nn.Module):
    
    def __init__(self):
        super(FashionCNN, self).__init__()
        
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.fc1 = nn.Linear(in_features=64*6*6, out_features=600)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=10)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)
        
        return out


def see_accuracy_classes():
    class_correct = [0. for _ in range(10)]
    total_correct = [0. for _ in range(10)]

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            test = Variable(images)
            outputs = model(test)
            predicted = torch.max(outputs, 1)[1]
            c = (predicted == labels).squeeze()
            
            for i in range(100):
                label = labels[i]
                class_correct[label] += c[i].item()
                total_correct[label] += 1
            
    for i in range(10):
        print("Accuracy of {}: {:.2f}%".format(output_label(i), class_correct[i] * 100 / total_correct[i]))


def plot_iteration_loss():
    plt.plot(iteration_list, loss_list)
    plt.xlabel("No. of Iteration")
    plt.ylabel("Loss")
    plt.title("Iterations vs Loss")
    plt.show()


def plot_iteration_accuracy():
    filtered_iterations = [i for i, acc in zip(iteration_list, accuracy_list) if acc is not None]
    filtered_accuracies = [acc for acc in accuracy_list if acc is not None]
    plt.plot(filtered_iterations, filtered_accuracies)
    plt.xlabel("No. of Iteration")
    plt.ylabel("Accuracy")
    plt.title("Iterations vs Accuracy")    
    plt.show()

def print_confusion_matrix():
    predictions_l = [predictions_list[i].tolist() for i in range(len(predictions_list))]
    labels_l = [labels_list[i].tolist() for i in range(len(labels_list))]
    predictions_l = list(chain.from_iterable(predictions_l))
    labels_l = list(chain.from_iterable(labels_l))

    target_names = [output_label(i) for i in range(10)]  # Add this line to define target names

    confusion_matrix(labels_l, predictions_l)
    print("Classification report for CNN :\n%s\n"
          % (metrics.classification_report(labels_l, predictions_l, target_names=target_names)))

def create_analizyse():
    see_accuracy_classes()
    print_confusion_matrix()
    plot_iteration_accuracy()
    plot_iteration_loss()        


model = FashionCNN()
model.to(device)

error = nn.CrossEntropyLoss()

learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
print(model)

num_epochs = 5
count = 0
# Lists for visualization of loss and accuracy 
loss_list = []
iteration_list = []
accuracy_list = []

# Lists for knowing classwise accuracy
predictions_list = []
labels_list = []

target_accuracy = 87  # Set your target accuracy here
stop_training = False

# Initialize lists
loss_list = []
iteration_list = []
accuracy_list = []
predictions_list = []
labels_list = []

# Training loop
for epoch in range(num_epochs):
    for images, labels in train_loader:
        # Transfering images and labels to GPU if available
        images, labels = images.to(device), labels.to(device)
    
        train = Variable(images.view(100, 1, 28, 28))
        labels = Variable(labels)
        
        # Forward pass 
        outputs = model(train)
        loss = error(outputs, labels)
        
        # Initializing a gradient as 0 so there is no mixing of gradient among the batches
        optimizer.zero_grad()
        
        # Propagating the error backward
        loss.backward()
        
        # Optimizing the parameters
        optimizer.step()
    
        count += 1

        # Append loss and iteration count to the lists
        loss_list.append(loss.item())
        iteration_list.append(count)

        # Testing the model
        if not (count % 50):    # It's same as "if count % 50 == 0"
            total = 0
            correct = 0

            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Append predictions and labels to the lists
                predictions_list.append(predicted)
                labels_list.append(labels)

            accuracy = 100 * correct / total
            accuracy_list.append(accuracy)  # Append accuracy to the list
            print(f'Accuracy: {accuracy}%')

            if accuracy >= target_accuracy:
                print(f'Overall accuracy of {target_accuracy}% reached.')
                stop_training = True
                create_analizyse()
                break
        else:
            accuracy_list.append(None)  # Append None for iterations without accuracy calculation

    if stop_training:
        break
