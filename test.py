import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import datetime
import argparse
import numpy as np
from collections import Counter
from torchinfo import summary

# Import Pascal VOC 2012 Dataset
from torchvision.datasets import VOCSegmentation

# Import ResNet50
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from torchvision.transforms.v2 import PILToTensor

# Import custom model
from model4 import Model4

# Variables
batch_size = 8
learning_rate = 1e-4
n_epochs = 20
model_name = "s"
showGraphs = False

# Array of 20 RGB colors
colors = np.array([
    [0, 0, 0],
    [128, 0, 0],
    [0, 128, 0],
    [128, 128, 0],
    [0, 0, 128],
    [128, 0, 128],
    [0, 128, 128],
    [128, 128, 128],
    [64, 0, 0],
    [192, 0, 0],
    [64, 128, 0],
    [192, 128, 0],
    [64, 0, 128],
    [192, 0, 128],
    [64, 128, 128],
    [192, 128, 128],
    [0, 64, 0],
    [128, 64, 0],
    [0, 192, 0],
    [128, 192, 0],
    [0, 64, 128]
])

fillerColours = np.full((255, 3), 255, dtype=np.uint8)

# Append fillerColours to colors
colors = np.append(colors, fillerColours, axis=0)

# Inputs are labels in range 0-20 where 0 is the background and 1-20 are 20 unique classes
def meanIoU(output, target):
    total_loss = 0
    # Break down the batches
    for n in range(output.shape[0]):
        ious = []
        # Calculate for every class
        for i in range(1, 21):
            # Calculate intersection
            intersection = np.logical_and(output[n] == i, target[n][0] == i)
            union = np.logical_or(output[n] == i, target[n][0] == i)
            if union.sum() == 0:
                ious.append(float("nan"))
            else:
                ious.append(intersection.sum() / union.sum())
        total_loss += np.nanmean(ious)

    return total_loss

def test(model, loss_fn, test_loader, device):
    print('Testing ...')
    model.eval()

    total_loss = 0

    for imgs, labels in test_loader:
        if device == 'cuda':
            imgs = imgs.cuda()
            labels = labels.cuda()

        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)

        argmaxOutput = torch.argmax(outputs, dim=1)

        # Plot the output
        if (showGraphs):
            # Show the img, argmaxOutput, and labels side by side
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))
            fig.suptitle("Input, Ground Truth, and Prediction", fontsize=16)

            axs[0].imshow(imgs[0].permute(1, 2, 0).cpu().numpy())
            axs[0].set_title("Image")
            axs[0].axis("off")
            argmaxColours = colors[argmaxOutput[0].cpu().numpy()]
            argmaxColours = torch.tensor(argmaxColours)
            axs[1].imshow(argmaxColours)
            axs[1].set_title("Prediction")
            axs[1].axis("off")
            showlabels = colors[labels[0].cpu().numpy()]
            showlabels = torch.tensor(showlabels)
            axs[2].imshow(showlabels[0].cpu().numpy())
            axs[2].set_title("Label")
            axs[2].axis("off")

            plt.tight_layout()
            plt.show()
            plt.close(fig)


        # Calculate mean intersection over union loss
        loss = meanIoU(argmaxOutput.cpu().numpy(), labels.cpu().numpy())
        total_loss += loss

    # Divide total loss by number of samples
    print('Mean IoU: ', total_loss / len(test_loader.dataset))

def main():
    global n_epochs, model_name, showGraphs, batch_size

    # Get arguments
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-m', metavar='model', type=str, help='model to use [Solo (s), Response-Based (r), Feature-based (f)]', default='s')
    argParser.add_argument('-b', metavar='batch size', type=int, help='batch size', default=32)
    argParser.add_argument('-d', metavar='show pics', type=int, help='Show model outputs in matplotlib', default=0)

    args = argParser.parse_args()

    # Assign command line arguments to variables
    if args.m != None:
        model_name = args.m
    if args.b != None:
        batch_size = args.b
    if args.d != None:
        showGraphs = args.d

    # Configure device
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    print('USING DEVICE ', device)

    # Create pretrained model with weights for Pascal VOC 2012
    if model_name == 's':
        weights_file = "weights_solo.pth"
    elif model_name == 'r':
        weights_file = "weights_response_based.pth"
    elif model_name == 'f':
        weights_file = "weights_feature_based.pth"
    else:
        print("Invalid model name")
        exit()

    model = Model4(in_channels=3, out_channels=21)
    model.to(device)
    # summary(model, input_size=(batch_size, 3, 224, 224), col_names=("input_size", "output_size", "num_params"))

    # load save file
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(weights_file))
    else:
        model.load_state_dict(torch.load(weights_file, map_location=torch.device('cpu')))

    loss_fn = nn.CrossEntropyLoss()

    # Create transform for dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    label_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.ToTensor(), # For [0.0, 1.0]
        PILToTensor(), # For [0, 255]
    ])

    # Get datasets
    test_set = VOCSegmentation(root='./data', year='2012', image_set='val', download=True, transform=transform, target_transform=label_transform)

    # Create DataLoaders
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    print("Testing on ", len(test_loader), " batches")

    # Test model
    test(model, loss_fn, test_loader, device)

###################################################################

if __name__ == '__main__':
    main()
