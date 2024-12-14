from sched import scheduler

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.nn.functional as F
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
from torchvision.transforms import PILToTensor

from model import LZNet
from model2 import LZNet2
from model3 import Model3
from model4 import Model4

# Variables
batch_size = 8
learning_rate = 1e-3
n_epochs = 80

class DiceLoss(torch.nn.Module):
    def __init__(self, ignore_index=255, class_weights=None):
        super().__init__()
        self.ignore_index = ignore_index
        self.class_weights = class_weights

    def forward(self, logits, targets):
        smooth = 1.0  # Smoothing factor to avoid division by zero
        num_classes = logits.shape[1]

        # Mask for valid labels (excluding the void class)
        valid_mask = (targets != self.ignore_index)

        # Clamp targets to valid range for one-hot encoding
        targets_clamped = targets.clamp(0, num_classes - 1)

        # Convert targets to one-hot format
        targets_one_hot = F.one_hot(targets_clamped, num_classes=num_classes)  # Shape: [batch_size, H, W, num_classes]
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2)  # Shape: [batch_size, num_classes, H, W]

        # Apply the mask to exclude void pixels from the loss calculation
        valid_mask = valid_mask.unsqueeze(1)  # Shape: [batch_size, 1, H, W]
        targets_one_hot = targets_one_hot * valid_mask  # Zero-out void regions in the one-hot labels
        probs = F.softmax(logits, dim=1) * valid_mask  # Zero-out void regions in the predictions

        # Compute intersection and union
        intersection = (probs * targets_one_hot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))

        # Compute Dice coefficient
        if self.class_weights is not None:
            dice = ((2.0 * intersection + smooth) / (union + smooth)) * self.class_weights
        else:
            dice = ((intersection + smooth) / (union + smooth))
        return 1 - dice.mean()

# Custom Combined Loss function
class CombinedLoss(nn.Module):
    def __init__(self, ce_weight=1.0, dice_weight=1.0, weights=None):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=255, weight=weights)
        self.dice_loss = DiceLoss()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        ce = self.ce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        return self.ce_weight * ce + self.dice_weight * dice

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

def train(n_epochs, optimizer, model, loss_fn, train_loader, scheduler, device, validation_loss_fn=None, validation_loader=None, save_file=None, plot_file=None):
    print('Training ...')

    losses_train = []
    losses_val = []

    # Training Loop
    for epoch in range(1, n_epochs + 1):
        model.train()
        print('Epoch {}/{}'.format(epoch, n_epochs))
        epoch_loss = 0.0
        i = 0

        # Training set
        for imgs, labels in train_loader:
            print(f"Batch {i + 1}/{len(train_loader)}", end='\r')
            i += 1
            optimizer.zero_grad()

            if device == 'cuda':
                imgs = imgs.cuda()
                labels = labels.cuda()

            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)

            labels = labels.squeeze(1).long()

            loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()

            # Add to epoch loss total
            epoch_loss += loss.item()

        # Add epoch loss to list for plotting
        losses_train += [epoch_loss / len(train_loader)]

        print('{} Epoch {}, Training loss {}'.format(
            datetime.datetime.now(), epoch, epoch_loss / len(train_loader)))

        val_loss = 0.0
        if validation_loss_fn is not None:
            # Validation set
            model.eval()

            for imgs, labels in validation_loader:
                if device == 'cuda':
                    imgs = imgs.cuda()
                    labels = labels.cuda()

                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)

                labels = labels.squeeze(1).long()

                loss = validation_loss_fn(outputs, labels)
                val_loss += loss.item()

            losses_val += [val_loss / len(validation_loader)]

            print('Epoch {}, Validation loss {}'.format(
                epoch, val_loss / len(validation_loader)))

        # Step scheduler
        if scheduler is not None:
            # scheduler.step(epoch_loss)
            scheduler.step()

        if save_file != None:
            torch.save(model.state_dict(), "weights/weights_" + str(epoch) + ".pth")
            if (val_loss <= min(losses_val) and val_loss != 0.0) or epoch == 1:
                torch.save(model.state_dict(), "weights/weights_min_val.pth")

        if plot_file != None:
            plt.figure(2, figsize=(12, 7))
            plt.clf()

            plt.plot(losses_train, label='train')
            if validation_loss_fn is not None:
                plt.plot(losses_val, label='validation')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.legend(loc=1)
            print('saving ', plot_file)
            plt.savefig(plot_file)

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0.01)
    elif type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0.01)

def compute_class_weights(labels, num_classes, ignore_index=255):
    pixel_counts = Counter()
    total_pixels = 0

    for label in labels:
        unique, counts = np.unique(label[label != ignore_index], return_counts=True)
        pixel_counts.update(dict(zip(unique, counts)))
        total_pixels += label[label != ignore_index].size

    class_frequencies = np.array([pixel_counts.get(i, 0) / total_pixels for i in range(num_classes)])
    median_frequency = np.median(class_frequencies[class_frequencies > 0])
    class_weights = median_frequency / (class_frequencies + 1e-8)  # Avoid divide-by-zero
    return torch.tensor(class_weights, dtype=torch.float32)

def main():
    global n_epochs, batch_size

    # Get arguments
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-b', metavar='batch size', type=int, help='batch size', default=32)
    argParser.add_argument('-e', metavar='num epochs', type=int, help='number of epochs', default=40)

    args = argParser.parse_args()

    # Assign command line arguments to variables
    if args.b != None:
        batch_size = args.b
    if args.e != None:
        n_epochs = args.e

    # Configure device
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'
    print('USING DEVICE ', device)

    # Create pretrained model with weights for Pascal VOC 2012
    # model = fcn_resnet50(weights=FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1)
    model = Model4(in_channels=3, out_channels=21)
    model.apply(init_weights)
    model.to(device)
    summary(model, input_size=(batch_size, 3, 224, 224), col_names=("input_size", "output_size", "num_params"))

    class_weights = [0.0006 * 2, 0.0630, 0.1572, 0.0539, 0.0757, 0.0770, 0.0263, 0.0326, 0.0173,
        0.0405, 0.0552, 0.0356, 0.0275, 0.0500, 0.0409, 0.0097, 0.0717, 0.0524,
        0.0321, 0.0292, 0.0515]
    class_weights = torch.tensor(class_weights)
    class_weights = class_weights / class_weights.sum()

    if device == 'cuda':
        class_weights = torch.FloatTensor(class_weights).cuda()
    else:
        class_weights = torch.FloatTensor(class_weights)
    class_weights = class_weights.to(device)
    # loss_fn = nn.CrossEntropyLoss(ignore_index=255, weight=class_weights)
    loss_fn = CombinedLoss(ce_weight=0.1, dice_weight=1, weights=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Create transform for dataset
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    label_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        # transforms.ToTensor(), # For [0.0, 1.0]
        PILToTensor(),  # For [0, 255]
    ])

    # Get datasets
    train_set = VOCSegmentation(root='./data', year='2012', image_set='train', download=True, transform=transform, target_transform=label_transform)
    test_set = VOCSegmentation(root='./data', year='2012', image_set='val', download=True, transform=transform, target_transform=label_transform)

    # Create DataLoaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    print("Training on ", len(train_loader), " batches for ", n_epochs, " epochs")

    # Train model
    train(n_epochs,
          optimizer,
          model,
          loss_fn,
          train_loader,
          scheduler,
          device,
          validation_loss_fn=loss_fn,
          validation_loader=test_loader,
          save_file='weights.pth',
          plot_file='plot.png')

###################################################################

if __name__ == '__main__':
    main()
