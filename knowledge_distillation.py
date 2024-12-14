#import student model and pretrained teacher model
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

from torchinfo import summary
# Import Pascal VOC 2012 Dataset
from torchvision.datasets import VOCSegmentation

# Import ResNet50
from torchvision.models.segmentation import fcn_resnet50, FCN_ResNet50_Weights
from torchvision.transforms.v2 import PILToTensor

from model4 import Model4

# Variables
batch_size = 8
learning_rate = 1e-3
n_epochs = 50
showGraphs = False

def train_kd(teacher, student, train_loader, num_epochs, optimizer, scheduler, loss_fn, device, temperature, soft_weight, label_weight, validation_loader=None, save_file=None, plot_file=None):
    teacher.eval()
    student.train()

    losses_train = []
    losses_val = []

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        running_loss = 0.0

        print(f"Epoch {epoch+1}/{num_epochs}")
        for i, (inputs, labels) in enumerate(train_loader):
            print(f"Batch {i + 1}/{len(train_loader)}", end='\r')
            inputs = inputs.to(device)
            labels = labels.to(device)

            labels = labels.squeeze(1).long()

            optimizer.zero_grad()

            # Forward pass with the teacher model
            with torch.no_grad():
                teacher_outputs = teacher(inputs)
                teacher_outputs = teacher_outputs['out']

            # Forward pass with the student model
            student_outputs = student(inputs)

            #softmax logits
            soft_targets = nn.functional.softmax(teacher_outputs / temperature, dim=1)
            soft_prob = nn.functional.softmax(student_outputs / temperature, dim=1)

            loss_fn.ignore_index = -100
            soft_loss = loss_fn(soft_targets, soft_prob)

            #calculate label loss
            loss_fn.ignore_index = 255
            label_loss = loss_fn(student_outputs, labels)

            #weighted sum of both losses
            loss = (soft_weight * soft_loss) + (label_weight * label_loss)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)

        losses_train += [epoch_loss]

        print('{} Epoch {}, Training loss {}'.format(
            datetime.datetime.now(), epoch, epoch_loss))

        # Step scheduler
        if scheduler is not None:
            scheduler.step()

        if save_file != None:
            torch.save(student.state_dict(), "weights_" + str(epoch) + ".pth")

        if plot_file != None:
            plt.figure(2, figsize=(12, 7))
            plt.clf()

            plt.plot(losses_train, label='train')
            if validation_loader is not None:
                plt.plot(losses_val, label='validation')
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.legend(loc=1)
            print('saving ', plot_file)
            plt.savefig(plot_file)

    print("Finished training")
    #loss for final epoch
    print("Final loss: ", epoch_loss)


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

    #Teacher model

    # Create pretrained model with weights for Pascal VOC 2012
    weights = FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
    teacher_model = fcn_resnet50(weights=weights, num_classes=21)
    teacher_model.to(device)
    #summary(teacher_model, input_size=(batch_size, 3, 224, 224), col_names=("input_size", "output_size", "num_params"), depth=2)

    #Student model
    student_model = Model4(in_channels=3, out_channels=21, returnBackbone=False)
    student_model.to(device)
    #summary(student_model, input_size=(batch_size, 3, 224, 224), col_names=("input_size", "output_size", "num_params"), depth=2)


    loss_fn = nn.CrossEntropyLoss(ignore_index=255)
    teacher_optimizer = optim.Adam(teacher_model.parameters(), lr=learning_rate)
    student_optimizer = optim.Adam(student_model.parameters(), lr=learning_rate)
    teacher_scheduler = optim.lr_scheduler.StepLR(teacher_optimizer, step_size=30, gamma=0.1)
    student_scheduler = optim.lr_scheduler.StepLR(student_optimizer, step_size=30, gamma=0.1)

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
    train_set = VOCSegmentation(root='./data', year='2012', image_set='train', download=True, transform=transform, target_transform=label_transform)
    # test_set = VOCSegmentation(root='./data', year='2012', image_set='val', download=True, transform=transform, target_transform=label_transform)

    # Create DataLoaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # Train model
    train_kd(
        teacher=teacher_model,
        student=student_model,
        train_loader=train_loader,
        num_epochs=n_epochs,
        optimizer=student_optimizer,
        scheduler = None,
        loss_fn=loss_fn,
        device=device,
        temperature=2,
        soft_weight=0.25,
        label_weight=0.75,
        validation_loader=None,
        save_file='weights.pth',
        plot_file='r_kd_plot.png'
    )


###################################################################

if __name__ == '__main__':
    main()