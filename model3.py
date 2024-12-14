import torch
import torch.nn as nn

class Model3(nn.Module):
    def __init__(self, num_classes):
        super(Model3, self).__init__()
        prevSize = 3

        sizes = [64, 128, 256, 780]
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()


        # Simple 4 layer Conv + MaxPool Down
        for size in sizes:
            self.encoder.append(
                nn.Sequential(
                    nn.Conv2d(prevSize, size, kernel_size=3, padding=1, stride=1),
                    nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
                )
            )
            prevSize = size

        # Upscale with ConvTranspose2d
        # for size in sizes[::-1]:
        #     self.decoder.append(
        #         nn.Sequential(
        #             nn.ConvTranspose2d(prevSize, size, kernel_size=3, stride=2, padding=1, output_padding=1),
        #         )
        #     )
        #     prevSize = size
        self.decoder.append(
            nn.Sequential(
                nn.ConvTranspose2d(780, 256, kernel_size=2, stride=2, padding=0),
                nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0),
                nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0),
                nn.ConvTranspose2d(64, num_classes, kernel_size=2, stride=2, padding=0),
            )
        )

    def forward(self, x):
        # Encoder
        for layer in self.encoder:
            x = layer(x)

        # Decoder
        for layer in self.decoder:
            x = layer(x)

        return x
        
