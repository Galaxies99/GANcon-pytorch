import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, channel):
        super(ResBlock, self).__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(channel, channel, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(channel, channel, kernel_size = 3, stride = 1, padding = 1)
        )
        self.relu = nn.ReLU(inplace = True)
    
    def forward(self, x):
        y = self.residual(x)
        return self.relu(x + y)


class GeneratorEncoder(nn.Module):
    def __init__(self, in_channel, dropout = -1):
        super(GeneratorEncoder, self).__init__()
        self.res_block = ResBlock(in_channel)
        if 0 < dropout < 1:
            self.middle_layer = nn.Sequential(
                nn.Conv2d(in_channel, in_channel, kernel_size = 3, stride = 1, padding = 1),
                nn.ReLU(inplace = True),
                nn.Dropout(p = dropout, inplace = False)
            )
        else:
            self.middle_layer = nn.Sequential(
                nn.Conv2d(in_channel, in_channel, kernel_size = 3, stride = 1, padding = 1),
                nn.ReLU(inplace = True)
            )
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)
    
    def forward(self, x):
        x = self.res_block(x)
        x = self.middle_layer(x)
        y = self.pool(x)
        return x, y
    

class GeneratorDecoder(nn.Module):
    def __init__(self, in_channel):
        super(GeneratorDecoder, self).__init__()
        self.up_sampling = nn.ConvTranspose2d(in_channel, in_channel, kernel_size = 2, stride = 2)
        self.middle_layer = nn.Sequential(
            nn.Conv2d(in_channel + in_channel, in_channel, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(inplace = True)
        )
        self.res_block = ResBlock(in_channel)

    def forward(self, x, y):
        x = self.up_sampling(x)
        x = torch.cat([x, y], dim = 1)
        x = self.middle_layer(x)
        x = self.res_block(x)
        return x


class ContactMapGenerator(nn.Module):
    def __init__(self, args):
        super(ContactMapGenerator, self).__init__()
        in_channel = args.get('input_channel', 441)
        out_channel = args.get('output_channel', 10)
        dropout_rate = args.get('dropout_rate', 0.5)

        self.encoder1 = GeneratorEncoder(in_channel)
        self.encoder2 = GeneratorEncoder(in_channel)
        self.encoder3 = GeneratorEncoder(in_channel)
        self.encoder4 = GeneratorEncoder(in_channel, dropout = dropout_rate)

        self.middle_layer = nn.Sequential(
            ResBlock(in_channel),
            nn.Conv2d(in_channel, in_channel, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(inplace = True),
            nn.Dropout(p = dropout_rate)
        )

        self.decoder1 = GeneratorDecoder(in_channel)
        self.decoder2 = GeneratorDecoder(in_channel)
        self.decoder3 = GeneratorDecoder(in_channel)
        self.decoder4 = GeneratorDecoder(in_channel)

        self.final = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(inplace = True),
            nn.Conv2d(out_channel, out_channel, kernel_size = 1, stride = 1),
            nn.Softmax(dim = 1)
        )
    
    def forward(self, x):
        # x: n * input_channel * L * L
        # skip1: n * input_channel * L * L
        # x1: n * input_channel * (L / 2) * (L / 2)
        skip1, x1 = self.encoder1(x)
        # skip2: n * input_channel * (L / 2) * (L / 2)
        # x2: n * input_channel * (L / 4) * (L / 4)
        skip2, x2 = self.encoder2(x1)
        # skip3: n * input_channel * (L / 4) * (L / 4)
        # x3: n * input_channel * (L / 8) * (L / 8)
        skip3, x3 = self.encoder3(x2)
        # skip4: n * input_channel * (L / 8) * (L / 8)
        # x4: n * input_channel * (L / 16) * (L / 16)
        skip4, x4 = self.encoder4(x3)
        # y0: n * input_channel * (L / 16) * (L / 16)
        y0 = self.middle_layer(x4)
        # y1: n * input_channel * (L / 8) * (L / 8)
        y1 = self.decoder1(y0, skip4)
        # y2: n * input_channel * (L / 4) * (L / 4)
        y2 = self.decoder2(y1, skip3)
        # y3: n * input_channel * (L / 2) * (L / 2)
        y3 = self.decoder3(y2, skip2)
        # y4: n * input_channel * L * L
        y4 = self.decoder4(y3, skip1)
        # res: n * output_channel * L * L
        res = self.final(y4)
        return res
        

class ContactMapDiscriminator(nn.Module):
    def __init__(self, args = {}):
        super(ContactMapDiscriminator, self).__init__()
        in_channel = args.get('input_channel', 441 + 10)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size = 3, stride = 1, padding = 1),
            nn.LeakyReLU(inplace = True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size = 3, stride = 1, padding = 1),
            nn.LeakyReLU(inplace = True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size = 3, stride = 1, padding = 1),
            nn.LeakyReLU(inplace = True)
        )
        self.final = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size = 1, stride = 1),
            nn.Sigmoid()
        )
    
    def forward(self, feature, contact_map):
        x = torch.cat([feature, contact_map], dim = 1)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.final(x)
        return x
