from . import config
from torch.nn import ConvTranspose2d
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import ReLU
from torchvision.transforms import CenterCrop
from torch.nn import functional as F
import torch

class Block(Module):
    def __init__(self, inchannels, outchannels):
        super().__init__()
        self.conv1 = Conv2d(inchannels, outchannels, 3)
        self.relu = ReLU()
        self.conv2 = Conv2d(outchannels, outchannels, 3)

    def forward(self, x):
        output = self.conv1(x)
        output = self.relu(output)
        output = self.conv2(output)
        
        return output

class Encoder(Module):
    def __init__(self, channels=(3, 16, 32, 64)):
        super().__init__()
        self.enc_blocks = ModuleList(
            [Block(channels[i], channels[i + 1]) for i in range(len(channels)-1)]
        )
        
        #enc_blocks = {
        #   Block1(3, 16) --> conv( 3, 16) / relu / conv(16, 16)
        #   Block2(16, 32) -> conv(16, 32) / relu / conv(32, 32)
        #   Block3(32, 64) -> conv(32, 64) / relu / conv(64, 64)
        # }

        self.pool = MaxPool2d(2)

    def forward(self, x):
        #print('---------------starting decoding process!!---------------')

        blockOutputs = [] 
        for block in self.enc_blocks:
            x = block(x)
            #print('after block data', x.shape)
            blockOutputs.append(x)
            x = self.pool(x)
            
        #for i in range(len(blockOutputs)):
            #print(blockOutputs[i].shape)
        return blockOutputs
        #blockOutputs
        # {
            #[124,124,16]
            #[58, 58, 32]
            #[25, 25, 64]
        # }

class Decoder(Module):
    def __init__(self, channels=(64, 32, 16)): #channels decrease by 1/2
        super().__init__()
        self.channels = channels
        self.upconvs = ModuleList(
            [ConvTranspose2d(channels[i], channels[i+1], 2, 2) for i in range(len(channels) -1)]
        )#image size doubled & channel halved!!
        # upconvs = [
        #   ConvTranspose2d(64, 32, 2, 2)
        #   ConvTranspose2d(32, 16, 2, 2)
        # ]

        self.dec_blocks = ModuleList(
            [Block(channels[i], channels[i+1]) for i in range(len(channels)-1)]
        )#channel size halved!!
        # dec_blocks = [
        #   Block = conv(64, 32) / relu / conv(32, 32)
        #   Block = conv(32, 16) / relu / conv(16, 16)
        # ]


    def forward(self, x, encFeatures):
        #print('---------------starting decoding process!!---------------')
        
        for i in range(len(self.channels)-1): #2 iterations
            #print('{}th input x shape'.format(i+1), x.shape)
            x=self.upconvs[i](x) 
            encFeature = self.crop(encFeatures[i], x) #crop encoder_features same size as x
            #print('after crop encFeat', encFeat.shape)
            x = torch.cat([x, encFeature], dim = 1) #concat encoder image & upsampled decoder image
            #print('after concat', x.shape)
            x = self.dec_blocks[i](x) #2 layer conv(channel halves)
            #print('after dec_block', x.shape)

        return x
    
    def crop(self, encFeatures, x):
        (_, _, H, W) = x.shape
        encFeatures = CenterCrop([H, W])(encFeatures)

        return encFeatures

#Encoder과 Decoder를 미리 정의하고 UNet에서 객체화시킴!! (좋은 코드..!!)
class UNet(Module):
    def __init__(self, encChannels=(3, 16, 32, 64), decChannels=(64, 32, 16),
        nbClasses=1, retainDim=True,
        outSize=(config.INPUT_IMAGE_HEIGHT, config.INPUT_IMAGE_WIDGH)):
        super().__init__()
        self.encoder = Encoder(encChannels)
        self.decoder = Decoder(decChannels)

        self.fc_layer = Conv2d(decChannels[-1], nbClasses, 1)
        self.retainDim = retainDim
        self.outSize = outSize

    def forward(self, x):
        encFeatures = self.encoder(x)
        decoder_result = self.decoder(encFeatures[::-1][0] ,encFeatures[::-1][1:]) 
        #encFeatures에 저장되어 있던 featureMap들을 역순으로 사용
        #first featureMap is used as Decoder's input, left featureMaps are used in skip_connection!

        prediction = self.fc_layer(decoder_result)
        if self.retainDim:
            final_map = F.interpolate(prediction, self.outSize) #outsize=(H, W) / interpolate = 크기에 맞게 자연스럽게 사이즈 확대?
        
        #print('after mapping', map.shape)
        return final_map
