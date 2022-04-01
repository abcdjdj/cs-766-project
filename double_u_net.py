import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
from torchvision import models

def squeeze_and_excite(inputs, ratio = 8):
    init = inputs  #(b, 32, 128, 128)
    channel_axis = 1
    filters = init.shape[channel_axis]

    #print(f'Filter shape - {filters}') #32
    
    se = nn.AvgPool2d(kernel_size = (init.shape[2], init.shape[3]))(init) # (b, 32) -> (b,4)

    se = se.view(init.shape[0] , filters)

    #print(f'After avg pool {se.shape}') # (b,32)

    se = nn.Sequential(nn.Linear(filters, filters//ratio, bias = False), nn.ReLU(), nn.Linear(filters//ratio, filters, bias = False), nn.Sigmoid())(se) # (b,32)

    #print(f'Shape after  - {se.shape}')

    print(se.shape)

    se = se.view(init.shape[0],filters,1,1) #(b, 32, 1, 1)

    print(se.shape)

    return torch.mul(init,se) #(b,32,128,128)

def ASPP(x, filter_count):
    se = nn.AvgPool2d(kernel_size = (x.shape[2], x.shape[3]))(x)
    se = nn.Conv2d(in_channels = se.shape[1], out_channels = filter_count, kernel_size = 1, padding='same')(se)
    se = nn.BatchNorm2d(num_features = se.shape[1])(se)
    se = nn.ReLU()(se)
    se = nn.UpsamplingBilinear2d(size=(x.shape[2], x.shape[3]))(se)
    print(se.shape)

    y1 = nn.Conv2d(dilation=1, in_channels = x.shape[1], out_channels = filter_count, kernel_size = 1, padding='same', bias=False)(x)
    y1 = nn.BatchNorm2d(num_features = y1.shape[1])(y1)
    y1 = nn.ReLU()(y1)
    print(y1.shape)

    y2 = nn.Conv2d(dilation=6, in_channels = x.shape[1], out_channels = filter_count, kernel_size = 1, padding='same', bias=False)(x)
    y2 = nn.BatchNorm2d(num_features = y2.shape[1])(y2)
    y2 = nn.ReLU()(y2)
    print(y2.shape)

    y3 = nn.Conv2d(dilation=12, in_channels = x.shape[1], out_channels = filter_count, kernel_size = 1, padding='same', bias=False)(x)
    y3 = nn.BatchNorm2d(num_features = y3.shape[1])(y3)
    y3 = nn.ReLU()(y3)
    print(y3.shape)

    y4 = nn.Conv2d(dilation=18, in_channels = x.shape[1], out_channels = filter_count, kernel_size = 1, padding='same', bias=False)(x)
    y4 = nn.BatchNorm2d(num_features = y4.shape[1])(y4)
    y4 = nn.ReLU()(y4)
    print(y4.shape)

    y = torch.cat([se, y1, y2, y3, y4], dim=1)
    y = nn.Conv2d(dilation=1, in_channels = y.shape[1], out_channels = filter_count, kernel_size = 1, padding='same', bias=False)(y)
    y = nn.BatchNorm2d(num_features = y.shape[1])(y)
    y = nn.ReLU()(y)
    return y


"""
function: This is Encoder 1
params: Medical Image Input
return: Output of Encoder1, 4 Skip Conns for Decoder 1
"""
def encoder1(inputs):
    model = models.vgg19()
    #print(summary(model,(3,256,256)))

    #skip connections from pre-trained VGG-19
    names = ["ReLU-4", "ReLU-9", "ReLU-18", "ReLU-27", "ReLU-36"]

    indices = [3, 8, 17, 26, 35]

    skip_connections = []

    def encoder1_receive_outputs(layer, _, output):
        skip_connections.append(output)

    for name, layer in model.named_children():
        for idx in indices:
            layer[idx].register_forward_hook(encoder1_receive_outputs)
        break

    model(inputs)

    return skip_connections[-1], skip_connections[0:-1]

"""
Function: 2 Blocks of Convolution + BN + ReLU
Input:
Output:
"""
def conv_block(x, filters):
    x = nn.Conv2d(in_channels = x.shape[1], out_channels = filters, kernel_size = 3, padding='same')(x)
    x = nn.BatchNorm2d(num_features = x.shape[1])(x)
    x = nn.ReLU()(x)

    x = nn.Conv2d(in_channels = x.shape[1], out_channels = filters, kernel_size = 3, padding='same')(x)
    x = nn.BatchNorm2d(num_features = x.shape[1])(x)
    x = nn.ReLU()(x)

    return x

"""
Function: Decoder 1
Params:
Output:
"""
def decoder1(inputs, skip_connections):
    num_filters = [256, 128, 64, 32]

    skip_connections.reverse()

    x = inputs

    for i,f in enumerate(num_filters):
        x = nn.UpsamplingBilinear2d(size=(2*x.shape[2], 2*x.shape[3]))(x)
        x = torch.cat([x, skip_connections[i]], dim=1)
        x = conv_block(x, f)

    return x

'''
Tester functions
'''
def test_encoder1():
    output, skip_connections = encoder1(torch.ones(1,3,256,256))
    print("Encoder1 Output = ")
    print(output.shape)
    print("Skip connections = ")
    for conn in skip_connections:
        print(conn.shape)

def test_decoder1():
    output, skip_connections = encoder1(torch.ones(1,3,256,256))
    print(decoder1(output,skip_connections).shape)
    #Output we get is (1,32,256,256)

def main():
    test_decoder1()

if __name__ == '__main__':
    main()