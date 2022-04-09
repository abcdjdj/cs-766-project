import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchsummary import summary
from torchvision import models
from imageio import imread as imread
import matplotlib.pyplot as plt
import utils

"""
Function: Squueze and Excite Block
Inputs: Activation Maps
Outputs: Relevant Activation Maps retained, redundant dropped
"""
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

    #print(se.shape)

    se = se.view(init.shape[0],filters,1,1) #(b, 32, 1, 1)

    #print(se.shape)

    return torch.mul(init,se) #(b,32,128,128)
"""
Function: ASPP to get high resolution feature maps
Inputs: feature maps, output channels desired 
Outputs: High Res feature maps
"""
def ASPP(x, filter_count):
    se = nn.AvgPool2d(kernel_size = (x.shape[2], x.shape[3]))(x)
    se = nn.Conv2d(in_channels = se.shape[1], out_channels = filter_count, kernel_size = 1, padding='same')(se)
    se = nn.BatchNorm2d(num_features = se.shape[1])(se)
    se = nn.ReLU()(se)
    se = nn.UpsamplingBilinear2d(size=(x.shape[2], x.shape[3]))(se)
    #print(se.shape)

    y1 = nn.Conv2d(dilation=1, in_channels = x.shape[1], out_channels = filter_count, kernel_size = 1, padding='same', bias=False)(x)
    y1 = nn.BatchNorm2d(num_features = y1.shape[1])(y1)
    y1 = nn.ReLU()(y1)
    #print(y1.shape)

    y2 = nn.Conv2d(dilation=6, in_channels = x.shape[1], out_channels = filter_count, kernel_size = 1, padding='same', bias=False)(x)
    y2 = nn.BatchNorm2d(num_features = y2.shape[1])(y2)
    y2 = nn.ReLU()(y2)
    #print(y2.shape)

    y3 = nn.Conv2d(dilation=12, in_channels = x.shape[1], out_channels = filter_count, kernel_size = 1, padding='same', bias=False)(x)
    y3 = nn.BatchNorm2d(num_features = y3.shape[1])(y3)
    y3 = nn.ReLU()(y3)
    #print(y3.shape)

    y4 = nn.Conv2d(dilation=18, in_channels = x.shape[1], out_channels = filter_count, kernel_size = 1, padding='same', bias=False)(x)
    y4 = nn.BatchNorm2d(num_features = y4.shape[1])(y4)
    y4 = nn.ReLU()(y4)
    #print(y4.shape)

    y = torch.cat([se, y1, y2, y3, y4], dim=1)
    y = nn.Conv2d(dilation=1, in_channels = y.shape[1], out_channels = filter_count, kernel_size = 1, padding='same', bias=False)(y)
    y = nn.BatchNorm2d(num_features = y.shape[1])(y)
    y = nn.ReLU()(y)
    #print(y.shape)
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
Input: Input Activation Map, Desired output channels
Output: Convolved Activation Maps
"""
def conv_block(x, filters):
    x = nn.Conv2d(in_channels = x.shape[1], out_channels = filters, kernel_size = 3, padding='same')(x)
    x = nn.BatchNorm2d(num_features = x.shape[1])(x)
    x = nn.ReLU()(x)

    x = nn.Conv2d(in_channels = x.shape[1], out_channels = filters, kernel_size = 3, padding='same')(x)
    x = nn.BatchNorm2d(num_features = x.shape[1])(x)
    x = nn.ReLU()(x)

    x = squeeze_and_excite(x)

    return x

"""
Function: Decoder 1
Params: ASPP Output, Skip Connections from Encoder1
Output: To be passed through output_block to get mask
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

"""
Function: To get mask from decoder1 output
Input: Decoder1 output
Output: Mask for Network1
"""
def output_block(inputs):
    x = nn.Conv2d(in_channels = inputs.shape[1], out_channels = 1, kernel_size = 1, padding = "same")(inputs)
    x = nn.Sigmoid()(x)
    return x


def encoder2(inputs):
    num_filters = [32, 64, 128, 256]
    skip_connections = []
    x = inputs

    for f in num_filters:
        x = conv_block(x, f)
        skip_connections.append(x)
        x = nn.MaxPool2d(kernel_size = (2,2))(x)

    return x, skip_connections

def decoder2(inputs, skip_1, skip_2):
    num_filters = [256, 128, 64, 32]

    skip_2.reverse()

    x = inputs

    for i,f in enumerate(num_filters):
        x = nn.UpsamplingBilinear2d(size=(2*x.shape[2], 2*x.shape[3]))(x)
        #print(f"X -> {x.shape}")
        #print(f"Skip1 -> {torch.Tensor(skip_1[i]).shape}")
        #print(f"Skip2 -> {torch.Tensor(skip_2[i]).shape}")
        x = torch.cat([x, skip_1[i], skip_2[i]], dim=1)
        x = conv_block(x, f)

    return x

"""
Function: Network 1 pipeline
Input: Batches of Images
Output: Input * Mask
"""
def build_model(inputs):
    encoder1_op, encoder1_skip_conns = encoder1(inputs)
    #print(f"Encoder 1 o/p shape {encoder1_op.shape}")
    aspp_op = ASPP(encoder1_op, 64)
    #print(f"ASPP o/p shape {aspp_op.shape}")
    decoder1_op = decoder1(aspp_op, encoder1_skip_conns)
    #print(f"Decoder 1 o/p shape {decoder1_op.shape}")
    mask = output_block(decoder1_op)
    #print(f"Mask shape {mask.shape}")
    network1_op = inputs * mask
    #print(f"Network 1 o/p shape {network1_op.shape}")
    encoder2_op,encoder2_skip_conns = encoder2(network1_op)
    #print(f"Encoder2 o/p shape {encoder2_op.shape}")
    aspp2_op = ASPP(encoder2_op, 64)
    #print(f"ASPP2 o/p shape {aspp2_op.shape}")
    decoder2_op = decoder2(aspp2_op, encoder1_skip_conns, encoder2_skip_conns)
    #print(f"Decoder2 o/p shape {decoder2_op.shape}")
    network2_op = output_block(decoder2_op)
    #print(f"Network 2 o/p shape {network2_op.shape}")
    final_output = torch.cat([mask, network2_op], dim = 1)
    print(f"Final o/p shape {final_output.shape}")
    return final_output



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
    inputs = utils.read_img("cvc-DB/PNG/Original/1.png")
    inputs1 = utils.img_to_tensor(inputs)
    inputs2 = inputs1
    ip = torch.cat([inputs1, inputs2])
    print(build_model(ip.float()))

if __name__ == '__main__':
    main()