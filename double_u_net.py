import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#model = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=True)

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