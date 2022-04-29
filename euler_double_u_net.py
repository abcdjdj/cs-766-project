import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SqueezeAndExcite(nn.Module):
  def __init__(self, x, ratio = 8):
    super(SqueezeAndExcite, self).__init__()

    channel_axis = 1
    filters = x.shape[channel_axis]
    # Architecture
    self.avgpool2d = nn.AvgPool2d(kernel_size = (x.shape[2], x.shape[3]))
    self.sequential = nn.Sequential(nn.Linear(filters, filters//ratio, bias = False), nn.ReLU(), nn.Linear(filters//ratio, filters, bias = False), nn.Sigmoid())

  def forward(self, x):
    init = x
    #channel_axis = 1
    filters = init.shape[1]
    x = self.avgpool2d(x)
    x = x.view(init.shape[0] , filters)
    x = self.sequential(x)
    x = x.view(init.shape[0], filters, 1, 1)

    return torch.mul(init, x)

class ConvBlock(nn.Module):
  def __init__(self, x, filters):
      super(ConvBlock, self).__init__()

      self.layer1_conv2d = nn.Conv2d(in_channels = x.shape[1], out_channels = filters, kernel_size = 3, padding='same')
      x = self.layer1_conv2d(x)
      self.layer1_batchnorm2d = nn.BatchNorm2d(num_features = x.shape[1])
      x = self.layer1_batchnorm2d(x)
      self.layer1_relu = nn.ReLU()
      x = self.layer1_relu(x)

      self.layer2_conv2d = nn.Conv2d(in_channels = x.shape[1], out_channels = filters, kernel_size = 3, padding='same')
      x = self.layer2_conv2d(x)
      self.layer2_batchnorm2d = nn.BatchNorm2d(num_features = x.shape[1])
      x = self.layer2_batchnorm2d(x)
      self.layer2_relu = nn.ReLU()
      x = self.layer2_relu(x)

      self.squeeze_and_excite = SqueezeAndExcite(x)

  def forward(self, x):
      x = self.layer1_conv2d(x)
      x = self.layer1_batchnorm2d(x)
      x = self.layer1_relu(x)

      x = self.layer2_conv2d(x)
      x = self.layer2_batchnorm2d(x)
      x = self.layer2_relu(x)

      x = self.squeeze_and_excite.forward(x)
      return x

class ASPP(nn.Module):
    def __init__(self, x, filter_count):
      super(ASPP, self).__init__()

      self.layer1_avgpool2d = nn.AvgPool2d(kernel_size = (x.shape[2], x.shape[3]))
      se = self.layer1_avgpool2d(x)
      self.layer1_conv2d = nn.Conv2d(in_channels = se.shape[1], out_channels = filter_count, kernel_size = 1, padding='same')
      se = self.layer1_conv2d(se)
      self.layer1_batchnorm2d = nn.BatchNorm2d(num_features = se.shape[1])
      se = self.layer1_batchnorm2d(se)
      self.layer1_relu = nn.ReLU()
      se = self.layer1_relu(se)
      self.layer1_upsampling = nn.UpsamplingBilinear2d(size=(x.shape[2], x.shape[3]))
      se = self.layer1_upsampling(se)

      self.layer2_conv2d = nn.Conv2d(dilation=1, in_channels = x.shape[1], out_channels = filter_count, kernel_size = 1, padding='same', bias=False)
      y1 = self.layer2_conv2d(x)
      self.layer2_batchnorm2d = nn.BatchNorm2d(num_features = y1.shape[1])
      y1 = self.layer2_batchnorm2d(y1)
      self.layer2_relu = nn.ReLU()
      y1 = self.layer2_relu(y1)

      self.layer3_conv2d = nn.Conv2d(dilation=6, in_channels = x.shape[1], out_channels = filter_count, kernel_size = 1, padding='same', bias=False)
      y2 = self.layer3_conv2d(x)
      self.layer3_batchnorm2d = nn.BatchNorm2d(num_features = y2.shape[1])
      y2 = self.layer3_batchnorm2d(y2)
      self.layer3_relu = nn.ReLU()
      y2 = self.layer3_relu(y2)

      self.layer4_conv2d = nn.Conv2d(dilation=12, in_channels = x.shape[1], out_channels = filter_count, kernel_size = 1, padding='same', bias=False)
      y3 = self.layer4_conv2d(x)
      self.layer4_batchnorm2d = nn.BatchNorm2d(num_features = y3.shape[1])
      y3 = self.layer4_batchnorm2d(y3)
      self.layer4_relu = nn.ReLU()
      y3 = self.layer4_relu(y3)

      self.layer5_conv2d = nn.Conv2d(dilation=18, in_channels = x.shape[1], out_channels = filter_count, kernel_size = 1, padding='same', bias=False)
      y4 = self.layer5_conv2d(x)
      self.layer5_batchnorm2d = nn.BatchNorm2d(num_features = y4.shape[1])
      y4 = self.layer5_batchnorm2d(y4)
      self.layer5_relu = nn.ReLU()
      y4 = self.layer5_relu(y4)

      y = torch.cat([se, y1, y2, y3, y4], dim=1)
      self.layer6_conv2d = nn.Conv2d(dilation=1, in_channels = y.shape[1], out_channels = filter_count, kernel_size = 1, padding='same', bias=False)
      y = self.layer6_conv2d(y)
      self.layer6_batchnorm2d = nn.BatchNorm2d(num_features = y.shape[1])
      y = self.layer6_batchnorm2d(y)
      self.layer6_relu = nn.ReLU()
      y = self.layer6_relu(y)

    def forward(self, x, filter_count):
      se = self.layer1_avgpool2d(x)
      se = self.layer1_conv2d(se)
      se = self.layer1_batchnorm2d(se)
      se = self.layer1_relu(se)
      se = self.layer1_upsampling(se)
      #print(se.shape)

      y1 = self.layer2_conv2d(x)
      y1 = self.layer2_batchnorm2d(y1)
      y1 = self.layer2_relu(y1)
      #print(y1.shape)

      y2 = self.layer3_conv2d(x)
      y2 = self.layer3_batchnorm2d(y2)
      y2 = self.layer3_relu(y2)
      #print(y2.shape)

      y3 = self.layer4_conv2d(x)
      y3 = self.layer4_batchnorm2d(y3)
      y3 = self.layer4_relu(y3)
      #print(y3.shape)

      y4 = self.layer5_conv2d(x)
      y4 = self.layer5_batchnorm2d(y4)
      y4 = self.layer5_relu(y4)
      #print(y4.shape)

      y = torch.cat([se, y1, y2, y3, y4], dim=1)
      del x, se, y1, y2, y3, y4
      y = self.layer6_conv2d(y)
      y = self.layer6_batchnorm2d(y)
      y = self.layer6_relu(y)
      #print(y.shape)
      return y

class Encoder1(nn.Module):
    def __init__(self):
      super(Encoder1, self).__init__()
      # self.model = models.vgg19(pretrained = True)
      # self.vgg19_final_op = None
    
    def forward(self, inputs):
      #skip connections from pre-trained VGG-19
      names = ["ReLU-4", "ReLU-9", "ReLU-18", "ReLU-27", "ReLU-36"]
      model = models.vgg19(pretrained = True)
      
      if inputs.device == device:
        model = model.to(device)

      indices = [3, 8, 17, 26, 35]

      skip_connections = []

      def encoder1_receive_outputs(layer, _, output):
          skip_connections.append(output.detach())
          

      for name, layer in model.named_children():
          for idx in indices:
              layer[idx].register_forward_hook(encoder1_receive_outputs)
          break

      #self.model(inputs)
      model(inputs)

      # print(f'[Encoder1] Op size = {skip_connections[-1].shape}')
      # print(f'[Encoder1] 0 size = {skip_connections[0].shape}')
      # print(f'[Encoder1] 1 size = {skip_connections[1].shape}')
      # print(f'[Encoder1] 2 size = {skip_connections[2].shape}')
      # print(f'[Encoder1] 3 size = {skip_connections[3].shape}')
      return skip_connections[-1], skip_connections[0:-1]
      # op = torch.ones(3, 512, 18, 24)
      # skip_connections = [torch.ones(3, 64, 288, 384), torch.ones(3, 128, 144, 192), torch.ones(3, 256, 72, 96), torch.ones(3, 512, 36, 48)]
      # return op, skip_connections

class Decoder1(nn.Module):
    def __init__(self, inputs, skip_connections):
        super(Decoder1, self).__init__()
        num_filters = [256, 128, 64, 32]
        skip_connections.reverse()
        x = inputs

        self.layer1_upsampling = nn.UpsamplingBilinear2d(size=(2*x.shape[2], 2*x.shape[3]))
        x = self.layer1_upsampling(x)
        x = torch.cat([x, skip_connections[0]], dim=1)
        self.layer1_convblock = ConvBlock(x, num_filters[0])
        x = self.layer1_convblock.forward(x)

        self.layer2_upsampling = nn.UpsamplingBilinear2d(size=(2*x.shape[2], 2*x.shape[3]))
        x = self.layer2_upsampling(x)
        x = torch.cat([x, skip_connections[1]], dim=1)
        self.layer2_convblock = ConvBlock(x, num_filters[1])
        x = self.layer2_convblock.forward(x)

        self.layer3_upsampling = nn.UpsamplingBilinear2d(size=(2*x.shape[2], 2*x.shape[3]))
        x = self.layer3_upsampling(x)
        x = torch.cat([x, skip_connections[2]], dim=1)
        self.layer3_convblock = ConvBlock(x, num_filters[2])
        x = self.layer3_convblock.forward(x)

        self.layer4_upsampling = nn.UpsamplingBilinear2d(size=(2*x.shape[2], 2*x.shape[3]))
        x = self.layer4_upsampling(x)
        x = torch.cat([x, skip_connections[3]], dim=1)
        self.layer4_convblock = ConvBlock(x, num_filters[3])
        x = self.layer4_convblock.forward(x)

        # Undo the reversal so that forward passes don't get screwed
        skip_connections.reverse()
    
    def forward(self, x, skip_connections):
        num_filters = [256, 128, 64, 32]
        skip_connections.reverse()

        x = self.layer1_upsampling(x)
        x = torch.cat([x, skip_connections[0]], dim=1)
        x = self.layer1_convblock.forward(x)

        x = self.layer2_upsampling(x)
        x = torch.cat([x, skip_connections[1]], dim=1)
        x = self.layer2_convblock.forward(x)

        x = self.layer3_upsampling(x)
        x = torch.cat([x, skip_connections[2]], dim=1)
        x = self.layer3_convblock.forward(x)

        x = self.layer4_upsampling(x)
        x = torch.cat([x, skip_connections[3]], dim=1)
        x = self.layer4_convblock.forward(x)

        del skip_connections

        return x

class Encoder2(nn.Module):
    def __init__(self, inputs):
        super(Encoder2, self).__init__()
        num_filters = [32, 64, 128, 256]
        x = inputs

        self.layer1_convblock = ConvBlock(x, num_filters[0])
        x = self.layer1_convblock.forward(x)
        self.layer1_maxpool2d = nn.MaxPool2d(kernel_size = (2,2))
        x = self.layer1_maxpool2d(x)

        self.layer2_convblock = ConvBlock(x, num_filters[1])
        x = self.layer2_convblock.forward(x)
        self.layer2_maxpool2d = nn.MaxPool2d(kernel_size = (2,2))
        x = self.layer2_maxpool2d(x)

        self.layer3_convblock = ConvBlock(x, num_filters[2])
        x = self.layer3_convblock.forward(x)
        self.layer3_maxpool2d = nn.MaxPool2d(kernel_size = (2,2))
        x = self.layer3_maxpool2d(x)

        self.layer4_convblock = ConvBlock(x, num_filters[3])
        x = self.layer4_convblock.forward(x)
        self.layer4_maxpool2d = nn.MaxPool2d(kernel_size = (2,2))
        x = self.layer4_maxpool2d(x)
    
    def forward(self, x):
        num_filters = [32, 64, 128, 256]
        skip_connections = []

        x = self.layer1_convblock.forward(x)
        skip_connections.append(x)
        x = self.layer1_maxpool2d(x)

        x = self.layer2_convblock.forward(x)
        skip_connections.append(x)
        x = self.layer2_maxpool2d(x)

        x = self.layer3_convblock.forward(x)
        skip_connections.append(x)
        x = self.layer3_maxpool2d(x)

        x = self.layer4_convblock.forward(x)
        skip_connections.append(x)
        x = self.layer4_maxpool2d(x)

        return x, skip_connections

class Decoder2(nn.Module):
      def __init__(self, inputs, skip_1, skip_2):
          super(Decoder2, self).__init__()
          num_filters = [256, 128, 64, 32]

          skip_2.reverse()
          x = inputs

          self.layer1_upsampling = nn.UpsamplingBilinear2d(size=(2*x.shape[2], 2*x.shape[3]))
          x = self.layer1_upsampling(x)
          x = torch.cat([x, skip_1[0], skip_2[0]], dim=1)
          self.layer1_convblock = ConvBlock(x, num_filters[0])
          x = self.layer1_convblock.forward(x)

          self.layer2_upsampling = nn.UpsamplingBilinear2d(size=(2*x.shape[2], 2*x.shape[3]))
          x = self.layer2_upsampling(x)
          x = torch.cat([x, skip_1[1], skip_2[1]], dim=1)
          self.layer2_convblock = ConvBlock(x, num_filters[1])
          x = self.layer2_convblock.forward(x)

          self.layer3_upsampling = nn.UpsamplingBilinear2d(size=(2*x.shape[2], 2*x.shape[3]))
          x = self.layer3_upsampling(x)
          x = torch.cat([x, skip_1[2], skip_2[2]], dim=1)
          self.layer3_convblock = ConvBlock(x, num_filters[2])
          x = self.layer3_convblock.forward(x)

          self.layer4_upsampling = nn.UpsamplingBilinear2d(size=(2*x.shape[2], 2*x.shape[3]))
          x = self.layer4_upsampling(x)
          x = torch.cat([x, skip_1[3], skip_2[3]], dim=1)
          self.layer4_convblock = ConvBlock(x, num_filters[3])
          x = self.layer4_convblock.forward(x)

          skip_2.reverse() # Undo the reverse so we don't screw up forward()
      
      def forward(self, x, skip_1, skip_2):
          num_filters = [256, 128, 64, 32]

          skip_2.reverse()

          x = self.layer1_upsampling(x)
          x = torch.cat([x, skip_1[0], skip_2[0]], dim=1)
          x = self.layer1_convblock.forward(x)

          x = self.layer2_upsampling(x)
          x = torch.cat([x, skip_1[1], skip_2[1]], dim=1)
          x = self.layer2_convblock.forward(x)

          x = self.layer3_upsampling(x)
          x = torch.cat([x, skip_1[2], skip_2[2]], dim=1)
          x = self.layer3_convblock.forward(x)

          x = self.layer4_upsampling(x)
          x = torch.cat([x, skip_1[3], skip_2[3]], dim=1)
          x = self.layer4_convblock.forward(x)

          del skip_1, skip_2

          return x

class OutputBlock(nn.Module):
      def __init__(self, inputs):
          super(OutputBlock, self).__init__()
          self.conv2d = nn.Conv2d(in_channels = inputs.shape[1], out_channels = 1, kernel_size = 1, padding = "same")
          self.sigmoid = nn.Sigmoid()
      
      def forward(self, x):
          x = self.conv2d(x)
          x = self.sigmoid(x)
          return x

class DoubleUNet(nn.Module):
  def __init__(self, inputs):
      with torch.no_grad():
        super(DoubleUNet, self).__init__()

        # Encoder 1
        self.encoder1 = Encoder1()
        encoder1_op, encoder1_skip_conns = self.encoder1.forward(inputs)

        # ASPP
        self.aspp1 = ASPP(encoder1_op, 64)
        aspp_op = self.aspp1.forward(encoder1_op, 64)

        # Decoder 1
        self.decoder1 = Decoder1(aspp_op, encoder1_skip_conns)
        decoder1_op = self.decoder1.forward(aspp_op, encoder1_skip_conns)

        # Output 1
        self.outputblock1 = OutputBlock(decoder1_op)
        mask = self.outputblock1.forward(decoder1_op)
        network1_op = inputs * mask

        # Encoder 2
        self.encoder2 = Encoder2(network1_op)
        encoder2_op,encoder2_skip_conns = self.encoder2.forward(network1_op)

        # ASPP 2
        self.aspp2 = ASPP(encoder2_op, 64)
        aspp2_op = self.aspp2.forward(encoder2_op, 64)

        # Decoder 2
        self.decoder2 = Decoder2(aspp2_op, encoder1_skip_conns, encoder2_skip_conns)
        decoder2_op = self.decoder2.forward(aspp2_op, encoder1_skip_conns, encoder2_skip_conns)

        # Output 2
        self.outputblock2 = OutputBlock(decoder2_op)

  
  def forward(self, inputs):
      # Encoder 1
      encoder1_op, encoder1_skip_conns = self.encoder1.forward(inputs)
      encoder1_op = encoder1_op.to(device)
      encoder1_skip_conns[0] = encoder1_skip_conns[0].to(device)
      encoder1_skip_conns[1] = encoder1_skip_conns[1].to(device)
      encoder1_skip_conns[2] = encoder1_skip_conns[2].to(device)
      encoder1_skip_conns[3] = encoder1_skip_conns[3].to(device)

      # ASPP
      aspp_op = self.aspp1.forward(encoder1_op, 64)
      del encoder1_op

      # Decoder 1
      decoder1_op = self.decoder1.forward(aspp_op, encoder1_skip_conns)
      del aspp_op

      # Output 1
      mask = self.outputblock1.forward(decoder1_op)
      network1_op = inputs * mask

      # Encoder 2
      encoder2_op,encoder2_skip_conns = self.encoder2.forward(network1_op)
      del network1_op

      # ASPP 2
      aspp2_op = self.aspp2.forward(encoder2_op, 64)
      del encoder2_op

      # Decoder 2
      decoder2_op = self.decoder2.forward(aspp2_op, encoder1_skip_conns, encoder2_skip_conns)
      del aspp2_op, encoder1_skip_conns, encoder2_skip_conns

      # Output 2
      network2_op = self.outputblock2.forward(decoder2_op)

      final_output = torch.cat([mask, network2_op], dim = 1)
      del mask, network2_op
      return final_output