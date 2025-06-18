import torch
import torch.nn as nn
from torchvision.ops.misc import Conv2dNormActivation

#from models.helpers.utils import make_divisible

from enum import Enum
import torch.nn.functional as F
    
class ChannelSELayer(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*

    """

    def __init__(self, num_channels, reduction_ratio=2):
        """

        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        """

        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output tensor
        """
        batch_size, num_channels, H, W = input_tensor.size()
        # Average along each channel
        squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)

        # channel excitation
        fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        a, b = squeeze_tensor.size()
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor


class SpatialSELayer(nn.Module):
    """
    Re-implementation of SE block -- squeezing spatially and exciting channel-wise described in:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018*
    """

    def __init__(self, num_channels):
        """

        :param num_channels: No of input channels
        """
        super(SpatialSELayer, self).__init__()
        self.conv = nn.Conv2d(num_channels, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor, weights=None):
        """

        :param weights: weights for few shot learning
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output_tensor
        """
        # spatial squeeze
        batch_size, channel, a, b = input_tensor.size()

        if weights is not None:
            weights = torch.mean(weights, dim=0)
            weights = weights.view(1, channel, 1, 1)
            out = F.conv2d(input_tensor, weights)
        else:
            out = self.conv(input_tensor)
        squeeze_tensor = self.sigmoid(out)

        # spatial excitation
        # print(input_tensor.size(), squeeze_tensor.size())
        squeeze_tensor = squeeze_tensor.view(batch_size, 1, a, b)
        output_tensor = torch.mul(input_tensor, squeeze_tensor)
        #output_tensor = torch.mul(input_tensor, squeeze_tensor)
        return output_tensor


class ChannelSpatialSELayer(nn.Module):
    """
    Re-implementation of concurrent spatial and channel squeeze & excitation:
        *Roy et al., Concurrent Spatial and Channel Squeeze & Excitation in Fully Convolutional Networks, MICCAI 2018, arXiv:1803.02579*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """

        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        """
        super(ChannelSpatialSELayer, self).__init__()
        self.cSE = ChannelSELayer(num_channels, reduction_ratio)
        self.sSE = SpatialSELayer(num_channels)

    def forward(self, input_tensor):
        """

        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output_tensor
        """
        output_tensor = torch.max(self.cSE(input_tensor), self.sSE(input_tensor))
        return output_tensor


class SELayer(Enum):
    """
    Enum restricting the type of SE Blockes available. So that type checking can be adding when adding these blockes to
    a neural network::

        if self.se_block_type == se.SELayer.CSE.value:
            self.SELayer = se.ChannelSpatialSELayer(params['num_filters'])

        elif self.se_block_type == se.SELayer.SSE.value:
            self.SELayer = se.SpatialSELayer(params['num_filters'])

        elif self.se_block_type == se.SELayer.CSSE.value:
            self.SELayer = se.ChannelSpatialSELayer(params['num_filters'])
    """
    NONE = 'NONE'
    CSE = 'CSE'
    SSE = 'SSE'
    CSSE = 'CSSE'  

def initialize_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, mode="fan_in") # frrom fan_out
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        #nn.init.normal_(m.weight, 0, 0.01)
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class ChannelShuffle(nn.Module):
    def __init__(self, groups = 1):
        super(ChannelShuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        batch_size, num_channels, height, width = x.size()
        assert num_channels % self.groups == 0, "The number of channels must be divisible by the number of groups"
        
        #if torch.rand(1).item() < self.p:
        channels_per_group = num_channels // self.groups

        # Reshape the tensor to (batch_size, groups, channels_per_group, height, width)
        x = x.view(batch_size, self.groups, channels_per_group, height, width)

        # Permute the tensor to (batch_size, channels_per_group, groups, height, width)
        x = x.permute(0, 2, 1, 3, 4).contiguous()      

        # Reshape the tensor back to (batch_size, num_channels, height, width)
        x = x.view(batch_size, num_channels, height, width)

        return x
 

                 
class Network_test(nn.Module):
    def __init__(self, config):
        super(Network_test, self).__init__()
        #n_classes = config['n_classes']
        in_channels = config['in_channels']
        base_channels = config['base_channels']
        channels_multiplier = config['channels_multiplier']
        expansion_rate = config['expansion_rate']
        divisor = config['divisor']

        self.conv2d_1_0 = Conv2dNormActivation(in_channels = in_channels,
                                             out_channels = 7,
                                             norm_layer = nn.BatchNorm2d,
                                             activation_layer = None, #torch.nn.ReLU,
                                             kernel_size = 3,
                                             stride = 1,
                                             padding= 1,
                                             inplace=False)
        
        self.conv2d_1_1 = Conv2dNormActivation(in_channels = 7,
                                             out_channels = int(base_channels/2),
                                             norm_layer = nn.BatchNorm2d,
                                             activation_layer = None, #torch.nn.ReLU,
                                             kernel_size = 1,
                                             stride = 1,
                                             padding= 0,
                                             inplace=False)
        
        self.cs_1 = ChannelShuffle(int(base_channels/4)) #int(10/2))
        
        self.codw2d_1L1 = Conv2dNormActivation(in_channels = int(base_channels/2),
                                               out_channels = base_channels,
                                               norm_layer = nn.BatchNorm2d,
                                               activation_layer = None, #torch.nn.ReLU,
                                               groups = int(base_channels/2),
                                               kernel_size = (3,1),
                                               stride = (1,1),
                                               #dilation = (2,2),
                                               padding= (1,0),
                                               inplace = False)  
        
        self.codw2d_1R1 = Conv2dNormActivation(in_channels = base_channels,
                                               out_channels = base_channels,
                                               norm_layer = nn.BatchNorm2d,
                                               activation_layer = None, #torch.nn.ReLU,
                                               groups = int(base_channels/2),
                                               kernel_size = (1,3),
                                               stride = (1,1),
                                               #dilation = (2,2),
                                               padding= (0,1),
                                               inplace = False)          
                
        self.se_1_1     = ChannelSELayer(base_channels)

        self.conv2d_2_1 = Conv2dNormActivation(in_channels = base_channels,
                                               out_channels = int(base_channels * channels_multiplier),
                                               norm_layer = nn.BatchNorm2d, #None,
                                               activation_layer = None,                                                           
                                               kernel_size = (1,1),
                                               stride = (1,1),
                                               padding = 0,
                                               inplace = False)
        self.codw2d_2_1L = Conv2dNormActivation(in_channels = int(base_channels * channels_multiplier),
                                               out_channels = int(base_channels * channels_multiplier * expansion_rate),
                                               norm_layer = nn.BatchNorm2d,
                                               activation_layer = None, #torch.nn.ReLU,
                                               groups = int(base_channels * channels_multiplier),
                                               kernel_size = (5,1),
                                               stride = (1,1),
                                               #dilation = (2,2),
                                               padding= (2,0),
                                               inplace = False)
        self.codw2d_2_1R = Conv2dNormActivation(in_channels = int(base_channels * channels_multiplier * expansion_rate),
                                               out_channels = int(base_channels * channels_multiplier * expansion_rate),
                                               norm_layer = nn.BatchNorm2d,
                                               activation_layer = None, #torch.nn.ReLU,
                                               groups = int(base_channels * channels_multiplier * expansion_rate),
                                               kernel_size = (1,5),
                                               stride = (2,2),
                                               #dilation = (2,2),
                                               padding= (0,2),
                                               inplace = False)        
        
        self.conv2d_2_2 = Conv2dNormActivation(in_channels = base_channels,
                                               out_channels = int(base_channels * channels_multiplier),
                                               norm_layer = nn.BatchNorm2d, #None,
                                               activation_layer = None,                                                           
                                               kernel_size = (1,1),
                                               stride = (1,1),
                                               padding = 0,
                                               inplace = False)
        self.codw2d_2_2L = Conv2dNormActivation(in_channels = int(base_channels * channels_multiplier),
                                               out_channels = int(base_channels * channels_multiplier * expansion_rate),
                                               norm_layer = nn.BatchNorm2d,
                                               activation_layer = None, #torch.nn.ReLU,
                                               groups = int(base_channels * channels_multiplier),
                                               kernel_size = (5,1),
                                               stride = (1,1),
                                               #dilation = (2,2),
                                               padding= (2,0),
                                               inplace = False)       
        self.codw2d_2_2R = Conv2dNormActivation(in_channels = int(base_channels * channels_multiplier * expansion_rate),
                                               out_channels = int(base_channels * channels_multiplier * expansion_rate),
                                               norm_layer = nn.BatchNorm2d,
                                               activation_layer = None, #torch.nn.ReLU,
                                               groups = int(base_channels * channels_multiplier * expansion_rate),
                                               kernel_size = (1,5),
                                               stride = (2,2),
                                               #dilation = (2,2),
                                               padding= (0,2),
                                               inplace = False)              

        self.se_2_1     = ChannelSELayer(int(base_channels * channels_multiplier * expansion_rate))
        self.se_2_2     = ChannelSELayer(int(base_channels * channels_multiplier * expansion_rate))

        self.conv2d_3_1 = Conv2dNormActivation(in_channels = int(base_channels * channels_multiplier * expansion_rate),
                                               out_channels = int(base_channels * channels_multiplier * expansion_rate),
                                               norm_layer = nn.BatchNorm2d,
                                               activation_layer = None, # adding relu degrades perf                              
                                               kernel_size = (1,1),
                                               stride = 1,
                                               padding = 0,
                                               inplace = False)
        self.bthnorm2d_3 = nn.BatchNorm2d(int(base_channels * channels_multiplier * expansion_rate))
        
        self.cs_3 = ChannelShuffle(int(base_channels * channels_multiplier * expansion_rate/2))

        self.codw2d_3_1 = Conv2dNormActivation(in_channels = int(base_channels * channels_multiplier * expansion_rate),
                                               out_channels = int(base_channels * channels_multiplier * expansion_rate * expansion_rate),
                                               norm_layer = nn.BatchNorm2d,
                                               activation_layer = None, #torch.nn.ReLU,   
                                               groups = int(base_channels * channels_multiplier * expansion_rate),
                                               kernel_size = (5,1), #change from 7 to 3 as kernel size is bigger than input size
                                               stride = (2,1),
                                               padding = (2,0),
                                               inplace = False)            
                
        self.conv2d_4_1 = Conv2dNormActivation(in_channels = int(base_channels * channels_multiplier * expansion_rate  * expansion_rate),
                                             out_channels = int(base_channels * channels_multiplier * expansion_rate * expansion_rate / divisor),
                                             norm_layer = nn.BatchNorm2d,
                                             activation_layer = None, #torch.nn.ReLU,
                                             kernel_size = (5,1),
                                             stride = (1,1),
                                             padding = (2,0),
                                             inplace = False)     
                
        self.bgru2d_4_1 = nn.GRU(16, 64, num_layers=1, batch_first = True, bidirectional=False)         
             
        self.drop2d_0_3 = nn.Dropout2d(p = 0.3)
        self.drop2d_0_5 = nn.Dropout2d(p = 0.5)
        self.drop1d_0_5 = nn.Dropout1d(p = 0.5)
        self.drop1d_0_3 = nn.Dropout1d(p = 0.3)   

        self.maxp2d_2_2 = nn.MaxPool2d(kernel_size = (2,2))
        self.avgp2d_2_2 = nn.AvgPool2d(kernel_size = (2,2))   
  

        self.param1 = nn.Parameter(torch.tensor(0.5))
        self.param2 = nn.Parameter(torch.tensor(0.5))
        self.param3 = nn.Parameter(torch.tensor(0.5))
        self.param4 = nn.Parameter(torch.tensor(0.5))

        self.line2d_5_1 = nn.Linear(int(base_channels * channels_multiplier * expansion_rate * expansion_rate / divisor),
                                out_features = 10)

        self.conv1d_5_2 = nn.Conv1d(1, #int(base_channels * channels_multiplier * expansion_rate * channels_multiplier * expansion_rate/ divisor),
                                             out_channels = 10,
                                             kernel_size = int(base_channels * channels_multiplier * expansion_rate * expansion_rate / divisor),
                                             stride = 1,
                                             padding = 0
                                             )     
        
        #self.apply(initialize_weights) # default pytorch initializatin seens to work better

    def forward(self, x):
        x = self.conv2d_1_0(x) 
        x = self.conv2d_1_1(x) 
        x = (self.codw2d_1L1(x)) 
        x = F.silu(self.codw2d_1R1(x)) 

        x = (self.se_1_1(x))
        
        xm = self.maxp2d_2_2(x)
        xa = self.avgp2d_2_2(x)

        xm = self.conv2d_2_1(xm)
        xm = (self.codw2d_2_1L(xm))
        xm = F.silu(self.codw2d_2_1R(xm))

        xa = self.conv2d_2_2(xa)
        xa = (self.codw2d_2_2L(xa))
        xa = F.silu(self.codw2d_2_2R(xa))

        xm = self.se_2_1(xm)
        xa = self.se_2_2(xa)            
                             
        xm1 = self.maxp2d_2_2(xm)
        xm2 = self.avgp2d_2_2(xm)
        xa1 = self.avgp2d_2_2(xa)
        xa2 = self.maxp2d_2_2(xa)

        xa = self.param1 * xa1 + (1 - self.param1) * xm2
        xm = self.param2 * xm1 + (1- self.param2) * xa2
        
        x1 = (self.param4 * xa + (1 - self.param4) * xm)
        x = self.conv2d_3_1(x1) + x1
        x = F.silu(self.codw2d_3_1(x))  
        x = self.drop2d_0_3(x) 
        
        x = F.silu(self.conv2d_4_1(x))
        x1  = torch.mean(x, dim = 3)

        x, _ = self.bgru2d_4_1(x1)
        x2 = (self.drop1d_0_5(x))

        x1 = torch.mean(x1, dim = 2)
        x2 = torch.mean(x2, dim = 2)
        
        x = (self.param3 * x1 + (1 - self.param3) * x2) # better than concat
        
        x = x.unsqueeze(1)
        logits = self.conv1d_5_2(x)
        logits = logits.view(logits.size(0), -1)     

        return logits          
                 
def get_ntu_model(n_classes=10, in_channels=1, n_blocks=(3, 2, 1), 
                    base_channels = 16, channels_multiplier = 1.5, expansion_rate = 2, divisor = 3, strides=None):
    
    model_config = {
        "n_classes": n_classes,
        "in_channels": in_channels,
        "base_channels": base_channels,
        "channels_multiplier": channels_multiplier,
        "expansion_rate": expansion_rate,
        "divisor": divisor,
        "n_blocks": n_blocks,
        "strides": strides
    }
    m = Network_test(model_config)
    return m