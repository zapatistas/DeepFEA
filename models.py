import torch
from torch import nn
from torch.nn.init import kaiming_normal_, constant_
import random

class ConvLSTMCell(nn.Module):

    def __init__(self, in_channels, out_channels, 
    kernel_size, padding, activation, frame_size):

        super(ConvLSTMCell, self).__init__()  

        if activation == "tanh":
            self.activation = torch.tanh
            non_linear_a = 0
            non_linearity = 'selu'
        elif activation == "relu":
            self.activation = torch.relu
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "selu":
            self.activation = nn.SELU()
        elif activation == "prelu":
            self.activation = nn.PReLU()
        elif activation == "elu":
            self.activation = nn.ELU()
        else:
            self.activation = activation
        self.out = nn.Sigmoid()
        self.out_channels = out_channels

        self.conv = nn.Conv2d(
            in_channels=in_channels + out_channels, 
            out_channels=4 * out_channels, 
            kernel_size=kernel_size,padding=padding)

        # Initialize weights for Hadamard Products
        if type(activation) == nn.PReLU or type(activation) == nn.LeakyReLU:
            non_linear_a = 0.2
            non_linearity = 'leaky_relu'
        elif type(activation) == nn.Tanh:
            non_linear_a = 0
            non_linearity = 'selu'
        elif type(activation) == nn.ReLU: 
            non_linear_a = 0
            non_linearity = 'relu'
        elif type(activation) == nn.SELU:
            non_linear_a = 0
            non_linearity = 'selu'

        self.W_ci = kaiming_normal_(nn.Parameter(torch.zeros(out_channels, *frame_size)),a=non_linear_a,nonlinearity=non_linearity)
        self.W_co = kaiming_normal_(nn.Parameter(torch.zeros(out_channels, *frame_size)),a=non_linear_a,nonlinearity=non_linearity)
        self.W_cf = kaiming_normal_(nn.Parameter(torch.zeros(out_channels, *frame_size)),a=non_linear_a,nonlinearity=non_linearity)
        self.bias_ci = constant_(nn.Parameter(torch.zeros(out_channels,1,1)), 0)
        self.bias_co = constant_(nn.Parameter(torch.zeros(out_channels,1,1)), 0)
        self.bias_cf = constant_(nn.Parameter(torch.zeros(out_channels,1,1)), 0)
        
    def forward(self, X, h_prev, c_prev):

        # Idea adapted from https://github.com/ndrplz/ConvLSTM_pytorch
        conv_output = self.conv(torch.cat([X, h_prev], dim=1))
        i_conv, f_conv, c_conv, o_conv = torch.chunk(conv_output, chunks=4, dim=1)
    
        input_gate = self.out(i_conv + self.W_ci * c_prev)

        forget_gate = self.out(f_conv +  self.W_cf * c_prev)
        # Current Cell output
        c = forget_gate*c_prev + input_gate * self.activation(c_conv)

        output_gate = self.out(o_conv + self.W_co*c)
        # Current Hidden State
        h = output_gate * self.activation(c)
        
        if torch.isnan(h).sum() > 0:
            print("nan")
        return h, c

class ConvLSTMCell_3D(nn.Module):

    def __init__(self, in_channels, out_channels, 
    kernel_size, padding, activation, frame_size):

        super(ConvLSTMCell_3D, self).__init__()  

        if activation == "tanh":
            self.activation = torch.tanh 
            non_linear_a = 0
            non_linearity = 'selu'
        elif activation == "relu":
            self.activation = torch.relu
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "selu":
            self.activation = nn.SELU()
        elif activation == "prelu":
            self.activation = nn.PReLU()
        elif activation == "elu":
            self.activation = nn.ELU()
        else:
            self.activation = activation
        self.out = torch.sigmoid
        self.out_channels = out_channels

        self.conv = nn.Conv3d(
            in_channels=in_channels + out_channels, 
            out_channels=4 * out_channels, 
            kernel_size=kernel_size,padding=padding)

        # Initialize weights for Hadamard Products
        if type(activation) == nn.PReLU or type(activation) == nn.LeakyReLU:
            non_linear_a = 0.2
            non_linearity = 'leaky_relu'
        elif type(activation) == nn.Tanh:
            non_linear_a = 0
            non_linearity = 'selu'
        elif type(activation) == nn.ReLU: 
            non_linear_a = 0
            non_linearity = 'relu'
        elif type(activation) == nn.SELU:
            non_linear_a = 0
            non_linearity = 'selu'

        self.W_ci = kaiming_normal_(nn.Parameter(torch.zeros(out_channels, *frame_size)),a=non_linear_a,nonlinearity=non_linearity)
        self.W_co = kaiming_normal_(nn.Parameter(torch.zeros(out_channels, *frame_size)),a=non_linear_a,nonlinearity=non_linearity)
        self.W_cf = kaiming_normal_(nn.Parameter(torch.zeros(out_channels, *frame_size)),a=non_linear_a,nonlinearity=non_linearity)
        
    def forward(self, X, h_prev, c_prev):

        # Idea adapted from https://github.com/ndrplz/ConvLSTM_pytorch
        conv_output = self.conv(torch.cat([X, h_prev], dim=1))
        # Idea adapted from https://github.com/ndrplz/ConvLSTM_pytorch
        i_conv, f_conv, c_conv, o_conv = torch.chunk(conv_output, chunks=4, dim=1)
    
        input_gate = self.out(i_conv + self.W_ci * c_prev)

        forget_gate = self.out(f_conv +  self.W_cf * c_prev)
        # Current Cell output
        c = forget_gate*c_prev + input_gate * self.activation(c_conv)

        output_gate = self.out(o_conv + self.W_co*c)

        # Current Hidden State
        h = output_gate * self.activation(c)
        del output_gate
        return h, c

class ConvLSTM_3D(nn.Module):

    def __init__(self, in_channels,out_channels,layer_channels, kernel_size,padding,activation,frame_size,
    out_act='tanh'):
        super().__init__()
    
        self.num_layers = len(layer_channels)
        self.layers_channels = layer_channels
        self.out_channels = out_channels
        self.conv3d = nn.Conv3d(layer_channels[self.num_layers-1],out_channels,1)
    
        self.conv3d_2 = nn.Conv3d(layer_channels[self.num_layers-1],2,2)
        self.padding = padding
        
        self.frame_size = frame_size
        self.out_act = out_act
        self.cell_list = nn.ModuleList()
        if kernel_size[0] == 5:
            padding = 2
        elif kernel_size[0] == 9:
            padding = 4
        self.cell_list.append(ConvLSTMCell_3D(in_channels, layer_channels[0], 
        kernel_size[0], padding,activation, frame_size))
        for i in range(1,self.num_layers):
            if kernel_size[i] == 5:
                padding = 2
            elif kernel_size[i] == 9:
                padding = 4
            else:
                padding = self.padding
            self.cell_list.append(ConvLSTMCell_3D(layer_channels[i-1], layer_channels[i], 
            kernel_size[i], padding, activation, frame_size))
        
    def forward(self, X, device, h_o =None, c_o=None, prob = 1):

        # X is a frame sequence (batch_size, num_channels, seq_len, height, width)

        # Get the dimensions
        batch_size, seq_len, channels, height, width, depth = X.size()

        # Initialize output
        output = torch.zeros(batch_size, seq_len, self.out_channels,#layers_channels, #out_channels,#
        height, width,depth, device=device)
        
        h_list = []
        c_list = []

        for i in range(self.num_layers):

            # Initialize Hidden State
            h_list.append(torch.zeros(batch_size, self.layers_channels[i], 
            height, width, depth, device=device))

            # Initialize Cell Input
            c_list.append(torch.zeros(batch_size,self.layers_channels[i], #self.frame_size[0], self.frame_size[1],
            height, width, depth, device=device))
        else:
            H = h_o
            C = c_o

        if X.dtype == torch.float64:
            H = H.double()
            C = C.double()
            output = output.double()
        
        eff = torch.zeros(batch_size, seq_len, 2,
        height-1, width-1,depth-1, device=device)
        bounded_nodes = torch.where(X[:,:,channels-1,...] == -1)

        # Unroll over time steps
        for time_step in range(seq_len):
            
            p = random.random()
            

            if time_step > 0  and p > prob:
                
                # prob = 1 always use gt, prob=0 always use previous H
                # we use clone to avoid back propagation error
                X[bounded_nodes[0],time_step,0:self.out_channels,bounded_nodes[2],bounded_nodes[3],bounded_nodes[4]] = TEMP_H[bounded_nodes[0],:,bounded_nodes[2],bounded_nodes[3],bounded_nodes[4]].clone()
                        
            h_list[0],c_list[0] = self.cell_list[0](X[:,time_step,:], h_list[0],c_list[0])
            for i in range(1,self.num_layers):
                h_list[i],c_list[i] = self.cell_list[i](h_list[i-1], h_list[i],c_list[i])
            
            
            TEMP_H =  torch.tanh(self.conv3d(h_list[self.num_layers-1])) 
            eff[:,time_step,:]  = torch.tanh(self.conv3d_2(h_list[self.num_layers-1]))
            output[:,time_step,:] = TEMP_H
                
            if torch.isnan(TEMP_H).sum()>0:
                print("TEMP H is nan")
        del TEMP_H
       
        return output,eff,h_list,c_list

class ConvLSTM(nn.Module):

    def __init__(self, in_channels,out_channels, layer_channels, kernel_size,padding,activation,frame_size,
    out_act='tanh'):
        super().__init__()

        self.num_layers = len(layer_channels)
        self.padding = padding
        self.layers_channels = layer_channels
        self.out_channels = out_channels
        
        self.conv2d = nn.Conv2d(layer_channels[self.num_layers-1],out_channels,1)
        self.conv2d_2 = nn.Conv2d(layer_channels[self.num_layers-1] ,out_channels,2)

    
        self.frame_size = frame_size
        self.out_act = out_act
       
        self.cell_list = nn.ModuleList()
        if kernel_size[0] == 5:
            padding = 2
        elif kernel_size[0] == 9:
            padding = 4
        self.cell_list.append(ConvLSTMCell(in_channels, layer_channels[0], 
        kernel_size[0], padding,activation, frame_size))
        for i in range(1,self.num_layers):
            if kernel_size[i] == 5:
                padding = 2
            elif kernel_size[i] == 9:
                padding = 4
            else:
                padding = self.padding
            self.cell_list.append(ConvLSTMCell(layer_channels[i-1], layer_channels[i], 
            kernel_size[i], padding, activation, frame_size))
        
    def forward(self, X, device, h_o =None, c_o=None, prob = 1):

        # X is a frame sequence (batch_size, num_channels, seq_len, height, width)

        # Get the dimensions
        batch_size, seq_len, channels, height, width = X.size()

        # Initialize output
        output = torch.zeros(batch_size, seq_len, self.out_channels,
        height, width, device=device)
        h_list = []
        c_list = []

        for i in range(self.num_layers):

            # Initialize Hidden State
            h_list.append(torch.zeros(batch_size, self.layers_channels[i], 
            height, width, device=device))

            # Initialize Cell Input
            c_list.append(torch.zeros(batch_size,self.layers_channels[i],
            height, width, device=device))
        else:
            H = h_o
            C = c_o

        eff = torch.zeros(batch_size, seq_len, 2,
        height-1, width-1, device=device)
        bounded_nodes = torch.where(X[:,:,channels-1,...] == -1)
        # bounded_nodes = torch.where(torch.diff(X[:,:,0:2,:,:],dim=(1)).sum(axis=(1,2)) == 0)
        # Unroll over time steps
        for time_step in range(seq_len):
            
            p = random.random()
            if time_step > 0  and p > prob:
                
                # prob = 1 always use gt, prob=0 always use previous H
                # we use clone to avoid back propagation error
                # We only replace not bounded nodes
                X[bounded_nodes[0],time_step,0:self.out_channels,bounded_nodes[2],bounded_nodes[3]] = TEMP_H[bounded_nodes[0],:,bounded_nodes[2],bounded_nodes[3]].clone()

            h_list[0],c_list[0] = self.cell_list[0](X[:,time_step,:], h_list[0],c_list[0])
            for i in range(1,self.num_layers):
                h_list[i],c_list[i] = self.cell_list[i](h_list[i-1], h_list[i],c_list[i])
            
            TEMP_H =  torch.tanh(self.conv2d(h_list[self.num_layers-1])) 
            eff[:,time_step,:]  = torch.tanh(self.conv2d_2(h_list[self.num_layers-1]))
            output[:,time_step,:] = TEMP_H.clone() 
            
            if torch.isnan(TEMP_H).sum()>0:
                print("TEMP H is nan")
        
        del TEMP_H
        
        return output,eff,h_list,c_list


class NEP_Net_2D(nn.Module):
    def __init__(self,config):
        super().__init__()
        (dis_in,dis_out,conv_layers,num_channels,kernels,padding,frame, act_out,lstm_act) = config

        self.out_channels = num_channels
        self.out_features = dis_out
        self.kernel_size = kernels
        self.num_cnn_layers = conv_layers
       
        self.height = frame[0]
        self.width = frame[1]
        self.convlstm = ConvLSTM(dis_in,dis_out,num_channels,kernels,
                    padding,lstm_act,frame,out_act=act_out)
    def forward(self,x,device,prob = 0):
        
        y_t,eff,_,_ = self.convlstm(x,device,prob=prob)

        return y_t,eff


class NEP_Net_3D(nn.Module):
    def __init__(self,config,act_out=nn.Tanh(),lstm_act=nn.Tanh()):
        super().__init__()
        (dis_in,dis_out,num_channels,kernels,padding,frame) = config

        self.out_channels = num_channels
        self.out_features = dis_out
        self.kernel_size = kernels
        
        self.height = frame[0]
        self.width = frame[1]
        self.depth = frame[2]
   
      
        self.convlstm = ConvLSTM_3D(dis_in,dis_out,num_channels,kernels,
                    padding,lstm_act,frame,out_act=act_out)
      
    def forward(self,x,device,prob = 0):
        
        y_t,eff,_,_ = self.convlstm(x,device,prob=prob) 
            
        return y_t,eff
