import torch
import torch.nn as nn

from base import GazeModel
from neuron import AdaptiveSpikingNeuron, ParametricSpikingNeuron

class GazeSCRNN(GazeModel):
    """
    GazeSCRNN is a spiking convolutional recurrent neural network model for gaze tracking.
    This version is supposed to be used with the full sized input frames.

    Args:
        input_size (tuple): The size of the input tensor (channels, height, width). Default is (2, 260, 346).
        hidden_size (int): The number of features in the hidden state of the recurrent layers. Default is 256.
        *args: Additional arguments for the GazeModel base class.
        **kwargs: Additional keyword arguments for the GazeModel base class.
    """
    def __init__(
        self,
        input_size=(2, 260, 346),
        hidden_size=256,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.input_size = input_size
        self.hidden_size = hidden_size

        size = input_size
        self.conv0 = nn.Conv2d(2, 32, kernel_size=3, stride=2, padding=1)
        size = self.conv0.out_channels, (size[1] + 2*self.conv0.padding[0] - self.conv0.kernel_size[0]) // self.conv0.stride[0] + 1, (size[2] + 2*self.conv0.padding[1] - self.conv0.kernel_size[1]) // self.conv0.stride[1] + 1
        self.bn0 = nn.BatchNorm2d(32)
        self.pool0 = nn.MaxPool2d(2, stride=2)
        size = size[0], size[1]//2, size[2]//2
        self.lif0 = AdaptiveSpikingNeuron(channels=32, kernel_size=3)
        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1)
        size = self.conv1.out_channels, (size[1] + 2*self.conv1.padding[0] - self.conv1.kernel_size[0]) // self.conv1.stride[0] + 1, (size[2] + 2*self.conv1.padding[1] - self.conv1.kernel_size[1]) // self.conv1.stride[1] + 1
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        size = size[0], size[1]//2, size[2]//2
        self.lif1 = AdaptiveSpikingNeuron(channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        size = self.conv2.out_channels, (size[1] + 2*self.conv2.padding[0] - self.conv2.kernel_size[0]) // self.conv2.stride[0] + 1, (size[2] + 2*self.conv2.padding[1] - self.conv2.kernel_size[1]) // self.conv2.stride[1] + 1
        self.bn2 = nn.BatchNorm2d(64)
        self.lif2 = AdaptiveSpikingNeuron(channels=64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1)
        size = self.conv3.out_channels, (size[1] + 2*self.conv3.padding[0] - self.conv3.kernel_size[0]) // self.conv3.stride[0] + 1, (size[2] + 2*self.conv3.padding[1] - self.conv3.kernel_size[1]) // self.conv3.stride[1] + 1
        self.bn3 = nn.BatchNorm2d(64)
        self.lif3 = AdaptiveSpikingNeuron(channels=64, kernel_size=3)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        size = self.conv4.out_channels, (size[1] + 2*self.conv4.padding[0] - self.conv4.kernel_size[0]) // self.conv4.stride[0] + 1, (size[2] + 2*self.conv4.padding[1] - self.conv4.kernel_size[1]) // self.conv4.stride[1] + 1
        self.bn4 = nn.BatchNorm2d(128)
        self.lif4 = AdaptiveSpikingNeuron(channels=128, kernel_size=3)
        self.dropout = nn.Dropout(0.1)
        self.flatten = nn.Flatten()
        self.recurrent1 = nn.Linear((size[0]*size[1]*size[2])+self.hidden_size, self.hidden_size)
        self.bn5 = nn.BatchNorm1d(self.hidden_size)
        self.lif5 = AdaptiveSpikingNeuron(linear_features=self.hidden_size, concatenate=True)
        self.recurrent2 = nn.Linear(2*self.hidden_size, self.hidden_size)
        self.bn6 = nn.BatchNorm1d(self.hidden_size)
        self.lif6 = AdaptiveSpikingNeuron(linear_features=self.hidden_size, concatenate=True)
        self.recurrent3 = nn.Linear(2*self.hidden_size, self.hidden_size)
        self.bn7 = nn.BatchNorm1d(self.hidden_size)
        self.lif7 = AdaptiveSpikingNeuron(linear_features=self.hidden_size, concatenate=True)
        self.recurrent4 = nn.Linear(2*self.hidden_size, self.hidden_size)
        self.bn8 = nn.BatchNorm1d(self.hidden_size)
        self.lif8 = AdaptiveSpikingNeuron(linear_features=self.hidden_size, concatenate=True)
        self.fc1 = nn.Linear(self.hidden_size, self.output_size)
        self.lif9 = ParametricSpikingNeuron(disable_reset=True)

    def forward_timestep(self, x_t):
        x_t = self.conv0(x_t)
        x_t = self.bn0(x_t)
        x_t = self.pool0(x_t)
        x_t = self.lif0(x_t)
        x_t = self.conv1(x_t)
        x_t = self.bn1(x_t)
        x_t = self.pool1(x_t)
        x_t = self.lif1(x_t)
        x_t = self.conv2(x_t)
        x_t = self.bn2(x_t)
        x_t = self.lif2(x_t)
        x_t = self.conv3(x_t)
        x_t = self.bn3(x_t)
        x_t = self.lif3(x_t)
        x_t = self.conv4(x_t)
        x_t = self.bn4(x_t)
        x_t = self.lif4(x_t)
        x_t = self.dropout(x_t)
        x_t = self.flatten(x_t)
        x_t = self.recurrent1(torch.cat((x_t, self.lif5.spk if self.lif5.spk.shape == (*x_t.shape[:-1], self.lif5.linear_features) else torch.zeros((*x_t.shape[:-1], self.lif5.linear_features), device=x_t.device)), dim=-1))
        x_t = self.bn5(x_t)
        x_t = self.lif5(x_t)
        x_t = self.recurrent2(torch.cat((x_t, self.lif6.spk if self.lif6.spk.shape == (*x_t.shape[:-1], self.lif6.linear_features) else torch.zeros((*x_t.shape[:-1], self.lif6.linear_features), device=x_t.device)), dim=-1))
        x_t = self.bn6(x_t)
        x_t = self.lif6(x_t)
        x_t = self.recurrent3(torch.cat((x_t, self.lif7.spk if self.lif7.spk.shape == (*x_t.shape[:-1], self.lif7.linear_features) else torch.zeros((*x_t.shape[:-1], self.lif7.linear_features), device=x_t.device)), dim=-1))
        x_t = self.bn7(x_t)
        x_t = self.lif7(x_t)
        x_t = self.recurrent4(torch.cat((x_t, self.lif8.spk if self.lif8.spk.shape == (*x_t.shape[:-1], self.lif8.linear_features) else torch.zeros((*x_t.shape[:-1], self.lif8.linear_features), device=x_t.device)), dim=-1))
        x_t = self.bn8(x_t)
        x_t = self.lif8(x_t)
        x_t = self.fc1(x_t)
        x_t = self.lif9(x_t)
        return x_t