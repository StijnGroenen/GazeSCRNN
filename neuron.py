import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpikingActivation(torch.autograd.Function):
    """
    Custom autograd function for spiking activation.
    This function implements a spiking activation function with a Gaussian-based surrogate gradient for backpropagation.
    (Based on https://github.com/byin-cwi/sFPTT)
    """
    @staticmethod
    def forward(ctx, input, scale=6.0, height=0.15, lens=0.3, gamma=0.5):
        """
        Forward pass for the custom neuron function.

        Returns:
            torch.Tensor: Output tensor where each element is 1 if the corresponding input element is greater than 0, otherwise 0.
        """
        ctx.save_for_backward(input)
        ctx.scale = scale
        ctx.height = height
        ctx.lens = lens
        ctx.gamma = gamma
        return input.gt(0).float()
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Performs the backward pass for the custom spiking activation function using a surrogate gradient.

        Args:
            ctx: The context object that contains saved tensors and other information.
            grad_output: The gradient of the loss with respect to the output of the forward pass.

        Returns:
            A tuple containing the gradient of the loss with respect to the input of the forward pass,
            and None for other parameters (to match the expected return signature).
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = SpikingActivation.gaussian(input, mu=0., sigma=ctx.lens) * (1. + ctx.height) \
               - SpikingActivation.gaussian(input, mu=ctx.lens, sigma=ctx.scale * ctx.lens) * ctx.height \
               - SpikingActivation.gaussian(input, mu=-ctx.lens, sigma=ctx.scale * ctx.lens) * ctx.height
        return grad_input * temp.float() * ctx.gamma, None, None, None, None
    
    @staticmethod
    def gaussian(x, mu=0., sigma=.5):
        """
        Computes the Gaussian function.

        Args:
            x (torch.Tensor): The input tensor.
            mu (float, optional): The mean of the Gaussian distribution. Default is 0.
            sigma (float, optional): The standard deviation of the Gaussian distribution. Default is 0.5.

        Returns:
            torch.Tensor: The result of applying the Gaussian function to the input tensor.
        """
        return torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / torch.sqrt(2 * torch.tensor(np.pi)) / sigma


class AdaptiveSpikingNeuron(nn.Module):
    """
    AdaptiveSpikingNeuron is a PyTorch module that models an Adaptive Leaky Integrate and Fire neuron.

    Args:
        channels (int): Number of input channels for convolutional layers.
        linear_features (int): Number of input features for linear layers.
        kernel_size (int): Size of the convolutional kernel.
        concatenate (bool): Whether to concatenate inputs with previous states.
        disable_reset (bool): Whether to disable the reset mechanism.
        b_j0 (float): Initial threshold value.
        beta (float): Scaling factor for the adaptive threshold.
    """
    def __init__(self, channels=None, linear_features=None, kernel_size=None, concatenate=False, disable_reset=False, b_j0=0.5, beta=1.8):
        super(AdaptiveSpikingNeuron, self).__init__()

        self.channels = channels
        self.linear_features = linear_features
        self.kernel_size = kernel_size
        self.concatenate = concatenate
        self.disable_reset = disable_reset

        self.b_j0 = b_j0
        self.beta = beta

        assert (
            (
                linear_features != None and
                kernel_size == None
            ) or
            (
                linear_features == None and
                kernel_size != None
            )
        ), "Specify either linear features or specify kernel_size, not both!"

        if linear_features == None:
            self.layer_tau_m = nn.Conv2d(channels * (2 if concatenate else 1), channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
            self.batch_norm_tau_m = nn.BatchNorm2d(channels)

            if not self.disable_reset:
                self.layer_tau_adp = nn.Conv2d(channels * (2 if concatenate else 1), channels, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
                self.batch_norm_tau_adp = nn.BatchNorm2d(channels)
        else:
            self.layer_tau_m = nn.Linear(linear_features * (2 if concatenate else 1), linear_features)
            self.batch_norm_tau_m = nn.BatchNorm1d(linear_features)

            if not self.disable_reset:
                self.layer_tau_adp = nn.Linear(linear_features * (2 if concatenate else 1), linear_features)
                self.batch_norm_tau_adp = nn.BatchNorm1d(linear_features)

        self.activation_tau_m = nn.Sigmoid()
        if not self.disable_reset:
            self.activation_tau_adp = nn.Sigmoid()

        nn.init.xavier_uniform_(self.layer_tau_m.weight)
        if not self.disable_reset:
            nn.init.xavier_uniform_(self.layer_tau_adp.weight)
        if linear_features == None:
            nn.init.constant_(self.layer_tau_m.bias,0)
            if not self.disable_reset:
                nn.init.constant_(self.layer_tau_adp.bias,0)

        spk = torch.zeros(0)
        self.register_buffer("spk", spk, False)

        mem = torch.zeros(0)
        self.register_buffer("mem", mem, False)

        b = torch.zeros(0)
        self.register_buffer("b", b, False)


    def forward(self, x):
        """
        Perform the forward pass of the neuron model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the neuron model.
        """
        if not self.spk.shape == x.shape:
            self.spk = torch.zeros_like(x, device=self.spk.device)

        if not self.mem.shape == x.shape:
            self.mem = torch.zeros_like(x, device=self.mem.device)

        if not self.b.shape == x.shape:
            self.b = torch.full_like(x, self.b_j0, device=self.b.device)

        tau_m = self.activation_tau_m(self.batch_norm_tau_m(self.layer_tau_m(torch.cat((x, self.mem), dim=(1 if self.linear_features == None else -1)) if self.concatenate else (x + self.mem))))

        d_mem = -self.mem + x
        self.mem += d_mem * tau_m

        if self.disable_reset:
            return self.mem
        else:
            tau_adp = self.activation_tau_adp(self.batch_norm_tau_adp(self.layer_tau_adp(torch.cat((x, self.b), dim=(1 if self.linear_features == None else -1)) if self.concatenate else (x + self.b))))
        
            self.b = tau_adp * self.b + (1 - tau_adp) * self.spk
            B = self.b_j0 + self.beta * self.b

            self.spk = SpikingActivation.apply(self.mem - B)
            self.mem = (1 - self.spk) * self.mem

            return self.spk


    def reset_mem(self):
        """
        Resets the neuron state variables to their initial values.

        This method resets the following attributes:
        - `spk`: Sets the spike tensor to zeros, maintaining the same shape and device.
        - `mem`: Sets the membrane potential tensor to zeros, maintaining the same shape and device.
        - `b`: Resets the bias tensor to its initial value `b_j0`, maintaining the same shape and device.
        """
        self.spk = torch.zeros_like(self.spk, device=self.spk.device)
        self.mem = torch.zeros_like(self.mem, device=self.mem.device)
        self.b = torch.full_like(self.b, self.b_j0, device=self.b.device)
    

    def detach_hidden(self):
        """
        Detach the hidden state variables from the current computation graph.

        This method detaches the spiking activity (spk), membrane potential (mem), and bias (b) from the current computation graph,
        which is useful for truncated Backpropagation Through Time.
        """
        self.spk.detach_()
        self.mem.detach_()
        self.b.detach_()


class ParametricSpikingNeuron(nn.Module):
    def __init__(self, disable_reset=False, b_j0=0.5, beta=1.8):
        super(ParametricSpikingNeuron, self).__init__()

        self.disable_reset = disable_reset

        self.b_j0 = b_j0
        self.beta = beta

        self.tau_m = nn.Parameter(torch.tensor(0.5), requires_grad=True)
        if not self.disable_reset:
            self.tau_adp = nn.Parameter(torch.tensor(0.5), requires_grad=True)

        spk = torch.zeros(0)
        self.register_buffer("spk", spk, False)

        mem = torch.zeros(0)
        self.register_buffer("mem", mem, False)

        b = torch.zeros(0)
        self.register_buffer("b", b, False)

    def forward(self, x):
        """
        Perform the forward pass of the neuron model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying the neuron model.
        """
        if not self.spk.shape == x.shape:
            self.spk = torch.zeros_like(x, device=self.spk.device)

        if not self.mem.shape == x.shape:
            self.mem = torch.zeros_like(x, device=self.mem.device)

        if not self.b.shape == x.shape:
            self.b = torch.full_like(x, self.b_j0, device=self.b.device)

        tau_m = F.sigmoid(self.tau_m)

        d_mem = -self.mem + x
        self.mem += d_mem * tau_m

        if self.disable_reset:
            return self.mem
        else:
            tau_adp = F.sigmoid(self.tau_adp)
            
            self.b = tau_adp * self.b + (1 - tau_adp) * self.spk
            B = self.b_j0 + self.beta * self.b

            self.spk = SpikingActivation.apply(self.mem - B)
            self.mem = (1 - self.spk) * self.mem

            return self.spk


    def reset_mem(self):
        """
        Resets the neuron state variables to their initial values.

        This method resets the following attributes:
        - `spk`: Sets the spike tensor to zeros, maintaining the same shape and device.
        - `mem`: Sets the membrane potential tensor to zeros, maintaining the same shape and device.
        - `b`: Resets the bias tensor to its initial value `b_j0`, maintaining the same shape and device.
        """
        self.spk = torch.zeros_like(self.spk, device=self.spk.device)
        self.mem = torch.zeros_like(self.mem, device=self.mem.device)
        self.b = torch.full_like(self.b, self.b_j0, device=self.b.device)
    
    
    def detach_hidden(self):
        """
        Detach the hidden state variables from the current computation graph.

        This method detaches the spiking activity (spk), membrane potential (mem), and bias (b) from the current computation graph,
        which is useful for truncated Backpropagation Through Time.
        """
        self.spk.detach_()
        self.mem.detach_()
        self.b.detach_()