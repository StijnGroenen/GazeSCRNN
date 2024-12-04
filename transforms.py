import torch

class RandomNoiseEvents():
    """
    A transform to add random noise events to frames with a specified probability.

    Args:
        probability (float): The probability of adding noise to each pixel. Default is 0.01.
    """
    def __init__(self, probability=0.01, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.probability = probability

    def __call__(self, frames):
        """
        Adds random noise events to the input frames with a specified probability.
        
        Args:
            frames (torch.Tensor): Input tensor with shape (C, H, W) or (B, C, H, W),
                                   where B is the batch size, C is the number of channels,
                                   H is the height, and W is the width.
        Returns:
            torch.Tensor: The transformed tensor with noise events added.
        """
        assert (len(frames.shape) == 3 or len(frames.shape) == 4), "Only inputs with 3 or 4 dimensions are supported!"
        
        if len(frames.shape) == 3:
            noise = torch.bernoulli(torch.ones(frames.shape[1:]) * self.probability)
            noise_channel = torch.bernoulli(torch.ones(noise.shape) * 0.5).to(torch.bool)
            frames[0] = frames[0] + noise * (~noise_channel) * (1-frames[0])
            frames[1] = frames[1] + noise * noise_channel * (1-frames[1])
            return frames
        
        noise = torch.bernoulli(torch.ones((frames.shape[0], frames.shape[2], frames.shape[3])) * self.probability)
        noise_channel = torch.bernoulli(torch.ones(noise.shape)*0.5).to(torch.bool)
        frames[:,0] = frames[:,0] + noise * (~noise_channel) * (1-frames[:,0])
        frames[:,1] = frames[:,1] + noise * noise_channel * (1-frames[:,1])
        return frames