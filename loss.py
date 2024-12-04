import numpy as np
import torch


def angles_to_coords(angles):
    """
    Converts angles to 3D coordinates.

    Args:
        angles (torch.Tensor): The vector angles in radians. Expected shape is either (B, 2) or (B, T, 2),
            where B is the batch size, T is the number of time steps,
            and 2 represents the angles or the gaze vector.

    Returns:
        torch.Tensor: A tensor of shape (B, 3) or (B, T, 3) containing the 3D coordinates corresponding to the input angles.
    """
    if len(angles.shape) == 3:
        coordinates = torch.zeros((angles.shape[0], angles.shape[1], 3), device=angles.device)
        coordinates[:,:,0] = torch.sin(np.pi/2 - angles[:,:,1]) * torch.sin(-angles[:,:,0])
        coordinates[:,:,1] = torch.cos(np.pi/2 - angles[:,:,1])
        coordinates[:,:,2] = torch.sin(np.pi/2 - angles[:,:,1]) * torch.cos(-angles[:,:,0])
    else:
        coordinates = torch.zeros((angles.shape[0], 3), device=angles.device)
        coordinates[:,0] = torch.sin(np.pi/2 - angles[:,1]) * torch.sin(-angles[:,0])
        coordinates[:,1] = torch.cos(np.pi/2 - angles[:,1])
        coordinates[:,2] = torch.sin(np.pi/2 - angles[:,1]) * torch.cos(-angles[:,0])
    return coordinates


def loss_distance(prediction, target, mask=None):
    """
    Computes a masked mean squared error loss between the predicted and target gaze vector origin.

    Args:
        prediction (torch.Tensor): The predicted gaze vector. Expected shape is either (B, 5) or (B, T, 5),
            where B is the batch size, T is the number of time steps,
            and 5 represents the 3 origin coordinates (in mm) and 2 angles of the gaze vector (in radians).
        target (torch.Tensor): The ground truth vector. Expected shape is either (B, 5) or (B, T, 5),
            where B is the batch size, T is the number of time steps,
            and 5 represents the 3 origin coordinates (in mm) and 2 angles of the gaze vector (in radians).
        mask (torch.Tensor, optional): A mask tensor to apply to the loss. Expected shape is either (B) or (B, T),
            where B is the batch size, T is the number of time steps.
            If None, no values are masked. Default is None.

    Returns:
        torch.Tensor: The computed masked mean squared error loss.
    """
    if mask is None:
        mask = torch.ones(prediction.shape[:-1])
    if len(prediction.shape) == 3:
        mean_squared_error = torch.square(prediction[:,:,:3] - target[:,:,:3]).mean(dim=-1)
    else:
        mean_squared_error = torch.square(prediction[:,:3] - target[:,:3]).mean(dim=-1)
    return (mean_squared_error * mask).sum() / mask.sum()


def loss_angle(prediction, target, mask=None):
    """
    Computes a masked cosine distance loss between the predicted and target gaze vector angles.
    
    Args:
        prediction (torch.Tensor): The predicted gaze vector. Expected shape is either (B, 5) or (B, T, 5),
            where B is the batch size, T is the number of time steps,
            and 5 represents the 3 origin coordinates (in mm) and 2 angles of the gaze vector (in radians).
        target (torch.Tensor): The ground truth vector. Expected shape is either (B, 5) or (B, T, 5),
            where B is the batch size, T is the number of time steps,
            and 5 represents the 3 origin coordinates (in mm) and 2 angles of the gaze vector (in radians).
        mask (torch.Tensor, optional): A mask tensor to apply to the loss. Expected shape is either (B) or (B, T),
            where B is the batch size, T is the number of time steps.
            If None, no values are masked. Default is None.
    
    Returns:
        torch.Tensor: The computed masked mean cosine distance loss, scaled by 1000.
    """
    if mask is None:
        mask = torch.ones(prediction.shape[:-1])
    if len(prediction.shape) == 3:
        prediction = angles_to_coords(prediction[:,:,3:])
        target = angles_to_coords(target[:,:,3:])
        cosine_distance = (1 - (prediction * target).sum(dim=-1))
    else:
        prediction = angles_to_coords(prediction[:,3:])
        target = angles_to_coords(target[:,3:])
        cosine_distance = (1 - (prediction * target).sum(dim=-1))
    return (cosine_distance * mask).sum() / mask.sum() * 1000


def metric_distance(prediction, target, mask=None):
    """
    Computes the masked Euclidean distance between the predicted and target gaze vector origin.

    Args:
        prediction (torch.Tensor): The predicted gaze vector. Expected shape is either (B, 5) or (B, T, 5),
            where B is the batch size, T is the number of time steps,
            and 5 represents the 3 origin coordinates (in mm) and 2 angles of the gaze vector (in radians).
        target (torch.Tensor): The ground truth vector. Expected shape is either (B, 5) or (B, T, 5),
            where B is the batch size, T is the number of time steps,
            and 5 represents the 3 origin coordinates (in mm) and 2 angles of the gaze vector (in radians).
        mask (torch.Tensor, optional): A mask tensor to apply to the loss. Expected shape is either (B) or (B, T),
            where B is the batch size, T is the number of time steps.
            If None, no values are masked. Default is None.

    Returns:
        torch.Tensor: The computed masked mean Euclidean distance.
    """
    if mask is None:
        mask = torch.ones(prediction.shape[:-1])
    if len(prediction.shape) == 3:
        distance = torch.sqrt(torch.square(prediction[:,:,:3] - target[:,:,:3]).sum(dim=-1))
    else:
        distance = torch.sqrt(torch.square(prediction[:,:3] - target[:,:3]).sum(dim=-1))
    return (distance * mask).sum() / mask.sum()


def metric_angle(prediction, target, mask=None):
    """
    Computes the angular difference between predicted and target gaze vector angles.

    Args:
        prediction (torch.Tensor): The predicted gaze vector. Expected shape is either (B, 5) or (B, T, 5),
            where B is the batch size, T is the number of time steps,
            and 5 represents the 3 origin coordinates (in mm) and 2 angles of the gaze vector (in radians).
        target (torch.Tensor): The ground truth vector. Expected shape is either (B, 5) or (B, T, 5),
            where B is the batch size, T is the number of time steps,
            and 5 represents the 3 origin coordinates (in mm) and 2 angles of the gaze vector (in radians).
        mask (torch.Tensor, optional): A mask tensor to apply to the loss. Expected shape is either (B) or (B, T),
            where B is the batch size, T is the number of time steps.
            If None, no values are masked. Default is None.

    Returns:
        torch.Tensor: The computed masked mean angular difference.
    """
    if mask is None:
        mask = torch.ones(prediction.shape[:-1])
    if len(prediction.shape) == 3:
        prediction = angles_to_coords(prediction[:,:,3:])
        target = angles_to_coords(target[:,:,3:])
        angle = torch.rad2deg(torch.arccos(torch.clamp((prediction * target).sum(dim=-1), -1., 1.)))
    else:
        prediction = angles_to_coords(prediction[:,3:])
        target = angles_to_coords(target[:,3:])
        angle = torch.rad2deg(torch.arccos(torch.clamp((prediction * target).sum(dim=-1), -1., 1.)))
    return (angle * mask).sum() / mask.sum()