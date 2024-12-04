import numpy as np
import torch
import torch.nn.functional as F
import lightning

from loss import loss_angle, loss_distance, metric_angle, metric_distance
from neuron import AdaptiveSpikingNeuron, ParametricSpikingNeuron

class GazeModel(lightning.LightningModule):
    """
    A PyTorch Lightning module for a gaze tracking model.
    
    Args:
        output_size (int): The size of the output layer. Default is 5.
        backprop_timesteps (int): The number of time steps for truncated Backpropagation Through Time. Default is 5.
        fptt (bool): Flag to enable/disable Forward Propagation Through Time. Default is True.
        clip (float): Gradient clipping value. Default is 1.0.
        time_window_threshold (float): Time window threshold for masking unreliable interpolated targets. Default is infinity, meaning no targets are masked.
        *args: Additional arguments for the LightningModule.
        **kwargs: Additional keyword arguments for the LightningModule.
    """
    def __init__(
            self,
            output_size=5,
            backprop_timesteps=5,
            fptt=True,
            clip=1.,
            time_window_threshold=float('inf'),
            *args,
            **kwargs,
        ):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()

        self.automatic_optimization = False

        self.output_size = output_size

        self.backprop_timesteps = backprop_timesteps

        self.fptt = fptt

        self.clip = clip

        self.time_window_threshold = time_window_threshold

        self.named_params = None


    def forward_timestep(self, x_t):
        """
        Processes the input for a single timestep.

        Args:
            x_t: The input at the current timestep.

        Returns:
            The processed output for the current timestep.
        """
        return x_t


    def detach_hidden(self):
        """
        Detaches the state of the spiking neurons of the model.
        """
        for module in self.modules():
            if isinstance(module, AdaptiveSpikingNeuron) or isinstance(module, ParametricSpikingNeuron):
                module.detach_hidden()


    def reset_mem(self):
        """
        Resets the membrane potential of the spiking neurons of the model.
        """
        for module in self.modules():
            if isinstance(module, AdaptiveSpikingNeuron) or isinstance(module, ParametricSpikingNeuron):
                module.reset_mem()


    def get_stats_named_params(self):
        """
        Collects and returns a dictionary of named parameters, used for Forward Propagation Through Time.
        (Based on https://github.com/byin-cwi/sFPTT)

        Returns:
            dict: A dictionary where keys are parameter names and values are tuples 
                containing the original parameter tensor, its detached clone, 
                and a zero-initialized tensor of the same shape.
        """
        named_params = {}
        for name, param in self.named_parameters():
            sm, lm = param.detach().clone(), torch.zeros_like(param, device=param.device)
            named_params[name] = (param, sm, lm)
        return named_params


    def reset_named_params(self):
        """
        Resets the names parameters to their initial values, used for Forward Propagation Through Time.
        (Based on https://github.com/byin-cwi/sFPTT)
        """
        for name in self.named_params:
            param, sm, _ = self.named_params[name]
            param.data.copy_(sm.data)
    

    def get_regularizer_named_params(self, alpha=0.1, rho=0.0, _lambda=1.0):
        """
        Calculate the regularization term for the loss, used for Forward Propagation Through Time.
        (Based on https://github.com/byin-cwi/sFPTT)

        Returns:
            torch.Tensor: The computed regularization term.
        """
        regularization = torch.zeros([], device=self.device)
        for name in self.named_params:
            param, sm, lm = self.named_params[name]
            regularization += (rho-1.) * torch.sum( param * lm )
            r_p = _lambda * 0.5 * alpha * torch.sum( torch.square(param - sm) )
            regularization += r_p
        return regularization


    def post_optimizer_updates(self, alpha=0.1, beta=0.5):
        """
        Applies post-optimization updates to the parameters, used for Forward Propagation Through Time.
        (Based on https://github.com/byin-cwi/sFPTT)
        """
        for name in self.named_params:
            param, sm, lm = self.named_params[name]
            lm.data.add_( -alpha * (param - sm) )
            sm.data.mul_( (1.0-beta) )
            sm.data.add_( beta * param - (beta/alpha) * lm )


    def forward(self, x, train=False, targets=None):
        B, T, C, H, W = x.shape

        outputs = torch.zeros((B, T, self.output_size if self.output_size > 0 else self.embed_dims), device=x.device)
        sparsity = torch.zeros((T), device=x.device)

        if train:
            total_distance_loss = 0
            total_angle_loss = 0

        self.reset_mem()

        for timestep in range(T):
            x_t = x[:,timestep]
            x_t = self.forward_timestep(x_t)
            x_t[:,3] = F.tanh(x_t[:,3]) * np.pi
            x_t[:,4] = F.tanh(x_t[:,4]) * np.pi/2
            outputs[:,timestep] = x_t

            sparsity[timestep] = self.get_timestep_sparsity()

            if train and (timestep+1) % self.backprop_timesteps == 0:
                time_window_mask = targets[:,timestep,1] <= self.time_window_threshold
                distance_loss = loss_distance(x_t, targets[:,timestep,2:], time_window_mask)
                angle_loss = loss_angle(x_t, targets[:,timestep,2:], time_window_mask)
    
                total_distance_loss += distance_loss.item()
                total_angle_loss += angle_loss.item()

                loss = distance_loss + angle_loss

                if self.fptt:
                    regularizer = self.get_regularizer_named_params()
                    loss += regularizer

                optimizer = self.optimizers()
                optimizer.zero_grad()
                
                self.manual_backward(loss)
                    
                if self.clip > 0:
                    self.clip_gradients(optimizer, gradient_clip_val=self.clip, gradient_clip_algorithm="norm")
                    
                optimizer.step()

                if self.fptt:
                    self.post_optimizer_updates()

                self.detach_hidden()

        return (outputs, sparsity) if not train else (outputs, sparsity, total_distance_loss/(T/self.backprop_timesteps), total_angle_loss/(T/self.backprop_timesteps))


    def training_step(self, batch):
        if self.fptt and self.named_params is None:
            self.named_params = self.get_stats_named_params()

        frames, targets = batch
        frames = frames.float()
        targets = targets.float()

        _, sparsity, distance_loss, angle_loss = self.forward(frames, targets=targets, train=True)
        
        optimizer = self.optimizers()
        
        self.log_dict({
            "train_distance_loss": distance_loss,
            "train_angle_loss": angle_loss,
            "train_loss": distance_loss + angle_loss,
            "lr": optimizer.param_groups[0]["lr"],
            "train_sparsity": sparsity.mean()
        }, prog_bar=True, sync_dist=True)

        if self.fptt and self.trainer.is_last_batch:
            self.reset_named_params()


    def on_train_epoch_end(self):
        scheduler = self.lr_schedulers()
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(self.trainer.callback_metrics["train_angle_loss"])
        else:
            scheduler.step()


    def validation_step(self, batch):
        frames, targets = batch
        frames = frames.float()
        targets = targets.float()

        outputs, sparsity = self.forward(frames)

        time_window_mask = targets[:,:,1] <= self.time_window_threshold

        distance_loss = loss_distance(outputs, targets[:,:,2:], time_window_mask)
        angle_loss = loss_angle(outputs, targets[:,:,2:], time_window_mask)
        distance = metric_distance(outputs, targets[:,:,2:], time_window_mask)
        angle = metric_angle(outputs, targets[:,:,2:], time_window_mask)

        self.log_dict({
            "val_distance_loss": distance_loss,
            "val_angle_loss": angle_loss,
            "val_loss": distance_loss + angle_loss,
            "val_distance": distance,
            "val_angle": angle,
            "val_sparsity": sparsity.mean()
        }, prog_bar=True, sync_dist=True)


    def test_step(self, batch):
        frames, targets = batch
        frames = frames.float()
        targets = targets.float()

        outputs, sparsity = self.forward(frames)

        time_window_mask = targets[:,:,1] <= self.time_window_threshold

        distance = metric_distance(outputs, targets[:,:,2:], time_window_mask)
        angle = metric_angle(outputs, targets[:,:,2:], time_window_mask)

        self.log_dict({
            "test_distance": distance,
            "test_angle": angle,
            "test_sparsity": sparsity.mean()
        }, prog_bar=True, sync_dist=True)
    

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }
    

    def get_timestep_sparsity(self):
        """
        Calculates the mean firing rate of the spiking neurons at the current time step.

        Returns:
            float: The ratio of total spikes to total neurons, representing the sparsity.
        """
        total_spikes = 0
        total_neurons = 0
        for module in self.modules():
            if (isinstance(module, AdaptiveSpikingNeuron) or isinstance(module, ParametricSpikingNeuron)) and not module.disable_reset:
                total_spikes += module.spk.sum()
                total_neurons += module.spk.numel()
        return total_spikes / total_neurons