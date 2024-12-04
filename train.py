import numpy as np

import torch
from torch.utils.data import DataLoader

import tonic.transforms
import torchvision.transforms
import transforms

import lightning

from dataset import EVEyeDataset, CachedEyeDataset, BatchedEyeDataset
from model import GazeSCRNN

import argparse

torch.manual_seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("experiment_name", help="Experiment name")

parser.add_argument("-b", "--batch_size", default=80, help="Batch size")
parser.add_argument("-e", "--epochs", default=200, help="Epochs")

parser.add_argument("-g", "--gpus", help="The GPUs to use")
parser.add_argument("-d", "--dev", action='store_true', help="Enable dev mode, only run one batch for debugging purposes")

parser.add_argument("-bt", "--backprop_timesteps", default=8, help="The number of truncated Backpropagation Through Time steps")
parser.add_argument("-f", "--fptt", action='store_true', help="Enable Forward Propagation Through Time")
parser.add_argument("-twt", "--time_window_threshold", default=float('inf'), help="Set a threshold for the maximum time window of interpolation between two ground truth gaze references. All above the threshold will be masked from the loss and metric computations.")

parser.add_argument("-fu", "--full", action='store_true', help="Use full size inputs")
parser.add_argument("-p", "--plif", action='store_true', help="Use PLIF neurons instead of ALIF neurons")

parser.add_argument("-ey", "--eye", default="left", help="Which eye to use (left, right)")

parser.add_argument("-d_p", "--data_preload", action='store_true', help="Preload dataset (useful for pre-caching)")
parser.add_argument("-d_t", "--data_time_window", default=None, help="Time window (in µs) per frame")
parser.add_argument("-d_ec", "--data_event_count", default=None, help="The number of events per frame")
parser.add_argument("-d_fc", "--data_frame_count", default=1000, help="The number of frames per sequence")
parser.add_argument("-d_aa", "--data_augment_affine", action='store_true', help="Enable affine data augmentation")
parser.add_argument("-d_an", "--data_augment_noise", default=0.0, help="Enable random noise events data augmentation")

args = vars(parser.parse_args())

experiment_name = args['experiment_name']

batch_size = int(args['batch_size'])
epochs = int(args['epochs'])

gpus = list(map(int, args['gpus'].split(','))) if args['gpus'] else []
fast_dev_run = bool(args['dev'])

backprop_timesteps = int(args['backprop_timesteps'])
fptt = bool(args['fptt'])
time_window_threshold = float(args['time_window_threshold'])

full = bool(args['full'])
plif = bool(args['plif'])

eye = args['eye']

data_preload = bool(args['data_preload'])
data_frame_count = int(args['data_frame_count'])
data_time_window = int(args['data_time_window']) if args['data_time_window'] is not None else None
data_event_count = (int(args['data_event_count']) if args['data_event_count'] is not None else 300) if data_time_window is None else None
data_augment_affine = bool(args['data_augment_affine'])
data_augment_noise = float(args['data_augment_noise'])

def print_main_process(*args, **kwargs):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(*args, **kwargs)
    else:
        print(*args, **kwargs)

print_main_process('name', experiment_name)

print_main_process('batch_size', batch_size)
print_main_process('epochs', epochs)

print_main_process('gpus', gpus)
print_main_process('fast_dev_run', fast_dev_run)

print_main_process('backprop_timesteps', backprop_timesteps)
print_main_process('fptt', fptt)
print_main_process('time_window_threshold', time_window_threshold)

print_main_process('full', full)
print_main_process('plif', plif)

print_main_process('eye', eye)

print_main_process('data_preload', data_preload)
print_main_process('data_frame_count', data_frame_count)
print_main_process('data_time_window', data_time_window)
print_main_process('data_event_count', data_event_count)
print_main_process('data_augment_affine', data_augment_affine)
print_main_process('data_augment_noise', data_augment_noise)

downsample = tonic.transforms.Downsample(spatial_factor=0.5)
dataset = CachedEyeDataset(
    EVEyeDataset(
        interpolate=True,
        time_window=data_time_window,
        event_count=data_event_count,
        eye=eye,
        sensor_size=(173, 130, 2) if not full else (346, 260, 2),
        events_transform=downsample if not full else None,
    ),
    cache_path=f'./cache/EVEye{"_half" if not full else ""}_{eye}_{data_event_count if data_event_count is not None else str(data_time_window)+"us"}_interpolate/',
    use_disk_cache=True
)
if data_preload and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
    print("Preload dataset")
    dataset.preload(4)

data_augmentations = []
if data_augment_affine:
    data_augmentations.append(
        torchvision.transforms.RandomAffine(
            degrees=2,
            translate=(0.01, 0.01),
            scale=(0.99, 1.01),
            shear=2,
            interpolation=torchvision.transforms.InterpolationMode.NEAREST,
        )
    )
if data_augment_noise > 0:
    data_augmentations.append(
        transforms.RandomNoiseEvents(
            probability=data_augment_noise,
        )
    )
transform = torchvision.transforms.Compose(data_augmentations) if len(data_augmentations) > 0 else None

train_dataset = CachedEyeDataset(
    BatchedEyeDataset(dataset, frame_count=data_frame_count),
    cache_path=f'./cache/EVEye{"_half" if not full else ""}_{eye}_{data_event_count if data_event_count is not None else str(data_time_window)+"us"}_interpolate_{data_frame_count}_frames/train/',
    use_disk_cache=True,
    transform=transform,
)
val_dataset = CachedEyeDataset(
    BatchedEyeDataset(dataset, frame_count=data_frame_count, split='val'),
    cache_path=f'./cache/EVEye{"_half" if not full else ""}_{eye}_{data_event_count if data_event_count is not None else str(data_time_window)+"us"}_interpolate_{data_frame_count}_frames/val/',
    use_disk_cache=True
)
test_dataset = CachedEyeDataset(
    BatchedEyeDataset(dataset, frame_count=data_frame_count, split='test'),
    cache_path=f'./cache/EVEye{"_half" if not full else ""}_{eye}_{data_event_count if data_event_count is not None else str(data_time_window)+"us"}_interpolate_{data_frame_count}_frames/test/',
    use_disk_cache=True
)
if data_preload and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
    train_dataset.preload(8)
    val_dataset.preload(8)
    test_dataset.preload(8)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

if full:
    from model_full import GazeSCRNN
if plif:
    from model_plif import GazeSCRNN

model = GazeSCRNN(
    output_size=5,
    backprop_timesteps=backprop_timesteps,
    fptt=fptt,
    time_window_threshold = time_window_threshold,
)

logger = lightning.pytorch.loggers.tensorboard.TensorBoardLogger('logs', name='', version=experiment_name)
summary_callback = lightning.pytorch.callbacks.ModelSummary(max_depth=1)
checkpoint_callback_angle = lightning.pytorch.callbacks.ModelCheckpoint(
    dirpath=f"checkpoints/{experiment_name}",
    save_top_k=10,
    monitor='val_angle',
    filename=experiment_name+"-{epoch:02d}-{val_angle:.2f}",
)
checkpoint_callback_distance = lightning.pytorch.callbacks.ModelCheckpoint(
    dirpath=f"checkpoints/{experiment_name}",
    save_top_k=10,
    monitor='val_distance',
    filename=experiment_name+"-{epoch:02d}-{val_distance:.2f}",
)
checkpoint_callback_loss = lightning.pytorch.callbacks.ModelCheckpoint(
    dirpath=f"checkpoints/{experiment_name}",
    save_top_k=10,
    monitor='val_loss',
    filename=experiment_name+"-{epoch:02d}-{val_loss:.2f}",
)

trainer = lightning.Trainer(logger=logger, callbacks=[summary_callback, checkpoint_callback_angle, checkpoint_callback_distance, checkpoint_callback_loss], log_every_n_steps=1, max_epochs=epochs, devices=gpus if len(gpus) > 0 else 'auto', accelerator='gpu' if len(gpus) > 0 else 'cpu', fast_dev_run=fast_dev_run)
trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    test_trainer = lightning.Trainer(logger=logger, devices=[gpus[0]] if len(gpus) > 0 else 'auto', accelerator='gpu' if len(gpus) > 0 else 'cpu', num_nodes=1, fast_dev_run=fast_dev_run)
    test_trainer.test(model=model, ckpt_path=checkpoint_callback_angle.best_model_path if not fast_dev_run else None, dataloaders=test_dataloader)

if torch.distributed.is_initialized():
    torch.distributed.destroy_process_group()