import numpy as np

import torch
from torch.utils.data import DataLoader

import tonic.transforms

import lightning

from dataset import EVEyeDataset, CachedEyeDataset, BatchedEyeDataset
from model import GazeSCRNN

import argparse

torch.manual_seed(0)
np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("checkpoint_path", help="Checkpoint path")
parser.add_argument("experiment_name", help="Experiment name")

parser.add_argument("-l", "--log", action='store_true', help="Enable logger")

parser.add_argument("-b", "--batch_size", default=80, help="Batch size")
parser.add_argument("-e", "--epochs", default=200, help="Epochs")

parser.add_argument("-g", "--gpus", help="GPUs")
parser.add_argument("-d", "--dev", action='store_true', help="Dev mode")

parser.add_argument("-bt", "--backprop_timesteps", default=4, help="Backprop timesteps")
parser.add_argument("-f", "--fptt", action='store_true', help="FPTT")
parser.add_argument("-twt", "--time_window_threshold", default=float('inf'), help="Time window threshold")

parser.add_argument("-fu", "--full", action='store_true', help="Use full size inputs")
parser.add_argument("-p", "--plif", action='store_true', help="Use PLIF neurons instead of ALIF neurons")

parser.add_argument("-ey", "--eye", default="left", help="Eye")

parser.add_argument("-d_p", "--data_preload", action='store_true', help="Preload dataset")
parser.add_argument("-d_t", "--data_time_window", default=None, help="Time window (in µs)")
parser.add_argument("-d_ec", "--data_event_count", default=None, help="Event count")
parser.add_argument("-d_fc", "--data_frame_count", default=1000, help="Frame count")
parser.add_argument("-d_aa", "--data_augment_affine", action='store_true', help="Enable affine data augmentation")
parser.add_argument("-d_an", "--data_augment_noise", default=0.0, help="Enable random noise events data augmentation")

args = vars(parser.parse_args())

checkpoint_path = args['checkpoint_path']
experiment_name = args['experiment_name']

log = bool(args['log'])

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

print_main_process('checkpoint_path', checkpoint_path)
print_main_process('name', experiment_name)

print_main_process('log', log)

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

test_dataset = CachedEyeDataset(
    BatchedEyeDataset(dataset, frame_count=data_frame_count, split='test'),
    cache_path=f'./cache/EVEye{"_half" if not full else ""}_{eye}_{data_event_count if data_event_count is not None else str(data_time_window)+"us"}_interpolate_{data_frame_count}_frames/test/',
    use_disk_cache=True
)
if data_preload and (not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0):
    test_dataset.preload(8)

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

logger = lightning.pytorch.loggers.tensorboard.TensorBoardLogger('logs', name='', version=experiment_name) if log else False

if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    test_trainer = lightning.Trainer(logger=logger, devices=[gpus[0]] if len(gpus) > 0 else 'auto', accelerator='gpu' if len(gpus) > 0 else 'cpu', num_nodes=1, fast_dev_run=fast_dev_run)
    test_trainer.test(model=model, ckpt_path=checkpoint_path, dataloaders=test_dataloader)

if torch.distributed.is_initialized():
    torch.distributed.destroy_process_group()