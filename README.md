# GazeSCRNN: Event-based Near-eye Gaze Tracking using a Spiking Neural Network

GazeSCRNN is a spiking convolutional recurrent neural network designed for event-based near-eye gaze tracking. This repository contains the official implementation of the model, training scripts, and evaluation metrics.

## Requirements

To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

Besides the Python dependencies, training and testing the GazeSCRNN models requires the EV-Eye dataset to be present in the [EV_Eye_dataset](EV_Eye_dataset) directory. The EV-Eye dataset can be obtained by following the steps [here](https://github.com/Ningreka/EV-Eye).

## Training

To train the GazeSCRNN model, run the [train.py](train.py) script with the desired parameters. For example:

```bash
python train.py Experiment1 --data_preload --gpus 0 --fptt
```

Alternatively, you can train the GazeSCRNN model with one of the predefined configurations. For example:

```bash
xargs python train.py --gpus 0 < configs/GazeSCRNN-Events300-FPTT-Backprop8.txt
```

## Testing

To test a checkpoint of the GazeSCRNN model, run the [test.py](test.py) script with the desired parameters:

```bash
python test.py <experiment_name> <path_to_checkpoint_file> --gpus <gpu_id>
```

This will output the evaluation metrics such as Mean Angle Error (MAE), Mean Pupil Error (MPE), and Mean Firing Rate (MFR).

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
