import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
from tqdm import tqdm

class EyeDataset(Dataset):
    """
    A dataset class for handling eye-tracking data.
    Args:
        sensor_size (tuple): The size of the sensor (width, height, channels). Default is (346, 260, 2).
        eye (str): Specifies which eye data to use ('left', 'right', or 'both'). Default is 'left'.
        events_transform (callable, optional): A function/transform to apply to the events data.
        transform (callable, optional): A function/transform to apply to the frames data.
        interpolate (bool): If True, interpolates the target values. Default is True.
        time_window (int): The time window for segmenting the data in microseconds. Default is 10000.
        event_count (int, optional): The number of events to use for segmenting the data. If None, time_window is used.
        target_shape (int): The shape of the target data. Default is 5.
    """
    def __init__(self, sensor_size=(346, 260, 2), eye='left', events_transform=None, transform=None, interpolate=True, time_window=10000, event_count=None, target_shape=5):
        self.sensor_size = sensor_size
        self.eye = eye
        
        self.events_transform = events_transform
        self.transform = transform
        self.interpolate = interpolate
        self.time_window = time_window if event_count is None else None
        self.event_count = event_count
        self.target_shape = target_shape


    def get_session_lengths(self):
        """
        Calculate the lengths of all sessions in the dataset.

        This method iterates over the dataset and computes the number of frames
        for each session, storing the lengths in an array.

        Returns:
            numpy.ndarray: An array of session lengths, where each entry corresponds
                        to the number of frames in a session.
        """
        session_lengths = np.zeros(len(self), dtype=np.uint16)
        for i in range(len(self)):
            frames, __ = self[i]
            session_lengths[i] = frames.shape[0]
        return session_lengths


    def load_combined_events(self, index):   
        """
        Load and combine events from left and/or right eye based on the specified eye configuration.

        Args:
            index (int): The index of the events to load.

        Returns:
            numpy.ndarray: Combined events from the specified eye(s). If both eyes are specified, the events are concatenated and sorted by time.
        """
        if self.eye == 'left' or self.eye == 'both':
            events_left = self.load_events(index, 'left')
            if self.events_transform is not None:
                events_left = self.events_transform(events_left)
            events = events_left
        if self.eye == 'right' or self.eye == 'both':
            events_right = self.load_events(index, 'right')
            if self.events_transform is not None:
                events_right = self.events_transform(events_right)
            events_right['x'] = self.sensor_size[0] - 1 - events_right['x'] # Flip right eye events
            events = events_right
        if self.eye == 'both':
            events = np.concatenate((events_left, events_right))
            events.sort(kind='stable', order='t')
        return events


    def events_to_frames(self, events, event_indices):
        """
        Converts a list of events into frames.

        Args:
            events (numpy.ndarray): A structured numpy array containing the event data with fields:
                        - 'x': X-coordinate of the event.
                        - 'y': Y-coordinate of the event.
                        - 'p': Polarity of the event.
                        - 'eye': Eye identifier (0 for left, 1 for right).
            event_indices (numpy.ndarray): An array of indices indicating the start and end of each event segment corresponding to a frame.

        Returns:
            numpy.ndarray: A 4D numpy array representing the frames. The shape of the array is (number of events, channels, height, width).
                The number of channels depends on whether 'both' eyes are considered or not.
        """
        events = [events[event_indices[i]:event_indices[i+1]] for i in range(len(event_indices)-1)]
        frames = np.zeros((len(events), self.sensor_size[2]*2 if self.eye == 'both' else self.sensor_size[2], self.sensor_size[1], self.sensor_size[0]), dtype=np.uint8)
        for i, e in enumerate(events):
            channel = e['p'] + (self.sensor_size[2]*e['eye'] if self.eye == 'both' else 0)
            e = e[frames[(i, channel, e['y'], e['x'])] < 255]
            np.add.at(
                frames,
                (i, channel, e['y'], e['x']),
                1,
            )
        return frames


    def load_session_by_timewindows(self, index):
        """
        Load session data by time windows.
        This function processes gaze data and divides it into time windows. It interpolates
        the gaze data if required and aligns it with event data.

        Args:
            index (int): The index of the session to load.

        Returns:
            tuple: A tuple containing:
                - frames (numpy.ndarray): The frames corresponding to the events.
                - targets (numpy.ndarray): The processed gaze data divided into time windows.
        """
        gaze = self.load_gaze(index)
        
        targets = np.zeros((int((gaze[-1,0] - gaze[0,0]) / self.time_window * 1.1), 2+self.target_shape))
        targets[0,0] = gaze[0,0] - self.time_window
        current_index = 1
        for i in range(gaze.shape[0]):
            next_timestamp = gaze[i,0]
            next_target = gaze[i,1:]
            previous_target = targets[current_index-1,2:]
            previous_timestamp = targets[current_index-1,0]
            time_window = next_timestamp - previous_timestamp

            if time_window > self.time_window*0.8:
                num_windows = int(np.round(time_window / self.time_window))
                for j in range(num_windows-1, 0, -1):
                    timestamp = targets[current_index-1,0] + (time_window / num_windows)
                    targets[current_index,0] = timestamp
                    targets[current_index,1] = time_window
                    if self.interpolate:
                        targets[current_index,2:] = ((next_timestamp - timestamp) / time_window) * previous_target + ((timestamp - previous_timestamp) / time_window) * next_target
                    else:
                        targets[current_index,2:] = next_target
                    current_index += 1
                targets[current_index,0] = next_timestamp
                targets[current_index,1] = 0
                targets[current_index,2:] = next_target
                current_index += 1
        targets = targets[:current_index]
        targets[:,0] = np.round(targets[:,0])
        targets[:,1] = np.round(targets[:,1])

        events = self.load_combined_events(index)
        indices = np.searchsorted(events['t'], targets[:,0])
        frames = self.events_to_frames(events, indices)
        targets = targets[1:]

        return frames, targets


    def load_session_by_events(self, index):
        """
        Load session data by events for a given index.

        Args:
            index (int): The index of the session to load.

        Returns:
            tuple: A tuple containing:
                - frames (numpy.ndarray): The frames generated from the events.
                - targets (numpy.ndarray): The target values corresponding to the frames.
        """
        gaze = self.load_gaze(index)
        
        events = self.load_combined_events(index)
        events = events[(events['t'] < gaze[0,0]).sum()-self.event_count+1:]
        events = events[events['t'] <= gaze[-1,0]]
        
        distances = np.full((self.event_count), 0)
        for i in range(self.event_count):
            timestamps = events['t'][np.arange(self.event_count-1+i, events.shape[0]-1, self.event_count)]
            gaze_timestamps = gaze[:,0].astype(np.int32)
            indices = np.searchsorted(timestamps, gaze_timestamps)
            distances[i] = np.abs(timestamps[indices-1]-gaze_timestamps).sum()
        optimal_offset = np.argmin(distances)
        events = events[optimal_offset:]

        indices = np.arange(events.shape[0] // self.event_count + 1) * self.event_count
        frames = self.events_to_frames(events, indices)
        
        targets = np.zeros((frames.shape[0], 2+self.target_shape))
        targets[:,0] = events['t'][indices[1:]-1]
        for i in range(targets.shape[0]):
            timestamp = targets[i,0]
            
            gaze_index = np.sum(gaze[:,0] < timestamp)
            next_timestamp = gaze[gaze_index,0]
            next_target = gaze[gaze_index,1:]
            previous_timestamp = gaze[gaze_index-1,0]
            previous_target = gaze[gaze_index-1,1:]
            
            time_window = next_timestamp - previous_timestamp

            targets[i,1] = time_window
            if self.interpolate:
                targets[i,2:] = ((next_timestamp - timestamp) / time_window) * previous_target + ((timestamp - previous_timestamp) / time_window) * next_target
            else:
                targets[i,2:] = next_target if next_timestamp-timestamp <= timestamp-previous_timestamp else previous_target

        return frames, targets


    def __getitem__(self, index):
        """
        Retrieves the dataset item at the specified index.

        Depending on the presence of `event_count`, this method will load session data either by time windows or by events.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing:
                - frames (torch.Tensor): The frames of the session.
                - targets (torch.Tensor): The targets associated with the frames.
        """
        if self.event_count is None:
            frames, targets = self.load_session_by_timewindows(index)
        else:
            frames, targets = self.load_session_by_events(index)

        frames = torch.tensor(frames)
        targets = torch.tensor(targets)

        if self.transform != None:
            frames = self.transform(frames)

        return frames, targets


class CachedEyeDataset(Dataset):
    """
    CachedEyeDataset is a dataset wrapper that provides caching mechanisms for eye-tracking data. It supports both memory and disk caching to speed up data retrieval.
    
    Args:
        dataset (EyeDataset): The underlying dataset to be wrapped.
        cache_path (str): Path to the directory where cache files will be stored.
        use_memory_cache (bool): Flag to enable caching in memory. Default is False.
        use_disk_cache (bool): Flag to enable caching on disk. Default is False.
        transform (callable, optional): A function/transform to apply to the data.
    """
    def __init__(self, dataset, cache_path='./cache/eye_dataset/', use_memory_cache=False, use_disk_cache=False, transform=None):
        self.dataset = dataset
        
        self.cache_path = cache_path
        
        self.use_memory_cache = use_memory_cache
        if self.use_memory_cache:
            self.cache = {}
            self.session_lengths = None
        self.use_disk_cache = use_disk_cache
        if self.use_disk_cache and not os.path.isdir(self.cache_path):
            os.makedirs(self.cache_path)

        self.transform = transform


    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.dataset)


    def preload(self, num_workers=8):
        """
        Preloads and caches the dataset.

        Args:
            num_workers (int, optional): Number of worker threads to use for data loading. Defaults to 8.
        """
        dataloader = DataLoader(self, batch_size=1, shuffle=False, num_workers=num_workers)
        session_lengths = np.zeros(len(self), dtype=np.uint16)
        for i, batch in enumerate(tqdm(dataloader)):
            frames, _ = batch
            session_lengths[i] = frames.shape[1]
        if self.use_disk_cache:
            file_path = os.path.join(self.cache_path, f"session_lengths.hdf5")
            with h5py.File(file_path, "w") as f:
                f.create_dataset('session_lengths', data=session_lengths, compression="lzf")
        if self.use_memory_cache:
            self.session_lengths = session_lengths


    def get_session_lengths(self):
        """
        Retrieves the lengths of sessions.
        This method calculates or retrieves the lengths of sessions, either from memory, disk cache, or by computing them directly.

        Returns:
            numpy.ndarray: An array containing the lengths of each session.
        """
        if self.use_memory_cache and self.session_lengths is not None:
            return self.session_lengths
            
        if self.use_disk_cache:
            file_path = os.path.join(self.cache_path, f"session_lengths.hdf5")
            try:
                with h5py.File(file_path, "r") as f:
                    session_lengths = f['session_lengths'][()]
                if self.use_memory_cache:
                    self.session_lengths = session_lengths
                return session_lengths
            except (FileNotFoundError, OSError) as _:
                session_lengths = np.zeros(len(self), dtype=np.uint16)
                for i in range(len(self)):
                    frames, __ = self[i]
                    session_lengths[i] = frames.shape[0]
                with h5py.File(file_path, "w") as f:
                    f.create_dataset('session_lengths', data=session_lengths, compression="lzf")
                if self.use_memory_cache:
                    self.session_lengths = session_lengths
                return session_lengths

        session_lengths = np.zeros(len(self), dtype=np.uint16)
        for i in range(len(self)):
            frames, __ = self[i]
            session_lengths[i] = frames.shape[0]
        if self.use_memory_cache:
            self.session_lengths = session_lengths
        return session_lengths


    def __getitem__(self, index):
        """
        Retrieve the item at the specified index from the dataset.
        This method supports three modes of data retrieval:
        1. Memory cache: If the item is cached in memory, it retrieves it from there.
        2. Disk cache: If the item is not in memory but is cached on disk, it retrieves it from the disk.
        3. Direct dataset access: If the item is neither in memory nor on disk, it retrieves it directly from the dataset.

        Args:
            index (int): The index of the item to retrieve.

        Returns:
            tuple: A tuple containing:
                - frames (torch.Tensor): The frames tensor, either dense or transformed.
                - targets (torch.Tensor): The targets tensor.
        """
        if self.use_memory_cache and index in self.cache:
            frames, targets = self.cache[index]
            frames = frames.to_dense()
            if self.transform != None:
                frames = self.transform(frames)
            return frames, targets

        if self.use_disk_cache:
            file_path = os.path.join(self.cache_path, f"{index}.hdf5")
            try:
                with h5py.File(file_path, "r") as f:
                    frames_indices, frames_values, frames_size, targets = f['frames_indices'][()], f['frames_values'][()], f['frames_size'][()], f['targets'][()]
                    frames = torch.sparse_coo_tensor(frames_indices, frames_values, tuple(frames_size))
                    targets = torch.tensor(targets)
                    if self.use_memory_cache:
                        self.cache[index] = frames, targets
                    frames = frames.to_dense()
                    if self.transform != None:
                        frames = self.transform(frames)
                    return frames, targets
            except (FileNotFoundError, OSError) as _:
                frames, targets = self.dataset[index]
                sparse_frames = frames.to_sparse()
                with h5py.File(file_path, "w") as f:
                    f.create_dataset('frames_indices', data=sparse_frames.indices(), compression="lzf")
                    f.create_dataset('frames_values', data=sparse_frames.values(), compression="lzf")
                    f.create_dataset('frames_size', data=sparse_frames.size(), compression="lzf")
                    f.create_dataset('targets', data=targets, compression="lzf")
                if self.use_memory_cache:
                    self.cache[index] = sparse_frames, targets
                if self.transform != None:
                    frames = self.transform(frames)
                return frames, targets
                
        frames, targets = self.dataset[index]
        if self.use_memory_cache:
            self.cache[index] = frames.to_sparse(), targets
        if self.transform != None:
            frames = self.transform(frames)
        return frames, targets


class BatchedEyeDataset(Dataset):
    """
    BatchedEyeDataset is a dataset wrapper that handles batching of eye-tracking data for training, validation, and testing, as well as splitting sessions into sequences of a fixed size.

    Args:
        dataset (EyeDataset): The original dataset containing the eye-tracking data.
        frame_count (int): The number of frames in each sequence batch. Default is 1000.
        include_partial (bool): Whether to include partial batches that do not meet the frame_count. Default is False.
        split (str): The dataset split to use ('train', 'val', 'test'). Default is 'train'.
    """
    def __init__(self, dataset, frame_count=1000, include_partial=False, split='train'):
        self.dataset = dataset
        
        self.frame_count = frame_count
        self.include_partial = include_partial
        
        self.split = split
        
        self.session_lengths = self.dataset.get_session_lengths()
        self.session_batch_counts = (-1*(-1*self.session_lengths // self.frame_count) if self.include_partial else self.session_lengths // self.frame_count).astype(int)
        self.session_indices = np.cumsum(self.session_batch_counts)
        self.length = np.sum(self.session_batch_counts)
        self.test_indices = np.random.default_rng(42).choice(self.length, size=int(self.length * 0.3) // 2 * 2, replace=False, shuffle=False)


    def __len__(self):
        """
        Returns the number of samples in the selected split ('train', 'val', 'test') of the dataset.

        Returns:
            int: Number of samples in the dataset.
        """
        return self.length - self.test_indices.shape[0] if self.split == 'train' else self.test_indices.shape[0] // 2


    def __getitem__(self, index):
        """
        Retrieves the frames and targets for a given index.

        Args:
            index (int): The index of the data to retrieve.

        Returns:
            tuple: A tuple containing:
                - frames (torch.Tensor): The frames corresponding to the given index.
                - targets (torch.Tensor): The targets corresponding to the given index.
        """
        if self.split == 'train':
            index += np.sum(self.test_indices <= index)
        elif self.split == 'test':
            index = self.test_indices[index]
        elif self.split == 'val':
            index = self.test_indices[index + (self.test_indices.shape[0] // 2)]

        session_index = np.sum(self.session_indices <= index)
        frames, targets = self.dataset[session_index]
        session_start = int(self.session_indices[session_index-1]) if session_index > 0 else 0
        frame_index = (index - session_start) * self.frame_count
        if frame_index + self.frame_count <= frames.shape[0]:
            frames = frames[frame_index:frame_index+self.frame_count]
            targets = targets[frame_index:frame_index+self.frame_count]
        else:
            frames = frames[frame_index:]
            frames = torch.cat((frames, torch.zeros((self.frame_count-frames.shape[0], *frames.shape[1:]), dtype=torch.uint8)))
            targets = targets[frame_index:]
            targets = torch.cat((targets, torch.full((self.frame_count-targets.shape[0], *targets.shape[1:]), -1)))

        return frames, targets


class EVEyeDataset(EyeDataset):
    """
    EVEyeDataset is a dataset class for handling eye-tracking data from the EV-Eye dataset.

    Args:
        dataset_path (str): Path to the dataset directory.
        target_gaze3d (bool): Flag indicating whether to target 3D gaze coordinates. Default is False.
    """
    def __init__(self, dataset_path='./EV_Eye_dataset/', target_gaze3d=False, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dataset_path = dataset_path
        self.target_gaze3d = target_gaze3d
        self.target_shape = 3 if self.target_gaze3d else (10 if self.eye == 'both' else 5)

        self.raw_data_path = os.path.join(self.dataset_path, 'raw_data')
        self.data_davis_path = os.path.join(self.raw_data_path, 'Data_davis')
        self.data_gaze_path = os.path.join(self.raw_data_path, 'Data_tobii')
        
        self.num_users = 48 # 48 users, 4 sessions per user
        self.num_sessions = 4


    def __len__(self):
        """
        Returns the total number of sessions in the dataset.

        The total number of sessions is calculated as the product of the number of users and the number of sessions per user.

        Returns:
            int: The total number of sessions in the dataset.
        """
        return self.num_users * self.num_sessions


    def get_session_pattern_index(self, index):
        """
        Calculate the user, user session index, session, and pattern based on the given index.

        Args:
            index (int): The index to be converted.

        Returns:
            tuple: A tuple containing:
                - user (int): The user number.
                - user_session_index (int): The index within the user's sessions.
                - session (int): The session number.
                - pattern (int): The pattern number.
        """
        user = index // self.num_sessions + 1
        user_session_index = index % self.num_sessions
        session = user_session_index // 2 + 1
        pattern = user_session_index % 2 + 1
        return user, user_session_index, session, pattern


    def load_events(self, index, eye):   
        """
        Load and process event data for a given session and eye.

        Parameters:
            index (int): The index of the session pattern.
            eye (str): The eye for which to load events ('left' or 'right').

        Returns:
            numpy.ndarray: A structured numpy array containing the event data with fields:
                - 't' (int32): Timestamp of the event.
                - 'x' (int16): X-coordinate of the event.
                - 'y' (int16): Y-coordinate of the event.
                - 'p' (int8): Polarity of the event.
                - 'eye' (int8): Eye identifier (0 for left, 1 for right).
        """
        user, user_session_index, session, pattern = self.get_session_pattern_index(index)
        events_folder_path = os.path.join(self.data_davis_path, f'user{user}', eye, f'session_{session}_0_{pattern}', 'events')
        events_path = os.path.join(events_folder_path, 'events.txt')
        if not os.path.exists(events_path):
            events_path = os.path.join(events_folder_path, 'events.txt.gz')
        events_dtype = np.dtype([('t', np.int32), ('x', np.int16), ('y', np.int16), ('p', np.int8)])
        events = pd.read_csv(events_path, sep=' ', names=['t', 'x', 'y', 'p'], dtype=events_dtype)
        events['eye'] = 0 if eye == 'left' else 1
        events_dtype = np.dtype([('t', np.int32), ('x', np.int16), ('y', np.int16), ('p', np.int8), ('eye', np.int8)])
        events = np.array(events.to_records(index=False), dtype=events_dtype)
        
        creation_time_path = os.path.join(self.data_davis_path, f'user{user}', eye, 'creation_time.txt')
        gaze_send_path = os.path.join(self.data_gaze_path, f'user{user}', 'tobiisend.txt')
        creation_time = pd.read_csv(creation_time_path, names=['creation_time'])['creation_time'].to_numpy()
        gaze_send = pd.read_csv(gaze_send_path, names=['gaze_send'])['gaze_send'].to_numpy()
        time_offset = (creation_time[user_session_index] - gaze_send[user_session_index]) * 1000000
        events['t'] = events['t'] - events['t'][0] + time_offset
        return events


    def load_gaze(self, index):
        """
        Loads gaze data for a given index and processes it into a structured format.

        Args:
            index (int): The index of the data to load.

        Returns:
            numpy.ndarray: A numpy array containing the processed gaze data.
        """
        user, _, session, pattern = self.get_session_pattern_index(index)
        gaze_path = os.path.join(self.data_gaze_path, f'user{user}', f'session_{session}_0_{pattern}', 'gazedata')
        if not os.path.exists(gaze_path):
            gaze_path = os.path.join(self.data_gaze_path, f'user{user}', f'session_{session}_0_{pattern}', 'gazedata.gz')
        gaze = pd.read_json(gaze_path, lines=True)
        gaze_timestamps = (gaze['timestamp'] * 1000000).astype(np.int32)
        gaze = pd.json_normalize(gaze['data'], max_level=1)
        gaze['t'] = gaze_timestamps
        if self.target_gaze3d:
            gaze = gaze[['gaze3d', 't']]
        elif self.eye == 'both':
            gaze = gaze[['eyeleft.gazeorigin', 'eyeleft.gazedirection', 'eyeright.gazeorigin', 'eyeright.gazedirection', 't']]
        else:
            gaze = gaze[[f'eye{self.eye}.gazeorigin', f'eye{self.eye}.gazedirection', 't']]
        gaze = gaze.dropna(ignore_index=True)
        
        def gaze_to_target(row):
            target = [row['t']]
            if self.target_gaze3d:
                return np.concatenate((target, np.array(row['gaze3d'])))
            if self.eye == 'left' or self.eye == 'both':
                gaze_left_direction = row['eyeleft.gazedirection']
                gaze_left_angles = -np.arctan2(gaze_left_direction[0], gaze_left_direction[2]), np.pi/2 - np.arccos(gaze_left_direction[1])
                target = np.concatenate((target, row['eyeleft.gazeorigin'], gaze_left_angles))
            if self.eye == 'right' or self.eye == 'both':
                gaze_right_direction = row['eyeright.gazedirection']
                gaze_right_angles = -np.arctan2(gaze_right_direction[0], gaze_right_direction[2]), np.pi/2 - np.arccos(gaze_right_direction[1])
                target = np.concatenate((target, row['eyeright.gazeorigin'], gaze_right_angles))
            return target
            
        gaze = np.stack(gaze.apply(gaze_to_target, axis=1))
        return gaze