import numpy as np
import pandas as pd

def jitter(data, sigma=0.01):
    noise = np.random.normal(loc=0, scale=sigma, size=data.shape)
    return data + noise

def scaling(data, sigma=0.1):
    factor = np.random.normal(loc=1.0, scale=sigma, size=(data.shape[0], 1))
    return data * factor

def time_warp(data, sigma=0.2):
    orig_steps = np.arange(data.shape[0])
    random_steps = np.sort(np.random.normal(loc=1.0, scale=sigma, size=data.shape[0]))
    new_steps = np.cumsum(random_steps)
    new_steps = (new_steps / new_steps[-1]) * orig_steps[-1]
    return np.interp(orig_steps, new_steps, data)

def augment_time_series_data(data, augmentations=['jitter', 'scaling', 'time_warp'], n_augmented=5):
    augmented_data = []
    for _ in range(n_augmented):
        aug_data = data.copy()
        for aug in augmentations:
            if aug == 'jitter':
                aug_data = jitter(aug_data)
            elif aug == 'scaling':
                aug_data = scaling(aug_data)
            elif aug == 'time_warp':
                aug_data = time_warp(aug_data)
        augmented_data.append(aug_data)
    return augmented_data
