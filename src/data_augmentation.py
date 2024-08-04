import numpy as np

def jitter(data, sigma=0.01):
    """
    Add random noise to the data (jittering).

    Parameters:
    - data: np.ndarray, input data
    - sigma: float, standard deviation of the Gaussian noise

    Returns:
    - np.ndarray, jittered data
    """
    noise = np.random.normal(loc=0, scale=sigma, size=data.shape)
    return data + noise

def scaling(data, sigma=0.1):
    """
    Scale the data by a random factor.

    Parameters:
    - data: np.ndarray, input data
    - sigma: float, standard deviation of the scaling factor

    Returns:
    - np.ndarray, scaled data
    """
    factor = np.random.normal(loc=1.0, scale=sigma, size=(data.shape[0], 1))
    return data * factor

def time_warp(data, sigma=0.2):
    """
    Apply time warping to the data.

    Parameters:
    - data: np.ndarray, input data
    - sigma: float, standard deviation for the random steps

    Returns:
    - np.ndarray, time-warped data
    """
    orig_steps = np.arange(data.shape[0])
    random_steps = np.sort(np.random.normal(loc=1.0, scale=sigma, size=data.shape[0]))
    new_steps = np.cumsum(random_steps)
    new_steps = (new_steps / new_steps[-1]) * orig_steps[-1]

    if data.ndim == 2:  # If data is 2D (time steps, features)
        return np.array([np.interp(orig_steps, new_steps, data[:, i]) for i in range(data.shape[1])]).T
    elif data.ndim == 1:  # If data is 1D (time steps)
        return np.interp(orig_steps, new_steps, data)
    else:
        raise ValueError("Data should be either 1D or 2D array")

def augment_time_series_data(data, augmentations=['jitter', 'scaling', 'time_warp'], n_augmented=5):
    """
    Generate augmented data for time series using various augmentation techniques.

    Parameters:
    - data: np.ndarray, input data
    - augmentations: list, list of augmentations to apply
    - n_augmented: int, number of augmented copies to generate

    Returns:
    - np.ndarray, augmented data
    """
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
    return np.concatenate(augmented_data, axis=0)
