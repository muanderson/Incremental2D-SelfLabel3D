# data_loader.py

"""
Responsible for loading and preprocessing NIfTI files for the self-training
pipeline. Provides functions to create tf.data.Dataset objects for training
and prediction.
"""

import os
import glob
import logging
import numpy as np
import nibabel as nib
import tensorflow as tf
import albumentations as A

def standardize_slice(image_slice):
    """
    Standardizes a single 2D slice to have a mean of 0 and a standard deviation of 1.
    """
    if np.std(image_slice) > 1e-6:  # Avoid division by zero
        return (image_slice - np.mean(image_slice)) / np.std(image_slice)
    return image_slice

def load_nifti_and_process_slices(image_path, mask_path):
    """
    Loads a 3D NIfTI image and mask, and processes them into a list of 2D slices.
    This function is designed to be wrapped by tf.py_function.
    """
    # Decode tensor paths to strings
    image_path = image_path.numpy().decode('utf-8')
    mask_path = mask_path.numpy().decode('utf-8')

    try:
        # Load NIfTI volumes
        img_vol = nib.load(image_path).get_fdata(dtype=np.float32)
        mask_vol = nib.load(mask_path).get_fdata(dtype=np.float32)

        processed_images = []
        processed_masks = []

        for i in range(img_vol.shape[2]):
            img_slice = img_vol[:, :, i]
            mask_slice = mask_vol[:, :, i]

            # Standardize the image slice
            img_slice = standardize_slice(img_slice)
            processed_images.append(img_slice)
            processed_masks.append(mask_slice)

        images = np.stack(processed_images, axis=0)
        masks = np.stack(processed_masks, axis=0)

        # Add channel dimension
        images = np.expand_dims(images, axis=-1)
        masks = np.expand_dims(masks, axis=-1)

        return images.astype(np.float32), masks.astype(np.float32)

    except Exception as e:
        logging.error(f"Error loading or processing {image_path}: {e}")
        return np.array([]), np.array([])


def create_training_dataset(image_dir, mask_dir, batch_size):
    """
    Creates a tf.data.Dataset for training.

    Args:
        image_dir (str): Path to the directory with training images (NIfTI).
        mask_dir (str): Path to the directory with training masks (NIfTI).
        batch_size (int): The batch size for training.

    Returns:
        tf.data.Dataset: A dataset ready for training.
    """
    image_paths = sorted(glob.glob(os.path.join(image_dir, '*.nii.gz')))
    mask_paths = sorted(glob.glob(os.path.join(mask_dir, '*.nii.gz')))

    if not image_paths or not mask_paths:
        raise ValueError("No NIfTI files found in the provided directories.")
    if len(image_paths) != len(mask_paths):
        raise ValueError("The number of images and masks must be the same.")

    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    dataset = dataset.shuffle(buffer_size=len(image_paths))

    def py_func_wrapper(img_path, mask_path):
        return tf.py_function(
            load_nifti_and_process_slices,
            [img_path, mask_path],
            (tf.float32, tf.float32)
        )

    # Use flat_map to flatten the slices from each volume into the dataset
    dataset = dataset.flat_map(
        lambda img, msk: tf.data.Dataset.from_tensor_slices(py_func_wrapper(img, msk))
    )

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset

def create_prediction_dataset(image_path, slice_indices, batch_size):
    """
    Creates a tf.data.Dataset for predicting on specific unlabeled slices.

    Args:
        image_path (str): Path to the NIfTI image volume.
        slice_indices (tuple): A (start, end) tuple for the slices to predict on.
        batch_size (int): Batch size for prediction.

    Returns:
        tf.data.Dataset: A dataset of slices ready for prediction.
    """
    img_vol = nib.load(image_path).get_fdata(dtype=np.float32)
    start, end = slice_indices
    slices_to_predict = img_vol[:, :, start:end]

    processed_slices = []
    for i in range(slices_to_predict.shape[2]):
        img_slice = standardize_slice(slices_to_predict[:, :, i])
        processed_slices.append(img_slice)

    images = np.stack(processed_slices, axis=0)
    images = np.expand_dims(images, axis=-1) # Add channel dimension

    dataset = tf.data.Dataset.from_tensor_slices(images.astype(np.float32))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset