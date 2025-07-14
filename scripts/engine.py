# engine.py

"""
Core training engine for the self-labeling process.
Manages the iterative loop of training, predicting, and expanding the dataset.
"""

import os
import sys
import glob
import logging
import numpy as np
import nibabel as nib
import tensorflow as tf

from model import build_unet_2d
from data_loader import create_training_dataset, create_prediction_dataset

class SelfTrainingEngine:
    """
    Manages the self-training lifecycle.
    """
    def __init__(self, args):
        self.args = args
        self.strategy = tf.distribute.MirroredStrategy()
        logging.info(f"Number of devices: {self.strategy.num_replicas_in_sync}")

        self.state = {}
        self.iteration = 0

    def _initialize_state(self):
        """
        Initialises the state dictionary to track the labeled slice boundaries
        for each volume in the original dataset.
        """
        logging.info("Initializing self-training state...")
        image_files = glob.glob(os.path.join(self.args.original_data_dir, 'imagesTr', '*.nii.gz'))
        if not image_files:
            raise FileNotFoundError(f"No original images found in {self.args.original_data_dir}")

        for img_path in image_files:
            filename = os.path.basename(img_path)
            volume = nib.load(img_path)
            depth = volume.shape[2]
            
            # Start with only the central ground truth slice
            center_slice_idx = depth // 2
            
            self.state[filename] = {
                'path': img_path,
                'affine': volume.affine,
                'header': volume.header,
                'depth': depth,
                'labeled_from': center_slice_idx,
                'labeled_to': center_slice_idx + 1, # Slicing is exclusive at the end
                'is_complete': False
            }
        logging.info(f"State initialized for {len(self.state)} volumes.")


    def _run_training_step(self):
        """
        Runs one cycle of training on the current set of labeled data.
        """
        logging.info(f"--- Starting Training Iteration {self.iteration} ---")
        
        train_image_dir = os.path.join(self.args.output_dir, 'imagesTr')
        train_mask_dir = os.path.join(self.args.output_dir, 'labelsTr')

        # Create the training dataset
        train_dataset = create_training_dataset(
            image_dir=train_image_dir,
            mask_dir=train_mask_dir,
            batch_size=self.args.batch_size * self.strategy.num_replicas_in_sync
        )

        with self.strategy.scope():
            # We rebuild the model and optimizer in each iteration to ensure
            # they are created within the strategy's scope.
            model = build_unet_2d(input_shape=(self.args.img_height, self.args.img_width, 1))
            optimizer = tf.keras.optimizers.Adam(learning_rate=self.args.lr)
            
            # Load weights if a checkpoint exists
            if os.path.exists(self.args.checkpoint_path):
                logging.info(f"Loading weights from {self.args.checkpoint_path}")
                model.load_weights(self.args.checkpoint_path)

            model.compile(
                optimizer=optimizer,
                loss='binary_crossentropy', # Or custom Dice loss
                metrics=['accuracy'] # Add custom Dice metric
            )
        
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                self.args.checkpoint_path, monitor='loss', save_best_only=True,
                save_weights_only=True, mode='min'
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='loss', factor=0.2, patience=5, min_lr=1e-6
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor='loss', patience=10, verbose=1
            )
        ]

        model.fit(
            train_dataset,
            epochs=self.args.epochs_per_iteration,
            callbacks=callbacks,
            verbose=1
        )
        return model

    def _prepare_initial_dataset(self):
        """
        Creates the initial dataset in the output directory.

        If args.labels_are_2d_slices is True, it pairs the central image slice
        with the provided 2D label. Otherwise, it extracts the central slice
        from both the 3D image and 3D label volumes.
        """
        logging.info("Preparing initial dataset with central slices...")
        
        output_img_dir = os.path.join(self.args.output_dir, 'imagesTr')
        output_label_dir = os.path.join(self.args.output_dir, 'labelsTr')
        os.makedirs(output_img_dir, exist_ok=True)
        os.makedirs(output_label_dir, exist_ok=True)
        
        original_label_dir = os.path.join(self.args.original_data_dir, 'labelsTr')

        for filename, data in self.state.items():
            # --- Image Handling (always the same) ---
            img_vol_3d = nib.load(data['path'])
            img_data_3d = img_vol_3d.get_fdata()
            
            # Extract the single central image slice
            start, end = data['labeled_from'], data['labeled_to']
            img_slice_data = img_data_3d[:, :, start:end]
            
            # Create a new NIfTI object for the single image slice
            # We need to update the affine to reflect the new 1-slice depth
            new_affine = img_vol_3d.affine.copy()
            img_slice_nifti = nib.Nifti1Image(img_slice_data, new_affine, data['header'])
            nib.save(img_slice_nifti, os.path.join(output_img_dir, filename))

            # --- Label Handling (conditional) ---
            label_path = os.path.join(original_label_dir, filename)
            
            if self.args.labels_are_2d_slices:
                # SCENARIO 1: Load the provided 2D label directly
                logging.info(f"Loading pre-extracted 2D label for {filename}")
                label_slice_nifti = nib.load(label_path)
                # Ensure the data is shaped correctly as a 1-slice 3D volume
                label_data_2d = label_slice_nifti.get_fdata()
                if label_data_2d.ndim == 2:
                    label_data_3d = np.expand_dims(label_data_2d, axis=2)
                else:
                    label_data_3d = label_data_2d
                
                # Create new NIfTI object with compatible header info
                label_nifti_to_save = nib.Nifti1Image(label_data_3d, new_affine, data['header'])
                
            else:
                # SCENARIO 2: Extract central slice from full 3D label volume
                logging.info(f"Extracting central slice from 3D label for {filename}")
                label_vol_3d = nib.load(label_path)
                label_data_3d = label_vol_3d.get_fdata()
                label_slice_data = label_data_3d[:, :, start:end]
                label_nifti_to_save = nib.Nifti1Image(label_slice_data, new_affine, data['header'])

            nib.save(label_nifti_to_save, os.path.join(output_label_dir, filename))

    def _run_prediction_step(self, model):
        """
        Uses the trained model to predict labels for neighboring slices.
        """
        logging.info(f"--- Starting Prediction & Expansion Step {self.iteration} ---")
        all_volumes_complete = True

        for filename, data in self.state.items():
            if data['is_complete']:
                continue

            all_volumes_complete = False
            
            # 1. Predict slice below the current labeled chunk
            slice_to_predict_below = data['labeled_from'] - 1
            if slice_to_predict_below >= 0:
                pred_dataset = create_prediction_dataset(data['path'], (slice_to_predict_below, data['labeled_from']), self.args.batch_size)
                prediction = model.predict(pred_dataset)
                pseudo_mask = (prediction[0] >= self.args.confidence_threshold).astype(np.uint8)

                # Load current data and prepend
                current_img_vol = nib.load(os.path.join(self.args.output_dir, 'imagesTr', filename)).get_fdata()
                current_mask_vol = nib.load(os.path.join(self.args.output_dir, 'labelsTr', filename)).get_fdata()
                
                slice_from_original = nib.load(data['path']).get_fdata()[:, :, slice_to_predict_below:data['labeled_from']]

                new_img_vol = np.concatenate([slice_from_original, current_img_vol], axis=2)
                new_mask_vol = np.concatenate([pseudo_mask, current_mask_vol], axis=2)
                
                # Update state and save
                self.state[filename]['labeled_from'] -= 1
                nib.save(nib.Nifti1Image(new_img_vol, data['affine'], data['header']), os.path.join(self.args.output_dir, 'imagesTr', filename))
                nib.save(nib.Nifti1Image(new_mask_vol, data['affine'], data['header']), os.path.join(self.args.output_dir, 'labelsTr', filename))

            # 2. Predict slice above the current labeled chunk
            slice_to_predict_above = data['labeled_to']
            if slice_to_predict_above < data['depth']:
                pred_dataset = create_prediction_dataset(data['path'], (slice_to_predict_above, slice_to_predict_above + 1), self.args.batch_size)
                prediction = model.predict(pred_dataset)
                pseudo_mask = (prediction[0] >= self.args.confidence_threshold).astype(np.uint8)

                # Load current data and append
                current_img_vol = nib.load(os.path.join(self.args.output_dir, 'imagesTr', filename)).get_fdata()
                current_mask_vol = nib.load(os.path.join(self.args.output_dir, 'labelsTr', filename)).get_fdata()

                slice_from_original = nib.load(data['path']).get_fdata()[:, :, slice_to_predict_above:slice_to_predict_above + 1]

                new_img_vol = np.concatenate([current_img_vol, slice_from_original], axis=2)
                new_mask_vol = np.concatenate([current_mask_vol, pseudo_mask], axis=2)
                
                # Update state and save
                self.state[filename]['labeled_to'] += 1
                nib.save(nib.Nifti1Image(new_img_vol, data['affine'], data['header']), os.path.join(self.args.output_dir, 'imagesTr', filename))
                nib.save(nib.Nifti1Image(new_mask_vol, data['affine'], data['header']), os.path.join(self.args.output_dir, 'labelsTr', filename))

            # Check if this volume is now fully labeled
            if self.state[filename]['labeled_from'] == 0 and self.state[filename]['labeled_to'] == data['depth']:
                self.state[filename]['is_complete'] = True
                logging.info(f"Volume {filename} is now fully labeled.")
        
        return all_volumes_complete


    def run(self):
        """
        The main entry point to start the self-training process.
        """
        self._initialize_state()
        self._prepare_initial_dataset()

        while True:
            # 1. Train the model on the current dataset
            model = self._run_training_step()

            # 2. Predict on neighbors and expand the dataset
            all_done = self._run_prediction_step(model)
            
            # 3. Check for termination condition
            if all_done:
                logging.info("All volumes have been fully labeled. Self-training complete.")
                break
            
            self.iteration += 1
