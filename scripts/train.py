# train.py

"""
Entry point for initiating the self-labeling U-Net training process.
Configures and launches the SelfTrainingEngine.

Example usage:
python train.py \
    --original_data_dir /path/to/original_dataset \
    --output_dir /path/to/training_output \
    --epochs_per_iteration 15 \
    --batch_size 16 \
    --lr 1e-4 \
    --confidence_threshold 0.8
"""

import os
import argparse
import logging
import shutil
from engine import SelfTrainingEngine

def main(args):
    """
    Sets up the environment and starts the training engine.
    """
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        stream=sys.stdout
    )

    # Create output directories and handle checkpoint path
    os.makedirs(args.output_dir, exist_ok=True)
    args.checkpoint_path = os.path.join(args.output_dir, 'checkpoints', 'best_model.h5')
    os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)
    
    # Clean output directories if starting from scratch
    if not args.resume:
        logging.warning(f"Starting fresh. Deleting contents of {args.output_dir}")
        shutil.rmtree(os.path.join(args.output_dir, 'imagesTr'), ignore_errors=True)
        shutil.rmtree(os.path.join(args.output_dir, 'labelsTr'), ignore_errors=True)
        shutil.rmtree(os.path.join(args.output_dir, 'checkpoints'), ignore_errors=True)

    # Initialize and run the engine
    engine = SelfTrainingEngine(args)
    engine.run()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Self-labelling U-Net Training")

    # --- Data and Paths ---
    parser.add_argument('--original_data_dir', type=str, required=True,
                        help="Path to the original dataset with 'imagesTr' and 'labelsTr'.")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Directory to save generated pseudo-labels, checkpoints, and logs.")
    parser.add_argument('--labels_are_2d_slices', action='store_true',
                        help="Set this flag if the ground-truth labels are pre-extracted 2D central slices, not full 3D volumes.")

    # --- Model Hyperparameters ---
    parser.add_argument('--img_width', type=int, default=176, help="Image width.")
    parser.add_argument('--img_height', type=int, default=144, help="Image height.")

    # --- Training Parameters ---
    parser.add_argument('--epochs_per_iteration', type=int, default=20,
                        help="Number of epochs to train in each self-labeling iteration.")
    parser.add_argument('--batch_size', type=int, default=16,
                        help="Batch size per GPU. Total batch size will be this * num_gpus.")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate.")
    parser.add_argument('--confidence_threshold', type=float, default=0.8,
                        help="Confidence threshold for accepting a pseudo-label.")
    parser.add_argument('--resume', action='store_true',
                        help="Resume training from the last state in the output directory.")

    args = parser.parse_args()
    main(args)
