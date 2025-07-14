# evaluate.py

"""
Evaluates a trained model on a test dataset.

Example usage:
python evaluate.py /
    --model_weights /path/to/training_output/checkpoints/best_model.h5 /
    --test_data_dir /path/to/test_dataset /
    --batch_size 32
"""
import os
import argparse
import logging
import numpy as np
import tensorflow as tf
from model import build_unet_2d
from data_loader import create_training_dataset # Re-using this for evaluation is fine

def dice_coefficient(y_true, y_pred, smooth=1e-6):
    """
    Calculates the Dice coefficient for a batch of predictions.
    """
    y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
    y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
    
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

def main(args):
    """
    Loads model, runs evaluation, and prints metrics.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

    # --- Load Model ---
    logging.info("Loading model and weights...")
    model = build_unet_2d(input_shape=(args.img_height, args.img_width, 1))
    model.load_weights(args.model_weights)
    logging.info("Model loaded successfully.")

    # --- Load Test Data ---
    logging.info("Creating test dataset...")
    test_dataset = create_training_dataset(
        image_dir=os.path.join(args.test_data_dir, 'imagesTs'),
        mask_dir=os.path.join(args.test_data_dir, 'labelsTs'),
        batch_size=args.batch_size
    )

    # --- Evaluation Loop ---
    all_dice_scores = []
    logging.info("Starting evaluation...")
    for i, (images, true_masks) in enumerate(test_dataset):
        logging.info(f"Processing batch {i+1}...")
        pred_masks = model.predict(images)
        pred_masks_binary = (pred_masks > 0.5).astype(np.uint8)
        
        dice = dice_coefficient(true_masks, pred_masks_binary)
        all_dice_scores.append(dice.numpy())

    # --- Report Results ---
    mean_dice = np.mean(all_dice_scores)
    std_dice = np.std(all_dice_scores)

    logging.info("--- Evaluation Complete ---")
    logging.info(f"Mean Dice Coefficient: {mean_dice:.4f}")
    logging.info(f"Standard Deviation of Dice: {std_dice:.4f}")
    logging.info(f"Evaluated on {len(all_dice_scores)} batches.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a trained U-Net model.")
    parser.add_argument('--model_weights', type=str, required=True,
                        help="Path to the saved .h5 model weights file.")
    parser.add_argument('--test_data_dir', type=str, required=True,
                        help="Path to the test dataset directory with 'imagesTs' and 'labelsTs'.")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for evaluation.")
    parser.add_argument('--img_width', type=int, default=176, help="Image width.")
    parser.add_argument('--img_height', type=int, default=144, help="Image height.")
    
    args = parser.parse_args()
    main(args)