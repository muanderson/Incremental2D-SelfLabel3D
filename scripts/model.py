# model.py

"""
Defines the 2D U-Net model architecture using TensorFlow/Keras.
"""

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    UpSampling2D,
    concatenate,
    SpatialDropout2D
)

def build_unet_2d(
    input_shape=(144, 176, 4),
    num_classes=1,
    filters=[8, 16, 32, 64, 128]
):
    """
    Builds a flexible 2D U-Net model.

    Args:
        input_shape (tuple): The shape of the input tensor (height, width, channels).
        num_classes (int): The number of output classes (channels in the final mask).
        filters (list): A list of integers specifying the number of filters at each
                        convolutional block in the contracting path. The expanding
                        path will mirror this.

    Returns:
        tf.keras.Model: A compiled Keras U-Net model.
    """
    if len(filters) < 2:
        raise ValueError("The 'filters' list must have at least two elements.")

    inputs = Input(shape=input_shape)
    skips = []
    x = inputs

    # --- Contracting Path (Encoder) ---
    for f in filters[:-1]:
        x = Conv2D(f, 3, activation='relu', padding='same')(x)
        x = Conv2D(f, 3, activation='relu', padding='same')(x)
        skips.append(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

    # --- Bottleneck ---
    x = Conv2D(filters[-1], 3, activation='relu', padding='same')(x)
    x = Conv2D(filters[-1], 3, activation='relu', padding='same')(x)
    x = SpatialDropout2D(0.5)(x)

    # --- Expanding Path (Decoder) ---
    for f in reversed(filters[:-1]):
        x = Conv2D(f, 2, activation='relu', padding='same')(UpSampling2D(size=(2, 2))(x))
        # Retrieve the corresponding skip connection
        skip_x = skips.pop()
        x = concatenate([skip_x, x], axis=3)
        x = Conv2D(f, 3, activation='relu', padding='same')(x)
        x = Conv2D(f, 3, activation='relu', padding='same')(x)

    # --- Output Layer ---
    activation = 'sigmoid' if num_classes == 1 else 'softmax'
    outputs = Conv2D(num_classes, 1, activation=activation)(x)

    model = Model(inputs=inputs, outputs=outputs, name="U-Net-2D")
    return model

if __name__ == '__main__':
    model = build_unet_2d(input_shape=(256, 256, 1))
    model.summary()