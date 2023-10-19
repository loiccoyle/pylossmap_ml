from typing import List, Optional, Tuple

import keras_tuner as kt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Sequential


def build_model_window(
    vector_length: int = 33,
    kernel_sizes: List[int] = [3, 3],
    maxpool_sizes: List[Optional[int]] = [3, 5],
    regression: bool = False,
):
    """
    Create a tunable rolling window ufo detection model.

    Args:
        vector_length: the legnth of the input vectors, i.e. the number of BLMs
        n_classes: the number of output classes
        kernel_sizes: the size of the convolutional kernels
        maxpool_sizes: the sizes of the maxpool layers

    Returns:
        A tunable model.

    Example:
        Tune model using random search:
        >>> tuner = kt.RandomSearch(
            hypermodel=build_model_rolling_window,
            objective="val_accuracy",
            max_trials=32,
            directory="tuner",
            project_name="supervised_ufo"
            )
    """

    def build(hp):
        with_batch_norm = hp.Boolean("batchnorm")
        activation_conv = "relu"  # hp.Choice("activation_conv", ["relu", "tanh"])
        activation_dense = "relu"  # hp.Choice("activation_dense", ["relu", "tanh"])

        model = keras.models.Sequential()
        for i, (kernal_size, maxpool_size) in enumerate(
            zip(kernel_sizes, maxpool_sizes)
        ):
            if i == 0:
                model.add(
                    layers.Conv1D(
                        filters=hp.Int(
                            f"filters_{i}", min_value=4, max_value=64, step=8
                        ),
                        kernel_size=kernal_size,
                        input_shape=(vector_length, 1),
                    )
                )
            else:
                model.add(
                    layers.Conv1D(
                        filters=hp.Int(
                            f"filters_{i}", min_value=4, max_value=32, step=4
                        ),
                        kernel_size=kernal_size,
                    )
                )
            if with_batch_norm:
                model.add(layers.BatchNormalization())
            model.add(layers.Activation(activation_conv))
            if maxpool_size is not None:
                model.add(layers.MaxPooling1D(maxpool_size))
            model.add(layers.Dropout(hp.Float("dropout", min_value=0, max_value=0.5)))

        model.add(
            layers.Dense(
                units=hp.Int("units", min_value=4, max_value=32, step=4),
                activation=activation_dense,
            )
        )
        model.add(layers.Dense(1, activation="sigmoid"))
        # model.add(layers.Flatten())

        model.summary()
        if regression:
            # MAE, MSE, MRE
            model.compile(optimizer="adam", loss="MAE")
        else:
            model.compile(
                optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
            )
        return model

    return build


def build_model_full_lm(
    vector_length: int = 3595,
    kernel_sizes: List[int] = [64, 32, 16],
    strides: List[int] = [16, 12, 8],
    maxpool_sizes: List[Optional[int]] = [None, None, None],
):
    """
    Create a tunable full lossmap ufo detection model.

    Args:
        vector_length: the legnth of the input vectors, i.e. the number of BLMs
        n_classes: the number of output classes
        kernel_sizes: the size of the convolutional kernels
        strides: the strides of the cnovolution layers
        maxpool_sizes: the sizes of the maxpool layers

    Returns:
        A tunable model.

    Example:
        Tune model using random search:
        >>> tuner = kt.RandomSearch(
            hypermodel=build_model_full_lm,
            objective="val_accuracy",
            max_trials=32,
            directory="tuner",
            project_name="supervised_ufo"
            )
    """

    def build(hp):

        with_batch_norm = hp.Boolean("batchnorm")
        activation_conv = "relu"  # hp.Choice("activation_conv", ["relu", "tanh"])
        activation_dense = "relu"  # hp.Choice("activation_dense", ["relu", "tanh"])

        model = keras.models.Sequential()
        for i, (kernal_size, stride, maxpool_size) in enumerate(
            zip(kernel_sizes, strides, maxpool_sizes)
        ):
            if i == 0:
                model.add(
                    layers.Conv1D(
                        filters=hp.Int(
                            f"filters_{i}", min_value=4, max_value=32, step=8
                        ),
                        kernel_size=kernal_size,
                        strides=stride,
                        input_shape=(vector_length, 1),
                    )
                )
            else:
                model.add(
                    layers.Conv1D(
                        filters=hp.Int(
                            f"filters_{i}", min_value=4, max_value=32, step=8
                        ),
                        kernel_size=kernal_size,
                        strides=stride,
                    )
                )
            if with_batch_norm:
                model.add(layers.BatchNormalization())
            model.add(layers.Activation(activation_conv))
            if maxpool_size is not None:
                model.add(layers.MaxPooling1D(maxpool_size))
            model.add(layers.Dropout(hp.Float("dropout", min_value=0, max_value=0.5)))
        output_dense_units = hp.Int("output_dense", min_value=4, max_value=32, step=4)

        model.add(layers.Dense(units=output_dense_units, activation=activation_dense))
        model.add(layers.Dense(1, activation="sigmoid"))

        model.summary()
        model.compile(
            optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
        )
        return model

    return build


# vision transformer model, from:
# https://keras.io/examples/vision/image_classification_with_vision_transformer
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, 1, self.patch_size, 1],
            strides=[1, 1, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        # embedding here is maybe a problem ?
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


def build_vit_model(
    input_shape: Tuple[int],
    patch_size: int,
    num_patches: int,
    projection_dim: int,
    num_heads: int,
    transformer_layers: int,
    transformer_units: int,
    mlp_head_units: List[int],
    num_classes: int = 1,
):
    """

    learning_rate = 0.001
    weight_decay = 0.0001
    batch_size = 256
    num_epochs = 100
    image_size = 72  # We'll resize input images to this size
    patch_size = 6  # Size of the patches to be extract from the input images
    num_patches = (image_size // patch_size) ** 2
    projection_dim = 64
    num_heads = 4
    transformer_units = [
        projection_dim * 2,
        projection_dim,
    ]  # Size of the transformer layers
    transformer_layers = 8
    mlp_head_units = [2048, 1024]  # Size of the dense layers of the final classifier



    """
    inputs = layers.Input(shape=input_shape)
    # # Augment data.
    # augmented = data_augmentation(inputs)
    # normalized = layers.Normalization()(inputs)
    # Create patches.
    patches = Patches(patch_size)(inputs)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # Classify outputs.
    logits = layers.Dense(num_classes)(features)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model
