from typing import List, Optional

import keras_tuner as kt
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Sequential


def build_model_window(
    vector_length: int = 33,
    kernel_sizes: List[int] = [3, 3],
    maxpool_sizes: List[Optional[int]] = [3, 5],
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
            directory="supervised_ufo_tuner",
            project_name="supervised_ufo"
            )
    """

    def build(hp):
        with_batch_norm = hp.Boolean("batchnorm")
        activation_conv = hp.Choice("activation_conv", ["relu", "tanh"])
        activation_dense = hp.Choice("activation_dense", ["relu", "tanh"])

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
            directory="supervised_ufo_tuner",
            project_name="supervised_ufo"
            )
    """

    def build(hp):

        with_batch_norm = hp.Boolean("batchnorm")
        activation_conv = hp.Choice("activation_conv", ["relu", "tanh"])
        activation_dense = hp.Choice("activation_dense", ["relu", "tanh"])

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
