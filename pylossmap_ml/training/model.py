from typing import List, Optional

from tensorflow.keras.layers import Conv1D, Conv1DTranspose, Dropout, Input
from tensorflow.keras.models import Sequential


def create_model(
    sequence_length: int,
    n_features: int = 1,
    activation: str = "relu",
    activation_last_layer: Optional[str] = None,
    encoder_kernel_sizes: List[int] = [15, 10],
    encoder_sizes: List[int] = [16, 32],
    encoder_strides: List[int] = [2, 2],
    encoder_dropout: Optional[float] = None,
    decoder_kernel_sizes: Optional[List[int]] = None,
    decoder_sizes: Optional[List[int]] = [16, 1],
    decoder_strides: Optional[List[int]] = None,
    decoder_dropout: Optional[float] = None,
):
    """Create a Convolutional AutoEncoder anomaly detection model.


    Note:
        The returned model is uncompiled, to compile:
            >>> model.compile(loss="mse", optimizer="adam")

        Original model had a learning rate of 0.001:
            >>> keras.optimizers.Adam(learning_rate=0.001)
    """
    if decoder_sizes is None:
        decoder_sizes = encoder_sizes[::-1]
    if decoder_strides is None:
        decoder_strides = encoder_strides[::-1]
    if decoder_kernel_sizes is None:
        decoder_kernel_sizes = encoder_kernel_sizes[::-1]

    layers = [Input(shape=(sequence_length, n_features))]
    for i, (kernel_size, layer_size, stride) in enumerate(
        zip(encoder_kernel_sizes, encoder_sizes, encoder_strides)
    ):
        encoder_layer = Conv1D(
            filters=layer_size,
            kernel_size=kernel_size,
            padding="same",
            strides=stride,
            activation=activation,
        )
        layers.append(encoder_layer)
        if encoder_dropout is not None and i != len(encoder_sizes) - 1:
            # don't add a dropout after the last encoder layer
            encoder_dropout_layer = Dropout(encoder_dropout)
            layers.append(encoder_dropout_layer)

    for i, (kernel_size, layer_size, stride) in enumerate(
        zip(decoder_kernel_sizes, decoder_sizes, decoder_strides)
    ):
        if i != len(decoder_sizes) - 1:
            activation = activation
        else:
            activation = activation_last_layer

        decoder_layer = Conv1DTranspose(
            filters=layer_size,
            kernel_size=kernel_size,
            padding="same",
            strides=stride,
            activation=activation,
        )
        layers.append(decoder_layer)
        if decoder_dropout is not None and i != len(decoder_sizes) - 1:
            # don't add a dropout after the last decoder layer
            decoder_dropout_layer = Dropout(decoder_dropout)
            layers.append(decoder_dropout_layer)

    # layers.append(
    #     Conv1DTranspose(filters=n_features, kernel_size=7, padding="same"),
    # )
    return Sequential(layers)
