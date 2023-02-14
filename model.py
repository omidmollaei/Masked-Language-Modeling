"""
Define the model components here and build an encoder-only transformer model.
"""

import tensorflow as tf


def create_padding_mask(x: tf.Tensor):
    mask = tf.cast(tf.math.equal(x, 0), dtype=tf.float32)
    return mask[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(x: tf.Tensor):
    seq_len = tf.shape(x)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(
        tf.ones((seq_len, seq_len), dtype=tf.float32), -1, 0
    )
    padding_mask = create_padding_mask(x)
    return tf.maximum(look_ahead_mask, padding_mask)


class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position: int, d_model: int, **kwargs):
        """
        Sins and cosines positional embedding class.
        Args:
            position: number of input vocab size. (required to build the embedding matrix.)
            d_model: model dimension. (required to build the embedding matrix.)
        """
        super(PositionalEncoding, self).__init__(**kwargs)
        self.position = position
        self.d_model = d_model
        self.pos_encoding = self.positional_encoding(position, d_model)   # example final shape (1, 35000, 512)

    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        config.update({"position": self.position, "d_model": self.d_model})
        return config

    def get_angles(self, position: tf.Tensor, i: tf.Tensor, d_model: tf.Tensor):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / d_model)
        return position * angles

    def positional_encoding(self, position: int, d_model: int):
        angle_rads = self.get_angles(
            position=tf.cast(tf.range(position)[:, tf.newaxis], dtype=tf.float32),
            i=tf.cast(tf.range(d_model)[tf.newaxis, :], dtype=tf.float32),
            d_model=tf.cast(d_model, dtype=tf.float32),
        )

        # apply sin to even index in the array
        sines = tf.math.sin(angle_rads[:, 0::2])
        # apply cos to odd index in the array
        cosines = tf.math.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return pos_encoding

    def call(self, inputs: tf.Tensor):
        return inputs + self.pos_encoding[:, : tf.shape(inputs)[1], :]
