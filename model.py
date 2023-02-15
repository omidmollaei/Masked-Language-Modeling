"""
Define the model components here and build an encoder-only transformer model.
"""

import tensorflow as tf
from typing import Any, NewType
from dataclasses import dataclass


#@dataclass
#class Params:
#    d_model: int = field(default=512)

DataClassType = NewType("DataClassType", Any)


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


def scaled_dot_product_attention(query, key, value, mask):
    """
    Calculate the attention weights
    Args:
        query: query vectors . (shape: [batch_size, num_attention_heads, seq_length, depth])
        key: key vectors. (shape: [batch_size, num_attention_heads, seq_length, depth])
        value: value vectors. (shape: [batch_size, num_attention_heads, seq_length, depth])
        mask: attention mask to prevent positions look at padding tokens.
    Returns:
        scaled dot product attention of input vectors .
    """
    matmul_qk = tf.matmul(query, key, transpose_b=True)
    depth = tf.cast(tf.shape(key)[-1], dtype=tf.float32)
    logits = matmul_qk / depth
    if mask is not None:
        logits += mask * -1e9
    attention_weights = tf.nn.softmax(logits, axis=-1)
    output = tf.matmul(attention_weights, value)
    return output


class MultiHeadAttentionLayer(tf.keras.layers.Layer):
    def __init__(self, num_heads: int, d_model: int, **kwargs):
        """
        Args:
            num_heads: number of attention heads.
            d_model: model dimension.
        """
        assert d_model % num_heads == 0
        super(MultiHeadAttentionLayer, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model
        self.depth = self.d_model // self.num_heads
        self.query_dense = tf.keras.layers.Dense(self.d_model)
        self.key_dense = tf.keras.layers.Dense(self.d_model)
        self.value_dense = tf.keras.layers.Dense(self.d_model)
        self.dense = tf.keras.layers.Dense(self.d_model)

    def get_config(self):
        config = super(MultiHeadAttentionLayer, self).get_config()
        config.update({"num_heads": self.num_heads, "d_model": self.d_model})
        return config

    def split_heads(self, inputs: tf.Tensor, batch_size: int):
        inputs = tf.keras.layers.Lambda(
            lambda x: tf.reshape(x, shape=(batch_size, -1, self.num_heads, self.depth))
        )(inputs)

        return tf.keras.layers.Lambda(
            lambda x: tf.transpose(x, perm=[0, 2, 1, 3])
        )(inputs)

    def call(self, inputs: tf.Tensor):
        query, key, value, mask = (
            inputs['query'],
            inputs['key'],
            inputs['value'],
            inputs['mask'],
        )
        batch_size = tf.shape(query)

        # -- linear layers
        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        # -- split heads
        query = self.split_head(query, batch_size)
        key = self.split_head(key, batch_size)
        value = self.split_head(value, batch_size)

        # -- scaled dot-product attention
        scaled_attention = scaled_dot_product_attention(query, key, value, mask)
        scaled_attention = tf.keras.layers.Lambda(
            lambda x: tf.transpose(x, perm=[0, 2, 1, 3])
        )(scaled_attention)

        # -- concatenation of heads
        concat_attention = tf.keras.layers.Lambda(
            lambda x: tf.reshape(
                x, (batch_size, -1, self.d_model)
            )
        )(scaled_attention)

        # -- final linear layer
        outputs = self.dense(concat_attention)
        return outputs


def encoder_layer(params: DataClassType, name: str = "encoder_layer"):
    """
    The main function to build each of the encoder blocks.
    Args:
        params: A dataclass with required fields, containing the config for building the model.
        name: An optional name for this encoder sub-block.
    Returns:
         The built encoder sub-block with keras functional api.
    """
    inputs = tf.keras.layers.Input(shape=(None, params.model_dim), name="inputs")
    padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")

    attention = MultiHeadAttentionLayer(
        num_heads=params.num_attention_heads, d_model=params.model_dim, name="attention"
    )({
        "query": inputs,
        "key": inputs,
        "value": inputs,
        "mask": padding_mask,
    })
    attention = tf.keras.layers.Dropout(params.dropout_rate)(attention)
    attention += tf.cast(inputs, dtype=tf.float32)
    attention = tf.keras.layers.LayerNormalization(epsilon=params.layer_norm_eps)(attention)

    outputs = tf.keras.layers.Dense(params.intermediate_dense_size, activation=params.activation)(attention)
    outputs = tf.keras.layers.Dense(params.model_dim)(outputs)
    outputs = tf.keras.layers.Dropout(params.dropout_rate)(outputs)
    outputs += attention
    outputs = tf.keras.layers.LayerNormalization(epsilon=params.layer_norm_eps)(outputs)
    return tf.keras.Model(inputs=[inputs, padding_mask], outputs=outputs, name=name)
