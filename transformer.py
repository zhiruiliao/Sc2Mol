import logging
import tensorflow as tf
import tensorflow.keras.layers as layers

logging.getLogger('tensorflow').setLevel(logging.ERROR)


def scaled_dot_product_attention(q, k, v, mask=None):
    """Calculate the scaled dot product attention.

        Attention(q, k, v) = softmax(Q @ K.T / sqrt(dimension_k), axis=-1) @ V

    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
         q: Query tensor. shape = `[..., seq_len_q, dimension]`.
         k: Key tensor. shape = `[..., seq_len_k, dimension]`.
         v: Value tensor. shape = `[..., seq_len_v, dimension_v]`.
         mask: Float tensor with shape broadcastable
             to `[..., seq_len_k]`. Defaults to None.

    Returns:
        output: Output tensor. shape = `[..., seq_len_q, dimension_v]`.
        attention_weights: Attention weight tensor. shape = `[..., seq_len_q, seq_len_k]`.
    """
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, v)
    return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),
        tf.keras.layers.Dense(d_model)
    ])


class TransformerEmbedding(layers.Layer):
    """Embedding Layer for tokens and token index (positions).
    `embed_token(inputs) = inputs @ token_embed_matrix`
    `embed_pos(positions) = lookup(pos_embed_matrix, positions)
    `output = embed_token(inputs) + embed_pos(positions)`
    
    Arguments:
      max_len: an integer, the maximum length of input tensor.
      vocab_size: an integer, the size of vocabulary.
      d_model: an integer, the dimension of output tensor.
    
    Inputs:
      X: a 3D tensor with shape: `(batch_size, max_len, input_vocab_size)`, like one-hot.
    
    Outputs:
      A 3D tensor with shape: `(batch_size, max_len, d_model)`.
    """
    def __init__(self, max_len, vocab_size, d_model, **kwargs):
        super(TransformerEmbedding, self).__init__(**kwargs)
        self.token_emb = layers.Dense(units=d_model, activation=None, use_bias=True)
        self.pos_emb = layers.Embedding(input_dim=max_len, output_dim=d_model, input_length=max_len)
        self.max_len = max_len
        self.d_model = d_model 
        self.embed_config = {'max_len': max_len, 'vocab_size': vocab_size, 'd_model': d_model}
    
    def get_config(self):
        base_config = super(TransformerEmbedding, self).get_config()
        self.embed_config.update(base_config)
        return self.embed_config
    
    def call(self, inputs):
        
        seq_len = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=self.max_len, delta=1)
        
        x = self.token_emb(inputs)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_emb(positions)
        return x


class MultiHeadAttention(layers.Layer):
    def __init__(self, d_model, num_heads, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0
        self.inner_dim = d_model // num_heads

        self.wq = layers.Dense(units=d_model)
        self.wk = layers.Dense(units=d_model)
        self.wv = layers.Dense(units=d_model)

        self.dense = layers.Dense(units=d_model)

    def get_config(self):
        config = {
            "Output dimension": self.d_model,
            "Num heads": self.num_heads
        }
        base_config = super(MultiHeadAttention, self).get_config()
        config.update(base_config)
        return config
    
    def split_heads(self, x, batch_size):
        """Split the last dimension into `[num_heads, inner_dim]`,
         and transpose the result such that the shape is `[batch_size, num_heads, seq_len, inner_dim]`
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.inner_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def concatenate_heads(self, x, batch_size):
        """Merge the last dimension into `d_model`,
           and transpose the result such that the shape is `[batch_size, seq_len, d_model]`
        """
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        return tf.reshape(x, (batch_size, -1, self.d_model))

    def call(self, q, k, v, mask, **kwargs):

        batch_size = tf.shape(q)[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        concat_attention = self.concatenate_heads(scaled_attention, batch_size)
        output = self.dense(concat_attention)

        return output, attention_weights


class EncoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1, **kwargs):
        super(EncoderLayer, self).__init__(**kwargs)

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm_1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm_2 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout_1 = layers.Dropout(dropout_rate)
        self.dropout_2 = layers.Dropout(dropout_rate)

        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate

    def get_config(self):
        config = {
            "Output dimension": self.d_model,
            "Num heads": self.num_heads,
            "Feed forward dimension": self.dff,
            "Dropout rate": self.dropout_rate
        }
        base_config = super(EncoderLayer, self).get_config()
        config.update(base_config)
        return config

    def call(self, x, mask, training, **kwargs):

        attn_output, attn_weights_block = self.mha(x, x, x, mask)
        attn_output = self.dropout_1(attn_output, training=training)
        out_1 = self.layernorm_1(x + attn_output)

        ffn_output = self.ffn(out_1)
        ffn_output = self.dropout_2(ffn_output, training=training)
        out_2 = self.layernorm_2(out_1 + ffn_output)

        return out_2


class DecoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1, **kwargs):
        super(DecoderLayer, self).__init__(**kwargs)

        self.mha_1 = MultiHeadAttention(d_model, num_heads)
        self.mha_2 = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm_1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm_2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm_3 = layers.LayerNormalization(epsilon=1e-6)

        self.dropout_1 = layers.Dropout(dropout_rate)
        self.dropout_2 = layers.Dropout(dropout_rate)
        self.dropout_3 = layers.Dropout(dropout_rate)

        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.dropout_rate = dropout_rate

    def get_config(self):
        config = {
            "Output dimension": self.d_model,
            "Num heads": self.num_heads,
            "Feed forward dimension": self.dff,
            "Dropout rate": self.dropout_rate
        }
        base_config = super(DecoderLayer, self).get_config()
        config.update(base_config)
        return config

    def call(self, x, enc_output,
             look_ahead_mask, padding_mask, training, **kwargs):

        attn_1, attn_weights_block_1 = self.mha_1(x, x, x, look_ahead_mask)
        attn_1 = self.dropout_1(attn_1, training=training)
        out_1 = self.layernorm_1(attn_1 + x)

        attn_2, attn_weights_block_2 = self.mha_2(out_1, enc_output, enc_output, padding_mask)
        attn_2 = self.dropout_2(attn_2, training=training)
        out_2 = self.layernorm_2(attn_2 + out_1)

        ffn_output = self.ffn(out_2)
        ffn_output = self.dropout_3(ffn_output, training=training)
        out_3 = self.layernorm_3(ffn_output + out_2)

        return out_3, attn_weights_block_1, attn_weights_block_2


class Encoder(layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 maximum_position_encoding, dropout_rate=0.1, **kwargs):
        super(Encoder, self).__init__(**kwargs)

        self.embedding_layer = TransformerEmbedding(max_len=maximum_position_encoding,
                                                    vocab_size=input_vocab_size,
                                                    d_model=d_model)
        self.dropout = layers.Dropout(dropout_rate)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, dropout_rate)
                           for _ in range(num_layers)]

        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.input_vocab_size = input_vocab_size
        self.maximum_position_encoding = maximum_position_encoding
        self.dropout_rate = dropout_rate

    def get_config(self):
        config = {
            "Num layers": self.num_layers,
            "Output dimension": self.d_model,
            "Num heads": self.num_heads,
            "Feed forward dimension": self.dff,
            "Input vocab size": self.input_vocab_size,
            "Max input length": self.maximum_position_encoding,
            "Dropout rate": self.dropout_rate
        }
        base_config = super(Encoder, self).get_config()
        config.update(base_config)
        return config

    def call(self, x, mask, training, **kwargs):

        x = self.embedding_layer(x)
        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, mask, training)

        return x


class Decoder(layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                 maximum_position_encoding, dropout_rate=0.1, **kwargs):
        super(Decoder, self).__init__(**kwargs)

        self.token_embedding = layers.Embedding(input_dim=target_vocab_size, output_dim=d_model)
        self.position_embedding = layers.Embedding(input_dim=maximum_position_encoding, output_dim=d_model)
        self.dropout = layers.Dropout(dropout_rate)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, dropout_rate)
                           for _ in range(num_layers)]

        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.target_vocab_size = target_vocab_size
        self.maximum_position_encoding = maximum_position_encoding
        self.dropout_rate = dropout_rate

    def get_config(self):
        config = {
            "Num layers": self.num_layers,
            "Output dimension": self.d_model,
            "Num heads": self.num_heads,
            "Feed forward dimension": self.dff,
            "Target vocab size": self.target_vocab_size,
            "Max target length": self.maximum_position_encoding,
            "Dropout rate": self.dropout_rate
        }
        base_config = super(Decoder, self).get_config()
        config.update(base_config)
        return config

    def call(self, x, enc_output, look_ahead_mask, padding_mask,
             training, **kwargs):

        seq_len = tf.shape(x)[1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        attention_weights = {}

        x = self.token_embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.position_embedding(positions)

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, att_block_1, att_block_2 = self.dec_layers[i](x, enc_output,
                                                             look_ahead_mask, padding_mask,
                                                             training)

            attention_weights['decoder_layer_{}_block_1st'.format(i + 1)] = att_block_1
            attention_weights['decoder_layer_{}_block_2nd'.format(i + 1)] = att_block_2

        return x, attention_weights


class Transformer(tf.keras.models.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                 target_vocab_size, max_len_input, max_len_target, dropout_rate=0.1, **kwargs):
        super(Transformer, self).__init__(**kwargs)

        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, max_len_input, dropout_rate)
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, max_len_target, dropout_rate)
        self.final_layer = layers.Dense(target_vocab_size)

        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.input_vocab_size = input_vocab_size
        self.target_vocab_size = target_vocab_size
        self.max_len_input = max_len_input
        self.max_len_target = max_len_target
        self.dropout_rate = dropout_rate

    def get_config(self):
        config = {
            "Num layers": self.num_layers,
            "Model dimension": self.d_model,
            "Num heads": self.num_heads,
            "Feed forward dimension": self.dff,
            "Input vocab size": self.input_vocab_size,
            "Target vocab size": self.target_vocab_size,
            "Max input length": self.max_len_input,
            "Max target length": self.max_len_target,
            "Dropout rate": self.dropout_rate
        }
        return config

    def call(self, inp, tar, enc_padding_mask,
             look_ahead_mask, dec_padding_mask, training, **kwargs):

        enc_output = self.encoder(inp, enc_padding_mask, training)
        dec_output, attention_weights_dict = self.decoder(
            tar, enc_output, look_ahead_mask, dec_padding_mask, training)
        final_output = self.final_layer(dec_output)

        return final_output, attention_weights_dict
