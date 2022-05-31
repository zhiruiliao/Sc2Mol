import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import activations
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.python.ops import math_ops


class TokenAndPositionEmbedding(layers.Layer):
    """Embedding Layer for tokens and token index (positions).
    `output = embed_token(inputs) + embed_pos(positions)`
    
    Arguments:
      max_len: an integer, the maximum length of input tensor.
      vocab_size: an integer, the size of vocabulary.
      embed_dim: an integer, the dimension of output tensor.
    
    Inputs:
      X: a 2D tensor with shape: `(batch_size, max_len)`.
    
    Outputs:
      A 3D tensor with shape: `(batch_size, max_len, embed_dim)`.
    """
    def __init__(self, max_len, vocab_size, embed_dim, **kwargs):
        super(TokenAndPositionEmbedding, self).__init__(**kwargs)
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=max_len, output_dim=embed_dim)
        self.max_len = max_len
        self.embed_config = {'max_len': max_len, 'vocab_size': vocab_size, 'embed_dim': embed_dim}
    
    def get_config(self):
        base_config = super(TokenAndPositionEmbedding, self).get_config()
        self.embed_config.update(base_config)
        return self.embed_config
    
    def call(self, inputs):
        positions = tf.range(start=0, limit=self.max_len, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(inputs)
        return x + positions


class GatedConv1D(layers.Layer):
    """Gated Convolutional Layer.
    `output = (X conv W + b) * sigmoid(X conv V + c)`
    where `W` and `V` are convolutional kernels; * is the element-wise product.
    
    Arguments:
      kernel_size: an integer, the size of convolution kernel.
      output_dim: an integer, the dimension of output tensor.
      use_residual: a boolean, if `True`, input will be added to output.
    
    Inputs:
      X: a 3D tensor with shape: `(batch_size, input_len, input_dim)`.
    
    Outputs:
      A 3D tensor with shape: `(batch_size, input_len, output_dim)`.
    """
    def __init__(
        self,
        output_dim,
        kernel_size,
        strides=1,
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        kernel_regularizer=None,
        use_residual=False,
        **kwargs):
        super(GatedConv1D, self).__init__(**kwargs)
        self.conv1d = layers.Conv1D(output_dim * 2, kernel_size, strides=strides, padding=padding,
                                    activation=None, use_bias=use_bias,
                                    kernel_initializer=kernel_initializer, bias_initializer=bias_initializer,
                                    kernel_regularizer=kernel_regularizer)
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.use_residual = use_residual
        self.strides = strides
        self.padding = padding
        self.use_bias = use_bias
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        
        
    def get_config(self):
        base_config = super(GatedConv1D, self).get_config()
        gated_conv1d_config = {
            'output_dim': self.output_dim,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'use_bias': self.use_bias,
            'use_residual': self.use_residual,
            'kernel_initializer': self.kernel_initializer,
            'bias_initializer': self.bias_initializer,
            'kernel_regularizer': self.kernel_regularizer
            }
        gated_conv1d_config.update(base_config)
        return gated_conv1d_config
    
    def call(self, inputs):
        conv_1, conv_2 = tf.split(self.conv1d(inputs), num_or_size_splits=2, axis=-1)
        conv = conv_1 * tf.math.sigmoid(conv_2)
        if self.use_residual:
            return conv + inputs
        else:
            return conv


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the latent vector.
    `output = z_mean + exp(0.5 * z_log_var) * epsilon`
    where `epsilon` is a sample from a standard normal distribution.
    
    Inputs:
      z_mean: a 2D tensor with shape: `(batch_size, input_dim)`.
      z_log_var: a 2D tensor with shape: `(batch_size, input_dim)`.
    
    Outputs:
      A 2D tensor with shape: `(batch_size, input_dim)`.
    """
    
    def __init__(self, **kwargs):
        super(Sampling, self).__init__(**kwargs)
    
    def get_config(self):
        config = {}
        base_config = super(Sampling, self).get_config()
        config.update(base_config)
        return config
    
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch_size = tf.shape(z_mean)[0]
        input_dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch_size, input_dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VariationalEncoder(layers.Layer):
    """Uses gated convolutional layers to encode sequences.
    
    Arguments:
      kernel_size: an integer, the size of convolution kernel.
      hidden_dim: an integer, the dimension of hidden tensor.
      n_layers: an integer, the number of gated conv1d layers.
      latent_dim: an integer, the dimension of output normal distribution.
      pooling: a string, in one of `max` (default) or `average`.
               The method of pooling operation.
    
    Inputs:
      X: a 3D tensor with shape: `(batch_size, input_len, input_dim)`.
    
    Outputs:
      z_mean: a 2D tensor with shape: `(batch_size, latent_dim)`.
      z_log_var: a 2D tensor with shape: `(batch_size, latent_dim)`.
      z_sample: a 2D tensor with shape: `(batch_size, latent_dim)
    """
    def __init__(self, kernel_size, hidden_dim, n_layers, latent_dim, pooling='max', **kwargs):
        super(VariationalEncoder, self).__init__(**kwargs)
        if pooling == 'max':
            self.pooling_layer = layers.GlobalMaxPooling1D()
        elif pooling == 'average':
            self.pooling_layer = layers.GlobalAveragePooling1D()
        else:
            raise ValueError("Unknown pooling method")
        
        self.gated_conv1d_layers = []
        for i in range(n_layers):
            self.gated_conv1d_layers.append(
                GatedConv1D(output_dim=hidden_dim, kernel_size=kernel_size, use_residual=True)
                )
        
        self.dense_mean_log_var = layers.Dense(units=latent_dim * 2)
        self.sampling_layer = Sampling()
        self.kernel_size = kernel_size
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.n_layers = n_layers
        self.pooling = pooling
    
    def get_config(self):
        base_config = super(VariationalEncoder, self).get_config()
        encoder_config = {
            'kernel_size': self.kernel_size,
            'hidden_dim': self.hidden_dim,
            'latent_dim': self.latent_dim,
            'n_layers': self.n_layers,
            'pooling': self.pooling
             }
        encoder_config.update(base_config)
        return encoder_config
    
    def call(self, inputs):
        h = inputs
        for gated_conv1d_layer in self.gated_conv1d_layers:
            h = gated_conv1d_layer(h)
        h = self.pooling_layer(h)
        z_mean, z_log_var = tf.split(self.dense_mean_log_var(h), num_or_size_splits=2, axis=-1)
        z_sample = self.sampling_layer([z_mean, z_log_var])
        return z_mean, z_log_var, z_sample


class VariationalDecoder(layers.Layer):
    """Uses gated convolutional layers to deconde latent vectors.
    
    Arguments:
      kernel_size: an integer, the size of convolution kernel.
      hidden_dim: an integer, the dimension of hidden tensor.
      n_layers: an integer, the number of gated conv1d layers.
      output_len: an integer, the length of output tensor.
      output_dim: an integer, the dimension of output tensor.
    
    Inputs:
      Z: a 2D tensor with shape: `(batch_size, latent_dim)`.
    
    Outputs:
      X_reconstruct: a 3D tensor with shape: `(batch_size, output_len, output_dim)`.
    """
    def __init__(self, kernel_size, hidden_dim, n_layers, output_len, output_dim, **kwargs):
        super(VariationalDecoder, self).__init__(**kwargs)
        self.dense_in = layers.Dense(units=output_len * hidden_dim)
        self.reshape_layer = layers.Reshape(target_shape=(output_len, hidden_dim))
        self.dense_out = layers.Dense(units=output_dim, activation='softmax')
        self.gated_conv1d_layers = []
        for i in range(n_layers):
            self.gated_conv1d_layers.append(
                GatedConv1D(output_dim=hidden_dim, kernel_size=kernel_size, use_residual=True)
                )
        self.kernel_size = kernel_size
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.output_len = output_len
        self.n_layers = n_layers
    
    def get_config(self):
        base_config = super(VariationalDecoder, self).get_config()
        decoder_config = {
            'kernel_size': self.kernel_size,
            'hidden_dim': self.hidden_dim,
            'n_layers': self.n_layers,
            'output_len': self.output_len,
            'output_dim': self.output_dim
             }
        self.decoder_config.update(base_config)
        
        
        return self.decoder_config
    
    def call(self, inputs):
        h = self.dense_in(inputs)
        h = self.reshape_layer(h)
        for gated_conv1d_layer in self.gated_conv1d_layers:
            h = gated_conv1d_layer(h)
        h = self.dense_out(h)
        return h


class VariationalAutoEncoder(keras.Model):
    """Variational Auto Encoder.
    
    In `call`:
      embed = embed_layer(inputs)
      # `(batch_size, input_len)` -> `(batch_size, input_len, embed_dim)`
      
      z_mean, z_log_var, z = self.encoder(embed)
      # `(batch_size, input_len, embed_dim)` -> `(batch_size, latent_dim)`
      
      reconstruction = decoder(z)
      # `(batch_size, latent_dim)` -> `(batch_size, input_len, vocab_size)`
      
    Arguments:
      embed_layer: an embedding layer.
      encoder: an encoder layer.
      decoder: a decoder layer.
     
    Inputs:
      X: a 2D tensor with shape: `(batch_size, input_len)`.
    
    Outputs:
      Z_mean: a 2D tensor with shape: `(batch_size, latent_dim)`.
      Z_log_var: a 2D tensor with shape: `(batch_size, latent_dim)`.
      Z: a 2D tensor with shape: `(batch_size, latent_dim)`.
      X_reconstruct: a 3D tensor with shape: `(batch_size, input_len, vocab_size)`.
    """
    def __init__(self, embed_layer, encoder, decoder, **kwargs):
        super(VariationalAutoEncoder, self).__init__(**kwargs)
        self.embed_layer = embed_layer
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
    
    def get_config(self):
        return {"Embedding layer":self.embed_layer, "Encoder": self.encoder, "Decoder": self.decoder}
    
    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]
    
    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z, reconstruction = self(data)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.sparse_categorical_crossentropy(data, reconstruction)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result()
            }
    
    def call(self, inputs):
        embed = self.embed_layer(inputs)
        z_mean, z_log_var, z = self.encoder(embed)
        reconstruction = self.decoder(z)
        return z_mean, z_log_var, z, reconstruction
