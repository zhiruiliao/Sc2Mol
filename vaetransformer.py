import numpy as np
import tensorflow as tf
import transformer
import utils
import vae

class VAETransformer(tf.keras.models.Model):
    def __init__(self,
                 input_max_len, input_vocab_size,
                 target_max_len, target_vocab_size,
                 d_model, num_vae_layers, vae_kernel_size, latent_dim, pooling,
                 num_transformer_layers, num_heads, dff, dropout_rate=0.1,
                 **kwargs):
        super(VAETransformer, self).__init__(**kwargs)
        
        self.vae_embedding = vae.TokenAndPositionEmbedding(
            max_len=input_max_len,
            vocab_size=input_vocab_size,
            embed_dim=d_model)
        
        self.vae_encoder = vae.VariationalEncoder(
            kernel_size=vae_kernel_size,
            hidden_dim=d_model,
            n_layers=num_vae_layers,
            latent_dim=latent_dim,
            pooling=pooling)

        self.vae_decoder = vae.VariationalDecoder(
            kernel_size=vae_kernel_size,
            hidden_dim=d_model,
            n_layers=num_vae_layers,
            output_len=input_max_len,
            output_dim=input_vocab_size)
        
        self.transformer_encoder = transformer.Encoder(
            num_layers=num_transformer_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            input_vocab_size=input_vocab_size,
            maximum_position_encoding=input_max_len,
            dropout_rate=dropout_rate)
        
        self.transformer_decoder = transformer.Decoder(
            num_layers=num_transformer_layers,
            d_model=d_model,
            num_heads=num_heads,
            dff=dff,
            target_vocab_size=target_vocab_size,
            maximum_position_encoding=target_max_len,
            dropout_rate=dropout_rate)
        
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)
        self.input_vocab_size = input_vocab_size
        self.target_vocab_size = target_vocab_size
        self.target_max_len = target_max_len
        
        self.config = {
            'input_max_len': input_max_len,
            'input_vocab_size': input_vocab_size,
            'target_max_len': target_max_len,
            'target_vocab_size': target_vocab_size,
            'd_model': d_model,
            'num_vae_layers': num_vae_layers,
            'vae_kernel_size': vae_kernel_size,
            'latent_dim': latent_dim,
            'pooling': pooling,
            'num_transformer_layers': num_transformer_layers, 
            'num_heads': num_heads,
            'dff': dff,
            'dropout_rate': dropout_rate
            }
    
    def get_config(self):
        return self.config
    
    def call(self, inp, tar, enc_padding_mask, look_ahead_mask, dec_padding_mask, training, **kwargs):
        """
        Input:
            inp: Input chain SMILES. shape = `[batch_size, input_length]`.
            tar: Output full SMILES. shape = `[batch_size, target_length]`.
            enc_padding_mask: Input chain mask. shape = `[batch_size, 1, 1, input_length]`.
            look_ahead_mask: Look-ahead mask. shape = `[batch_size, 1, target_length, target_length]`.
            dec_padding_mask: Identical to input chain mask. shape = `[batch_size, 1, 1, input_length]`.
        
        Output:
            z_mean: Mean of latent vector. shape = `[batch_size, latent_dim]`
            z_log_var: Logarithmic variance of latent vector. shape = `[batch_size, latent_dim]`
            z: Latent vector in VAE. shape = `[batch_size, latent_dim]`
            mid_output: Reconstructed chain SMILES generated by VAE.
                shape = `[batch_size, input_length, input_vocab_size]`.
            final_output: Full SMILES generated by model.
                shape = `[batch_size, target_length, target_vocab_size]`.
            attention_weights_dict: Dictionary of atention weights in transformer decoder.
        """
    
        # `(batch_size, input_length)` -> `(batch_size, input_length, d_model)`
        vae_embed = self.vae_embedding(inp)
        
        # `(batch_size, input_length, d_model)` -> `(batch_size, latent_dim)`
        z_mean, z_log_var, z = self.vae_encoder(vae_embed)
        
        # `(batch_size, latent_dim)` -> `(batch_size, mid_len, input_vocab_size)`
        mid_output = self.vae_decoder(z)

        # TODO: gumbel reparameterization trick
        inp_one_hot = tf.one_hot(inp, depth=self.input_vocab_size)
        
        # `(batch_size, mid_len, input_vocab_size)` -> `(batch_size, mid_len, d_model)`
        enc_output = self.transformer_encoder(inp_one_hot, enc_padding_mask, training)
        
        dec_output, attention_weights_dict = self.transformer_decoder(
            tar, enc_output, look_ahead_mask, dec_padding_mask, training)
        final_output = self.final_layer(dec_output)
        
        return z_mean, z_log_var, z, mid_output, final_output, attention_weights_dict
    
    def encode(self, inp):
        vae_embed = self.vae_embedding(inp)
        z_mean, z_log_var, z = self.vae_encoder(vae_embed)
        return z_mean, z_log_var, z
    
    def run_vae(self, inp):
        vae_embed = self.vae_embedding(inp)
        z_mean, z_log_var, z = self.vae_encoder(vae_embed)
        mid_output = self.vae_decoder(z)
        return z_mean, z_log_var, z, mid_output
    
    def sample_from_gaussian(self, z):
        "Congratulations! Batch sampling now has been supported."
        batch_size = tf.shape(z)[0]
        vae_output = self.vae_decoder(z)
        
        inp = tf.argmax(vae_output, axis=-1)
        
        enc_padding_mask = utils.create_padding_mask(inp)
        dec_padding_mask = utils.create_padding_mask(inp)
        
        # `[batch_size, input_length, input_vocab_size]`
        inp_one_hot = tf.one_hot(inp, depth=self.input_vocab_size)
        
        enc_output = self.transformer_encoder(inp_one_hot, enc_padding_mask, training=False)
        # start is [SOS] (1)
        # end is [EOS] (2)
        # `[batch_size, current_length]`
        output = tf.ones([batch_size, 1], dtype=tf.int64)
        for i in range(self.target_max_len):
            look_ahead_mask = utils.create_look_ahead_mask(tf.shape(output)[1])
            dec_target_padding_mask = utils.create_padding_mask(output)
            look_ahead_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

            dec_output, attention_weights_dict = self.transformer_decoder(
            output, enc_output, look_ahead_mask, dec_padding_mask, training=False)
            
            predictions = self.final_layer(dec_output)
            predictions = predictions[:, -1:, :]           
            predicted_id = tf.argmax(predictions, axis=-1)            
            output = tf.concat([output, predicted_id], axis=-1)

        return output, attention_weights_dict
