import numpy as np
import tensorflow as tf


class CustomLearningRateSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps
        
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        arg_1 = tf.math.rsqrt(step)
        arg_2 = step * (self.warmup_steps ** -1.5)
        lr_1 = tf.math.rsqrt(self.d_model) * tf.math.minimum(arg_1, arg_2)
        lr_2 = 1e-4
        return tf.minimum(lr_1, lr_2)
        
    def get_config(self):
        config = {
            'Model dimension': self.d_model,
            'Warmup steps': self.warmup_steps
        }
        return config


class CustomLossSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, start, end, interval, delta, warmup_steps=4000):
        self.start = start
        self.end = end
        self.interval = interval
        self.delta = delta
        self.warmup_steps = warmup_steps
        
    def __call__(self, step):
        if step < self.warmup_steps:
            return tf.convert_to_tensor([self.start])
        else:
            n = float(int((step - self.warmup_steps) / self.interval) + 1)
            return tf.convert_to_tensor([tf.math.minimum(self.start + n * self.delta, self.end)])
        
        
        
    def get_config(self):
        config = {
            'Start': self.start,
            'End': self.end,
            'Interval': self.interval,
            'Delta': self.delta,
            'Warmup steps': self.warmup_steps
        }
        return config


def crossentropy_loss_function_logits(y_true, y_pred):
    """Compute masked cross entropy loss from logits.
    Args:
        y_true: Ground truth. shape = `[batch_size, seq_length]`.
        y_pred: Prediction probabilities. shape = `[batch_size, seq_length, category_num]`.
    Returns:
        loss: Mean scalar loss value.
    """
    mask = tf.math.logical_not(tf.math.equal(y_true, 0))
    loss_ = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)


def crossentropy_loss_function_prob(y_true, y_pred):
    """Compute masked cross entropy loss from probabilities.
    Args:
        y_true: Ground truth. shape = `[batch_size, seq_length]`.
        y_pred: Prediction probabilities. shape = `[batch_size, seq_length, category_num]`.
    Returns:
        loss: Mean scalar loss value.
    """
    
    loss_ = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=False)
    batch_size, seq_length = tf.shape(y_true)[0], tf.shape(y_true)[1]
    n = tf.cast(batch_size * seq_length, dtype=loss_.dtype)
    
    return tf.reduce_sum(loss_) / n


def kl_loss_function(z_mean, z_log_var):
    kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
    return kl_loss


def accuracy_function(y_true, y_pred):
    accuracies = tf.equal(y_true, tf.argmax(y_pred, axis=2))
    mask = tf.math.logical_not(tf.math.equal(y_true, 0))
    accuracies = tf.math.logical_and(mask, accuracies)

    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies) / tf.reduce_sum(mask)


def create_padding_mask(seq):
    """Create padding mask for input sequences.

    Args:
        seq: Input sequence indices. shape = `[batch_size, seq_length]`.

    Returns:
        mask: Padding mask. shape = `[batch_size, 1, 1, seq_length]`.

    Usage:
        >>> seq = tf.constant([[7, 6, 0, 0, 1], [1, 2, 3, 0, 0]])
        >>> mask = create_padding_mask(seq)
        >>> assert mask.shape == (2, 1, 1, 5)
        >>> mask.numpy()
        array([[[[0., 0., 1., 1., 0.]]],
        <BLANKLINE>
        <BLANKLINE>
               [[[0., 0., 0., 1., 1.]]]], dtype=float32)
    """
    mask = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return mask[:, tf.newaxis, tf.newaxis, :]


def create_look_ahead_mask(seq_length):
    """Create look-ahead mask for a sequence,
    which is used to mask future tokens in a sequence.

        Args:
            seq_length: An integer, the lenghth of the input sequence.

        Returns:
            mask: Look-ahead mask. shape = `[seq_length, seq_length]`.

        Usage:
            >>> seq = tf.constant([[7, 6, 0], [1, 2, 3]])
            >>> mask = create_look_ahead_mask(tf.shape(seq)[1])
            >>> assert mask.shape == (3, 3)
            >>> mask.numpy()
            array([[0., 1., 1.],
                   [0., 0., 1.],
                   [0., 0., 0.]], dtype=float32)
    """
    mask = 1 - tf.linalg.band_part(tf.ones((seq_length, seq_length)), -1, 0)
    return mask


def create_masks(inp, tar):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask


class CustomDataset(object):
    def __init__(self, npy_num, batch_size, input_npy, target_npy):
        self.npy_num = npy_num
        self.batch_size = batch_size
        self.input_npy = input_npy
        self.target_npy = target_npy
    
    def __iter__(self):
        self.npy_i = 0
        self.batch_i = 0
        while self.npy_i < self.npy_num:
            
            inp_numpy = np.load(f"{self.input_npy}_{self.npy_i:03}.npy", allow_pickle=True).astype(np.int64)
            tar_numpy = np.load(f"{self.target_npy}_{self.npy_i:03}.npy", allow_pickle=True).astype(np.int64)
            npy_size = np.shape(inp_numpy)[0]
            
            while self.batch_i * self.batch_size < npy_size:
                inp = inp_numpy[self.batch_i * self.batch_size: (self.batch_i + 1) * self.batch_size]
                tar = tar_numpy[self.batch_i * self.batch_size: (self.batch_i + 1) * self.batch_size]
                inp = tf.convert_to_tensor(inp)
                tar = tf.convert_to_tensor(tar)
                self.batch_i += 1
                yield inp, tar
            
            self.npy_i += 1
            self.batch_i = 0
