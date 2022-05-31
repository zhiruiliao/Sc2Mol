import argparse
import os
import time
import numpy as np
import tensorflow as tf
import transformer
import utils
import vae
import vaetransformer


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--input_max_len', type=int, default=64)
    parser.add_argument('--target_max_len', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=25)
    
    parser.add_argument('--kl_start', type=float, default=0.001)
    parser.add_argument('--kl_end', type=float, default=0.01)
    parser.add_argument('--kl_interval', type=int, default=5000)
    parser.add_argument('--kl_delta', type=float, default=0.0001)
    parser.add_argument('--kl_warmup', type=int, default=40000)
    
    parser.add_argument('--warmup', type=int, default=10000)
    
    parser.add_argument('--npy_num', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--input_npy', type=str)
    parser.add_argument('--target_npy', type=str)
    
    parser.add_argument('--ckpt_path', type=str, default='ckpt')
    parser.add_argument('--step_count', type=int, default=0)
    parser.add_argument('--batch_log', type=str, default='./batch_log.csv')
    parser.add_argument('--epoch_log', type=str, default='./epoch_log.csv')
    args = parser.parse_args()
    print(args)
    
    learning_rate = utils.CustomLearningRateSchedule(d_model=256, warmup_steps=args.warmup)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)
    
    kl_loss = tf.keras.metrics.Mean(name='kl_loss')
    vae_ce_loss = tf.keras.metrics.Mean(name='vae_ce_loss')
    vae_accuracy = tf.keras.metrics.Mean(name='vae_acc')
    transformer_ce_loss = tf.keras.metrics.Mean(name='transformer_ce_loss')
    
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

    model = vaetransformer.VAETransformer(
        input_max_len=args.input_max_len, input_vocab_size=28,
        target_max_len=args.target_max_len, target_vocab_size=28,
        d_model=512, num_vae_layers=3, vae_kernel_size=3, latent_dim=128, pooling='max',
        num_transformer_layers=3, num_heads=4, dff=2048, dropout_rate=0.1)
    
    checkpoint_path = os.path.join('.', 'checkpoints', args.ckpt_path)
    os.makedirs(checkpoint_path, exist_ok=True)

    ckpt = tf.train.Checkpoint(model=model,
                               optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        print(ckpt_manager.latest_checkpoint)
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print ('Latest checkpoint restored!!', ckpt_manager.latest_checkpoint)

    train_step_signature = [
        tf.TensorSpec(shape=(None, args.input_max_len), dtype=tf.int64),
        tf.TensorSpec(shape=(None, args.target_max_len), dtype=tf.int64),
        tf.TensorSpec(shape=(1, ), dtype=tf.float32)
    ]
    
    
    @tf.function(input_signature=train_step_signature)
    def train_step(inp, tar, gamma_kl):
        """Training function for one step.
        Args:
            inp: Input chain SMILES. shape = `[batch_size, input_length]`.
            tar: Output full SMILES. shape = `[batch_size, target_length]`.
            gamma_kl: Weight for KL-divergence loss.
        Returns:
            None.
        """
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        enc_padding_mask, combined_mask, dec_padding_mask = utils.create_masks(inp, tar_inp)

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z, mid_output, final_output, attention_weights_dict = model(inp,
                                                                                           tar_inp,
                                                                                           enc_padding_mask,
                                                                                           combined_mask,
                                                                                           dec_padding_mask,
                                                                                           training=True)
            
            kl_div = utils.kl_loss_function(z_mean, z_log_var)
            vae_loss = utils.crossentropy_loss_function_prob(inp, mid_output)
            transformer_loss = utils.crossentropy_loss_function_logits(tar_real, final_output)
            loss = gamma_kl * kl_div + vae_loss + transformer_loss
        
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        kl_loss(kl_div)
        vae_ce_loss(vae_loss)
        vae_accuracy(utils.accuracy_function(inp, mid_output))
        transformer_ce_loss(transformer_loss)
        train_loss(loss)
        train_accuracy(utils.accuracy_function(tar_real, final_output))
    
    
    gamma_kl_schedule = utils.CustomLossSchedule(
        args.kl_start,
        args.kl_end,
        args.kl_interval,
        args.kl_delta,
        args.kl_warmup)
    
    step_count = args.step_count
    
    train_batches = utils.CustomDataset(args.npy_num, args.batch_size, args.input_npy, args.target_npy)
    
    for epoch in range(args.epochs):
        begin = time.time()
        
        kl_loss.reset_states()
        vae_ce_loss.reset_states()
        vae_accuracy.reset_states()
        transformer_ce_loss.reset_states()
        train_loss.reset_states()
        train_accuracy.reset_states()
        
        for (batch_i, (inp, tar)) in enumerate(train_batches):
            gamma_kl = gamma_kl_schedule(step_count)
            train_step(inp, tar, gamma_kl)
            step_count += 1
            if (batch_i + 1) % 100 == 0:
                print(time.asctime( time.localtime(time.time()) ))
                print(f'[Epoch: {epoch + 1}] [Batch: {batch_i + 1}] [Step: {step_count}]\n'\
                      f'[KL-gamma: {gamma_kl.numpy()[0]:.4f}] [KL: {kl_loss.result():.4f}]\n'\
                      f'[VAE Accuracy: {vae_accuracy.result():.4f}]\n'\
                      f'[VAE Loss: {vae_ce_loss.result():.4f}] [Transformer Loss: {transformer_ce_loss.result():.4f}]\n'\
                      f'[Total Loss: {train_loss.result():.4f}] [Accuracy: {train_accuracy.result():.4f}]')
                print('-' * 32)
                with open(args.batch_log, 'a') as fbatchlog:
                    print(f'{epoch + 1},{step_count},'\
                          f'{gamma_kl.numpy()[0]:.4f},{kl_loss.result():.4f},'\
                          f'{vae_accuracy.result():.4f},'\
                          f'{vae_ce_loss.result():.4f},{transformer_ce_loss.result():.4f},'\
                          f'{train_loss.result():.4f},{train_accuracy.result():.4f}', file=fbatchlog)
        ckpt_save_path = ckpt_manager.save()
        
        print(f'[Epoch: {epoch + 1}] [Step: {step_count}]\n'\
              f'[KL-gamma: {gamma_kl.numpy()[0]:.4f}] [KL: {kl_loss.result():.4f}]\n'\
              f'[VAE Accuracy: {vae_accuracy.result():.4f}]\n'\
              f'[VAE Loss: {vae_ce_loss.result():.4f}] [Transformer Loss: {transformer_ce_loss.result():.4f}]\n'\
              f'[Total Loss: {train_loss.result():.4f}] [Accuracy: {train_accuracy.result():.4f}]')
        with open(args.epoch_log, 'a') as f_log:
            print(f'{epoch + 1},{step_count},'\
                  f'{gamma_kl.numpy()[0]:.4f},{kl_loss.result():.4f},'\
                  f'{vae_accuracy.result():.4f},'\
                  f'{vae_ce_loss.result():.4f},{transformer_ce_loss.result():.4f},'\
                  f'{train_loss.result():.4f},{train_accuracy.result():.4f}', file=f_log)
        print(f'Time taken for one epoch: {time.time() - begin:.2f} secs')
        print(f'Savedcheckpoint for epoch {epoch + 1} at {ckpt_save_path}\n')
