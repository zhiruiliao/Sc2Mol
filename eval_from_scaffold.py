import argparse
import os
import time
import numpy as np
import tensorflow as tf
import transformer
import utils
import vae
import vaetransformer

import token_utils

if __name__ == '__main__':
        
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_len', type=int, default=64)
    parser.add_argument('--num_sample', type=int, default=10)
    
    parser.add_argument('--input', type=str, default='./data/test_raw_chain_000.npy')
    parser.add_argument('--target', type=str, default='./data/test_raw_target_000.npy')
    parser.add_argument('--output', type=str, default='./eval_from_scaffold_output.csv')
    parser.add_argument('--ckpt', type=str, default='./ckpt/ckpt-25')
    parser.add_argument('--vocab', type=str, default='./vocab.txt')
    
    parser.add_argument('--lr_warmup', type=int, default=10000)
    
    args = parser.parse_args()
    print(args)
    learning_rate = utils.CustomLearningRateSchedule(d_model=256, warmup_steps=args.lr_warmup)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                         epsilon=1e-9)
    
    model = vaetransformer.VAETransformer(
        input_max_len=args.max_len, input_vocab_size=28,
        target_max_len=args.max_len, target_vocab_size=28,
        d_model=256, num_vae_layers=3, vae_kernel_size=3, latent_dim=64, pooling='max',
        num_transformer_layers=3, num_heads=4, dff=1024, dropout_rate=0.1)
    
    checkpoint_path = os.path.join('.', 'checkpoints', args.ckpt)
    print(checkpoint_path)
    ckpt = tf.train.Checkpoint(model=model,
                               optimizer=optimizer)
    
    ckpt.restore(checkpoint_path)
    print(f"Checkpoint has been restored from {checkpoint_path}")
    
    tokenizer = token_utils.Tokenizer(max_len=args.max_len, init_vocab_txt=args.vocab)
    print(tokenizer)
    
    
    x = np.load(args.input, allow_pickle=True).astype(np.int64)
    y = np.load(args.target, allow_pickle=True).astype(np.int64)
       
    fout = open(args.output, 'w')
    print("scaffold,target,output", file=fout)
    fout.close()
    
    batch_i = 0
    batch_size = 64  
    
    
    while True:

        results_batch = []
        batch_begin = batch_i * batch_size
        batch_end = (batch_i + 1) * batch_size
        
        if batch_begin >= args.num_sample:
            break
        
        x_batch = x[batch_begin: batch_end]
        y_batch = y[batch_begin: batch_end]
        z_mean, z_log_var, z_batch = model.encode(x_batch)
        del z_mean
        del z_log_var
        z_batch = z_batch.numpy()
        
        np.save(f"z_test/test_z_ckpt25_{batch_i}.npy", z_batch)

        for i in range(len(z_batch)):
            zi = z_batch[i:i+1, :]
            pred, _ = model.sample_from_gaussian(zi)
            pred = pred.numpy().squeeze()
            results_batch.append(pred)
            
        fout = open(args.output, 'a')
        
        for i in range(len(x_batch)):
            scaffold = x_batch[i]
            scaffold = tokenizer.ids_to_chars(scaffold)
            scaffold = token_utils.single_to_multi(scaffold)
            
            tar = y_batch[i]
            tar = tokenizer.ids_to_chars(tar)
            tar = token_utils.single_to_multi(tar)
            
            pred = results_batch[i]
            pred = tokenizer.ids_to_chars(pred)
            pred = token_utils.single_to_multi(pred)
            
            print(f"{scaffold},{tar},{pred}", file=fout)            
            
        fout.close()        
        batch_i += 1
        print(f"Batch {batch_i} done.")


