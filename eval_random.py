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
    parser.add_argument('--num_sample', type=int, default=100)
    
    parser.add_argument('--output', type=str, default='eval_random_output.csv')
    parser.add_argument('--ckpt', type=str, default='ckpt/ckpt-25')
    parser.add_argument('--vocab', type=str, default='vocab.txt')
    
    parser.add_argument('--lr_warmup', type=int, default=10000)
    
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--mean', type=float, default=0.0)
    parser.add_argument('--std', type=float, default=1.0)
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
    
    if args.seed is not None:
        np.random.seed(args.seed)
    z = np.random.normal(loc=args.mean, scale=args.std, size=(args.num_sample, 64))
    
    results = []
    for i in range(args.num_sample):
        zi = z[i:i+1, :]
        pred, _ = model.sample_from_gaussian(zi)
        pred = pred.numpy().squeeze()
        results.append(pred)
    fout = open(args.output, 'w')
    print("SMILES", file=fout)
    for i in range(args.num_sample):
        pred = results[i]
        pred = tokenizer.ids_to_chars(pred)
        pred = token_utils.single_to_multi(pred)
        print(pred, file=fout)
    fout.close()
    
