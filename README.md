# Sc2Mol: A Scaffold-based Two-step Molecule Generator with Variational Autoencoder and Transformer
## Recent update
2024-01-10: Fix a bug in `transformer.py`, which may cause error when `get_config` method is called.  
2023-09-11:  Code is modified to support batch random generation. Much more faster!

## Requirements
python 3.7  
tensorflow-gpu >= 2.3.0  
rdkit >= 2020.03.2.0  
pandas >= 1.0.3  
numpy >= 1.19.2 

## Data
See `data` fold to get data and preprocess data.

## Training
`python training.py --input_npy ./data/training_raw_chain --target_npy ./data/training_raw_target --npy_num 10`  
More arguments setting can be found and changed in code file `training.py`.  

## Evaluating (random generation)
`python eval_random.py --num_sample NUM_SAMPLE --output OUTPUT --ckpt CHECKPOINT`  
More arguments setting can be found and changed in code file `eval_random.py`.  

## Evaluating (given scaffold)
`python eval_from_scaffold.py --input INPUT_NPY --target TARGET_NPY --num_sample NUM_SAMPLE --output OUTPUT_FILE --ckpt CHECKPOINT_PATH`  
More arguments setting can be found and changed in code file `eval_from_scaffold.py`.  

## Citation
Zhirui Liao, Lei Xie, Hiroshi Mamitsuka, Shanfeng Zhu, Sc2Mol: a scaffold-based two-step molecule generator with variational autoencoder and transformer, Bioinformatics, Volume 39, Issue 1, January 2023, btac814, https://doi.org/10.1093/bioinformatics/btac814

## Contacts and bug reports
Please feel free to send bug reports or questions to Zhirui Liao: zrliao19@fudan.edu.cn and Prof. Shanfeng Zhu: zhusf@fudan.edu.cn

## Declaration
It is free for non-commercial use. For commercial use, please contact with Zhirui Liao and Prof. Shanfeng Zhu (zhusf@fudan.edu.cn).
