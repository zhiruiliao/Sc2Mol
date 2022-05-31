# Sc2Mol 

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
