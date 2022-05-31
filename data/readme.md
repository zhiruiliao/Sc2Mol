Firstly please unzip all \*.7z to get \*.txt raw data file.

Then run  
`python token_utils.py --input training_raw.txt --max_len 64 --split 10 --save_path .`
`python token_utils.py --input test_raw.txt --max_len 64 --split 1 --save_path .`
to get preprocessed data.

More details of data can be found at: https://github.com/molecularsets/moses
