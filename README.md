# Siamese-ure

Codes for ECAI2023 paper "Siamese Representation Learning for Unsupervised Relation Extraction".

The backbone of this project follows [SimSiam](https://github.com/facebookresearch/simsiam))[1] and borrows some of the ideas from [SCAN](https://github.com/wvangansbeke/Unsupervised-Classification)[2].


## Requirements

Packages listed below are required.

- Python (tested on 3.8.10)
- CUDA (tested on 11.0)
- [PyTorch](http://pytorch.org/) (tested on 1.7.0)
- [Transformers](https://github.com/huggingface/transformers) (tested on 4.2.2)
- numpy (tested on 1.17.4)
- scipy==1.4.1
- scikit_learn==0.24.1
- tqdm


## Usage
Train the model with:  
```
python main.py --dataset ###(tacred or fewrel) --aug-plus --fix-pred-lr --use-relation-span ./data/tacred/ --save_path save_path
```
