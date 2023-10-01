# Siamese-ure

Codes for ECAI2023 paper "Siamese Representation Learning for Unsupervised Relation Extraction".

The backbone of this project follows [SimSiam](https://github.com/facebookresearch/simsiam)[1].


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


## Data
* TACRED: The TAC Relation Extraction Dataset ([download](https://catalog.ldc.upenn.edu/LDC2018T24))<br>
* NYT+FB: This dataset is not open. You can contact ([Diego Marcheggiani](https://diegma.github.io/)).<br>

### Format
Each dataset is a folder under the ```./data``` folder:
```
./data
└── nyt+fb
    ├── train_sentence.json
    ├── train_label_id.json
    ├── dev_sentence.json
    ├── dev_label_id.json
    ├── test_sentence.json
    └── test_label_id.json
└── tacred
    ├── train_sentence.json
    ├── train_label_id.json
    ├── dev_sentence.json
    ├── dev_label_id.json
    ├── test_sentence.json
    └── test_label_id.json
```

## Usage
Train the model with:  
```
python main.py \
--aug-plus \
--fix-pred-lr \
--use-relation-span \
--save_path save_path \
{ path_to_data } ###(tacred or fewrel)
```
