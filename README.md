# NoPos
This repository contains the code for the analysis and models discussed in the paper "[Transformer Language Models without Positional Encodings Still Learn Positional Information](https://arxiv.org/abs/2203.16634)".

## Requirements and Installation
This repository is a fork of the [Fairseq] (https://github.com/facebookresearch/fairseq) repository and so has the same requirements.

# NoPos Models
The main models (including 1.3B parameters model) that were trained without position embeddings (NoPos models) are avilable in the following link: https://www.cs.tau.ac.il/~adihaviv1/nopos_models/

### Requirements and Installation

This repository is a fork of the [Fairseq](https://github.com/pytorch/fairseq) repository and so has the same requirements. 

Once you've installed the dependencies, you can install this repository by running:

```bash
pip install --editable .
```

## Datasets
## Canonical setting - wikitext-103 

To download and preprocess the data, run:
```bash
cd examples/language_model/
bash prepare-wikitext-103.sh
cd ../.

# This really just pre-processes the data
TEXT=examples/language_model/wikitext-103
fairseq-preprocess \
    --only-source \
    --trainpref $TEXT/wiki.train.tokens \
    --validpref $TEXT/wiki.valid.tokens \
    --testpref $TEXT/wiki.test.tokens \
    --destdir data-bin/wikitext-103 \
    --workers 20
```

**Note** I have been using `pip3 install numpy==1.23.5` which is just right (not too old, not too new) for the current version of fairseq.

## Large Scale Setting - The Pile
To reconstract The Pile subset data we used for the experiments in the paper see https://github.com/adihaviv/NoPos/tree/main/nopos_experiments/the_pile_construction

The preprocessed dataset can be found in https://www.cs.tau.ac.il/~adihaviv1/nopos_the_pile

### Training and Inference

To train a language model with attention with linear baises (ALiBi), on input sequences with 512 tokens, run:
```bash
python train.py --task language_modeling data-bin/wikitext-103 --save-dir wt103/ --arch transformer_lm_wiki103 --max-update 286000 --lr 1.0 --t-mult 2 --lr-period-updates 270000 --lr-scheduler cosine --lr-shrink 0.75 --warmup-updates 16000 --warmup-init-lr 1e-07 --stop-min-lr 1e-09 --optimizer nag --min-lr 0.0001 --clip-norm 0.1 --criterion adaptive_loss --max-tokens 9216 --update-freq 1 --tokens-per-sample 512 --seed 1 --sample-break-mode none --skip-invalid-size-inputs-valid-test --ddp-backend=legacy_ddp --fp16 --required-batch-size-multiple 1
```

For input sequences larger than 512 (and up to 2048) tokens, just change the --tokens-per-sample.
For input sequences larger than 512 (and up to 2048) tokens, just change the --tokens-per-sample.

To train the model with inputs of length 3072, the --update-freq parameter must be changed to 3 and the --max-tokens parameter must be reduced to 3072 (and --tokens-per-sample must also be set to 3072). 

**If you run out of memory while training:** set --max-tokens to be 0.5 times what it was perviously and set --update-freq to be 2 times what it was previously. This results in a batched computation that is mathematically equivalent to the original command but requires less memory. If that doesn't work, set --max-tokens to be 0.25 times what it was previously and set the --update-freq to be 4 times what it was previously, and so on... 

## Experiment log: 
1. June 17: AilBI + Wiki103 data: https://wandb.ai/clsp/nopos/reports/Pre-training-w-AliBI-weights-on-Wiki103--Vmlldzo0NjcxNTU4 
2. June 18: Absolute + Wiki103 data
3. June 18: Relative + Wiki103 data
4. June 18: None + Wiki103 data
    
## Citation

If you find this work helpful, please cite us
```@article{Haviv2022TransformerLM,
  title={Transformer Language Models without Positional Encodings Still Learn Positional Information},
  author={Adi Haviv and Ori Ram and Ofir Press and Peter Izsak and Omer Levy},
  journal={ArXiv},
  year={2022},
  volume={abs/2203.16634}
}
```

This repo is still improving. for any questions, please email adi.haviv@cs.tau.ac.il
