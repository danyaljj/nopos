#!/bin/bash

fairseq-train  /scratch4/danielk/danielk/nopos/data-bin/wikitext-103 \
	--task language_modeling \
	--sample-break-mode complete_doc \
	--skip-invalid-size-inputs-valid-test \
	--ddp-backend=legacy_ddp \
	--keep-best-checkpoints 5 \
	--max-update 30000 \
	--total-num-update 30000 \
	--required-batch-size-multiple 1 \
	--validate-interval-updates 1000 \
	--save-interval-updates 1000 \
	--checkpoint-activations \
	--memory-efficient-fp16 \
	--optimizer adam \
	--adam-betas '(0.9, 0.98)' \
	--weight-decay 0.01 \
	--clip-norm 0.0 \
	--lr-scheduler polynomial_decay \
	--warmup-updates 1000 \
	--lr 0.0002 \
	--criterion cross_entropy \
	--update-freq 2 \
	--max-tokens 12288 \
	--fp16 \
	--arch transformer_lm_gpt_xl \
	--seed 1 \
	--tokens-per-sample 1024 \
	--no-token-positional-embeddings --alibi \
	--save-dir checkpoints/alibi \
	--distributed-world-size 4 \
	--distributed-port 54186 \
	--wandb-project nopos