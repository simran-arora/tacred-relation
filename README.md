Position-aware Attention RNN Model for Relation Extraction with Bootleg Embeddings
=========================

This repo contains the *PyTorch* code for paper [Position-aware Attention and Supervised Data Improve Slot Filling](https://nlp.stanford.edu/pubs/zhang2017tacred.pdf), and incorporates Bootleg embeddings as a feature. 

**The TACRED dataset**: Details on the TAC Relation Extraction Dataset can be found on [this dataset website](https://nlp.stanford.edu/projects/tacred/).

## Requirements

- Python 3 (tested on 3.6.2)
- PyTorch (tested on 1.0.0)
- unzip, wget (for downloading only)

## Preparation

First, download and unzip GloVe vectors from the Stanford website, with:
```
chmod +x download.sh; ./download.sh
```

Then prepare vocabulary and initial word vectors with:
```
python prepare_vocab.py dataset/tacred dataset/vocab --glove_dir dataset/glove
```

This will write vocabulary and word vectors as a numpy matrix into the dir `dataset/vocab`.

## Bootleg Preparation

First run mention extraction on the TACRED data as described in the end2end_ned tutorial. The Bootleg model accepts data in ```.jsonl``` format, and we have provided a script ```convert_to_jsonl.py``` to prepare the data, which for convenience merges the training, dev, and test data into a single output file. Skip this step if you have predefined mentions for your task. 

Using the output file from mention extraction, next obtain the Bootleg contextual entity embeddings and disambiguated entity labels for the TACRED data. Run Bootleg inference on this data as described in the end2end_ned tutorial, to obtain the ```bootleg_embs.npy``` and ```bootleg_labels.jsonl``` outputs. 

For convenience, the steps covered in the end2end_ned tutorial are replicated in the following file: ```tacred_e2ebootleg_ned.py```

Next, add the Bootleg features to the TACRED datasets: we have provided the script ```add_bootleg_features.ipynb``` to do so. 

Once the TACRED data is prepared, prepare the Bootleg entity vocabulary and initial vectors with:
```python prepare_entity_vocab_bootleg.py /path/to/bootleg_embs.npy```

The specific modifications required in the code include adding the Bootleg feature to the dataloader and concatenating the embedding ti the RNN input.  


## Training

Train a position-aware attention RNN model with:
```
python train.py --data_dir dataset/tacred --vocab_dir dataset/vocab --id 00 --info "Position-aware attention model"
```

Use `--topn N` to finetune the top N word vectors only. The script will do the preprocessing automatically (word dropout, entity masking, etc.).

Train an LSTM model with:
```
python train.py --data_dir dataset/tacred --vocab_dir dataset/vocab --no-attn --id 01 --info "LSTM model" 
```

To train with Bootleg embeddings, add the flags: 
the flags to use Bootleg are: 
```
python eval.py saved_models/00 --dataset test --use_ctx_ent 
```

Additionally, prior work suggests that creating a feature for the first token in a span -- for example if an entity is Barack Obama, and I only attach the entity representation at the position Barack, rather than at both Barack and Obama -- the above Bootleg preparation includes a feature to only use the first token and the flag to signal this during training is ```--use_first_ent_span_tok```

Model checkpoints and logs will be saved to `./saved_models/00`.

## Evaluation

Run evaluation on the test set with:
```
python eval.py saved_models/00 --dataset test
```

Again the flags to use Bootleg are: 
```
python eval.py saved_models/00 --dataset test --use_ctx_ent --use_first_ent_span_tok
```

This will use the `best_model.pt` by default. Use `--model checkpoint_epoch_10.pt` to specify a model checkpoint file. Add `--out saved_models/out/test1.pkl` to write model probability output to files (for ensemble, etc.).

## Ensemble

Please see the example script `ensemble.sh`.

## License

All work contained in this package is licensed under the Apache License, Version 2.0. See the included LICENSE file.
