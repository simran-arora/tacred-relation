# Given a file (.jsonl) produced from the bootleg mention extractor, produce the {bootleg_labels.jsonl, bootleg_embs.npy, and 
# static_entity_embs.npy} files. 

import sys
import numpy as np 
import pandas as pd
import ujson
from utils import load_mentions
import logging
from importlib import reload
reload(logging)
logging.basicConfig(stream=sys.stdout, format='%(asctime)s %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
import os
import torch 
use_cpu = False

# FILL IN HERE
data_dir = # fill in path to data
bootleg_model_dir = # fill in path to model directory
bootleg_model_name = # fill in name of model file
bootleg_config_name = # fill in name of config file (e.g., "config.json")
cand_map_name = # fill in name of candidatae map (e.g., "alias2qids.json")
input_dir =  # fill in path to model resources (entity_db, entity_mappings)

cand_map = f'{input_dir}/entity_db/entity_mappings/{cand_map_name}'
outfile_name = 

# MENTION EXTRACTION
infile_name = # FILL IN THE FILE NAME OF THE JSONL TACRED DATA (e.g. 'all_tacred_bootinput.jsonl')
infile = f'{data_dir}{expt_dir}/{infile_name}'
outfile_name = # FILL IN THE OUTPUT FILE NAME FOR TACRED DATA WITH MENTIONS EXTRACTED (e.g. 'all_tacred_w_bootoutput.jsonl')
outfile = f'{data_dir}{expt_dir}/{outfile_name}'
from bootleg.extract_mentions import extract_mentions
extract_mentions(in_filepath=infile, out_filepath=outfile, cand_map_file=cand_map, logger=logger)  

# CONFIGS FOR BOOTLEG INFRENCE
from bootleg import run
from bootleg.utils.parser_utils import get_full_config
config_path = f'{bootleg_model_dir}/{bootleg_config_name}'
config_args = get_full_config(config_path)
config_args.run_config.init_checkpoint = f'{bootleg_model_dir}/{bootleg_model_name}.pt'
config_args.data_config.entity_dir = f'{input_dir}/entity_db'
config_args.data_config.alias_cand_map = cand_map_name 
# set the data path 
config_args.data_config.data_dir = f'{data_dir}{expt_dir}'
config_args.data_config.test_dataset.file = outfile_name
# set the embedding paths 
config_args.data_config.emb_dir =  f'{input_dir}/emb_data'
config_args.data_config.word_embedding.cache_dir =  f'{input_dir}/emb_data'
# set the save directory 
config_args.run_config.save_dir = f'{input_dir}/results'
config_args.run_config.cpu = use_cpu
config_args.run_config.perc_eval = 1.0

# BOOTLEG INFERENCE 
bootleg_label_file, bootleg_emb_file = run.model_eval(args=config_args, mode="dump_embs", logger=logger, is_writer=True)
contextual_entity_embs = np.load(bootleg_emb_file)

