import os
import sys
import numpy as np
import pickle

datafile = sys.argv[1]
dir = os.path.dirname(datafile)

embs_bootleg = {'ent':{'inname':'ctx_embeddings.npy', 'outname':'ent_embedding.npy', 'outvocab':'ent_vocab.pkl'}}

for k, names_lst in embs_bootleg.items():
    datafile = dir+"/"+names_lst['inname']

    if os.path.isfile(datafile): 
        emb_data = np.load(datafile)

        new_embs = np.zeros((emb_data.shape[0] + 2, emb_data.shape[1]), dtype='float')
        new_embs[2:] = emb_data
        np.save('{DIR}/{out}'.format(DIR=dir,out=names_lst['outname']), new_embs)

        v = [i-2 for i in range(new_embs.shape[0])]
        vocab_file = '{DIR}/{outvoc}'.format(DIR=dir,outvoc=names_lst['outvocab'])

        with open(vocab_file, 'wb') as outfile:
            pickle.dump(v, outfile)
        print("Saved {}".format(k))
