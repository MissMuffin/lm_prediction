import numpy as np
from config import Config

def get_embs_from_npz():
    embs = np.load(Config.filename_emb_trimmed_short.format(300))["embeddings"]
    print("Number of embeddings", len(embs))
    print("Embedding dimension:", len(embs[0]))
    print(embs[0])
    return embs
    
get_embs_from_npz()