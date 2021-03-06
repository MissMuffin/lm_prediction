import numpy as np

from config import Config
from load_model import LoadModel
from sklearn.decomposition import PCA


def load_vocab(file_vocab):
    try:
        with open(file_vocab) as f:
            return {word.strip(): idx for idx, word in enumerate(f)}
    except IOError:
        print("Unable to load vocab from file ", file_vocab)


def build_trimmed_emb(lm_vocab, conll_vocab, file_emb, file_trimmed, dim_in):
    print("Writing embeddings as compressed npz...")

    # load npy
    embeddings = np.load(file_emb)
    print("input dimensions:", embeddings.shape)

    # init trimmed emb array
    trimmed_emb = np.zeros([len(conll_vocab), dim_in])

    for token in lm_vocab:
        if token == Config.UNK:
            pos_conll = conll_vocab["$UNK$"]
            pos_lm = lm_vocab[token]
        elif token in conll_vocab:
            pos_conll = conll_vocab[token]
            pos_lm = lm_vocab[token] 
        else:
            continue

        trimmed_emb[pos_conll] = embeddings[pos_lm]

    # save as compressed npz
    np.savez_compressed(file_trimmed, embeddings=trimmed_emb)
    print("Done. Saved file to", file_trimmed)


def reduce_dimens(dim, embeddings):
    print("Reducing dimensions: intial shape:", embeddings.shape)
    pca = PCA(dim)
    embeddings_reduced = pca.fit_transform(embeddings)
    print("Dimensions after PCA:", embeddings_reduced.shape)
    return embeddings_reduced


def export_reduced_embeddings(file_emb, file_reduced, dim):
    embeddings = np.load(file_emb)["embeddings"]
    embeddings_reduced = reduce_dimens(dim, embeddings)
    np.savez_compressed(file_reduced, embeddings=embeddings_reduced)


# build embedding with 1024 dim
build_trimmed_emb(lm_vocab=load_vocab(Config.filename_vocab),
                  conll_vocab=load_vocab(Config.file_conll_vocab),
                  file_emb=Config.filename_emb_full,
                  file_trimmed=Config.filename_emb_trimmed,
                  dim_in=1024)

d = 50
export_reduced_embeddings(file_emb=Config.filename_emb_trimmed,
                          file_reduced=Config.filename_emb_reduced.format(d),
                          dim=d)

d = 100
export_reduced_embeddings(file_emb=Config.filename_emb_trimmed,
                          file_reduced=Config.filename_emb_reduced.format(d),
                          dim=d)

d = 200
export_reduced_embeddings(file_emb=Config.filename_emb_trimmed,
                          file_reduced=Config.filename_emb_reduced.format(d),
                          dim=d)

d = 300
export_reduced_embeddings(file_emb=Config.filename_emb_trimmed,
                          file_reduced=Config.filename_emb_reduced.format(d),
                          dim=d)

d = 1024
export_reduced_embeddings(file_emb=Config.filename_emb_trimmed,
                          file_reduced=Config.filename_emb_reduced.format(d),
                          dim=d)