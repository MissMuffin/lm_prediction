import os
import sys

import numpy as np
import tensorflow as tf
from google.protobuf import text_format

from config import Config
from language_model.lm_1b import data_utils
from load_model import LoadModel


def remove_non_ascii_tokens(vocab, vocab_length=-1):
    filtered_vocab = []
    filtered_vocab_ids = []

    if vocab_length != -1:
        vsize = vocab_length
    else:
        vsize = vocab.size

    for i in range(vsize):
        token = vocab.id_to_word(i)
        try:
            token.encode('ascii')
            filtered_vocab_ids.append(i)
            filtered_vocab.append(token)
        except UnicodeEncodeError:
            continue
    return filtered_vocab_ids, filtered_vocab


def build_filtered_vocab(vocab, vocab_length=-1):

    # set vocab output filename
    if vocab_length != -1:
        filename = Config.filename_vocab_short
    else:
        filename = Config.filename_vocab

    # remove non ASCII chars from vocab
    print("Building filtered vocab...")
    filtered_vocab_ids, filtered_vocab = remove_non_ascii_tokens(vocab, vocab_length)

    # write vocab to file
    print("Writing filtered vocab in {}...".format(filename))
    with open(filename, "w") as f:
        f.write("\n".join(filtered_vocab))
    print("- done. Wrote {} tokens".format(len(filtered_vocab)))

    return filtered_vocab_ids


def dump_lm(vocab, vocab_length=-1, dump_as_txt=False, dump_softmax=False, print_emb_status_every=100):

    '''
    Save softmax, fitlered vocab and embeddings to file 

    Args:
      vocab: Contains vocabulary size and converts word to ids
      vocab_length: shorten the filtered vocab (also shortens embs)
      dump_as_txt: save language model embeddings also as txt file
      dum_softmax: save softmax to npy file
      print_emb_status_every: prints a status msg every n token during embeddings loading
    '''

    inputs = np.zeros([Config.BATCH_SIZE, Config.NUM_TIMESTEPS], np.int32)
    targets = np.zeros([Config.BATCH_SIZE, Config.NUM_TIMESTEPS], np.int32)
    weights = np.ones([Config.BATCH_SIZE, Config.NUM_TIMESTEPS], np.float32)

    sess, t = LoadModel(Config.pbtxt, Config.ckpt)

    dim = Config.emb_dim

    # dump softmax to file
    if dump_softmax:
        dump_softmax(sess, t, weights)

    # create filtered vocab
    vocab_ids = build_filtered_vocab(vocab, vocab_length)

    # shorten vocab
    if vocab_length != -1:
        vocab_ids = vocab_ids[:vocab_length]

    # set output files
    if vocab_length != -1:
        filename_emb_npy = Config.filename_emb_short.format(dim)
        filename_emb_text = Config.filename_emb_text_short.format(dim)
    else:
        filename_emb_npy = Config.filename_emb.format(dim)
        filename_emb_text = Config.filename_emb_text.format(dim)

    # init embeddings tensor
    all_embs = np.zeros([len(vocab_ids), dim])

    # collect embeddings tensor for each token in vocab
    print("Starting to collect ", len(vocab_ids), " word embeddings...")
    for i, word_id in enumerate(vocab_ids):
        input_dict = {t['inputs_in']: inputs,
                      t['targets_in']: targets,
                      t['target_weights_in']: weights}
        if 'char_inputs_in' in t:
            input_dict[t['char_inputs_in']] = (
                vocab.word_char_ids[word_id].reshape([-1, 1, Config.MAX_WORD_LEN]))

        embs = sess.run(t['all_embs'], input_dict)
        all_embs[i, :] = embs

        if print_emb_status_every != -1:
            if (i+1) % print_emb_status_every == 0:
                print('Finished word embedding %d/%d - index[%d] %s' % (
                    i+1, len(vocab_ids), i, vocab.id_to_word(word_id)))

    print("Finished all", len(vocab_ids), "word embeddings")

    # write embeddings to compressed npz file
    np.savez_compressed(filename_emb_npy, embeddings=all_embs)
    print('Embeddings saved to npy file.')

    # write embeddings to txt file
    if dump_as_txt:
        np.savetxt(filename_emb_text, all_embs)
        print('Embeddings saved to txt file.')


def dump_softmax(sess, t, weights):
    softmax_weights = sess.run(t['softmax_weights'])
    fname = Config.save_dir + '/embeddings_softmax.npy'
    with tf.gfile.Open(fname, mode='w') as f:
        np.save(f, softmax_weights)
    print('Finished writing softmax weights to file.')


# Vocabulary containing character-level information.
vocab = data_utils.CharsVocabulary(Config.vocab_file, Config.MAX_WORD_LEN)

dump_lm(vocab, 
        vocab_length=10, 
        dump_as_txt=True,
        dump_softmax=False, 
        print_emb_status_every=6)
