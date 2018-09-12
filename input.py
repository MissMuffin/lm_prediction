import os
import sys

import numpy as np
import tensorflow as tf
from google.protobuf import text_format
from six.moves import xrange

# From lm_1b
from language_model.lm_1b import data_utils


#-------------------------------------------------------------------------------
# Adopted from lm_1b_eval.py
def LoadModel(gd_file, ckpt_file):
    """Load the model from GraphDef and Checkpoint.
    Args: gd_file: GraphDef proto text file. ckpt_file: TensorFlow Checkpoint file.
    Returns: TensorFlow session and tensors dict."""
    with tf.Graph().as_default():
        #class FastGFile: File I/O wrappers without thread locking.
        with tf.gfile.FastGFile(gd_file, 'r') as f:
            # Py 2: s = f.read().decode()
            s = f.read()
            # Serialized version of Graph
            gd = tf.GraphDef()
            # Merges an ASCII representation of a protocol message into a message.
            text_format.Merge(s, gd)

        tf.logging.info('Recovering Graph %s', gd_file)

        t = {}
        [t['states_init'], t['lstm/lstm_0/control_dependency'],
         t['lstm/lstm_1/control_dependency'], t['softmax_out'], t['class_ids_out'],
         t['class_weights_out'], t['log_perplexity_out'], t['inputs_in'],
         t['targets_in'], t['target_weights_in'], t['char_inputs_in'],
         t['all_embs'], t['softmax_weights'], t['global_step']
        ] = tf.import_graph_def(gd, {}, ['states_init',
                                         'lstm/lstm_0/control_dependency:0',
                                         'lstm/lstm_1/control_dependency:0',
                                         'softmax_out:0',
                                         'class_ids_out:0',
                                         'class_weights_out:0',
                                         'log_perplexity_out:0',
                                         'inputs_in:0',
                                         'targets_in:0',
                                         'target_weights_in:0',
                                         'char_inputs_in:0',
                                         'all_embs_out:0',
                                         'Reshape_3:0',
                                         'global_step:0'], name='')	


        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        sess.run('save/restore_all', {'save/Const:0': ckpt_file})
        sess.run(t['states_init'])

    return sess, t

#-------------------------------------------------------------------------------

# For saving demo resources, use batch size 1 and step 1.
BATCH_SIZE = 1
NUM_TIMESTEPS = 1
MAX_WORD_LEN = 50

'''
FLAGS
'''
# Sampling stops either when </S> is met or this number of steps has passed.
MAX_SAMPLE_WORDS = 100
# Number of samples to generate for the prefix.
NUM_SAMPLES = 1

# File Paths
vocab_file = "./language_model/data/vocab-2016-09-10.txt"
save_dir   = "./output"
pbtxt      = "./language_model/data/graph-2016-09-10.pbtxt"
ckpt       = "./language_model/data/ckpt-*"

# targets  = np.zeros([BATCH_SIZE, NUM_TIMESTEPS], np.int32)
# weights  = np.ones([BATCH_SIZE, NUM_TIMESTEPS], np.float32)
# inputs   = np.zeros([BATCH_SIZE, NUM_TIMESTEPS], np.int32)
# char_ids_inputs = np.zeros([BATCH_SIZE, NUM_TIMESTEPS, vocab.max_word_length], np.int32)

# Recovers the model from protobuf
# sess, t = LoadModel(pbtxt, ckpt)

#-------------------------------------------------------------------------------

def _DumpEmb(vocab):
  """Dump the softmax weights and word embeddings to files.

  Args:
    vocab: Vocabulary. Contains vocabulary size and converts word to ids.
  """
  inputs = np.zeros([BATCH_SIZE, NUM_TIMESTEPS], np.int32)
  targets = np.zeros([BATCH_SIZE, NUM_TIMESTEPS], np.int32)
  weights = np.ones([BATCH_SIZE, NUM_TIMESTEPS], np.float32)

  sess, t = LoadModel(pbtxt, ckpt)

  softmax_weights = sess.run(t['softmax_weights'])
  fname = save_dir + '/embeddings_softmax.npy'
  with tf.gfile.Open(fname, mode='w') as f:
    np.save(f, softmax_weights)
  sys.stderr.write('Finished softmax weights\n')

  approved_vocab_ids = build_filtered_vocab(vocab)
  all_embs = np.zeros([len(approved_vocab_ids), 1024])
  
  print("Starting to collect ",len(approved_vocab_ids), " word embeddings..." )
  counter = 0
  for i in approved_vocab_ids:
    input_dict = {t['inputs_in']: inputs,
                  t['targets_in']: targets,
                  t['target_weights_in']: weights}
    if 'char_inputs_in' in t:
      input_dict[t['char_inputs_in']] = (vocab.word_char_ids[i].reshape([-1, 1, MAX_WORD_LEN]))

    embs = sess.run(t['all_embs'], input_dict)

    all_embs[i, :] = embs
    if (counter+1) % 100 == 0:
      print('Finished word embedding %d/%d - index[%d] %s' % (counter+1, vocab.size, counter, vocab.id_to_word(i)))
    counter += 1
  print("Finished all", len(approved_vocab_ids), "word embeddings")
    
  fname = save_dir + '/embeddings_char_cnn.npy'
  with tf.gfile.Open(fname, mode='w') as f:
    np.save(f, all_embs)
  sys.stderr.write('Embedding file saved\n')

def remove_non_ascii_tokens(vocab):
  filtered_vocab = []
  filtered_vocab_ids = []

  for i in range(vocab.size):
    token = vocab.id_to_word(i)
    try: 
      token.encode('ascii')
      filtered_vocab_ids.append(i)
      filtered_vocab.append(token) 
    except UnicodeEncodeError: 
      continue
  return filtered_vocab_ids, filtered_vocab

def build_filtered_vocab(vocab):
  print("Building filtered vocab...")
  filename = "./output/filtered_vocab.txt"
  filtered_vocab_ids, filtered_vocab = remove_non_ascii_tokens(vocab)
  
  print("Writing filtered vocab in {}...".format(filename))
  with open(filename, "w") as f:
      f.write("\n".join(filtered_vocab))
  print("- done. Wrote {} tokens".format(len(filtered_vocab)))
  return filtered_vocab_ids

    
def _SampleSoftmax(softmax):
  return min(np.sum(np.cumsum(softmax) < np.random.rand()), len(softmax) - 1)


def _SampleModel(prefix_words, vocab):
  """Predict next words using the given prefix words.

  Args:
    prefix_words: Prefix words.
    vocab: Vocabulary. Contains max word chard id length and converts between
        words and ids.
  """
  targets = np.zeros([BATCH_SIZE, NUM_TIMESTEPS], np.int32)
  weights = np.ones([BATCH_SIZE, NUM_TIMESTEPS], np.float32)

  sess, t = LoadModel(pbtxt, ckpt)

  if prefix_words.find('<S>') != 0:
    prefix_words = '<S> ' + prefix_words

  prefix = [vocab.word_to_id(w) for w in prefix_words.split()]
  prefix_char_ids = [vocab.word_to_char_ids(w) for w in prefix_words.split()]

  for _ in xrange(NUM_SAMPLES):
    inputs = np.zeros([BATCH_SIZE, NUM_TIMESTEPS], np.int32)
    char_ids_inputs = np.zeros([BATCH_SIZE, NUM_TIMESTEPS, vocab.max_word_length], np.int32)
    samples = prefix[:]
    char_ids_samples = prefix_char_ids[:]
    sent = ''
    while True:
        inputs[0, 0] = samples[0]
        char_ids_inputs[0, 0, :] = char_ids_samples[0]
        samples = samples[1:]
        char_ids_samples = char_ids_samples[1:]

        softmax = sess.run(t['softmax_out'],
                            feed_dict={t['char_inputs_in']: char_ids_inputs,
                                        t['inputs_in']: inputs,
                                        t['targets_in']: targets,
                                        t['target_weights_in']: weights})

        sample = _SampleSoftmax(softmax[0])
        sample_char_ids = vocab.word_to_char_ids(vocab.id_to_word(sample))

        if not samples:
            samples = [sample]
            char_ids_samples = [sample_char_ids]
        sent += vocab.id_to_word(samples[0]) + ' '
        sys.stderr.write('%s\n' % sent)

        if (vocab.id_to_word(samples[0]) == '</S>' or
            len(sent) > MAX_SAMPLE_WORDS):
            break 

#Vocabulary containing character-level information.
vocab = data_utils.CharsVocabulary(vocab_file, MAX_WORD_LEN)

# _SampleModel("With", vocab)
# _SampleModel("Check", vocab)
# _SampleModel("About", vocab)
# _SampleModel("We", vocab)
# _SampleModel("It", vocab)

_DumpEmb(vocab)