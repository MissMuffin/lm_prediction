import numpy as np

from config import Config
from language_model.lm_1b import data_utils
from load_model import LoadModel


def _SampleSoftmax(softmax):
  return min(np.sum(np.cumsum(softmax) < np.random.rand()), len(softmax) - 1)

def _SampleModel(prefix_words, vocab):
  """Predict next words using the given prefix words.

  Args:
    prefix_words: Prefix words.
    vocab: Vocabulary. Contains max word chard id length and converts between
        words and ids.
  """
  targets = np.zeros([Config.BATCH_SIZE, Config.NUM_TIMESTEPS], np.int32)
  weights = np.ones([Config.BATCH_SIZE, Config.NUM_TIMESTEPS], np.float32)

  sess, t = LoadModel(Config.pbtxt, Config.ckpt)

  if prefix_words.find('<S>') != 0:
    prefix_words = '<S> ' + prefix_words

  prefix = [vocab.word_to_id(w) for w in prefix_words.split()]
  prefix_char_ids = [vocab.word_to_char_ids(w) for w in prefix_words.split()]

  for _ in range(Config.NUM_SAMPLES):
    inputs = np.zeros([Config.BATCH_SIZE, Config.NUM_TIMESTEPS], np.int32)
    char_ids_inputs = np.zeros([Config.BATCH_SIZE, Config.NUM_TIMESTEPS, vocab.max_word_length], np.int32)
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
        print('%s\n' % sent)

        if (vocab.id_to_word(samples[0]) == '</S>' or
            len(sent) > Config.MAX_SAMPLE_WORDS):
            break 

# Vocabulary containing character-level information.
vocab = data_utils.CharsVocabulary(Config.file_lm_vocab, Config.MAX_WORD_LEN)

_SampleModel("With", vocab)
_SampleModel("With", vocab)
_SampleModel("Check", vocab)
_SampleModel("Check", vocab)
_SampleModel("About", vocab)
_SampleModel("About", vocab)
_SampleModel("We", vocab)
_SampleModel("We", vocab)
_SampleModel("It", vocab)
_SampleModel("It", vocab)

# _SampleModel(" With even more new technologies", vocab)
# _SampleModel("Check back for", vocab)
# _SampleModel("We are aware of", vocab)

