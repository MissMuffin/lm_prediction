
class Config:

    # For saving demo resources, use batch size 1 and step 1.
    BATCH_SIZE = 1
    NUM_TIMESTEPS = 1
    MAX_WORD_LEN = 50

    # Sampling stops either when </S> is met or this number of steps has passed.
    MAX_SAMPLE_WORDS = 100

    # Number of tokens to generate for the input prefix.
    NUM_SAMPLES = 1

    # File Paths
    ## language model input
    vocab_file = "./language_model/data/vocab-2016-09-10.txt"
    save_dir   = "./output"
    pbtxt      = "./language_model/data/graph-2016-09-10.pbtxt"
    ckpt       = "./language_model/data/ckpt-*"

    ## output
    ## (this ouput has non ASCII char filtered from original lm1b)
    filename_vocab = save_dir + "/lm1b_vocab.txt"
    filename_emb = save_dir + "/lm1b_emb_d{}.npy"
    filename_emb_trimmed = save_dir + "/lm1b_emb_trimmed_d{}.npz"
    filename_emb_text = save_dir + "/lm1b_emb_d{}.txt"

    ### shortened for testing on laptop    
    filename_vocab_short = save_dir + "/lm1b_vocab_short.txt"
    filename_emb_short = save_dir + "/lm1b_emb_d{}_short.npy"
    filename_emb_trimmed_short = save_dir + "/lm1b_emb_trimmed_d{}_short.npz"
    filename_emb_text_short = save_dir + "/lm1b_emb_d{}_short.txt"
