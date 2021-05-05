# ---- Strings
DATASET_PATH = "./dataset"
SAVED_PATH   = "./saved"
BERT_NAME    = "bert-base-uncased"

ID   = "id"
TEXT  = "text"
LABEL = "label"
HATE_SPEECH = "HS"
LIKELIHOOD_FILENAME = "likelihood.json"

# ---- Numbers
THRESHOLD = 0.7
RANDOM_SEED = 43

# ---- Dictionaries
TARGET_MAP = {"NOT": 0, "OFF": 1}
TRAGET_TRAC1_MAP = {"OAG": "OFF", "CAG": "OFF", "NAG": "NOT"}

COLUMN_MAP = {
    "OLID"   : {"tweet": TEXT, "subtask_a": LABEL},
    "HatEval": {"HS": LABEL},
    "TRAC-1" : [ID, TEXT, LABEL],
}

RARE_WORDS = {"URL": "http"}
SPECIAL_TOKENS = {"PAD": "[PAD]", "SEP": "[SEP]", "CLS": "[CLS]"}

# ---- Lists
PRIMARY_FILES = ["olid_training_v1.0.tsv", "olid_testset_levela.tsv", "olid_labels_levela.csv"]
META_FILES    = ["hateval_train_en.tsv", "track1_en_train.csv"]
USELESS_PUNCS = [":", "_", "...", "…"]
POSSIBLE_METHODS = ["add_likelihood", "add_with_thresholding"]

