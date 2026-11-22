"""Shared settings used by generate_data.py, train.py and predict.py.

Keeping SIZE/MEAN/STD/CLASSES in one place means the three scripts can never
drift apart. train.py also records them inside the checkpoint, so predict.py
can verify a weights file was produced with the same preprocessing instead of
silently feeding the model wrongly normalised images.
"""

CLASSES = ["circle", "square", "triangle", "star", "rounded_square"]

SIZE = 64
MEAN, STD = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

ARCH = "TinyCNN"
CKPT_VERSION = 1

# What train.py's augmentation is allowed to do. Flips, 90-degree rotations and a
# bounded colour jitter only - nothing that can move a shape outside the frame
# generate_data.py kept it inside. See train_tf() in train.py.
AUG_BRIGHTNESS, AUG_CONTRAST = 0.2, 0.2

TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT = "train", "val", "test"

# Where the data and the weights live. train.py reads train/val from disk every
# epoch and test only once, when nothing that can influence the weights is left
# to decide.
DATA_DIR = "data"
MODEL_PATH = "shape_cnn.pt"

# Defaults for train.py / generate_data.py; every one is overridable by CLI.
SEED = 0
EPOCHS = 20
LR = 3e-3
BATCH_SIZE = 64
N_TRAIN, N_VAL, N_TEST = 2000, 400, 400
WORKERS = 2               # DataLoader worker processes decoding PNGs alongside
                          # the GPU/MPS; 0 loads in the main process
WEIGHT_DECAY = 1e-4       # AdamW (decoupled) weight decay
LABEL_SMOOTHING = 0.05    # cross-entropy label smoothing

# Re-balancing a lopsided training mix (e.g. a fine-tune set where some classes
# have far fewer images). The cap/floor pair and the sampler sit at their off
# values (0, 0, False) for a balanced set like the one generate_data.py writes,
# where plain shuffling already shows every image once per epoch. See train.py.
BALANCED_SAMPLING = False
CAP_TRAIN_IMAGES = 0
FLOOR_TRAIN_IMAGES = 0

# The class acceptance floors are tuned on val to this accuracy and then read
# against test. See train.py.
THRESHOLD_ACC = 0.95