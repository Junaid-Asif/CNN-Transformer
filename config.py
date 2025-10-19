import os

# ========= Base Paths =========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
IMAGES_DIR = os.path.join(RAW_DIR, "images")
META_FILE = os.path.join(RAW_DIR, "metadata.csv")

# Output folders
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")
METRICS_DIR = os.path.join(OUTPUT_DIR, "metrics")
GRADCAM_DIR = os.path.join(OUTPUT_DIR, "gradcam")

# Make sure required folders exist
for d in [DATA_DIR, RAW_DIR, PROCESSED_DIR, OUTPUT_DIR, CHECKPOINT_DIR, LOG_DIR, METRICS_DIR, GRADCAM_DIR]:
    os.makedirs(d, exist_ok=True)

# ========= Model & Training Settings =========
SEED = 42
TASK_TYPE = "regression"

IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 25
LEARNING_RATE = 0.0005
OPTIMIZER = "adam"
WEIGHT_DECAY = 0.0001
EARLY_STOPPING_PATIENCE = 10
SCHEDULER = "cosine"
MIXED_PRECISION = True

MODEL_NAME = "custom_cnn"
PRETRAINED = True
FREEZE_BACKBONE = False
DROPOUT = 0.3

# Normalization
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# Augmentation options
AUGMENT = {
    "horizontal_flip": True,
    "vertical_flip": False,
    "rotation_deg": 15,
    "brightness": 0.15,
    "contrast": 0.15,
    "jitter": True,
    "random_erasing": True,
}
