import os

SAVE_DIR = "saved_models" # Directory to save models
PLOT_DIR = "plots" # Directory to save plots
LOG_DIR = "logs" # Directory to save logs
IMAGES_SAVE_DIR = "reconstructions" # Directory to save images
SMALL_NORB_PATH = os.path.join("datasets", "smallNORB") # Directory to save smallNorb Dataset

# Default values for command arguments
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_ALPHA = 0.0005 # Scaling factor for reconstruction loss
DEFAULT_DATASET = "small_norb" # 'mnist', 'small_norb'
DEFAULT_DECODER = "FC" # 'FC' or 'Conv'
DEFAULT_BATCH_SIZE = 128
DEFAULT_EPOCHS = 300 
DEFAULT_USE_GPU = True
DEFAULT_ROUTING_ITERATIONS = 3
