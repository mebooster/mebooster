import os
import os.path as osp
from os.path import dirname, abspath

DEFAULT_SEED = 42 #42 #seed
DS_SEED = 123  # uses this seed when splitting datasets

# -------------- Paths
CONFIG_PATH = abspath(__file__)
SRC_ROOT = dirname(CONFIG_PATH)
PROJECT_ROOT = dirname(SRC_ROOT)
CACHE_ROOT = osp.join(SRC_ROOT, 'cache')
DATASET_ROOT = osp.join(PROJECT_ROOT, 'data')
DEBUG_ROOT = osp.join(PROJECT_ROOT, 'debug')
MODEL_DIR = osp.join(PROJECT_ROOT, 'models')
#VICTIM_DIR = osp.join(MODEL_DIR, 'victim\\cifar-vgg16')
VICTIM_DIR = osp.join(MODEL_DIR, 'victim\\cifar-resnet18')
# VICTIM_DIR = osp.join(MODEL_DIR, 'victim\\mnist-lenet5')
# VICTIM_DIR = osp.join(MODEL_DIR, 'victim\\fashionmnist-lenet5')

# -------------- URLs
ZOO_URL = 'http://datasets.d2.mpi-inf.mpg.de/blackboxchallenge'

# -------------- Dataset Stuff
# IMAGENET_MEAN = [0.485, 0.456, 0.406]
# IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGENET_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STD = [0.5, 0.5, 0.5]

batch_size = 32
epoch = 1 #150
k = 2000
num_iter = 1
over_factor = 5
attack_set = 'maze_ini'

use_default_initial = False#True

start_shadow_test = 0
start_shadow_out_test = 10000

initial_seed = 10000
log_interval = 2000

test_dataset = 'CIFAR10'#'MNIST'#'CIFAR10' #'FashionMNIST'
attack_model_arch ='over_resnet18'#'over_another_lenet'#'over_lenet5'#'over_resnet18'
victim_model_arch ='resnet18'# 'lenet_tl'#'resnet18'
queryset ='EMNIST,MNIST'#'CIFAR10-0,CIFAR100-0,DownSampleImagenet32-50000'#'
attack_model_dir = osp.join(MODEL_DIR, "adversary\\ADV_DIR")
transfer_set_out_dir = osp.join(MODEL_DIR, "adversary\\TRANSFER_SET")
seed = 1337
copy_one_hot = False #use softmax; true: use copy_one_hot

shadow_model_dir = osp.join(MODEL_DIR, "shadow")
sampling_method = ''
ma_method = ''
unsuper_data = 1000
trainingbound = 50000
