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
layer_by_layer = False#True; Not use
attack_set = 'maze_ini'

use_default_initial = False#True#False

start_shadow_test = 0
start_shadow_out_test = 10000

initial_seed = 10000

#shadow model(for a original 10000)
start_shadow = 0
start_shadow_out = 35000
shadow_data = 20000
read_shadow_from_path = False
read_attack_mia_model_from_path = True

log_interval = 2000
imp_vic_mem = False #attack2
imp_vic_mem_bound = False
vic_mem_method = 'shadow_model'#'unsupervised' #shadow modell,

transfer_mi = False
augment_mi = False #only need the first 3000

mi_test = True #test process in shadow model

trainingbound = 50000

# -------------- put config here
test_dataset = 'CIFAR10'#'MNIST'#'CIFAR10' #'FashionMNIST'
attack_model_arch ='over_resnet18'#'over_another_lenet'#'over_lenet5'#'over_resnet18'
victim_model_arch ='resnet18'# 'lenet_tl'#'resnet18'
queryset ='EMNIST,MNIST'#'CIFAR10-0,CIFAR100-0,DownSampleImagenet32-50000'#'
attack_model_dir = osp.join(MODEL_DIR, "adversary\\ADV_DIR")
transfer_set_out_dir = osp.join(MODEL_DIR, "adversary\\TRANSFER_SET")
shadow_model_dir = osp.join(MODEL_DIR, "shadow")
sampling_method = 'membership_attack'#'kcenter' #'membership_attack'#'random' #'kcenter' #uncetainty # adversarial #'membership_attack'
ma_method = ''# bayes, generative, gradient, unsupervised, shadow_model
seed = 1337
copy_one_hot = False #use softmax; true: use copy_one_hot

unsuper_data = 1000