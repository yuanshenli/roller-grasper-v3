import numpy as np
import torch
import random
from utils import log

# seed
np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log(f'device: {device}', 'g', 'B')
