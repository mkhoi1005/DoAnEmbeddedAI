import random 
import numpy as np 
import torch

def set_seed(seed=42):

    """
    Set the random seed for reproducibility.
    
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For GPU

    # Đảm bảo rằng các phép toán trên GPU cũng có thể tái lập được
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"[INFO] seed set to {seed}.")