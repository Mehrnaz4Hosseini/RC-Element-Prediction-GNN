import random
import numpy as np
import torch

def set_seed(seed=42):
    """
    ثابت نگه‌داشتن شرایط تصادفی برای تولید نتایج قابل تکرار
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # برای multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False