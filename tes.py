import torch
import torch.nn as nn
import numpy as np
# 创建一个张量
data = torch.from_numpy(np.random.randint(1, 30, size=(10, 10)))
data[:, 0] = 1
print(data)

