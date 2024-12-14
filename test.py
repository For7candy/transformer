import torch
import math
import numpy as np
import torch.nn as nn

def test_PE():
    max_len = 100
    d_model = 50

    pe = torch.zeros(max_len, d_model)                  # (100,50)
    position = torch.arange(0, max_len).unsqueeze(1)    # (100,1)
    div_term = torch.exp(torch.arange(0, d_model, 2) *
                         -(math.log(200) / d_model))    # (25)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    print(div_term.size())
    print(pe)
    print(type(div_term))


def test_attention():
    q = torch.arange(0, 24).reshape(2, 2, 2, 3)
    k = torch.arange(0, 24).reshape(2, 2, 3, 2)
    scores = torch.matmul(q, k)
    print(scores)

    q2 = np.arange(0, 24).reshape([2, 2, 2, 3])
    k2 = np.arange(0, 8).reshape([2, 2, 2, 1])
    scores = q2*k2
    print(scores)


def test_torch_matmul():
    tensor1 = torch.arange(5)
    tensor2 = torch.arange(10).reshape(5, 2)
    tensor3 = torch.arange(20).reshape(2, 5, 2)
    tensor4 = torch.arange(40).reshape(2, 2, 2, 5)

    print(torch.matmul(tensor1, tensor2))
    # print(torch.matmul(tensor1, tensor2).size())
    print(torch.matmul(tensor2, tensor3).size())
    print(torch.matmul(tensor3, tensor4).size())


def test_zip():
    a = [1, 2, 3]
    b = [4, 5, 6]
    c = [4, 5, 6, 7, 8]
    zipped = zip(a, b)  # 返回一个对象

    a1, a2 = zip(*zip(a, b))
    print(list(a1))
    print(list(a2))


def test_nn_Embedding():
    vocab = 20
    d_model = 3
    lut = nn.Embedding(vocab, d_model)
    x = torch.arange(150).reshape(10, 5, 3)

    # forward
    print(lut)
    print(x)
    print(lut(x))


if __name__ == "__main__":
    test_nn_Embedding()
