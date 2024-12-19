from .net import subsequent_mask
from torch.autograd import Variable
import torch


class Batch:
    def __init__(self, src, tgt=None, pad=0):
        """
        "Object for holding a batch of data with mask during training."
        处理batch， 使它送入时带有mask
        对于src，只需要作padding mask即可
        对于tgt，同时需要padding mask 和 sequence mask
        :param src: 源语言序列，  (batch_size, src.seq.len)
        :param tgt: 目标语言序列, (batch_size, tgt.seq.len)
                    默认为None是因为在自注意力中没有tgt
        :param pad: mask的填充值
        """
        self.src = src
        # (batch_size, len) -> (batch_size, 1, len)
        self.src_mask = (src != pad).unsqueeze(-2)

        if tgt is not None:
            # 取tgt除最后一个词外的所有
            self.tgt = tgt[:, :-1]
            # 取tgt除第一个词外的所有
            self.tgt_y = tgt[:, 1:]
            # 得到padding和sequence mask
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        # tgt != pad 会返回一个和tgt相同大小的tensor,内容为bool值
        # unsqueeze插入第二维,(30,10) -> (30,1,10)
        # subsequent_mask()得到一个tgt.size*tgt.size大小的下三角矩阵(10,10)
        # 两者之间的 与 运算会被广播
        # 即(30,1,10) & (1,10,10),两者满足广播约束条件
        # 运算结果维度为(30,10,10)，即一句话会对应一个(10,10)的mask
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))

        return tgt_mask


class NoamOpt:
    "Optim wrapper that implements rate."
    """自定义了一个学习率自调整优化器，
        factor是缩放因子，model_size是控制量之一
        最为重要的是warmup，随着step++
        当step<warmup时，学习率逐渐递增
        当step>warmup时，学习率逐渐减小
    """

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        # optimizer = Adam (Parameter Group 0
        #    amsgrad: False
        #    betas: (0.9, 0.98)
        #    eps: 1e-09
        #    lr: 0
        #    weight_decay: 0
        # )
        self._step = 0
        self.warmup = warmup # e.g., 4000 轮 热身
        self.factor = factor # e.g., 2
        self.model_size = model_size # 512
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate`(learning rate) above"
        if step is None:
            step = self._step
        # 2*(512**(-0.5)*min(...)),
        # min(...)中表达式相等时step=warmup
        # step<warmup, min(...)=step**(-0.5)
        return self.factor * \
            (self.model_size ** (-0.5) *
             min(step ** (-0.5), step * self.warmup ** (-1.5)))


def get_std_opt(model):
    """作者给定的标准优化器"""
    return NoamOpt(model.src_embed[0].d_model, 2, 4000,
                   torch.optim.Adam(model.parameters(), lr=0,
                                    betas=(0.9, 0.98), eps=1e-9))

