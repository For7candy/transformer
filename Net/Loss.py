import torch.nn as nn
import torch
from torch.autograd import Variable


class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    "损失函数"
    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx  # '<blank>' 的id
        self.confidence = 1.0 - smoothing  # 自留的概率值、得分 e.g. 0.6
        self.smoothing = smoothing  # 均分出去的概率值，得分 e.g. 0.4
        self.size = size  # target vocab size 目标语言词表大小
        self.true_dist = None

    def forward(self, x, target):
        "in real-world case: 真实情况下"
        #  x的shape为(batch.size * seq.len, target.vocab.size)
        # y的shape是(batch.size * seq.len)

        # x=logits，(seq.len, target.vocab.size)
        # 每一行，代表一个位置的词
        # 类似于：假设seq.len=3, target.vocab.size=5
        # x中保存的是log(prob)
        # x = tensor([[-20.7233,  -1.6094,  -0.3567,  -2.3026, -20.7233],
        # [-20.7233,  -1.6094,  -0.3567,  -2.3026, -20.7233],
        # [-20.7233,  -1.6094,  -0.3567,  -2.3026, -20.7233]])

        # target 类似于：
        # target = tensor([2, 1, 0])，torch.size=(3)

        assert x.size(1) == self.size  # 目标语言词表大小
        true_dist = x.data.clone()
        # true_dist = tensor([[-20.7233,  -1.6094,  -0.3567,  -2.3026, -20.7233],
        # [-20.7233,  -1.6094,  -0.3567,  -2.3026, -20.7233],
        # [-20.7233,  -1.6094,  -0.3567,  -2.3026, -20.7233]])

        true_dist.fill_(self.smoothing / (self.size - 2))
        # true_dist = tensor([[0.1333, 0.1333, 0.1333, 0.1333, 0.1333],
        # [0.1333, 0.1333, 0.1333, 0.1333, 0.1333],
        # [0.1333, 0.1333, 0.1333, 0.1333, 0.1333]])

        # 注意，这里分母target.vocab.size-2是因为
        # (1) 最优值 0.6要占一个位置；
        # (2) 填充词 <blank> 要被排除在外
        # 所以被激活的目标语言词表大小就是self.size-2

        true_dist.scatter_(1, target.data.unsqueeze(1).long(),
                           self.confidence)
        # target.data.unsqueeze(1) ->
        # tensor([[2],
        # [1],
        # [0]]); shape=torch.Size([3, 1])
        # self.confidence = 0.6

        # 根据target.data的指示，按照列优先(1)的原则，把0.6这个值
        # 填入true_dist: 因为target.data是2,1,0的内容，
        # 所以，0.6填入第0行的第2列（列号，行号都是0开始）
        # 0.6填入第1行的第1列
        # 0.6填入第2行的第0列：
        # true_dist = tensor([[0.1333, 0.1333, 0.6000, 0.1333, 0.1333],
        # [0.1333, 0.6000, 0.1333, 0.1333, 0.1333],
        # [0.6000, 0.1333, 0.1333, 0.1333, 0.1333]])

        true_dist[:, self.padding_idx] = 0
        # true_dist = tensor([[0.0000, 0.1333, 0.6000, 0.1333, 0.1333],
        # [0.0000, 0.6000, 0.1333, 0.1333, 0.1333],
        # [0.0000, 0.1333, 0.1333, 0.1333, 0.1333]])
        # 设置true_dist这个tensor的第一列的值全为0
        # 因为这个是填充词'<blank>'所在的id位置，不应该计入
        # 目标词表。需要注意的是，true_dist的每一列，代表目标语言词表
        # 中的一个词的id

        mask = torch.nonzero(target.data == self.padding_idx)
        # mask = tensor([[2]]), 也就是说，最后一个词 2,1,0中的0，
        # 因为是'<blank>'的id，所以通过上面的一步，把他们找出来

        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
            # 当target reference序列中有0这个'<blank>'的时候，则需要把
            # 这一行的值都清空。
            # 在一个batch里面的时候，可能两个序列长度不一，所以短的序列需要
            # pad '<blank>'来填充，所以会出现类似于(2,1,0)这样的情况
            # true_dist = tensor([[0.0000, 0.1333, 0.6000, 0.1333, 0.1333],
            # [0.0000, 0.6000, 0.1333, 0.1333, 0.1333],
            # [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]])
        self.true_dist = true_dist
        return self.criterion(x,
                              Variable(true_dist, requires_grad=False))
        # 这一步就是调用KL loss来计算
        # x = tensor([[-20.7233,  -1.6094,  -0.3567,  -2.3026, -20.7233],
        # [-20.7233,  -1.6094,  -0.3567,  -2.3026, -20.7233],
        # [-20.7233,  -1.6094,  -0.3567,  -2.3026, -20.7233]])

        # true_dist=tensor([[0.0000, 0.1333, 0.6000, 0.1333, 0.1333],
        # [0.0000, 0.6000, 0.1333, 0.1333, 0.1333],
        # [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]])
        # 之间的loss了。细节可以参考我的那篇illustrated transformer


class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion, opt=None):
        self.generator = generator  # Generator 对象, linear+softmax
        self.criterion = criterion  # LabelSmooth对象，计算loss
        self.opt = opt  # NormOpt对象，优化算法对象

    def __call__(self, x, y, norm):
        # e.g., x为(2,3,8), batch.size=2, seq.len=3, d_model=8
        # y = tensor([[4, 2, 1],
        # [4, 4, 4]], dtype=torch.int32)

        # norm: (y=trg_y中非'<blank>'的token的个数)
        "attention here"

        # x = self.generator(x)
        # loss = self.criterion(x.contiguous().view(-1, x.size(-1)),
        #                       y.contiguous().view(-1)) / norm.item()
        # 变形后，x类似于(batch.size*seq.len, target.vocab.size)
        # y为(target.vocab.size)
        # 然后调用LabelSmooth来计算loss
        # loss.backward()
        # if self.opt is not None:
        #     self.opt.step()
        #     self.opt.optimizer.zero_grad()
        # return loss.data[0] * norm
        # "attention here"
        # return loss.data.item() * norm.item()

        x = self.generator(x)
        sloss = (
                self.criterion(
                    x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)
                )
                / norm
        )
        return sloss.data * norm, sloss
    