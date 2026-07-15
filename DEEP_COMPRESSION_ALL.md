# Deep Compression PyTorch 复现 · 整合版

> 单文件整合本仓库全部内容:复现报告 + 修复后的源码 + 参数化测试脚本。
> 原始来源:mightydeveloper/Deep-Compression-PyTorch
> 对应论文:*Deep Compression: Compressing Deep Neural Networks with Pruning,
> Trained Quantization and Huffman Coding* (Song Han, Huizi Mao, William J. Dally)

---

## 目录

- [① 复现报告](#①-复现报告)
  - [1. 仓库概览](#1-仓库概览)
  - [2. 兼容性修复](#2-兼容性修复)
  - [3. 测试方法](#3-测试方法)
  - [4. 测试结果](#4-测试结果)
  - [5. 结论](#5-结论)
- [② 源码](#②-源码)
  - [pruning.py](#pruningpy)
  - [weight_share.py](#weight_sharepy)
  - [huffman_encode.py](#huffman_encodepy)
  - [util.py](#utilpy)
  - [net/models.py](#netmodelspy)
  - [net/prune.py](#netprunepy)
  - [net/quantization.py](#netquantizationpy)
  - [net/huffmancoding.py](#nethuffmancodingpy)
- [③ 测试脚本](#③-测试脚本)
  - [sweep.py](#sweeppy)

---

# ① 复现报告

## 1. 仓库概览

| 阶段 | 入口 | 核心实现 | 作用 |
|---|---|---|---|
| ① 剪枝 | `pruning.py` | `net/prune.py` | 训练 LeNet-300-100 → 按 `s × std(layer)` 阈值置零 → 重训;`MaskedLinear` 用 mask 保证剪枝连接不再生长 |
| ② 权值共享(量化) | `weight_share.py` | `net/quantization.py` | 对 CSC/CSR 稀疏矩阵的非零元做 K-means 聚类,默认 5 bit = 32 簇 |
| ③ Huffman 编码 | `huffman_encode.py` | `net/huffmancoding.py` | 对稀疏矩阵的 data/indices/indptr 分别 Huffman,统计最终压缩比 |

**已知局限**:bias 不参与任何压缩;`models.py` 中的 `LeNet_5` 有 bug(`Linear` 未定义),
仅 `LeNet`(全连接 784→300→100→10)可运行。

## 2. 兼容性修复

原码写于 ~2018,在新依赖下无法直接运行。本次复现修复了 3 处:

| 文件 | 问题 | 修复 |
|---|---|---|
| `weight_share.py` / `huffman_encode.py` | torch 2.6 起 `torch.load` 默认 `weights_only=True`,拒绝反序列化自定义类 | 加 `weights_only=False` |
| `net/quantization.py` | sklearn 1.x 移除了 `precompute_distances`、`algorithm="full"` | 改 `algorithm="lloyd"`,删掉 `precompute_distances` |
| `net/quantization.py` | 极端稀疏层(非零元 < 2^bits)时 KMeans 报 `n_samples < n_clusters` | 加 `n_clusters = min(2**bits, len(mat.data))` |

## 3. 测试方法

扫描脚本 `sweep.py`,从同一基线出发,扫描:

- 剪枝灵敏度 `s ∈ {0.25, 0.5, 1.0, 1.5, 2.0, 3.0}`
- 量化位数 `bits ∈ {2, 3, 4, 5}`

基线:LeNet-300-100,Adam(lr=1e-2, wd=1e-4),20 epochs;剪枝后重训 15 epochs。
CPU 运行,单组约 1-2 分钟。

## 4. 测试结果

基线精度:**95.71%**(20 epochs;论文用 100 epochs 约 98%,绝对值不同但相对趋势一致)。

```
 sens bits | pruneAcc retrain  wsAcc dropVsBase | sparse%   wtCR |   origB   compB huffCR  allCR
-----------------------------------------------------------------------------------------------
 0.25    2 |    95.56   94.48  88.28      +7.43 |    52.6    2.1 | 1012388  178454   5.67    6.0
 0.25    3 |    95.56   94.48  92.80      +2.91 |    52.6    2.1 | 1012388  187685   5.39    5.7
 0.25    4 |    95.56   94.48  94.42      +1.29 |    52.6    2.1 | 1012388  200319   5.05    5.3
 0.25    5 |    95.56   94.48  94.56      +1.15 |    52.6    2.1 | 1012388  211491   4.79    5.0
  0.5    2 |    95.30   96.24  90.24      +5.47 |    69.4    3.3 |  654388  118235   5.53    9.0
  0.5    3 |    95.30   96.24  95.51      +0.20 |    69.4    3.3 |  654388  124748   5.25    8.5
  0.5    4 |    95.30   96.24  96.27      -0.56 |    69.4    3.3 |  654388  134083   4.88    8.0
  0.5    5 |    95.30   96.24  96.25      -0.54 |    69.4    3.3 |  654388  140613   4.65    7.6
  1.0    2 |    93.06   96.33  89.74      +5.97 |    86.3    7.3 |  294844   56241   5.24   19.0
  1.0    3 |    93.06   96.33  94.91      +0.80 |    86.3    7.3 |  294844   59253   4.98   18.0
  1.0    4 |    93.06   96.33  95.97      -0.26 |    86.3    7.3 |  294844   62876   4.69   17.0
  1.0    5 |    93.06   96.33  96.30      -0.59 |    86.3    7.3 |  294844   65866   4.48   16.2
  1.5    2 |    82.82   96.82  92.43      +3.28 |    92.1   12.7 |  170828   35219   4.85   30.3
  1.5    3 |    82.82   96.82  96.52      -0.81 |    92.1   12.7 |  170828   37046   4.61   28.8
  1.5    4 |    82.82   96.82  96.61      -0.90 |    92.1   12.7 |  170828   39435   4.33   27.0
  1.5    5 |    82.82   96.82  96.63      -0.92 |    92.1   12.7 |  170828   41040   4.16   26.0
  2.0    2 |    60.58   96.50  89.71      +6.00 |    95.0   19.8 |  110740   24987   4.43   42.7
  2.0    3 |    60.58   96.50  95.66      +0.05 |    95.0   19.8 |  110740   26549   4.17   40.2
  2.0    4 |    60.58   96.50  96.58      -0.87 |    95.0   19.8 |  110740   27750   3.99   38.4
  2.0    5 |    60.58   96.50  96.49      -0.78 |    95.0   19.8 |  110740   28972   3.82   36.8  ← 论文默认
  3.0    2 |     9.74   85.52  65.30     +30.41 |    98.0   48.8 |   46908   12076   3.88   88.3
  3.0    3 |     9.74   85.52  81.61     +14.10 |    98.0   48.8 |   46908   12621   3.72   84.5
  3.0    4 |     9.74   85.52  85.61     +10.10 |    98.0   48.8 |   46900   13138   3.57   81.2
  3.0    5 |     9.74   85.52  85.19     +10.52 |    98.0   48.8 |   46900   13759   3.41   77.5
```

字段说明:`dropVsBase` 正值 = 掉点;`allCR` = 相对原始 32-bit 稠密权重的总压缩比。

## 5. 结论

1. **剪枝阶段几乎无损**:重训后精度能恢复到接近甚至超过基线。直到
   s=2.0(95% 稀疏,权重 19.8×)仍基本无损;s=3.0(98% 稀疏)重训也救不回(85.5%)。

2. **量化位数是精度下降主因**:bits=2 掉点明显(5~30%);bits=3 仅极端稀疏时
   小掉;bits=4/5 几乎无损(<1%,常为负 = 反而略升)。

3. **三阶段叠加效果显著**:总压缩比从 5× 到 88×,Huffman 在量化基础上再省 4~5×。

4. **最佳折中**(论文默认配置):s=2.0 + bits=5 = **36.8× 压缩,精度反升 0.78 点**;
   s=1.5 + bits=5 = 26× 压缩、精度升 0.92 点。这与原论文"LeNet-300-100 可大幅压缩
   而几乎不掉点"的结论一致。

---

# ② 源码

> 以下为修复后可直接运行的源码。文件路径标注在每节标题旁。

## pruning.py

```python
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import tqdm

from net.models import LeNet
from net.quantization import apply_weight_sharing
import util

os.makedirs('saves', exist_ok=True)

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST pruning from deep compression paper')
parser.add_argument('--batch-size', type=int, default=50, metavar='N',
                    help='input batch size for training (default: 50)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--log', type=str, default='log.txt',
                    help='log file name')
parser.add_argument('--sensitivity', type=float, default=2,
                    help="sensitivity value that is multiplied to layer's std in order to get threshold value")
args = parser.parse_args()

# Control Seed
torch.manual_seed(args.seed)

# Select Device
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else 'cpu')
if use_cuda:
    print("Using CUDA!")
    torch.cuda.manual_seed(args.seed)
else:
    print('Not using CUDA!!!')

# Loader
kwargs = {'num_workers': 5, 'pin_memory': True} if use_cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=False, **kwargs)


# Define which model to use
model = LeNet(mask=True).to(device)

print(model)
util.print_model_parameters(model)

# NOTE : `weight_decay` term denotes L2 regularization loss term
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)
initial_optimizer_state_dict = optimizer.state_dict()

def train(epochs):
    model.train()
    for epoch in range(epochs):
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for batch_idx, (data, target) in pbar:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            loss.backward()

            # zero-out all the gradients corresponding to the pruned connections
            for name, p in model.named_parameters():
                if 'mask' in name:
                    continue
                tensor = p.data.cpu().numpy()
                grad_tensor = p.grad.data.cpu().numpy()
                grad_tensor = np.where(tensor==0, 0, grad_tensor)
                p.grad.data = torch.from_numpy(grad_tensor).to(device)

            optimizer.step()
            if batch_idx % args.log_interval == 0:
                done = batch_idx * len(data)
                percentage = 100. * batch_idx / len(train_loader)
                pbar.set_description(f'Train Epoch: {epoch} [{done:5}/{len(train_loader.dataset)} ({percentage:3.0f}%)]  Loss: {loss.item():.6f}')


def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    return accuracy


# Initial training
print("--- Initial training ---")
train(args.epochs)
accuracy = test()
util.log(args.log, f"initial_accuracy {accuracy}")
torch.save(model, f"saves/initial_model.ptmodel")
print("--- Before pruning ---")
util.print_nonzeros(model)

# Pruning
model.prune_by_std(args.sensitivity)
accuracy = test()
util.log(args.log, f"accuracy_after_pruning {accuracy}")
print("--- After pruning ---")
util.print_nonzeros(model)

# Retrain
print("--- Retraining ---")
optimizer.load_state_dict(initial_optimizer_state_dict) # Reset the optimizer
train(args.epochs)
torch.save(model, f"saves/model_after_retraining.ptmodel")
accuracy = test()
util.log(args.log, f"accuracy_after_retraining {accuracy}")

print("--- After Retraining ---")
util.print_nonzeros(model)

```

## weight_share.py

```python
import argparse
import os

import torch

from net.models import LeNet
from net.quantization import apply_weight_sharing
import util

parser = argparse.ArgumentParser(description='This program quantizes weight by using weight sharing')
parser.add_argument('model', type=str, help='path to saved pruned model')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--output', default='saves/model_after_weight_sharing.ptmodel', type=str,
                    help='path to model output')
args = parser.parse_args()

use_cuda = not args.no_cuda and torch.cuda.is_available()


# Define the model
model = torch.load(args.model, weights_only=False)
print('accuracy before weight sharing')
util.test(model, use_cuda)

# Weight sharing
apply_weight_sharing(model)
print('accuacy after weight sharing')
util.test(model, use_cuda)

# Save the new model
os.makedirs('saves', exist_ok=True)
torch.save(model, args.output)

```

## huffman_encode.py

```python
import argparse

import torch

from net.huffmancoding import huffman_encode_model
import util

parser = argparse.ArgumentParser(description='Huffman encode a quantized model')
parser.add_argument('model', type=str, help='saved quantized model')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA')
args = parser.parse_args()

use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else 'cpu')

model = torch.load(args.model, weights_only=False)
huffman_encode_model(model)

```

## util.py

```python
import os
import torch
import math
import numpy as np
from torch.nn import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
from torchvision import datasets, transforms

def log(filename, content):
    with open(filename, 'a') as f:
        content += "\n"
        f.write(content)


def print_model_parameters(model, with_values=False):
    print(f"{'Param name':20} {'Shape':30} {'Type':15}")
    print('-'*70)
    for name, param in model.named_parameters():
        print(f'{name:20} {str(param.shape):30} {str(param.dtype):15}')
        if with_values:
            print(param)


def print_nonzeros(model):
    nonzero = total = 0
    for name, p in model.named_parameters():
        if 'mask' in name:
            continue
        tensor = p.data.cpu().numpy()
        nz_count = np.count_nonzero(tensor)
        total_params = np.prod(tensor.shape)
        nonzero += nz_count
        total += total_params
        print(f'{name:20} | nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%) | total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}')
    print(f'alive: {nonzero}, pruned : {total - nonzero}, total: {total}, Compression rate : {total/nonzero:10.2f}x  ({100 * (total-nonzero) / total:6.2f}% pruned)')


def test(model, use_cuda=True):
    kwargs = {'num_workers': 5, 'pin_memory': True} if use_cuda else {}
    device = torch.device("cuda" if use_cuda else 'cpu')
    test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=1000, shuffle=False, **kwargs)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    return accuracy

```

## net/models.py

```python
import torch.nn as nn
import torch.nn.functional as F

from .prune import PruningModule, MaskedLinear

class LeNet(PruningModule):
    def __init__(self, mask=False):
        super(LeNet, self).__init__()
        linear = MaskedLinear if mask else nn.Linear
        self.fc1 = linear(784, 300)
        self.fc2 = linear(300, 100)
        self.fc3 = linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x


class LeNet_5(PruningModule):
    def __init__(self, mask=False):
        super(LeNet_5, self).__init__()
        linear = MaskedLinear if mask else Linear
        self.conv1 = nn.Conv2d(1, 6, kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(5, 5))
        self.conv3 = nn.Conv2d(16, 120, kernel_size=(5,5))
        self.fc1 = linear(120, 84)
        self.fc2 = linear(84, 10)

    def forward(self, x):
        # Conv1
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=2)

        # Conv2
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=2)

        # Conv3
        x = self.conv3(x)
        x = F.relu(x)

        # Fully-connected
        x = x.view(-1, 120)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)

        return x

```

## net/prune.py

```python
import math

import numpy as np
import torch
from torch.nn import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F

class PruningModule(Module):
    def prune_by_percentile(self, q=5.0, **kwargs):
        """
        Note:
             The pruning percentile is based on all layer's parameters concatenated
        Args:
            q (float): percentile in float
            **kwargs: may contain `cuda`
        """
        # Calculate percentile value
        alive_parameters = []
        for name, p in self.named_parameters():
            # We do not prune bias term
            if 'bias' in name or 'mask' in name:
                continue
            tensor = p.data.cpu().numpy()
            alive = tensor[np.nonzero(tensor)] # flattened array of nonzero values
            alive_parameters.append(alive)

        all_alives = np.concatenate(alive_parameters)
        percentile_value = np.percentile(abs(all_alives), q)
        print(f'Pruning with threshold : {percentile_value}')

        # Prune the weights and mask
        # Note that module here is the layer
        # ex) fc1, fc2, fc3
        for name, module in self.named_modules():
            if name in ['fc1', 'fc2', 'fc3']:
                module.prune(threshold=percentile_value)

    def prune_by_std(self, s=0.25):
        """
        Note that `s` is a quality parameter / sensitivity value according to the paper.
        According to Song Han's previous paper (Learning both Weights and Connections for Efficient Neural Networks),
        'The pruning threshold is chosen as a quality parameter multiplied by the standard deviation of a layer’s weights'

        I tried multiple values and empirically, 0.25 matches the paper's compression rate and number of parameters.
        Note : In the paper, the authors used different sensitivity values for different layers.
        """
        for name, module in self.named_modules():
            if name in ['fc1', 'fc2', 'fc3']:
                threshold = np.std(module.weight.data.cpu().numpy()) * s
                print(f'Pruning with threshold : {threshold} for layer {name}')
                module.prune(threshold)


class MaskedLinear(Module):
    r"""Applies a masked linear transformation to the incoming data: :math:`y = (A * M)x + b`

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to False, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(N, *, in\_features)` where `*` means any number of
          additional dimensions
        - Output: :math:`(N, *, out\_features)` where all but the last dimension
          are the same shape as the input.

    Attributes:
        weight: the learnable weights of the module of shape
            (out_features x in_features)
        bias:   the learnable bias of the module of shape (out_features)
        mask: the unlearnable mask for the weight.
            It has the same shape as weight (out_features x in_features)

    """
    def __init__(self, in_features, out_features, bias=True):
        super(MaskedLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        # Initialize the mask with 1
        self.mask = Parameter(torch.ones([out_features, in_features]), requires_grad=False)
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        return F.linear(input, self.weight * self.mask, self.bias)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) \
            + ', bias=' + str(self.bias is not None) + ')'

    def prune(self, threshold):
        weight_dev = self.weight.device
        mask_dev = self.mask.device
        # Convert Tensors to numpy and calculate
        tensor = self.weight.data.cpu().numpy()
        mask = self.mask.data.cpu().numpy()
        new_mask = np.where(abs(tensor) < threshold, 0, mask)
        # Apply new weight and mask
        self.weight.data = torch.from_numpy(tensor * new_mask).to(weight_dev)
        self.mask.data = torch.from_numpy(new_mask).to(mask_dev)




```

## net/quantization.py

```python
import torch
import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse import csc_matrix, csr_matrix


def apply_weight_sharing(model, bits=5):
    """
    Applies weight sharing to the given model
    """
    for module in model.children():
        dev = module.weight.device
        weight = module.weight.data.cpu().numpy()
        shape = weight.shape
        mat = csr_matrix(weight) if shape[0] < shape[1] else csc_matrix(weight)
        if len(mat.data) == 0:
            continue
        min_ = min(mat.data)
        max_ = max(mat.data)
        n_clusters = min(2**bits, len(mat.data))
        space = np.linspace(min_, max_, num=n_clusters)
        kmeans = KMeans(n_clusters=n_clusters, init=space.reshape(-1,1), n_init=1, algorithm="lloyd")
        kmeans.fit(mat.data.reshape(-1,1))
        new_weight = kmeans.cluster_centers_[kmeans.labels_].reshape(-1)
        mat.data = new_weight
        module.weight.data = torch.from_numpy(mat.toarray()).to(dev)



```

## net/huffmancoding.py

```python
import os
from collections import defaultdict, namedtuple
from heapq import heappush, heappop, heapify
import struct
from pathlib import Path

import torch
import numpy as np
from scipy.sparse import csr_matrix, csc_matrix

Node = namedtuple('Node', 'freq value left right')
Node.__lt__ = lambda x, y: x.freq < y.freq

def huffman_encode(arr, prefix, save_dir='./'):
    """
    Encodes numpy array 'arr' and saves to `save_dir`
    The names of binary files are prefixed with `prefix`
    returns the number of bytes for the tree and the data after the compression
    """
    # Infer dtype
    dtype = str(arr.dtype)

    # Calculate frequency in arr
    freq_map = defaultdict(int)
    convert_map = {'float32':float, 'int32':int}
    for value in np.nditer(arr):
        value = convert_map[dtype](value)
        freq_map[value] += 1

    # Make heap
    heap = [Node(frequency, value, None, None) for value, frequency in freq_map.items()]
    heapify(heap)

    # Merge nodes
    while(len(heap) > 1):
        node1 = heappop(heap)
        node2 = heappop(heap)
        merged = Node(node1.freq + node2.freq, None, node1, node2)
        heappush(heap, merged)

    # Generate code value mapping
    value2code = {}

    def generate_code(node, code):
        if node is None:
            return
        if node.value is not None:
            value2code[node.value] = code
            return
        generate_code(node.left, code + '0')
        generate_code(node.right, code + '1')

    root = heappop(heap)
    generate_code(root, '')

    # Path to save location
    directory = Path(save_dir)

    # Dump data
    data_encoding = ''.join(value2code[convert_map[dtype](value)] for value in np.nditer(arr))
    datasize = dump(data_encoding, directory/f'{prefix}.bin')

    # Dump codebook (huffman tree)
    codebook_encoding = encode_huffman_tree(root, dtype)
    treesize = dump(codebook_encoding, directory/f'{prefix}_codebook.bin')

    return treesize, datasize


def huffman_decode(directory, prefix, dtype):
    """
    Decodes binary files from directory
    """
    directory = Path(directory)

    # Read the codebook
    codebook_encoding = load(directory/f'{prefix}_codebook.bin')
    root = decode_huffman_tree(codebook_encoding, dtype)

    # Read the data
    data_encoding = load(directory/f'{prefix}.bin')

    # Decode
    data = []
    ptr = root
    for bit in data_encoding:
        ptr = ptr.left if bit == '0' else ptr.right
        if ptr.value is not None: # Leaf node
            data.append(ptr.value)
            ptr = root

    return np.array(data, dtype=dtype)


# Logics to encode / decode huffman tree
# Referenced the idea from https://stackoverflow.com/questions/759707/efficient-way-of-storing-huffman-tree
def encode_huffman_tree(root, dtype):
    """
    Encodes a huffman tree to string of '0's and '1's
    """
    converter = {'float32':float2bitstr, 'int32':int2bitstr}
    code_list = []
    def encode_node(node):
        if node.value is not None: # node is leaf node
            code_list.append('1')
            lst = list(converter[dtype](node.value))
            code_list.extend(lst)
        else:
            code_list.append('0')
            encode_node(node.left)
            encode_node(node.right)
    encode_node(root)
    return ''.join(code_list)


def decode_huffman_tree(code_str, dtype):
    """
    Decodes a string of '0's and '1's and costructs a huffman tree
    """
    converter = {'float32':bitstr2float, 'int32':bitstr2int}
    idx = 0
    def decode_node():
        nonlocal idx
        info = code_str[idx]
        idx += 1
        if info == '1': # Leaf node
            value = converter[dtype](code_str[idx:idx+32])
            idx += 32
            return Node(0, value, None, None)
        else:
            left = decode_node()
            right = decode_node()
            return Node(0, None, left, right)

    return decode_node()



# My own dump / load logics
def dump(code_str, filename):
    """
    code_str : string of either '0' and '1' characters
    this function dumps to a file
    returns how many bytes are written
    """
    # Make header (1 byte) and add padding to the end
    # Files need to be byte aligned.
    # Therefore we add 1 byte as a header which indicates how many bits are padded to the end
    # This introduces minimum of 8 bits, maximum of 15 bits overhead
    num_of_padding = -len(code_str) % 8
    header = f"{num_of_padding:08b}"
    code_str = header + code_str + '0' * num_of_padding

    # Convert string to integers and to real bytes
    byte_arr = bytearray(int(code_str[i:i+8], 2) for i in range(0, len(code_str), 8))

    # Dump to a file
    with open(filename, 'wb') as f:
        f.write(byte_arr)
    return len(byte_arr)


def load(filename):
    """
    This function reads a file and makes a string of '0's and '1's
    """
    with open(filename, 'rb') as f:
        header = f.read(1)
        rest = f.read() # bytes
        code_str = ''.join(f'{byte:08b}' for byte in rest)
        offset = ord(header)
        if offset != 0:
            code_str = code_str[:-offset] # string of '0's and '1's
    return code_str


# Helper functions for converting between bit string and (float or int)
def float2bitstr(f):
    four_bytes = struct.pack('>f', f) # bytes
    return ''.join(f'{byte:08b}' for byte in four_bytes) # string of '0's and '1's

def bitstr2float(bitstr):
    byte_arr = bytearray(int(bitstr[i:i+8], 2) for i in range(0, len(bitstr), 8))
    return struct.unpack('>f', byte_arr)[0]

def int2bitstr(integer):
    four_bytes = struct.pack('>I', integer) # bytes
    return ''.join(f'{byte:08b}' for byte in four_bytes) # string of '0's and '1's

def bitstr2int(bitstr):
    byte_arr = bytearray(int(bitstr[i:i+8], 2) for i in range(0, len(bitstr), 8))
    return struct.unpack('>I', byte_arr)[0]


# Functions for calculating / reconstructing index diff
def calc_index_diff(indptr):
    return indptr[1:] - indptr[:-1]

def reconstruct_indptr(diff):
    return np.concatenate([[0], np.cumsum(diff)])


# Encode / Decode models
def huffman_encode_model(model, directory='encodings/'):
    os.makedirs(directory, exist_ok=True)
    original_total = 0
    compressed_total = 0
    print(f"{'Layer':<15} | {'original':>10} {'compressed':>10} {'improvement':>11} {'percent':>7}")
    print('-'*70)
    for name, param in model.named_parameters():
        if 'mask' in name:
            continue
        if 'weight' in name:
            weight = param.data.cpu().numpy()
            shape = weight.shape
            form = 'csr' if shape[0] < shape[1] else 'csc'
            mat = csr_matrix(weight) if shape[0] < shape[1] else csc_matrix(weight)

            # Encode
            t0, d0 = huffman_encode(mat.data, name+f'_{form}_data', directory)
            t1, d1 = huffman_encode(mat.indices, name+f'_{form}_indices', directory)
            t2, d2 = huffman_encode(calc_index_diff(mat.indptr), name+f'_{form}_indptr', directory)

            # Print statistics
            original = mat.data.nbytes + mat.indices.nbytes + mat.indptr.nbytes
            compressed = t0 + t1 + t2 + d0 + d1 + d2

            print(f"{name:<15} | {original:10} {compressed:10} {original / compressed:>10.2f}x {100 * compressed / original:>6.2f}%")
        else: # bias
            # Note that we do not huffman encode bias
            bias = param.data.cpu().numpy()
            bias.dump(f'{directory}/{name}')

            # Print statistics
            original = bias.nbytes
            compressed = original

            print(f"{name:<15} | {original:10} {compressed:10} {original / compressed:>10.2f}x {100 * compressed / original:>6.2f}%")
        original_total += original
        compressed_total += compressed

    print('-'*70)
    print(f"{'total':15} | {original_total:>10} {compressed_total:>10} {original_total / compressed_total:>10.2f}x {100 * compressed_total / original_total:>6.2f}%")


def huffman_decode_model(model, directory='encodings/'):
    for name, param in model.named_parameters():
        if 'mask' in name:
            continue
        if 'weight' in name:
            dev = param.device
            weight = param.data.cpu().numpy()
            shape = weight.shape
            form = 'csr' if shape[0] < shape[1] else 'csc'
            matrix = csr_matrix if shape[0] < shape[1] else csc_matrix

            # Decode data
            data = huffman_decode(directory, name+f'_{form}_data', dtype='float32')
            indices = huffman_decode(directory, name+f'_{form}_indices', dtype='int32')
            indptr = reconstruct_indptr(huffman_decode(directory, name+f'_{form}_indptr', dtype='int32'))

            # Construct matrix
            mat = matrix((data, indices, indptr), shape)

            # Insert to model
            param.data = torch.from_numpy(mat.toarray()).to(dev)
        else:
            dev = param.device
            bias = np.load(directory+'/'+name)
            param.data = torch.from_numpy(bias).to(dev)

```

---

# ③ 测试脚本

## sweep.py

```python
import copy, os, time, itertools
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from net.models import LeNet
from net.quantization import apply_weight_sharing
from net.huffmancoding import huffman_encode_model
import util

torch.manual_seed(42)
device = torch.device('cpu')
BASE_EPOCHS = int(os.environ.get('BASE_EPOCHS', '20'))
RETRAIN_EPOCHS = int(os.environ.get('RETRAIN_EPOCHS', '15'))
SENSITIVITIES = [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]
BITS_LIST = [2, 3, 4, 5]

tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True, transform=tf),
    batch_size=50, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=tf),
    batch_size=1000, shuffle=False)

def train(model, epochs):
    model.train()
    opt = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0001)
    for _ in range(epochs):
        for data, target in train_loader:
            opt.zero_grad()
            out = model(data)
            loss = F.nll_loss(out, target)
            loss.backward()
            for name, p in model.named_parameters():
                if 'mask' in name: continue
                t = p.data.cpu().numpy(); g = p.grad.data.cpu().numpy()
                g = np.where(t == 0, 0, g)
                p.grad.data = torch.from_numpy(g)
            opt.step()

def acc(model):
    model.eval(); correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            pred = model(data).data.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    return 100. * correct / len(test_loader.dataset)

def sparsity(model):
    nz = total = 0
    for name, p in model.named_parameters():
        if 'mask' in name or 'bias' in name: continue
        t = p.data.cpu().numpy(); nz += np.count_nonzero(t); total += np.prod(t.shape)
    return nz, total

# Baseline
print(f"=== Baseline training ({BASE_EPOCHS} epochs) ===", flush=True)
t0 = time.time()
base = LeNet(mask=True).to(device)
train(base, BASE_EPOCHS)
acc_base = acc(base)
nz, total = sparsity(base)
print(f"baseline acc={acc_base:.2f}%  params={total}  time={time.time()-t0:.1f}s", flush=True)

rows = []
for s in SENSITIVITIES:
    cache = f'saves/swept_s{s}.ptmodel'
    if os.path.exists(cache):
        m = torch.load(cache, weights_only=False)
        acc_prune = float('nan')
    else:
        m = copy.deepcopy(base)
        m.prune_by_std(s)
        acc_prune = acc(m)
        train(m, RETRAIN_EPOCHS)
        torch.save(m, cache)
    acc_retrain = acc(m)
    nz, total = sparsity(m)
    pr = 100.*(total-nz)/total
    cr = total/nz if nz else float('inf')
    print(f"s={s:<4} prune_acc={acc_prune:5.2f} retrain_acc={acc_retrain:5.2f} "
          f"drop_base={acc_base-acc_retrain:+.2f} sparsity={pr:5.1f}% cr_weight={cr:5.1f}x", flush=True)
    for bits in BITS_LIST:
        mq = copy.deepcopy(m)
        apply_weight_sharing(mq, bits=bits)
        acc_ws = acc(mq)
        # huffman size
        os.makedirs('encodings', exist_ok=True)
        import io, contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            huffman_encode_model(mq, directory='encodings/')
        # parse compressed total from buf
        comp = None; orig = None
        for line in buf.getvalue().splitlines():
            if line.strip().startswith('total'):
                rhs = line.split('|')[-1].split()
                orig = int(rhs[0]); comp = int(rhs[1])
        # weight param bytes (float32) of full model
        full_bytes = sum(p.numel()*4 for n,p in mq.named_parameters() if 'mask' not in n)
        rows.append((s, bits, acc_base, acc_prune, acc_retrain, acc_ws,
                     pr, cr, orig, comp, full_bytes))

print("\n=== SUMMARY ===")
print(f"baseline accuracy: {acc_base:.2f}%  (LeNet-300-100, {BASE_EPOCHS} epochs)")
print(f"{'sens':>5} {'bits':>4} | {'pruneAcc':>8} {'retrain':>7} {'wsAcc':>6} {'dropVsBase':>10} | {'sparse%':>7} {'wtCR':>6} | {'origB':>7} {'compB':>7} {'huffCR':>6} {'allCR':>6}")
print("-"*95)
for r in rows:
    s,bits,ab,ap,ar,aw,pr,cr,orig,comp,fb = r
    hcr = orig/comp if comp else 0
    allcr = fb/comp if comp else 0
    print(f"{s:>5} {bits:>4} | {ap:>8.2f} {ar:>7.2f} {aw:>6.2f} {ab-aw:>+10.2f} | {pr:>7.1f} {cr:>6.1f} | {orig:>7} {comp:>7} {hcr:>6.2f} {allcr:>6.1f}")

```
