# YE-Deep-Compression

PyTorch 复现 *Deep Compression* (Song Han et al.) 的三阶段压缩流水线
(剪枝 → 权值共享/量化 → Huffman 编码),并在 LeNet-300-100 / MNIST 上
实测压缩效率与精度下降。

## 目录

| 路径 | 内容 |
|---|---|
| [`deepcompression-source/`](deepcompression-source/) | 修复后的源码,可直接运行(见下"兼容性修复") |
| [`deepcompression-tests/`](deepcompression-tests/) | 参数化扫描脚本 `sweep.py`,扫描剪枝灵敏度 × 量化位数 |
| [`deepcompression-report/`](deepcompression-report/REPORT.md) | 复现报告:方法、结果表、结论 |

## 快速开始

```bash
pip install torch torchvision scikit-learn scipy tqdm numpy

# 三阶段流水线(原仓库用法)
cd deepcompression-source
python pruning.py --epochs 20 --no-cuda
python weight_share.py saves/model_after_retraining.ptmodel --no-cuda
python huffman_encode.py saves/model_after_weight_sharing.ptmodel --no-cuda

# 参数化扫描(压缩率 × 精度下降)
cd ../deepcompression-tests
python sweep.py
```

## 兼容性修复

原码写于 ~2018,在新依赖(torch 2.6+ / sklearn 1.x)下无法直接运行,已修复:

- `torch.load(..., weights_only=False)` — torch 2.6 起默认改为 `True`,拒绝自定义类
- `KMeans(algorithm="lloyd")` — sklearn 移除了 `"full"` 和 `precompute_distances`
- `n_clusters = min(2**bits, len(mat.data))` — 防极端稀疏层崩溃

详见 [报告 §2](deepcompression-report/REPORT.md#2-兼容性修复)。

## 核心结论

s=2.0 + bits=5(论文默认)= **36.8× 压缩,精度反升 0.78 点**;
bits≥4 时量化几乎无损,bits=2 掉点明显。详见
[报告 §5](deepcompression-report/REPORT.md#5-结论)。
