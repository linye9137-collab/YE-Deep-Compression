# sweep.py — 参数化压缩扫描脚本

从同一基线模型出发,扫描 6 个剪枝灵敏度 × 4 个量化位数,一次性测量
每组的剪枝率、权值共享后精度、Huffman 压缩比与总压缩比。

## 用法

```bash
# 默认:基线 20 epochs,重训 15 epochs
python sweep.py

# 自定义训练长度(经环境变量)
BASE_EPOCHS=30 RETRAIN_EPOCHS=20 python sweep.py
```

## 依赖

```
torch torchvision scikit-learn scipy tqdm numpy
```

## 输出

脚本逐组打印进度,最后输出汇总表:

```
 sens bits | pruneAcc retrain  wsAcc dropVsBase | sparse%   wtCR |   origB   compB huffCR  allCR
```

- `pruneAcc` — 剪枝后(重训前)精度
- `retrain`  — 重训后精度
- `wsAcc`    — 权值共享(量化)后精度
- `dropVsBase` — 相对基线的掉点(正=掉点,负=反升)
- `sparse%` / `wtCR` — 权重稀疏率 / 权重压缩比
- `origB` / `compB` — Huffman 前后字节数
- `huffCR` / `allCR` — Huffman 压缩比 / 相对原始 32-bit 稠密权重的总压缩比

## 缓存

剪枝+重训后的模型按灵敏度缓存到 `saves/swept_s{灵敏度}.ptmodel`,
重跑时跳过重训、直接复用,便于快速调整量化位数。
