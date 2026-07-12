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
