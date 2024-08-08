import time
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import deque
from typing import Dict, Optional, Literal
from grokfast_pytorch.grokfast import GrokFastAdamW

class RNG:
    def __init__(self, seed):
        self.generator = torch.Generator().manual_seed(seed)

    def randn(self, *size, mu=0.0, sigma=1.0):
        return mu + sigma * torch.randn(*size, generator=self.generator)

    def rand(self, *size, a=0.0, b=1.0):
        return a + (b - a) * torch.rand(*size, generator=self.generator)

    def random(self):
        return torch.rand(1, generator=self.generator).item()

class StepTimer:
    def __init__(self):
        self.start_time = None
        self.dt = 0.0

    def __enter__(self):
        self.start_time = time.time()

    def __exit__(self, type, value, traceback):
        self.dt = time.time() - self.start_time

    def get_dt(self):
        return self.dt

def gradfilter_ma(
    m: nn.Module,
    grads: Optional[Dict[str, deque]] = None,
    window_size: int = 100,
    lamb: float = 5.0,
    filter_type: Literal['mean', 'sum'] = 'mean',
    warmup: bool = True,
    trigger: bool = False,
) -> Dict[str, deque]:
    if grads is None:
        grads = {n: deque(maxlen=window_size) for n, p in m.named_parameters() if p.requires_grad and p.grad is not None}

    for n, p in m.named_parameters():
        if p.requires_grad and p.grad is not None:
            grads[n].append(p.grad.data.detach())

            if not warmup or len(grads[n]) == window_size and not trigger:
                if filter_type == "mean":
                    avg = sum(grads[n]) / len(grads[n])
                elif filter_type == "sum":
                    avg = sum(grads[n])
                else:
                    raise ValueError(f"Unrecognized filter_type {filter_type}")
                p.grad.data = p.grad.data + avg * lamb

    return grads

class MLPRaw:
    def __init__(self, vocab_size, context_length, embedding_size, hidden_size, rng):
        v, t, e, h = vocab_size, context_length, embedding_size, hidden_size
        self.embedding_size = embedding_size
        self.wte = torch.tensor(rng.randn(v * e, mu=0, sigma=1.0)).view(v, e)
        scale = 1 / math.sqrt(e * t)
        self.fc1_weights = torch.tensor(rng.rand(t * e * h, -scale, scale)).view(h, t * e).T
        self.fc1_bias = torch.tensor(rng.rand(h, -scale, scale))
        scale = 1 / math.sqrt(h)
        self.fc2_weights = torch.tensor(rng.rand(v * h, -scale, scale)).view(v, h).T
        self.fc2_bias = torch.tensor(rng.rand(v, -scale, scale))
        for p in self.parameters():
            p.requires_grad = True

    def parameters(self):
        return [self.wte, self.fc1_weights, self.fc1_bias, self.fc2_weights, self.fc2_bias]

    def __call__(self, idx, targets=None):
        return self.forward(idx, targets)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        emb = self.wte[idx]
        emb = emb.view(B, -1)
        hidden = torch.tanh(emb @ self.fc1_weights + self.fc1_bias)
        logits = hidden @ self.fc2_weights + self.fc2_bias
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)
        return logits, loss

class MLP(nn.Module):
    def __init__(self, vocab_size, context_length, embedding_size, hidden_size, rng):
        super().__init__()
        self.wte = nn.Embedding(vocab_size, embedding_size)
        self.mlp = nn.Sequential(
            nn.Linear(context_length * embedding_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, vocab_size)
        )
        self.reinit(rng)

    @torch.no_grad()
    def reinit(self, rng):
        def reinit_tensor_randn(w, mu, sigma):
            winit = torch.tensor(rng.randn(w.numel(), mu=mu, sigma=sigma))
            w.copy_(winit.view_as(w))

        def reinit_tensor_rand(w, a, b):
            winit = torch.tensor(rng.rand(w.numel(), a=a, b=b))
            w.copy_(winit.view_as(w))

        reinit_tensor_randn(self.wte.weight, mu=0, sigma=1.0)
        scale = (self.mlp[0].in_features)**-0.5
        reinit_tensor_rand(self.mlp[0].weight, -scale, scale)
        reinit_tensor_rand(self.mlp[0].bias, -scale, scale)
        scale = (self.mlp[2].in_features)**-0.5
        reinit_tensor_rand(self.mlp[2].weight, -scale, scale)
        reinit_tensor_rand(self.mlp[2].bias, -scale, scale)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        emb = self.wte(idx)
        emb = emb.view(B, -1)
        logits = self.mlp(emb)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)
        return logits, loss

def dataloader(tokens, context_length, batch_size):
    n = len(tokens)
    inputs, targets = [], []
    pos = 0
    while True:
        window = tokens[pos:pos + context_length + 1]
        inputs.append(window[:-1])
        targets.append(window[-1])
        if len(inputs) == batch_size:
            yield (torch.tensor(inputs), torch.tensor(targets))
            inputs, targets = [], []
        pos += 1
        if pos + context_length >= n:
            pos = 0

@torch.inference_mode()
def eval_split(model, tokens, max_batches=None):
    total_loss = 0
    num_batches = len(tokens) // batch_size
    if max_batches is not None:
        num_batches = min(num_batches, max_batches)
    data_iter = dataloader(tokens, context_length, batch_size)
    for _ in range(num_batches):
        inputs, targets = next(data_iter)
        logits, loss = model(inputs, targets)
        total_loss += loss.item()
    mean_loss = total_loss / num_batches
    return mean_loss

def softmax(logits):
    maxval = torch.max(logits)
    exps = torch.exp(logits - maxval)
    probs = exps / torch.sum(exps)
    return probs

def sample_discrete(probs, coinf):
    cdf = 0.0
    for i, prob in enumerate(probs):
        cdf += prob
        if coinf < cdf:
            return i
    return len(probs) - 1

# Main execution
if __name__ == "__main__":
    # Tokenization
    train_text = open('data/train.txt', 'r').read()
    assert all(c == '\n' or ('a' <= c <= 'z') for c in train_text)
    uchars = sorted(list(set(train_text)))
    vocab_size = len(uchars)
    char_to_token = {c: i for i, c in enumerate(uchars)}
    token_to_char = {i: c for i, c in enumerate(uchars)}
    EOT_TOKEN = char_to_token['\n']
    
    test_tokens = [char_to_token[c] for c in open('data/test.txt', 'r').read()]
    val_tokens = [char_to_token[c] for c in open('data/val.txt', 'r').read()]
    train_tokens = [char_to_token[c] for c in open('data/train.txt', 'r').read()]

    # Model creation
    context_length = 3
    embedding_size = 64
    hidden_size = 1024
    init_rng = RNG(1337)
    model = MLP(vocab_size, context_length, embedding_size, hidden_size, init_rng)

    # Optimizer
    learning_rate = 0.01
    optimizer = GrokFastAdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Initialize gradfilter_ma
    grads = None

    # Training loop
    timer = StepTimer()
    batch_size = 128
    num_steps = 50000
    print(f'num_steps {num_steps}, num_epochs {num_steps * batch_size / len(train_tokens):.2f}')
    train_data_iter = dataloader(train_tokens, context_length, batch_size)
    
    for step in range(num_steps):
        lr = learning_rate * 0.5 * (1 + math.cos(math.pi * step / num_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        last_step = step == num_steps - 1
        if step % 200 == 0 or last_step:
            train_loss = eval_split(model, train_tokens, max_batches=20)
            val_loss = eval_split(model, val_tokens)
            print(f'step {step:6d} | train_loss {train_loss:.6f} | val_loss {val_loss:.6f} | lr {lr:e} | time/step {timer.get_dt()*1000:.4f}ms')
        
        with timer:
            inputs, targets = next(train_data_iter)
            logits, loss = model(inputs, targets)
            loss.backward()
            grads = gradfilter_ma(model, grads)
            optimizer.step()
            optimizer.zero_grad()

    # Model inference
    sample_rng = RNG(42)
    prompt = "\nrichard"
    context = [char_to_token[c] for c in prompt]
    assert len(context) >= context_length
    context = context[-context_length:]
    print(prompt, end='', flush=True)
    
    with torch.inference_mode():
        for _ in range(200):
            context_tensor = torch.tensor(context).unsqueeze(0)
            logits, _ = model(context_tensor)
            probs = softmax(logits[0])
            coinf = sample_rng.random()
            next_token = sample_discrete(probs, coinf)
            context = context[1:] + [next_token]
            print(token_to_char[next_token], end='', flush=True)
    print()

    # Report test loss
    test_loss = eval_split(model, test_tokens)
    print(f'test_loss {test_loss}')
