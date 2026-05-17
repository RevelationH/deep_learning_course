from __future__ import annotations

from typing import Dict, Optional


def _variant_explanation(explanation: str, *, variant_index: int) -> str:
    text = explanation.strip()
    if not text:
        return text
    variant = variant_index % 3
    if variant == 0:
        return text
    if variant == 1:
        return "下面给你一个等价的简洁写法。 " + text
    return "换一种更贴近复习场景的写法来看。 " + text


def _variant_code(code: str, *, variant_index: int) -> str:
    normalized = code.strip()
    variant = variant_index % 3
    if variant == 0:
        return normalized
    if variant == 1:
        return "# Alternative minimal example\n" + normalized
    return "# Second minimal example\n" + normalized


def _wrap_answer(
    code: str,
    *,
    explanation: str = "",
    code_only: bool = False,
    variant_index: int = 0,
) -> Dict[str, str]:
    code_block = f"```python\n{_variant_code(code, variant_index=variant_index)}\n```"
    if code_only or not explanation.strip():
        return {"answer": code_block}
    return {"answer": f"{_variant_explanation(explanation, variant_index=variant_index)}\n\n{code_block}"}


def build_code_template(
    query: str,
    *,
    code_only: bool = False,
    variant_index: int = 0,
) -> Optional[Dict[str, str]]:
    lowered = " ".join(str(query or "").lower().split())

    if any(token in lowered for token in ["fully connected", "full connected", "全连接", "mlp", "multi-layer perceptron"]):
        return _wrap_answer(
            """
import torch
import torch.nn as nn


class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )

    def forward(self, x):
        return self.net(x)


model = SimpleMLP()
x = torch.tensor([[0.2, 0.5, -1.0, 1.2]], dtype=torch.float32)
logits = model(x)
print(logits)
            """,
            explanation="这是一份最简单的全连接网络示例：输入先经过一层线性映射和激活函数，再经过第二层线性映射得到输出。",
            code_only=code_only,
            variant_index=variant_index,
        )

    if ("pytorch" in lowered or "torch" in lowered) and ("convolution" in lowered or "conv" in lowered or "卷积" in lowered):
        return _wrap_answer(
            """
import torch
import torch.nn as nn

conv = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1)
x = torch.randn(1, 1, 28, 28)
y = conv(x)

print("input shape:", x.shape)
print("output shape:", y.shape)
            """,
            explanation="这段代码用 PyTorch 构造了一个最小卷积层，并演示了输入张量经过卷积后的输出形状变化。",
            code_only=code_only,
            variant_index=variant_index,
        )

    if "backprop" in lowered or "反向传播" in lowered:
        return _wrap_answer(
            """
import torch
import torch.nn as nn
import torch.optim as optim

model = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Linear(4, 1),
)

x = torch.tensor([[1.0, 2.0], [2.0, 1.0], [3.0, 4.0]], dtype=torch.float32)
y = torch.tensor([[1.0], [0.0], [1.0]], dtype=torch.float32)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

optimizer.zero_grad()
pred = model(x)
loss = criterion(pred, y)
loss.backward()
optimizer.step()

print("loss:", float(loss))
            """,
            explanation="这是一段最小反向传播训练代码：前向计算损失后调用 `backward()` 求梯度，再用优化器更新参数。",
            code_only=code_only,
            variant_index=variant_index,
        )

    if "batchnorm" in lowered or "batch normalization" in lowered or "批归一化" in lowered:
        return _wrap_answer(
            """
import torch
import torch.nn as nn

layer = nn.Sequential(
    nn.Conv2d(3, 8, kernel_size=3, padding=1),
    nn.BatchNorm2d(8),
    nn.ReLU(),
)

x = torch.randn(4, 3, 32, 32)
y = layer(x)
print(y.shape)
            """,
            explanation="这份示例把 BatchNorm 接在卷积层后面，展示了它在 CNN 里的典型用法。",
            code_only=code_only,
            variant_index=variant_index,
        )

    if "layernorm" in lowered or "layer normalization" in lowered or "层归一化" in lowered:
        return _wrap_answer(
            """
import torch
import torch.nn as nn

layer_norm = nn.LayerNorm(normalized_shape=8)
x = torch.randn(2, 5, 8)  # [batch, seq_len, hidden_dim]
y = layer_norm(x)

print(y.shape)
            """,
            explanation="这段代码展示了 LayerNorm 在序列表示上的最小用法，输入的最后一维会被逐样本归一化。",
            code_only=code_only,
            variant_index=variant_index,
        )

    if "dropout" in lowered:
        return _wrap_answer(
            """
import torch
import torch.nn as nn

drop = nn.Dropout(p=0.5)
x = torch.ones(2, 6)

drop.train()
print(drop(x))

drop.eval()
print(drop(x))
            """,
            explanation="这份代码用同一个输入分别演示了 Dropout 在训练模式和测试模式下的行为差异。",
            code_only=code_only,
            variant_index=variant_index,
        )

    if "transformer block" in lowered or ("transformer" in lowered and "code" in lowered):
        return _wrap_answer(
            """
import torch
import torch.nn as nn


class SimpleTransformerBlock(nn.Module):
    def __init__(self, dim=32, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 64),
            nn.ReLU(),
            nn.Linear(64, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        return self.norm2(x + ffn_out)


block = SimpleTransformerBlock()
x = torch.randn(2, 6, 32)
y = block(x)
print(y.shape)
            """,
            explanation="这是一份最小 Transformer block 示例，包含自注意力、残差连接、前馈网络和 LayerNorm。",
            code_only=code_only,
            variant_index=variant_index,
        )

    if "lstm" in lowered:
        return _wrap_answer(
            """
import torch
import torch.nn as nn

lstm = nn.LSTM(input_size=10, hidden_size=16, num_layers=1, batch_first=True)
x = torch.randn(3, 5, 10)
y, (h, c) = lstm(x)

print("output:", y.shape)
print("hidden:", h.shape)
print("cell:", c.shape)
            """,
            explanation="这段代码给出一个最小 LSTM 例子，展示输入序列经过 LSTM 后的输出、隐藏状态和记忆单元。",
            code_only=code_only,
            variant_index=variant_index,
        )

    if "gan" in lowered:
        return _wrap_answer(
            """
import torch
import torch.nn as nn
import torch.optim as optim

generator = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 8))
discriminator = nn.Sequential(nn.Linear(8, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid())

g_opt = optim.Adam(generator.parameters(), lr=1e-3)
d_opt = optim.Adam(discriminator.parameters(), lr=1e-3)
criterion = nn.BCELoss()

real = torch.randn(4, 8)
noise = torch.randn(4, 16)
fake = generator(noise)

# update discriminator
d_opt.zero_grad()
real_loss = criterion(discriminator(real), torch.ones(4, 1))
fake_loss = criterion(discriminator(fake.detach()), torch.zeros(4, 1))
d_loss = real_loss + fake_loss
d_loss.backward()
d_opt.step()

# update generator
g_opt.zero_grad()
g_loss = criterion(discriminator(fake), torch.ones(4, 1))
g_loss.backward()
g_opt.step()

print("d_loss:", float(d_loss))
print("g_loss:", float(g_loss))
            """,
            explanation="这是一份极简 GAN 训练框架，展示了判别器和生成器交替更新的基本流程。",
            code_only=code_only,
            variant_index=variant_index,
        )

    if "diffusion" in lowered or "扩散模型" in lowered or "denoise" in lowered or "去噪" in lowered:
        return _wrap_answer(
            """
import torch
import torch.nn as nn


class TinyDenoiser(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
        )

    def forward(self, noisy_x):
        return self.net(noisy_x)


model = TinyDenoiser()
clean_x = torch.randn(4, 8)
noise = 0.1 * torch.randn_like(clean_x)
noisy_x = clean_x + noise
pred_noise = model(noisy_x)

print("predicted noise shape:", pred_noise.shape)
            """,
            explanation="这是一份极简扩散模型去噪骨架：先构造带噪输入，再让网络预测噪声。",
            code_only=code_only,
            variant_index=variant_index,
        )

    return None
