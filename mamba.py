import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PurePyTorchMambaBlock(nn.Module):
    def __init__(self, d_model: int = 128, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = self.expand * self.d_model
        self.dt_rank = math.ceil(self.d_model / 16)

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=True
        )

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        xz = self.in_proj(x)
        u, z = xz.chunk(2, dim=-1)

        u_conv = u.transpose(1, 2)
        u_conv = self.conv1d(u_conv)[:, :, :seq_len]
        u_conv = F.silu(u_conv).transpose(1, 2)

        x_dbl = self.x_proj(u_conv)
        delta_raw, B_raw, C_raw = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        delta = F.softplus(self.dt_proj(delta_raw))
        A = -torch.exp(self.A_log)

        h = torch.zeros(batch, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        y_list = []

        for t in range(seq_len):
            dt = delta[:, t, :].unsqueeze(-1)
            b_t = B_raw[:, t, :].unsqueeze(1)
            c_t = C_raw[:, t, :].unsqueeze(-1)
            u_t = u_conv[:, t, :].unsqueeze(-1)

            A_bar = torch.exp(dt * A.unsqueeze(0))
            B_bar = dt * b_t

            h = A_bar * h + B_bar * u_t

            y_t = torch.matmul(h, c_t).squeeze(-1)
            y_t = y_t + self.D * u_conv[:, t, :]
            y_list.append(y_t)

        y = torch.stack(y_list, dim=1)
        out = self.out_proj(y * F.silu(z))
        return out

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PurePyTorchMambaBlock(d_model=64, d_state=16).to(device)
    dummy_input = torch.randn(2, 16, 64, device=device)
    out = model(dummy_input)
    assert out.shape == dummy_input.shape
    print(f"PurePyTorchMambaBlock self-check passed on {device.type}.")
