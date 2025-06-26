from __future__ import annotations

if __name__ == "__main__":
    import os
    import torch
    from torch.nn.attention.flex_attention import flex_attention

    os.environ["TORCH_LOGS"] = "+dynamo,+inductor"

    flex_attention = torch.compile(flex_attention, fullgraph=True, mode="max-autotune")
