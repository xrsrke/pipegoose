import torch
from pipegoose.nn.fusion.ops import FusedDummy

x = torch.arange(10, dtype=torch.float32)
fused_op = FusedDummy()
output = fused_op.forward(x)

torch.testing.assert_close(output, x)
