import torch
from dtypes import Dtypes
from linear_fp8 import LinearReplacer


input = torch.randn((4, 4), device='cuda')
linear = torch.nn.Linear(4, 8).cuda()
model = LinearReplacer.replace(linear, Dtypes.kfloat32)

output = linear(input)
fp8_output = model(input)

assert fp8_output.dtype == torch.float32
torch.testing.assert_close(fp8_output.size(), torch.Size((4, 8)))
torch.testing.assert_close(output, fp8_output, rtol=0, atol=0.1)
