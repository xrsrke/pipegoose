import subprocess
import sys


def test_init_tensor_parallel_group():
    command = [
        sys.executable,
        "-m",
        "torch.distributed.launch",
        "--nproc-per-node=8",
        "./tests/distributed/_initializers/init_tensor_parallel_group.py",
    ]
    result = subprocess.run(command, capture_output=True, text=True)

    assert result.returncode == 0, f"Command failed with output: {result.stdout}, {result.stderr}"
