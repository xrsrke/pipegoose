from torch import nn


class Callback:
    # NOTE: add more events
    # NOTE: READING
    # + Pytorch lightning's Callback

    def on_fit_start(self, trainer: "pipegoose.Trainer", pl_module: nn.Module) -> None:
        """Called when fit begins."""

    def on_fit_end(self, trainer: "pipegoose.Trainer", pl_module: nn.Module) -> None:
        """Called when fit ends."""
