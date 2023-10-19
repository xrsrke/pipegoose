class PipelineGradientFlowError(Exception):
    """The gradients can't flow to leaf tensors"""


class PipelineNoSavedActivationError(Exception):
    """Can't find saved activations to do backpropogation"""


class PipelineNoSavedInput(Exception):
    """Can't find the input activations to return the gradients"""


class PipelineInputNotRequiresGrad(Exception):
    """The input of the pipeline stage must requires grad in order for gradients to flow back"""
