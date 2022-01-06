import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numbers

from torch.nn.parameter import Parameter
from typing import Union, List
from torch.nn.modules.batchnorm import _NormBase

class LayerNormRepara(nn.Module):
    """
    A regular torch.nn.LayerNorm layer, except the multiplicative affine
    parameter may be reparameterized in a few different ways. Behavior
    matches torch.nn.LayerNorm, except elementwise_affine=True always,
    and 'parameterization' may take the following values:

    'default': Identical to torch.nn.LayerNorm with elementwise_affine=True
    'centered': Multiplicative weights are shifted to be centered around one
    'exponential': Multiplicative weights are exponentiated
    """

    def __init__(self, normalized_shape: Union[int, List[int], torch.Size], eps: float = 1e-5, parameterization: str = 'default', device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(LayerNormRepara, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.parameterization = parameterization

        self.weight = Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
        self.bias = Parameter(torch.empty(self.normalized_shape, **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.parameterization == 'default':
            init.ones_(self.weight)
        elif self.parameterization in ['centered', 'exponential']:
            init.zeros_(self.weight)
        else:
            raise ValueError(
                f"'parameterization' must be in ['default', 'centered', 'exponential'], was {self.parameterization}."
            )
        init.zeros_(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.parameterization == 'default':
            weight = self.weight
        elif self.parameterization == 'centered':
            weight = 1 + self.weight
        elif self.parameterization == 'exponential':
            weight = torch.exp(self.weight)
        else:
            raise ValueError(
                f"'parameterization' must be in ['default', 'centered', 'exponential'], was {self.parameterization}."
            )
        return F.layer_norm(
            input, self.normalized_shape, weight, self.bias, self.eps
        )

    def extra_repr(self) -> str:
        return "{normalized_shape}, eps={eps}, " \
            "parameterization={parameterization}".format(**self.__dict__)


class _BatchNormRepara(_NormBase):
    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        parameterization='default',
        track_running_stats=True,
        device=None,
        dtype=None
    ):
        self.parameterization = parameterization
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(_BatchNormRepara, self).__init__(
            num_features, eps, momentum, True, track_running_stats, **factory_kwargs
        )

    def reset_parameters(self) -> None:
        self.reset_running_stats()
        if self.parameterization == 'default':
            init.ones_(self.weight)
        elif self.parameterization in ['centered', 'exponential']:
            init.zeros_(self.weight)
        else:
            raise ValueError(
                f"'parameterization' must be in ['default', 'centered', 'exponential'], was {self.parameterization}."
            )
        init.zeros_(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked = self.num_batches_tracked + 1  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Reparameterize weights.
        """
        if self.parameterization == 'default':
            weight = self.weight
        elif self.parameterization == 'centered':
            weight = 1 + self.weight
        elif self.parameterization == 'exponential':
            weight = torch.exp(self.weight)
        else:
            raise ValueError(
                f"'parameterization' must be in ['default', 'centered', 'exponential'], was {self.parameterization}."
            )


        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        return F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean
            if not self.training or self.track_running_stats
            else None,
            self.running_var if not self.training or self.track_running_stats else None,
            weight,
            self.bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )

class BatchNorm2dRepara(_BatchNormRepara):
    """
    Reparameterized batch norm. Copied form pytorch code.
    """
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError("expected 4D input (got {}D input)".format(input.dim()))
