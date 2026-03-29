"""Ranking model for logging and target policy"""

import torch


class ScaledTanhLinear(torch.nn.Module):
    def __init__(self, scaling_factor, *args, **kwargs):
        super(ScaledTanhLinear, self).__init__(*args, **kwargs)
        self.scaling_factor = scaling_factor

    def forward(self, x, key_padding_mask=None, *args, **kwargs) -> torch.Tensor:
        """Note that the masking is of the torch kind (False = keep)"""
        ret = self.scaling_factor * torch.tanh(x)
        if key_padding_mask is not None:
            return torch.where(
                key_padding_mask.view(*key_padding_mask.shape, 1),
                torch.zeros_like(ret),
                ret,
            )
        return ret


class ScaledSigmoidLinear(torch.nn.Module):
    def __init__(self, scaling_factor, *args, **kwargs):
        super(ScaledSigmoidLinear, self).__init__(*args, **kwargs)
        self.scaling_factor = scaling_factor * 2

    def forward(self, x, key_padding_mask=None, *args, **kwargs) -> torch.Tensor:
        """Note that the masking is of the torch kind (False = keep)"""
        uniform_low, uniform_high = (
            torch.finfo(x.dtype).tiny,
            1 - torch.finfo(x.dtype).eps,
        )
        uniform_range = uniform_high - uniform_low
        ret = self.scaling_factor * (uniform_low + torch.sigmoid(x) * uniform_range)
        if key_padding_mask is not None:
            return torch.where(
                key_padding_mask.view(*key_padding_mask.shape, 1),
                torch.zeros_like(ret),
                ret,
            )
        return ret


class FixedDropoutLayer(torch.nn.Module):
    def __init__(self, mask_idx: torch.Tensor | list, n_feat: int):
        """
        mask: 1D tensor of idx denoting which values to KEEP (i.e. values not on the list will be zeroed out)
        """
        super().__init__()
        mask = torch.zeros(n_feat)
        mask[mask_idx] = 1.0
        self.register_buffer("mask", mask)
        print(f"Applying mask to input: {mask}")

    def forward(self, x):
        return x * self.mask


class VLPL(torch.nn.Module):
    def __init__(
        self,
        n_feat,
        n_hidden,
        k,
        bias_output=True,
        weight_initializer=None,
        bias_initializer=None,
        output_weight_initializer=None,
        output_bias_initializer=None,
        n_hidden_layers=2,
        nn_sequence=False,
        input_dropout=0.0,
        linear_dropout=0.0,
        constrain_outputs_fn="tanh",
        constrain_outputs=0.0,
        output_name="logits",
        padding_value=-torch.inf,
        activation="tanh",
        fixed_input_dropout_idx=None,
        *args,
        **kwargs,
    ):
        super(VLPL, self).__init__()
        self.n_hidden = n_hidden
        self.k = k
        self.n_feat = n_feat
        self.bias_output = bias_output
        self.n_hidden_layers = n_hidden_layers
        self.nn_sequence = nn_sequence
        self.input_dropout = input_dropout
        self.linear_dropout = linear_dropout
        self.constrain_outputs_fn = {
            "tanh": ScaledTanhLinear,
            "sigmoid": ScaledSigmoidLinear,
        }.get(constrain_outputs_fn, None)
        self.final_outputs = (
            [self.constrain_outputs_fn(constrain_outputs)] if constrain_outputs else []
        )
        self.output_name = output_name
        self.padding_value = padding_value
        self.activation = {
            "sigmoid": torch.nn.Sigmoid,
            "tanh": torch.nn.Tanh,
            "relu": torch.nn.ReLU,
        }.get(activation, torch.nn.Sigmoid)
        iterator = (
            (lambda x: torch.nn.Sequential(*x))
            if nn_sequence
            else torch.nn.ParameterList
        )
        self.model = iterator(
            (
                [FixedDropoutLayer(fixed_input_dropout_idx, self.n_feat)]
                if fixed_input_dropout_idx is not None
                else []
            )
            + ([torch.nn.Dropout(input_dropout)] if not nn_sequence else [])
            + [
                torch.nn.Linear(self.n_feat, self.n_hidden, dtype=torch.float32),
                self.activation(),
                *[
                    layer
                    for _ in range(n_hidden_layers - 1)
                    for layer in (
                        [
                            torch.nn.Linear(
                                self.n_hidden, self.n_hidden, dtype=torch.float32
                            ),
                            self.activation(),
                        ]
                        + (
                            [torch.nn.Dropout(linear_dropout)]
                            if not nn_sequence
                            else []
                        )
                    )
                ],
                torch.nn.Linear(
                    self.n_hidden, self.k, dtype=torch.float32, bias=bias_output
                ),
            ]
            + self.final_outputs
        )
        # Manually overwrite init, otherwise doing weird torch.nn.Linear init
        # https://github.com/pytorch/pytorch/blob/main/torch/nn/modules/linear.py
        with torch.no_grad():
            for i in range(len(self.model)):
                if isinstance(self.model[i], torch.nn.Linear):
                    if i + 1 < len(self.model):
                        if weight_initializer is not None:
                            weight_initializer(self.model[i].weight)
                        if (
                            self.model[i].bias is not None
                            and bias_initializer is not None
                        ):
                            bias_initializer(self.model[i].bias)
                    else:  # Final layer
                        if output_weight_initializer is not None:
                            output_weight_initializer(self.model[i].weight)
                        if (
                            self.model[i].bias is not None
                            and output_bias_initializer is not None
                        ):
                            output_bias_initializer(self.model[i].bias)
        return

    def forward(self, input, padding_mask=None, *args, **kwargs):
        x = input
        # Compatibility with old models
        if self.nn_sequence:
            x = self.model(x)
        else:
            for layer in self.model:
                if isinstance(layer, ScaledTanhLinear):
                    x = layer(x, key_padding_mask=~padding_mask, need_weights=False)
                else:
                    x = layer(x)
        if padding_mask is not None:
            x = torch.where(padding_mask[:, :, None], x, self.padding_value)
        if self.output_name is not None:
            x = {self.output_name: x}
        return x
