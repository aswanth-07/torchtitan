# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import nullcontext
from types import SimpleNamespace
from typing import cast
from unittest.mock import MagicMock, patch

import torch

from torchtitan.trainer import Trainer


def test_pp_forward_backward_step_returns_sentinel_without_last_stage():
    trainer = cast(
        Trainer,
        SimpleNamespace(
            pp_has_first_stage=False,
            pp_has_last_stage=False,
            pp_schedule=SimpleNamespace(step=lambda **kwargs: None),
            train_context=nullcontext,
            post_dataloading_process=lambda input_dict, labels: (
                input_dict["input"],
                labels,
                {},
            ),
            device=torch.device("cpu"),
        ),
    )

    loss = Trainer.pp_forward_backward_step(
        trainer,
        input_dict_mbs=[{"input": torch.ones(1)}],
        label_mbs=[torch.ones(1)],
        global_valid_tokens=1.0,
    )

    torch.testing.assert_close(loss, torch.tensor([-1.0]))


def test_cuda_graph_wrapper_decorates_fwd_bwd_and_clones_reused_output():
    class PassthroughCUDAGraphWrapper:
        def __init__(self, fn, example_inputs):
            self.fn = fn

        def __call__(self, *args):
            return self.fn(*args)

    graph_loss = torch.tensor(0.0)
    trainer = cast(
        Trainer,
        SimpleNamespace(
            device=torch.device("cpu"),
        ),
    )
    fwd_bwd = MagicMock(return_value=graph_loss)

    accumulated_losses = []
    with patch("torchtitan.trainer.CUDAGraphWrapper", PassthroughCUDAGraphWrapper):
        runner = Trainer._wrap_with_cuda_graph(trainer, fwd_bwd)
        for value in (1.0, 2.0, 3.0):
            graph_loss.fill_(value)
            loss = runner(
                torch.ones(1),
                torch.ones(1),
                1.0,
                {"position": torch.ones(1)},
            )
            accumulated_losses.append(loss.detach())

    torch.testing.assert_close(
        torch.sum(torch.stack(accumulated_losses)), torch.tensor(6.0)
    )
    assert fwd_bwd.call_count == 3
    _, _, global_valid_tokens, extra_kwargs = fwd_bwd.call_args.args
    torch.testing.assert_close(global_valid_tokens, torch.tensor(1))
    assert global_valid_tokens.dtype == torch.int64
    torch.testing.assert_close(extra_kwargs["position"], torch.ones(1))
