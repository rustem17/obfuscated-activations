import numpy as np
import torch
import torch.nn.functional as F

# Import the utility you want to test. Adjust the PYTHONPATH accordingly.
from src.utils import get_valid_token_mask

# Also import the base Probe class from probe_training.py,
# as we will use its compute_loss method.
from src.probe_training import Probe

# For the purpose of testing, we define a dummy probe that simply applies a linear transformation.
class DummyProbe(Probe):
    def __init__(self, d_model):
        super().__init__()
        self.linear = torch.nn.Linear(d_model, 1)
    def forward(self, x):
        out = self.linear(x).squeeze(-1)
        return out

def test_get_valid_token_mask_behavior():
    """
    Use a dummy predicate: a token is considered a “start” token if its value is greater than 10.
    In our test, we pass a token_range as [dummy_predicate, None]. (None never matches an end condition.)

    IMPORTANT: According to get_valid_token_mask’s implementation:
      - When a token triggers the start condition, that token is NOT marked valid.
      - Subsequent tokens are marked as valid until the end condition is met.

    Thus, given our input below, we expect:
      For the first row ([8, 11, 9, 15, 10]):
         j=0: 8 does not trigger → False.
         j=1: 11 triggers start, but is not marked → False.
         j=2: 9 is after start → True.
         j=3: 15 triggers start again → False.
         j=4: 10 is after start → True.
      For the second row ([12, 7, 13, 5, 20]):
         j=0: 12 triggers start → False.
         j=1: 7 is after start → True.
         j=2: 13 triggers start → False.
         j=3: 5 is after start → True.
         j=4: 20 triggers start → False.
    """
    def dummy_predicate(seq_idx, token, tokens):
        return token > 10

    token_range = [dummy_predicate, None]

    # Create a dummy input_ids tensor (batch_size=2, seq_len=5)
    input_ids = torch.tensor([
        [8, 11, 9, 15, 10],
        [12, 7, 13, 5, 20]
    ])

    mask = get_valid_token_mask(input_ids, token_range)

    # According to our logic above, the expected mask is:
    #
    # For row1: [False, False, True, False, True]
    # For row2: [False, True, False, True, False]
    expected_mask = torch.tensor([
        [False, False, True, False, True],
        [False, True, False, True, False]
    ])

    assert torch.equal(mask, expected_mask), f"Mask differs from expected.\nGot: {mask}\nExpected: {expected_mask}"
    print("test_get_valid_token_mask_behavior: PASS")

def test_loss_computation_with_mask():
    """
    Create dummy activations and targets so that the loss computed by compute_loss
    of DummyProbe is well defined and does not produce NaN.
    We simulate activations of shape (batch_size, seq_len, d_model),
    and we use a dummy mask computed by checking whether the first element of each activation
    is positive.
    """
    batch_size = 2
    seq_len = 5
    d_model = 4

    torch.manual_seed(0)
    activations = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    # Create dummy targets as ones.
    targets = torch.ones(batch_size, seq_len)

    # For this test, we compute a mask where positions are valid if activations[...,0] > 0.
    mask = activations[..., 0] > 0

    probe = DummyProbe(d_model)

    loss = probe.compute_loss(activations, targets, mask=mask)

    assert not torch.isnan(loss), f"Loss is NaN: {loss}"

    loss.backward()
    for name, param in probe.named_parameters():
        if param.grad is not None:
            assert not torch.isnan(param.grad).any(), f"Gradient for {name} contains NaN."
    print("test_loss_computation_with_mask: PASS")

if __name__ == "__main__":
    print("Running tests for the masking utility ...")
    test_get_valid_token_mask_behavior()
    test_loss_computation_with_mask()
    print("All tests passed!")
