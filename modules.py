from torch import nn
import torch
from torch.distributions import Bernoulli

class SampleModule(nn.Module):
    """
    A sub-network that to be loaded and trained inside of the main model.
    This module framework becomes useful as model complexity increases.
    For example, you may have one big model with a number of sub-networks that
    all combine to process your data. Then, it can be easier to define each in
    terms of solely their inputs and outputs as Modules which can be loaded
    into the main network. It can also be easier to swap out different versions
    of sub-network components this way. Enjoy!
    """

    def __init__(self, ninp, nout):
        super(SampleModule, self).__init__()

        # --- Mappings ---
        self.fc = nn.Linear(ninp, nout)

    def forward(self, x):
        y = self.fc(x)
        return y
