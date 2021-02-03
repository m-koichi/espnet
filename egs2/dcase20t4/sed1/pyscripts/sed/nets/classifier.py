import torch
from torch import Tensor, dropout
from egs2.dcase20t4.sed1.pyscripts.sed.abs_sed import AbsSED


class LinearClassifier(torch.nn.Module):
    """Simple linear layer module for classification"""

    def __init__(
        self,
        idim,
        n_classes,
    ):
        super(LinearClassifier, self).__init__()
        self.classifier = torch.nn.Sequential(
                        torch.nn.Linear(idim, n_classes),
                    )
 
    def forward(self, x: Tensor,) -> Tensor:
        x = self.classifier(x)
        out = {
            "weak": x[:, 0, :],
            "strong": x[:, 1:, :],
        }
        return out
