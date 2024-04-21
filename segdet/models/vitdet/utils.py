from torch import nn


class PrintShape(nn.Module):
    def __init__(self, base_msg: str = None):
        super().__init__()
        self.base_msg = base_msg

    def forward(self, x):
        if self.base_msg is not None:
            print(self.base_msg)
        print(x.shape)
        return x
