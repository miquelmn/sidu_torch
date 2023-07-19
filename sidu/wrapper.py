from sidu import method
from typing import Union

import numpy as np
import torch


def sidu_wrapper(
        net: torch.nn.Module,
        layer,
        image: Union[np.array, torch.Tensor],
        device: torch.device = None,
        *args,
        **kwargs
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    activation = {}

    def hook(model, input, output):
        activation["layer"] = output.detach()

    layer.register_forward_hook(hook)
    _ = net(image)

    return method.sidu(net, activation["layer"], image, device=device).cpu().detach().numpy()
