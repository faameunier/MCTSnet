import torch


def diff_argmax(x):
    # will not propagate grad to x
    indexes = torch.arange(0., end=x.size(1), requires_grad=True)
    indexes = indexes.repeat(x.size(0))
    y = x - torch.max(x)
    y = torch.sign(y)
    y.retain_grad()
    y = y + 1
    y = y * indexes
    return torch.max(y)


def softargmax(x, beta=1e10):
    x_range = torch.arange(0., end=x.size(1), requires_grad=True, dtype=x.dtype)
    return torch.sum(torch.nn.functional.softmax(x * beta) * x_range)
