from .tensor import Tensor


def _check_lengths(predictions, targets, fn_name):
    if len(predictions) != len(targets):
        raise ValueError(
            f"{fn_name}: predictions and targets must have the same length, "
            f"got {len(predictions)} and {len(targets)}"
        )


def mseLoss(predictions, targets):
    _check_lengths(predictions, targets, 'mseLoss')
    return sum([(yout - ygt) ** 2 for ygt, yout in zip(targets, predictions)])


def maeLoss(predictions, targets):
    _check_lengths(predictions, targets, 'maeLoss')
    return sum([abs(yout - ygt) for ygt, yout in zip(targets, predictions)])


def bceLoss(predictions, targets):
    _check_lengths(predictions, targets, 'bceLoss')
    losses = []
    for ygt, yout in zip(targets, predictions):
        if not isinstance(yout, Tensor):
            raise TypeError(f"bceLoss: predictions must be Tensor, got {type(yout).__name__}")
        p = yout.data.item()
        if not (0 < p < 1):
            raise ValueError(
                f"bceLoss: predictions must be in (0, 1), got {p}. "
                "Apply sigmoid before computing BCE loss."
            )
        losses.append(ygt * yout.log() + (Tensor(1.0) - ygt) * (Tensor(1.0) - yout).log())
    return -sum(losses)
