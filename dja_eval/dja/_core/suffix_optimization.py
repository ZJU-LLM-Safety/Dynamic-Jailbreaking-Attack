"""Helpers for suffix-only optimization workflows."""


def freeze_model_parameters_for_suffix_optimization(model):
    """Freeze model parameters while keeping forward/backward through inputs intact."""
    for param in model.parameters():
        param.requires_grad = False
    return model
