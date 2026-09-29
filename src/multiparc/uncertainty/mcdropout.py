import torch
import torch.nn.functional as F


def add_mc_dropout_hooks(model, p=0.1, layer_types=(torch.nn.Conv2d),
                         exclude_names=()):
    '''
    Registers forward hooks that inject MC dropout on the output of matching
    layers, without modifying the model's definition.

    Applies channel-wise dropout (F.dropout2d) to Conv2d layer outputs and
    element-wise dropout (F.dropout) to all other matched layer types.
    Dropout is applied with training=True regardless of the model's current
    mode, so it stays active even if the model is in eval() (e.g. to keep
    BatchNorm layers in inference mode while still sampling dropout masks).

    Args:
        model: the nn.Module to attach hooks to.
        p: dropout probability applied at every hooked layer.
        layer_types: tuple of nn.Module subclasses whose outputs should
            have dropout applied. Defaults to Conv2d only.
        exclude_names: names (as returned by model.named_modules()) to
            skip, e.g. the final output layer or a constraint-enforcing
            projection layer, so dropout isn't applied there.

    Returns:
        List of hook handles. Call handle.remove() on each to restore the
        model's original deterministic behavior.
    '''
    handles = []

    def make_hook(is_conv):
        def hook(module, input, output):
            if is_conv:
                return F.dropout2d(output, p=p, training=True)
            else:
                return F.dropout(output, p=p, training=True)
        return hook

    for name, module in model.named_modules():
        if isinstance(module, layer_types) and name not in exclude_names:
            is_conv = isinstance(module, torch.nn.Conv2d)
            handle = module.register_forward_hook(make_hook(is_conv))
            handles.append(handle)

    return handles
