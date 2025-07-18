import torch

from checkpoint import Checkpoint


def format_name(name, tensor):
    return f"{name}_{'_'.join(str(x) for x in tensor.size())}"


def export_layer(name, layer, weight_dtype=torch.float32, bias_dtype=torch.float32):
    ret = {}
    if hasattr(layer, "weight"):
        ret[format_name(f"{name}_w", layer.weight)] = layer.weight.to(weight_dtype)
    if hasattr(layer, "bias"):
        ret[format_name(f"{name}_b", layer.bias)] = layer.bias.to(bias_dtype)
    return ret


def export_model(model_class, checkpoint_path, output_path):
    checkpoint = Checkpoint(checkpoint_path, "cpu", model_class, None, {})
    best_model = checkpoint.load_best()
    for name, tensor in best_model.export().items():
        file_name = f"{output_path}\\{name}"
        with open(file_name, "wb") as f:
            f.write(tensor.contiguous().numpy().tobytes())
