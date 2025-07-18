import torch

from pipeline.checkpoint import Checkpoint

DTYPE_MAPPING = {
    torch.float32: "float",
}


def format_name(name, tensor):
    return f"{name}_{DTYPE_MAPPING[tensor.dtype]}_{'_'.join(str(x) for x in tensor.size())}"


def export_layer(name, layer, weight_dtype=None, bias_dtype=None):
    ret = {}
    if hasattr(layer, "weight"):
        weight = layer.weight.to(weight_dtype) if weight_dtype else layer.weight
        ret[format_name(f"{name}_w", weight)] = weight
    if hasattr(layer, "bias"):
        bias = layer.bias.to(bias_dtype) if bias_dtype else layer.bias
        ret[format_name(f"{name}_b", bias)] = bias
    return ret


def export_model(model_class, checkpoint_path, output_path):
    checkpoint = Checkpoint(checkpoint_path, "cpu", model_class, None, {})
    best_model = checkpoint.load_best()
    with torch.no_grad():
        for name, tensor in best_model.export().items():
            file_name = f"{output_path}\\{name}"
            with open(file_name, "wb") as f:
                f.write(tensor.contiguous().numpy().tobytes())
