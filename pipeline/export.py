from checkpoint import Checkpoint


def export_model(model_class, checkpoint_path, output_path):
    checkpoint = Checkpoint(checkpoint_path, "cpu", model_class, None, {})
    best_model = checkpoint.load_best()
    for name, tensor in best_model.export():
        file_name = f"{output_path}\\{name}"
        with open(file_name, "wb") as f:
            f.write(tensor.contiguous().numpy().tobytes())
