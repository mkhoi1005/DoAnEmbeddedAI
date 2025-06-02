import torch
import torch_pruning as tp
from ultralytics import YOLO
import argparse
import os

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to YOLOv8 .pt model')
    parser.add_argument('--data', type=str, required=True, help='Path to dataset .yaml file')
    parser.add_argument('--task', type=str, default='segment', help='Task type: detect, segment, classify')
    parser.add_argument('--prune_rate', type=float, default=0.5, help='Global pruning ratio')
    parser.add_argument('--epochs', type=int, default=40, help='Number of fine-tuning epochs after pruning')
    parser.add_argument('--output', type=str, default='pruned.pt', help='Output path for pruned model')
    return parser.parse_args()

def main(opt):
    assert opt.task == 'segment', "This script currently supports only instance segmentation (task=segment)."

    print(f"🔧 Loading model from {opt.model}")
    model = YOLO(opt.model)

    print("🧩 Building model graph for pruning...")
    example_inputs = torch.randn(1, 3, 640, 640).to(model.device)

    ignored_layers = [model.model.model[-1]]  # Avoid pruning the detection head
    DG = tp.DependencyGraph().build_dependency(model.model, example_inputs=example_inputs)

    prunable_layers = [
        m for m in model.model.modules()
        if isinstance(m, torch.nn.Conv2d)
        and m not in ignored_layers
        and m in DG.module2node
    ]   
    total = sum(m.weight.numel() for m in prunable_layers)

    for m in prunable_layers:
        # Tính số lượng kênh cần prune
        out_channels = m.weight.shape[0]
        n_prune = int(opt.prune_rate * out_channels)
        if n_prune < 1 or out_channels - n_prune < 1:
            continue  # bỏ qua nếu không đủ kênh để prune

        # L1 norm để chọn kênh nhỏ nhất
        weight_copy = m.weight.data.abs().mean(dim=(1,2,3))
        prune_idx = torch.argsort(weight_copy)[:n_prune].tolist()

        pruning_group = DG.get_pruning_group(m, tp.prune_conv_out_channels, prune_idx)
        if pruning_group is not None:
            pruning_group.prune()

    print("💾 Saving pruned model temporarily...")
    model.save("pruned_temp.pt")

    print("📚 Fine-tuning pruned model...")
    model = YOLO("pruned_temp.pt")
    model.train(data=opt.data, epochs=opt.epochs, task=opt.task)

    print(f"✅ Saving final pruned + fine-tuned model to {opt.output}")
    model.save(opt.output)

    os.remove("pruned_temp.pt")  # cleanup
    print("🧹 Temporary files removed.")

if __name__ == "__main__":
    opt = parse_opt()
    main(opt)