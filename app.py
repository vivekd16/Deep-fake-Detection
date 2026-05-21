import argparse
import io
import json
from pathlib import Path

import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import models, transforms


CLASS_NAMES = ["fake", "real"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_model(model_name: str, dropout: float) -> torch.nn.Module:
    if model_name == "resnet18":
        model = models.resnet18(weights=None)
        in_features = model.fc.in_features
    elif model_name == "resnet34":
        model = models.resnet34(weights=None)
        in_features = model.fc.in_features
    elif model_name == "resnet50":
        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(dropout),
        torch.nn.Linear(in_features, 256),
        torch.nn.ReLU(),
        torch.nn.Dropout(dropout),
        torch.nn.Linear(256, 64),
        torch.nn.ReLU(),
        torch.nn.Dropout(dropout),
        torch.nn.Linear(64, 2),
    )
    return model


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.forward_handle = target_layer.register_forward_hook(self._forward_hook)
        self.backward_handle = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor: torch.Tensor, class_idx: int) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)
        score = logits[:, class_idx].sum()
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(
            cam,
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        cam = cam.squeeze().cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam


def create_heatmap_overlay(image: Image.Image, cam: np.ndarray, alpha: float = 0.4) -> Image.Image:
    base = image.convert("RGB")
    cam_uint8 = np.uint8(cam * 255.0)

    heatmap = np.zeros((cam_uint8.shape[0], cam_uint8.shape[1], 3), dtype=np.uint8)
    heatmap[..., 0] = cam_uint8
    heatmap[..., 1] = np.uint8(np.clip(255 - np.abs(cam_uint8.astype(np.int16) - 128) * 2, 0, 255))
    heatmap[..., 2] = np.uint8(255 - cam_uint8)

    heatmap_image = Image.fromarray(heatmap).resize(base.size)
    return Image.blend(base, heatmap_image, alpha=alpha)


def figure_to_pil() -> Image.Image:
    buffer = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format="png", bbox_inches="tight", dpi=150)
    plt.close()
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def build_training_curves_image(history_path: Path) -> Image.Image | None:
    if not history_path.exists():
        return None

    history = json.loads(history_path.read_text(encoding="utf-8"))
    epochs = [item["epoch"] for item in history]
    train_acc = [item["train"]["accuracy"] for item in history]
    val_acc = [item["valid"]["accuracy"] for item in history]
    train_f1 = [item["train"]["f1"] for item in history]
    val_f1 = [item["valid"]["f1"] for item in history]

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_acc, marker="o", label="Train Accuracy")
    plt.plot(epochs, val_acc, marker="o", label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy by Epoch")
    plt.legend()
    plt.grid(alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_f1, marker="o", label="Train F1")
    plt.plot(epochs, val_f1, marker="o", label="Validation F1")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.title("F1 by Epoch")
    plt.legend()
    plt.grid(alpha=0.3)

    return figure_to_pil()


def build_confusion_matrix_image(metrics: dict) -> Image.Image | None:
    if "confusion_matrix" not in metrics:
        return None

    cm = np.array(metrics["confusion_matrix"])
    labels = ["Fake", "Real"]

    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Test Confusion Matrix")
    plt.colorbar()
    plt.xticks([0, 1], labels)
    plt.yticks([0, 1], labels)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")

    return figure_to_pil()


def build_roc_curve_image(metrics: dict) -> Image.Image | None:
    roc_curve = metrics.get("roc_curve")
    if not roc_curve:
        return None

    fpr = roc_curve.get("fpr")
    tpr = roc_curve.get("tpr")
    if not fpr or not tpr:
        return None

    auc_value = metrics.get("roc_auc")

    plt.figure(figsize=(6, 5))
    label = f"Best Model (AUC = {auc_value:.4f})" if auc_value is not None else "Best Model"
    plt.plot(fpr, tpr, linewidth=2, label=label, color="darkorange")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Baseline")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    return figure_to_pil()


def load_analytics_images(checkpoint_path: Path):
    history_path = checkpoint_path.parent / "history.json"
    metrics_path = checkpoint_path.parent / "test_metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}

    return {
        "confusion_matrix": build_confusion_matrix_image(metrics) if metrics else None,
        "roc_curve": build_roc_curve_image(metrics) if metrics else None,
        "training_curves": build_training_curves_image(history_path),
    }


def load_artifacts(checkpoint_path: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint["config"]
    class_to_idx = checkpoint.get("class_to_idx", {"fake": 0, "real": 1})
    class_names = [name for name, _ in sorted(class_to_idx.items(), key=lambda item: item[1])]
    model = build_model(config["model_name"], config["dropout"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((config["image_size"], config["image_size"])),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    grad_cam = GradCAM(model, model.layer4[-1])
    analytics_images = load_analytics_images(checkpoint_path)
    return model, transform, device, config, class_names, grad_cam, analytics_images


def create_predict_fn(model, transform, device, class_names, grad_cam):
    @torch.no_grad()
    def predict(image: Image.Image):
        if image is None:
            return "Upload an image to run prediction.", None

        image = image.convert("RGB")
        display_image = image.copy()
        tensor = transform(image).unsqueeze(0).to(device)
        model.zero_grad(set_to_none=True)
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().tolist()
        predicted_idx = int(torch.argmax(logits, dim=1).item())
        predicted_label = class_names[predicted_idx]
        confidence = probs[predicted_idx]
        details = f"Prediction: {predicted_label} | Confidence: {confidence:.2%}"

        with torch.enable_grad():
            grad_input = transform(display_image).unsqueeze(0).to(device)
            cam = grad_cam.generate(grad_input, predicted_idx)
        overlay = create_heatmap_overlay(display_image, cam)

        return details, overlay

    return predict


def build_interface(model, transform, device, config, class_names, grad_cam, analytics_images, checkpoint_path: Path):
    metrics_path = checkpoint_path.parent / "test_metrics.json"
    metrics_text = ""
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics_text = (
            f"Test accuracy: {metrics['accuracy']:.4f}\n"
            f"Test F1: {metrics['f1']:.4f}\n"
            f"Test ROC-AUC: {metrics['roc_auc']:.4f}"
        )

    description = (
        "Upload a face image to classify it as real or fake.\n\n"
        f"Backbone: {config['model_name']}\n"
        f"Input size: {config['image_size']}\n"
        f"{metrics_text}".strip()
    )

    predict_fn = create_predict_fn(model, transform, device, class_names, grad_cam)

    with gr.Blocks(title="Deepfake Face Detector") as demo:
        gr.Markdown("# Deepfake Face Detector")
        gr.Markdown(description)
        gr.Markdown("Grad-CAM overlay highlights the regions that most influenced the current prediction.")

        with gr.Row():
            image_input = gr.Image(type="pil", label="Face Image")
            with gr.Column():
                summary_output = gr.Textbox(label="Prediction Summary", interactive=False)
                gradcam_output = gr.Image(type="pil", label="Grad-CAM Overlay")

        predict_button = gr.Button("Analyze Image", variant="primary")
        clear_button = gr.Button("Clear")

        predict_button.click(
            fn=predict_fn,
            inputs=image_input,
            outputs=[summary_output, gradcam_output],
        )
        clear_button.click(
            fn=lambda: (None, "", None),
            inputs=None,
            outputs=[image_input, summary_output, gradcam_output],
        )

        gr.Markdown("## Descriptive Analytics")
        with gr.Row():
            confusion_matrix_output = gr.Image(
                value=analytics_images.get("confusion_matrix"),
                type="pil",
                label="Confusion Matrix",
                interactive=False,
                scale=1,
            )
            roc_curve_output = gr.Image(
                value=analytics_images.get("roc_curve"),
                type="pil",
                label="ROC Curve",
                interactive=False,
                scale=1,
            )
        with gr.Row():
            training_curves_output = gr.Image(
                value=analytics_images.get("training_curves"),
                type="pil",
                label="Training Curves by Epoch",
                interactive=False,
            )

    return demo


def parse_args():
    parser = argparse.ArgumentParser(description="Launch the Gradio app for the trained deepfake detector.")
    parser.add_argument("--checkpoint", default="artifacts/best_model.pt")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model, transform, device, config, class_names, grad_cam, analytics_images = load_artifacts(checkpoint_path)
    demo = build_interface(model, transform, device, config, class_names, grad_cam, analytics_images, checkpoint_path)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)
