import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Reuse the original baseline implementation without modifying it.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "baseline"))
from baseline import load_split, image_path, model_for, canny_boxes, iou


def boxes_separated(annotations, gap=0):
    """True when no two annotated boxes overlap or touch (optionally with a gap)."""
    for i, a in enumerate(annotations):
        x1, y1, w1, h1 = map(float, a["bbox"])
        for b in annotations[i + 1:]:
            x2, y2, w2, h2 = map(float, b["bbox"])
            # Touching boundaries count as NOT separated.
            if (x1 <= x2 + w2 + gap and x2 <= x1 + w1 + gap
                    and y1 <= y2 + h2 + gap and y2 <= y1 + h1 + gap):
                return False
    return True


def bbox_iou(a, b):
    x1, y1, w1, h1 = map(float, a)
    x2, y2, w2, h2 = map(float, b)
    ix = max(0.0, min(x1 + w1, x2 + w2) - max(x1, x2))
    iy = max(0.0, min(y1 + h1, y2 + h2) - max(y1, y2))
    intersection = ix * iy
    union = w1 * h1 + w2 * h2 - intersection
    return intersection / union if union > 0 else 0.0


def max_pairwise_iou(annotations):
    boxes = [a["bbox"] for a in annotations]
    return max((bbox_iou(a, b)
                for i, a in enumerate(boxes) for b in boxes[i + 1:]),
               default=0.0)


def overlap_group(max_iou):
    if max_iou < 0.1:
        return "low"
    if max_iou <= 0.3:
        return "moderate"
    return "high"


def evaluate(args):
    data, folder = load_split(args.data, "val")
    anns = defaultdict(list)
    for ann in data["annotations"]:
        anns[ann["image_id"]].append(ann)

    # Crucial: select the exact same seeded 1,000-image cohort as baseline,
    # THEN filter it, rather than choosing 1,000 easy images.
    images = list(data["images"])
    random.Random(42).shuffle(images)
    if args.limit > 0:
        images = images[:args.limit]
    cohort_size = len(images)
    groups = {"low": [], "moderate": [], "high": []}
    for im in images:
        groups[overlap_group(max_pairwise_iou(anns[im["id"]]))].append(im)
    args.output.mkdir(parents=True, exist_ok=True)
    selection = {
        "validation_images_considered": cohort_size,
        "definitions": {
            "low": "max pairwise bbox IoU < 0.1",
            "moderate": "0.1 <= max pairwise bbox IoU <= 0.3",
            "high": "max pairwise bbox IoU > 0.3"
        },
        "groups": {
            name: {"images": len(group),
                   "ground_truth_products": sum(len(anns[im["id"]]) for im in group),
                   "image_ids": [im["id"] for im in group]}
            for name, group in groups.items()
        }
    }
    (args.output / "selection.json").write_text(json.dumps(selection, indent=2))
    print("Image counts:", {name: len(group) for name, group in groups.items()})

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    classes = checkpoint["classes"]
    model = model_for(len(classes), pretrained=False)
    model.load_state_dict(checkpoint["state"])
    device = ("cuda" if torch.cuda.is_available() else
              "mps" if torch.backends.mps.is_available() else "cpu")
    model = model.to(device).eval()
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])

    all_metrics = {}
    for group_name, selected in groups.items():
        group_output = args.output / group_name
        group_output.mkdir(parents=True, exist_ok=True)
        all_metrics[group_name] = evaluate_group(
            selected, group_name, group_output, args, anns, folder, model, device, tf, classes)
    (args.output / "metrics.json").write_text(json.dumps(all_metrics, indent=2))
    print(json.dumps(all_metrics, indent=2))


def evaluate_group(selected, group_name, output, args, anns, folder, model, device, tf, classes):
    gt_total = pred_total = matched_total = correct_total = exact_counts = 0
    for idx, im in enumerate(tqdm(selected, desc=f"Evaluating {group_name} overlap")):
        bgr = cv2.imread(str(image_path(folder, im["file_name"])))
        if bgr is None:
            raise FileNotFoundError(im["file_name"])
        boxes, edges = canny_boxes(bgr, args.min_area)
        predictions = []
        for x, y, w, h in boxes:
            crop = bgr[y:y+h, x:x+w]
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            with torch.no_grad():
                scores = model(tf(Image.fromarray(rgb)).unsqueeze(0).to(device)).softmax(1)[0]
            j = int(scores.argmax().item())
            predictions.append({
                "bbox": [x, y, w, h],
                "category_id": classes[j],
                "confidence": float(scores[j].item())
            })

        gt = anns[im["id"]]
        gt_total += len(gt)
        pred_total += len(predictions)
        exact_counts += int(len(gt) == len(predictions))
        pairs = sorted(
            ((iou(p["bbox"], a["bbox"]), pi, ai)
             for pi, p in enumerate(predictions)
             for ai, a in enumerate(gt)),
            reverse=True
        )
        used_p, used_a = set(), set()
        for score, pi, ai in pairs:
            if score < .5:
                break
            if pi in used_p or ai in used_a:
                continue
            used_p.add(pi)
            used_a.add(ai)
            matched_total += 1
            correct_total += int(predictions[pi]["category_id"] == gt[ai]["category_id"])

        if idx < args.visualize:
            vis = bgr.copy()
            for p in predictions:
                x, y, w, h = p["bbox"]
                cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(vis, str(p["category_id"]), (x, max(15, y-5)),
                            cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 0, 255), 1)
            for a in gt:
                x, y, w, h = map(int, a["bbox"])
                cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.imwrite(str(output / f"{idx:03d}_boxes.jpg"), vis)
            cv2.imwrite(str(output / f"{idx:03d}_edges.jpg"), edges)

    result = {
        "images": len(selected),
        "ground_truth_products": gt_total,
        "predicted_products": pred_total,
        "localization_recall_iou50": matched_total / max(1, gt_total),
        "localization_precision_iou50": matched_total / max(1, pred_total),
        "classification_accuracy_on_matched": correct_total / max(1, matched_total),
        "end_to_end_recall_iou50": correct_total / max(1, gt_total),
        "exact_count_accuracy": exact_counts / len(selected) if selected else None,
    }
    (output / "metrics.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RPC: evaluate low/moderate/high bbox overlap groups")
    parser.add_argument("--data", type=Path, default=Path("data/retail_product_checkout"))
    parser.add_argument("--checkpoint", type=Path, default=Path("rpc_resnet18.pt"))
    parser.add_argument("--limit", type=int, default=0,
                        help="Number of validation images before filtering; 0 uses all 6,000")
    parser.add_argument("--min-area", type=int, default=900)
    parser.add_argument("--visualize", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("results_overlap_groups"))
    evaluate(parser.parse_args())
