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
    selected = [im for im in images
                if anns[im["id"]] and boxes_separated(anns[im["id"]], args.gap)]
    overlapping = cohort_size - len(selected)

    args.output.mkdir(parents=True, exist_ok=True)
    selection = {
        "validation_images_considered": cohort_size,
        "non_overlapping_images": len(selected),
        "overlapping_or_unannotated_images": overlapping,
        "non_overlapping_ground_truth_products": sum(len(anns[im["id"]]) for im in selected),
        "gap_pixels": args.gap,
        "selected_image_ids": [im["id"] for im in selected],
    }
    (args.output / "selection.json").write_text(json.dumps(selection, indent=2))
    print(json.dumps({k:v for k,v in selection.items() if k != "selected_image_ids"}, indent=2))
    if not selected:
        print("No qualifying images; no evaluation performed.")
        return

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

    gt_total = pred_total = matched_total = correct_total = exact_counts = 0
    for idx, im in enumerate(tqdm(selected, desc="Evaluating non-overlapping images")):
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
            cv2.imwrite(str(args.output / f"{idx:03d}_boxes.jpg"), vis)
            cv2.imwrite(str(args.output / f"{idx:03d}_edges.jpg"), edges)

    result = {
        "images": len(selected),
        "ground_truth_products": gt_total,
        "predicted_products": pred_total,
        "localization_recall_iou50": matched_total / max(1, gt_total),
        "localization_precision_iou50": matched_total / max(1, pred_total),
        "classification_accuracy_on_matched": correct_total / max(1, matched_total),
        "end_to_end_recall_iou50": correct_total / max(1, gt_total),
        "exact_count_accuracy": exact_counts / len(selected),
    }
    print(json.dumps(result, indent=2))
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RPC Iteration 1: non-overlapping validation subset")
    parser.add_argument("--data", type=Path, default=Path("data/retail_product_checkout"))
    parser.add_argument("--checkpoint", type=Path, default=Path("rpc_resnet18.pt"))
    parser.add_argument("--limit", type=int, default=0,
                        help="Number of validation images before filtering; 0 uses all 6,000")
    parser.add_argument("--gap", type=float, default=0,
                        help="Minimum separation between boxes in pixels; 0 excludes touching")
    parser.add_argument("--min-area", type=int, default=900)
    parser.add_argument("--visualize", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("results_iteration1"))
    evaluate(parser.parse_args())
