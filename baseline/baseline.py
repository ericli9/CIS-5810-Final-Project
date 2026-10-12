import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm


def locate(root, name):
    matches = list(root.rglob(name))
    if not matches:
        raise FileNotFoundError(f"Could not find {name} under {root}")
    return matches[0]


def load_split(root, split):
    ann_file = locate(root, f"instances_{split}2019.json")
    data = json.loads(ann_file.read_text())
    image_dir = locate(root, f"{split}2019")
    return data, image_dir


def image_path(folder, file_name):
    p = folder / file_name
    if not p.exists():
        p = folder / Path(file_name).name
    return p


class ProductDataset(Dataset):
    def __init__(self, root):
        data, self.folder = load_split(root, "train")
        self.images = {im["id"]: im for im in data["images"]}
        self.samples = [(self.images[a["image_id"]]["file_name"], int(a["category_id"])) for a in data["annotations"]]
        self.classes = sorted({cat["id"] for cat in data["categories"]})
        self.to_index = {c: i for i, c in enumerate(self.classes)}
        self.tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_name, category = self.samples[idx]
        with Image.open(image_path(self.folder, file_name)) as img:
            tensor = self.tf(img.convert("RGB"))
        return tensor, self.to_index[category]


def model_for(num_classes, pretrained):
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train(args):
    ds = ProductDataset(args.data)
    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    model = model_for(len(ds.classes), pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    print(f"Training {len(ds.samples)} samples, {len(ds.classes)} classes, device={device}")
    for epoch in range(args.epochs):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            total += len(y)
            correct += (logits.argmax(1) == y).sum().item()
            loss_sum += loss.item() * len(y)
        print(f"train loss={loss_sum/total:.4f}, train accuracy={correct/total:.4f}")
    torch.save({"state": model.cpu().state_dict(), "classes": ds.classes}, args.checkpoint)
    print(f"Saved {args.checkpoint}")


def canny_boxes(bgr, min_area=900):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 1)
    edges = cv2.Canny(blur, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = gray.shape
    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h < min_area or w < 15 or h < 15:
            continue
        if w > width * .95 and h > height * .95:
            continue
        boxes.append([x, y, w, h])
    return boxes, edges


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax+aw, bx+bw), min(ay+ah, by+bh)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    return inter / max(1, aw*ah + bw*bh - inter)


def main_eval(args):
    data, folder = load_split(args.data, "val")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    classes = checkpoint["classes"]
    model = model_for(len(classes), pretrained=False)
    model.load_state_dict(checkpoint["state"])
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = model.to(device).eval()
    tf = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    anns = defaultdict(list)
    for a in data["annotations"]:
        anns[a["image_id"]].append(a)
    images = data["images"][:]
    random.Random(42).shuffle(images)
    images = images[:args.limit] if args.limit > 0 else images
    args.output.mkdir(parents=True, exist_ok=True)
    gt_total = pred_total = matched_total = correct_total = exact_counts = 0
    for idx, im in enumerate(tqdm(images, desc="Evaluating")):
        bgr = cv2.imread(str(image_path(folder, im["file_name"])))
        if bgr is None:
            raise FileNotFoundError(im["file_name"])
        boxes, edges = canny_boxes(bgr, args.min_area)
        predictions = []
        for x,y,w,h in boxes:
            crop = bgr[y:y+h, x:x+w]
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            with torch.no_grad():
                scores = model(tf(Image.fromarray(rgb)).unsqueeze(0).to(device)).softmax(1)[0]
            j = int(scores.argmax().item())
            predictions.append({"bbox":[x,y,w,h], "category_id":classes[j], "confidence":float(scores[j].item())})
        gt = anns[im["id"]]
        gt_total += len(gt)
        pred_total += len(predictions)
        exact_counts += int(len(gt) == len(predictions))
        pairs = sorted([(iou(p["bbox"], a["bbox"]), pi, ai) for pi,p in enumerate(predictions) for ai,a in enumerate(gt)], reverse=True)
        used_p, used_a = set(), set()
        for score, pi, ai in pairs:
            if score < .5:
                break
            if pi in used_p or ai in used_a:
                continue
            used_p.add(pi); used_a.add(ai)
            matched_total += 1
            correct_total += int(predictions[pi]["category_id"] == gt[ai]["category_id"])
        if idx < args.visualize:
            vis = bgr.copy()
            for p in predictions:
                x,y,w,h = p["bbox"]
                cv2.rectangle(vis,(x,y),(x+w,y+h),(0,0,255),2)
                cv2.putText(vis,str(p["category_id"]),(x,max(15,y-5)),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,255),1)
            for a in gt:
                x,y,w,h = map(int,a["bbox"])
                cv2.rectangle(vis,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.imwrite(str(args.output/f"{idx:03d}_boxes.jpg"),vis)
            cv2.imwrite(str(args.output/f"{idx:03d}_edges.jpg"),edges)
    result = {"images":len(images),"ground_truth_products":gt_total,"predicted_products":pred_total,
              "localization_recall_iou50":matched_total/max(1,gt_total),
              "localization_precision_iou50":matched_total/max(1,pred_total),
              "classification_accuracy_on_matched":correct_total/max(1,matched_total),
              "end_to_end_recall_iou50":correct_total/max(1,gt_total),
              "exact_count_accuracy":exact_counts/max(1,len(images))}
    print(json.dumps(result,indent=2))
    (args.output/"metrics.json").write_text(json.dumps(result,indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train","eval"])
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--checkpoint", type=Path, default=Path("rpc_resnet18.pt"))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--min-area", type=int, default=900)
    parser.add_argument("--visualize", type=int, default=10)
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args()
    if args.mode == "train":
        train(args)
    else:
        main_eval(args)
