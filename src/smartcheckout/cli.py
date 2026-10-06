"""Command line entry point: ``demo``, ``run``, ``zones``, ``check``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from .catalog import Catalog
from .config import AppConfig
from .demo import build_demo
from .pipeline import Pipeline
from .viz import compose

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "demo.json"
DEFAULT_CATALOG = REPO_ROOT / "configs" / "catalog.json"
WINDOW = "Smart Checkout"


# -- helpers -----------------------------------------------------------------


def _load(args: argparse.Namespace) -> AppConfig:
    cfg = AppConfig.load(args.config)
    if getattr(args, "source", None) is not None:
        target = cfg.camera(args.camera) if getattr(args, "camera", None) else cfg.cameras[0]
        target.source = args.source
        target.detections = None  # an overridden source needs a real detector
    if getattr(args, "model", None):
        cfg.model = args.model
    if getattr(args, "conf", None) is not None:
        cfg.min_conf = args.conf
    if getattr(args, "device", None):
        cfg.device = args.device
    return cfg


def _print_summary(pipeline: Pipeline) -> None:
    catalog = pipeline.catalog
    receipt = pipeline.cart.receipt()
    print("\n--- receipt " + "-" * 44)
    if not receipt["items"]:
        print("  (nothing in the bin)")
    for item in receipt["items"]:
        line = f"  {item['qty']}x {item['display']:<26}"
        print(f"{line} {catalog.money(item['subtotal']):>10}")
    print(f"  {'TOTAL':<29} {catalog.money(receipt['total']):>10}")
    if receipt["needs_attention"]:
        print("  ! unrecognized item in the bin - attendant required")

    alerts = pipeline.gate.alerts
    print(f"\n--- loss prevention: {len(alerts)} alert(s) " + "-" * 24)
    for alert in alerts:
        print(
            f"  t={alert.time_s:6.2f}s  {alert.display} (track {alert.track_id} on"
            f" {alert.camera}) -> {alert.reason}"
        )
        if alert.snapshot:
            print(f"               snapshot: {alert.snapshot}")
    if not alerts:
        print("  everything that reached the bag was accounted for")
    print()


# -- commands ----------------------------------------------------------------


def cmd_demo(args: argparse.Namespace) -> int:
    data_dir = Path(args.data_dir)
    config_path = Path(args.config_out)
    catalog_rel = Path(DEFAULT_CATALOG).name
    path = build_demo(data_dir, config_path, catalog_rel=catalog_rel)
    print(f"demo feeds written to {data_dir}")
    print(f"demo config written to {path}")
    if args.run:
        run_args = argparse.Namespace(
            config=str(path),
            backend="auto",
            display=not args.no_display,
            save=args.save,
            report=args.report,
            alerts_dir=args.alerts_dir,
            loop=False,
            max_frames=None,
            view_width=720,
            speed=args.speed,
            source=None,
            camera=None,
            model=None,
            conf=None,
            device=None,
        )
        return cmd_run(run_args)
    print(f"\nnext:  python run.py run --config {path}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    cfg = _load(args)
    alert_dir = Path(args.alerts_dir) if args.alerts_dir else None
    pipeline = Pipeline(cfg, backend=args.backend, loop=args.loop, alert_dir=alert_dir)
    writer: cv2.VideoWriter | None = None
    paused = False
    delay = 1
    if args.speed and args.speed > 0:
        delay = max(1, int(round(1000.0 / (pipeline.fps_nominal * args.speed))))

    print(f"config   : {cfg.path}")
    print(f"catalog  : {cfg.catalog} ({len(pipeline.catalog.known_classes())} classes)")
    for cam in pipeline.cameras:
        roles = ", ".join(sorted({z.role for z in cam.cfg.zones})) or "no zones"
        print(f"camera   : {cam.cfg.id:<8} {cam.backend:<9} {cam.size[0]}x{cam.size[1]}  [{roles}]")
    if args.display:
        print("keys     : q quit, space pause, r reset, p snapshot")

    try:
        while True:
            if not paused:
                result = pipeline.step()
                if result.frames:
                    canvas = compose(
                        result.frames,
                        pipeline.catalog,
                        pipeline.cart,
                        pipeline.gate,
                        pipeline.stats(),
                        view_width=args.view_width,
                    )
                    for alert in result.alerts:
                        print(
                            f"[f{alert.frame_idx:05d}] ALERT  unscanned {alert.display}"
                            f" entered {alert.zone} on camera {alert.camera}"
                        )
                    if args.save:
                        if writer is None:
                            out = Path(args.save)
                            out.parent.mkdir(parents=True, exist_ok=True)
                            writer = cv2.VideoWriter(
                                str(out),
                                cv2.VideoWriter_fourcc(*"mp4v"),
                                pipeline.fps_nominal,
                                (canvas.shape[1], canvas.shape[0]),
                            )
                            if not writer.isOpened():
                                raise RuntimeError(f"cannot write video to {out}")
                        writer.write(canvas)
                if result.finished:
                    break
                if args.max_frames and pipeline.frame_idx >= args.max_frames:
                    break

            if args.display and result.frames:
                cv2.imshow(WINDOW, canvas)
                key = cv2.waitKey(delay if not paused else 50) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord(" "):
                    paused = not paused
                elif key == ord("r"):
                    pipeline.reset()
                    print("state reset")
                elif key == ord("p"):
                    snap = Path(args.alerts_dir or "out") / f"snapshot_{pipeline.frame_idx:05d}.png"
                    snap.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(snap), canvas)
                    print(f"snapshot -> {snap}")
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        if writer is not None:
            writer.release()
            print(f"video    : {args.save}")
        pipeline.close()
        if args.display:
            cv2.destroyAllWindows()

    _print_summary(pipeline)
    if args.report:
        out = Path(args.report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(pipeline.report(), indent=2) + "\n", encoding="utf-8")
        print(f"report   : {out}")
    return 0


def cmd_zones(args: argparse.Namespace) -> int:
    from .zone_editor import edit_zones

    cfg = _load(args)
    camera_id = args.camera or cfg.cameras[0].id
    saved = edit_zones(cfg, camera_id, Path(args.out) if args.out else None)
    return 0 if saved else 1


def cmd_check(args: argparse.Namespace) -> int:
    cfg = AppConfig.load(args.config)
    catalog = Catalog.load(cfg.catalog)
    print(f"config : {cfg.path}")
    print(f"catalog: {cfg.catalog}")
    print(f"  {len(catalog.known_classes())} priced classes: {', '.join(catalog.known_classes())}")
    print(f"detector: model={cfg.model} conf={cfg.min_conf} tracker={cfg.tracker}")
    print(
        "tuning  : "
        f"coverage>={cfg.coverage_threshold} confirm={cfg.confirm_frames}"
        f" exit={cfg.exit_frames} lost={cfg.lost_frames}"
    )
    for cam in cfg.cameras:
        backend = "scripted" if cam.detections else "yolo"
        print(f"\ncamera {cam.id!r} ({backend})  source={cam.source}")
        if cam.detections:
            print(f"  detections: {cam.detections} ({'ok' if cam.detections.exists() else 'MISSING'})")
        src = cam.capture_source
        if isinstance(src, str) and src not in ("blank", "none"):
            print(f"  video     : {'ok' if Path(src).exists() else 'MISSING'}")
        for zone in cam.zones:
            print(f"  zone {zone.name:<16} role={zone.role:<7} {len(zone.polygon)} points")
    return 0


# -- parser ------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smartcheckout",
        description="Vision-based grocery auto-checkout and scan-gate loss prevention.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_demo = sub.add_parser("demo", help="build the offline demo feeds (no camera, no weights)")
    p_demo.add_argument("--data-dir", default=str(REPO_ROOT / "data" / "demo"))
    p_demo.add_argument("--config-out", default=str(DEFAULT_CONFIG))
    p_demo.add_argument("--run", action="store_true", help="run the demo right after building it")
    p_demo.add_argument("--no-display", action="store_true", help="with --run, render headless")
    p_demo.add_argument("--save", default=None, help="with --run, write the composed video here")
    p_demo.add_argument("--report", default=None, help="with --run, write a JSON session report")
    p_demo.add_argument("--alerts-dir", default=str(REPO_ROOT / "out" / "alerts"))
    p_demo.add_argument("--speed", type=float, default=1.0)
    p_demo.set_defaults(func=cmd_demo)

    p_run = sub.add_parser("run", help="run the pipeline over a config")
    p_run.add_argument("--config", default=str(DEFAULT_CONFIG))
    p_run.add_argument(
        "--backend",
        choices=("auto", "yolo", "scripted"),
        default="auto",
        help="auto: scripted where a config gives a detections file, yolo otherwise",
    )
    p_run.add_argument("--source", default=None, help="override a camera source (webcam index or path)")
    p_run.add_argument("--camera", default=None, help="which camera id --source applies to")
    p_run.add_argument("--model", default=None, help="override the YOLO weights (e.g. yolo11s.pt)")
    p_run.add_argument("--conf", type=float, default=None, help="override detector confidence")
    p_run.add_argument("--device", default=None, help="torch device, e.g. cpu, 0, mps")
    p_run.add_argument("--no-display", dest="display", action="store_false", help="headless")
    p_run.add_argument("--save", default=None, help="write the composed view to this video file")
    p_run.add_argument("--report", default=None, help="write a JSON session report here")
    p_run.add_argument("--alerts-dir", default=str(REPO_ROOT / "out" / "alerts"))
    p_run.add_argument("--loop", action="store_true", help="restart the session when sources end")
    p_run.add_argument("--max-frames", type=int, default=None)
    p_run.add_argument("--view-width", type=int, default=720)
    p_run.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="playback speed multiplier; 0 runs as fast as possible",
    )
    p_run.set_defaults(func=cmd_run, display=True)

    p_zones = sub.add_parser("zones", help="draw zone polygons on a camera frame")
    p_zones.add_argument("--config", default=str(DEFAULT_CONFIG))
    p_zones.add_argument("--camera", default=None, help="camera id to edit (default: the first)")
    p_zones.add_argument("--source", default=None, help="grab the frame from here instead")
    p_zones.add_argument("--out", default=None, help="write to this config instead of in place")
    p_zones.set_defaults(func=cmd_zones)

    p_check = sub.add_parser("check", help="validate a config and print what it declares")
    p_check.add_argument("--config", default=str(DEFAULT_CONFIG))
    p_check.set_defaults(func=cmd_check)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError, KeyError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
