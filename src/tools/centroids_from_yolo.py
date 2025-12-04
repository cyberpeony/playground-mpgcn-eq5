# -*- coding: utf-8 -*-
import os, argparse, yaml
from pathlib import Path
import numpy as np
import cv2

from ultralytics import YOLO

def cam_from_filename(fname: str):
    # prefijo antes del primer '-'  -> "columpioscam1-2025-..." -> "columpioscam1"
    return Path(fname).name.split("-")[0]

def cluster_points(points, radius=40):
    """
    Agrupa puntos (x,y) por proximidad (greedy, sin sklearn).
    radius: en píxeles. Devuelve lista de clusters (cada uno es np.array de puntos).
    """
    pts = np.array(points, dtype=np.float32)
    if len(pts) == 0:
        return []
    used = np.zeros(len(pts), dtype=bool)
    clusters = []
    for i in range(len(pts)):
        if used[i]:
            continue
        center = pts[i]
        d = np.linalg.norm(pts - center, axis=1)
        members = (d <= radius)
        clusters.append(pts[members])
        used[members] = True
    return clusters

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="Downloads", help="Carpeta con mp4")
    ap.add_argument("--model",  default="yolov8n.pt", help="Pesos YOLO detección (no pose)")
    ap.add_argument("--classes", default="bench", help="Clases objetivo separadas por coma (ej: bench,chair)")
    ap.add_argument("--conf", type=float, default=0.35, help="Umbral de confianza YOLO")
    ap.add_argument("--stride", type=int, default=3, help="Procesar 1 cada N frames")
    ap.add_argument("--radius", type=int, default=40, help="Radio px para agrupar centros")
    ap.add_argument("--max_per_class", type=int, default=4, help="Máx. centroides por clase/cámara")
    ap.add_argument("--yaml_out", default="configs/objects_auto.yaml", help="Salida YAML")
    ap.add_argument("--preview_out", default="", help="Carpeta para videos con centroides dibujados (opcional)")
    args = ap.parse_args()

    target_names = [c.strip().lower() for c in args.classes.split(",") if c.strip()]

    videos = [str(Path(args.videos)/f) for f in os.listdir(args.videos) if f.endswith(".mp4")]
    if not videos:
        raise SystemExit(f"No hay .mp4 en {args.videos}")

    model = YOLO(args.model)
    out_yaml = {}

    preview_dir = Path(args.preview_out) if args.preview_out else None
    if preview_dir:
        preview_dir.mkdir(parents=True, exist_ok=True)

    for vpath in sorted(videos):
        cam = cam_from_filename(vpath)
        print(f"[INFO] Procesando {Path(vpath).name} (cam={cam})")
        out_yaml.setdefault(cam, {})

        # Para preview
        writer = None
        if preview_dir:
            cap = cv2.VideoCapture(vpath)
            if not cap.isOpened():
                print(f"[WARN] no pude abrir {vpath}")
                continue
            W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
            H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            preview_path = preview_dir / (Path(vpath).stem + "_centroides_auto.mp4")
            writer = cv2.VideoWriter(str(preview_path), fourcc, fps, (W, H))
            cap.release()

        # Acumular centros por clase
        centers_by_class = {name: [] for name in target_names}

        # Inferencia en streaming
        results = model.predict(
            source=vpath,
            stream=True,            # frame por frame
            conf=args.conf,
            vid_stride=args.stride,
            verbose=False
        )

        W = H = None
        for r in results:
            if W is None:
                # tomar tamaño del primer frame
                im = r.orig_img
                H, W = im.shape[:2]
            if r.boxes is None or len(r.boxes) == 0:
                continue

            xywh = r.boxes.xywh.cpu().numpy()      # (cx, cy, w, h)
            cls_ids = r.boxes.cls.cpu().numpy().astype(int)
            names = r.names

            for i in range(len(cls_ids)):
                name = names[cls_ids[i]].lower()
                if name in centers_by_class:
                    cx, cy, _, _ = xywh[i]
                    centers_by_class[name].append((float(cx), float(cy)))

        # Consolidar por clase -> centroides finales
        for name, pts in centers_by_class.items():
            if not pts:
                continue
            clusters = cluster_points(pts, radius=args.radius)
            # ordenar clusters por tamaño (más estable primero)
            clusters = sorted(clusters, key=lambda c: len(c), reverse=True)[:args.max_per_class]
            for idx, cluster in enumerate(clusters):
                cx = float(np.median(cluster[:,0]))
                cy = float(np.median(cluster[:,1]))
                # normalizar
                x_norm = round(cx / W, 4)
                y_norm = round(cy / H, 4)
                key = f"{name}_{idx+1}"
                out_yaml[cam][key] = {"type": "auto", "centroid": [x_norm, y_norm]}

        # Escribir preview si se pidió
        if writer:
            # Recorremos de nuevo el video para dibujar (rápido: solo pintamos los puntos finales)
            cap2 = cv2.VideoCapture(vpath)
            while True:
                ok, frame = cap2.read()
                if not ok:
                    break
                if cam in out_yaml:
                    for obj, data in out_yaml[cam].items():
                        xn, yn = data["centroid"]
                        px = int(xn * frame.shape[1])
                        py = int(yn * frame.shape[0])
                        cv2.circle(frame, (px,py), 12, (0,0,255), -1)
                        cv2.putText(frame, obj, (px+10, py-6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50,255,50), 2)
                writer.write(frame)
            cap2.release()
            writer.release()
            print(f"[OK] Preview → {preview_path}")

    # Guardar YAML
    Path(args.yaml_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.yaml_out, "w", encoding="utf-8") as f:
        yaml.safe_dump(out_yaml, f, sort_keys=False, allow_unicode=True)
    print(f"[OK] YAML → {args.yaml_out}")
    print("Ejemplo:", out_yaml if len(out_yaml) < 4 else list(out_yaml.keys()))
    
if __name__ == "__main__":
    main()
