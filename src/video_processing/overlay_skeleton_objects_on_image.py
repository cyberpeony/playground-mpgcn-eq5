# -*- coding: utf-8 -*-
import cv2
import yaml
import os
import argparse
from pathlib import Path
import pandas as pd

# Mismo esqueleto que usas en grafoPanoramico
SKELETON_EDGES = [
    (0, 1), (0, 2),
    (1, 3), (2, 4),
    (5, 6),
    (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
]
NUM_JOINTS = 17

def load_rois(yaml_path: str):
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"No existe el YAML: {yaml_path}")
    with open(yaml_path, "r", encoding="utf-8") as f:
        rois = yaml.safe_load(f) or {}
    if not isinstance(rois, dict):
        raise ValueError(
            "El YAML de centroides debe ser un dict por cámara, "
            "por ejemplo {camera: [...]} o {camera: {...}}"
        )
    return rois

def iter_rois_for_camera(rois_cam):
    """
    Soporta:
    - dict: {obj_name: {centroid: [...], ...}}
    - list: [{name,x,y,...}, ...]
    """
    if isinstance(rois_cam, dict):
        for obj_name, data in rois_cam.items():
            yield obj_name, data
    elif isinstance(rois_cam, list):
        for i, obj in enumerate(rois_cam):
            if not isinstance(obj, dict):
                continue
            name = obj.get("name", f"obj_{i}")
            yield name, obj

def get_centroid_from_data(data):
    if not isinstance(data, dict):
        return None
    if "centroid" in data:
        val = data["centroid"]
        if isinstance(val, (list, tuple)) and len(val) >= 2:
            return [float(val[0]), float(val[1])]
    if "x" in data and "y" in data:
        return [float(data["x"]), float(data["y"])]
    return None

def px_from_centroid(val, W, H, assume_normalized=True):
    """
    val: [x, y] normalizado (0..1) por default
    """
    x, y = float(val[0]), float(val[1])
    if assume_normalized:
        px = max(0, min(int(round(x * W)), W - 1))
        py = max(0, min(int(round(y * H)), H - 1))
    else:
        px = max(0, min(int(round(x)), W - 1))
        py = max(0, min(int(round(y)), H - 1))
    return px, py

def draw_crosshair(img, x, y, color=(0,0,255), r=8, thickness=2):
    cv2.circle(img, (x, y), r, color, -1)
    cv2.line(img, (x - 2*r, y), (x + 2*r, y), color, thickness)
    cv2.line(img, (x, y - 2*r), (x, y + 2*r), color, thickness)

def draw_skeleton(frame, row, color=(0,255,0)):
    """
    row: fila del CSV con columnas x0,y0,...,x16,y16 en PIXELES
    """
    H, W = frame.shape[:2]

    # Joints
    pts = []
    for j in range(NUM_JOINTS):
        x = float(row[f"x{j}"])
        y = float(row[f"y{j}"])
        if x < 0 or y < 0 or x >= W or y >= H:
            pts.append(None)
        else:
            pts.append((int(round(x)), int(round(y))))

    # Dibujar puntos
    for p in pts:
        if p is None:
            continue
        cv2.circle(frame, p, 4, color, -1)

    # Dibujar edges
    for j1, j2 in SKELETON_EDGES:
        p1 = pts[j1]
        p2 = pts[j2]
        if p1 is None or p2 is None:
            continue
        cv2.line(frame, p1, p2, color, 2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True,
                    help="CSV de keypoints EN PIXELES (frame, person_id, x0,y0,...,x16,y16)")
    ap.add_argument("--yaml", default="configs/objects_manual.yaml",
                    help="YAML con centroides por cámara")
    ap.add_argument("--video", required=True,
                    help="Video de la cámara (mp4)")
    ap.add_argument("--cam", required=True,
                    help="Nombre de la cámara en el YAML (ej. columpioscam3)")
    ap.add_argument("--frame", type=int, default=0,
                    help="Índice de frame a visualizar (por número de frame en el CSV)")
    ap.add_argument("--person-id", type=int, default=0,
                    help="person_id a dibujar")
    ap.add_argument("--out", default="overlay_frame.png",
                    help="Imagen de salida")
    args = ap.parse_args()

    # 1) Cargar YAML
    rois_all = load_rois(args.yaml)
    if args.cam not in rois_all:
        raise KeyError(f"La cámara {args.cam} no está en el YAML {args.yaml}")
    rois_cam = list(iter_rois_for_camera(rois_all[args.cam]))

    # 2) Cargar CSV de keypoints en PIXELES
    df = pd.read_csv(args.csv)
    # Filtrar por frame y persona
    df_frame = df[(df["frame"] == args.frame) & (df["person_id"] == args.person_id)]
    if df_frame.empty:
        raise ValueError(f"No hay filas para frame={args.frame}, person_id={args.person_id} en {args.csv}")
    row = df_frame.iloc[0]

    # 3) Tomar el frame correspondiente del video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"No pude abrir el video: {args.video}")

    # Nos apoyamos en que el 'frame' del CSV corresponde al índice de frame del video
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.frame))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"No pude leer el frame {args.frame} del video {args.video}")

    H, W = frame.shape[:2]

    # 4) Dibujar esqueleto
    draw_skeleton(frame, row, color=(0, 255, 0))

    # 5) Dibujar centroides de objetos (0..1 normalizados)
    for obj_name, data in rois_cam:
        centroid = get_centroid_from_data(data)
        if centroid is None:
            print(f"[WARN] {args.cam}.{obj_name} sin centroid/x/y")
            continue
        cx, cy = px_from_centroid(centroid, W, H, assume_normalized=True)
        color = (0, 255, 255) if "bench" in obj_name.lower() or "seat" in obj_name.lower() else (0, 0, 255)
        draw_crosshair(frame, cx, cy, color=color, r=8, thickness=2)
        cv2.putText(
            frame,
            obj_name,
            (cx + 8, cy - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (50, 255, 50),
            2,
            cv2.LINE_AA,
        )

    # 6) Guardar y mostrar
    cv2.imwrite(args.out, frame)
    print(f"[OK] Guardado overlay en {args.out}")

    cv2.imshow("Skeleton + objetos", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
