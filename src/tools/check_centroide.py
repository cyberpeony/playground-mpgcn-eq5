# -*- coding: utf-8 -*-
import cv2
import yaml
import os
import argparse
from pathlib import Path

def load_rois(yaml_path: str):
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"No existe el YAML: {yaml_path}")
    with open(yaml_path, "r", encoding="utf-8") as f:
        rois = yaml.safe_load(f) or {}
    if not isinstance(rois, dict):
        raise ValueError(
            "El YAML de centroides debe ser un dict por cámara, "
            "por ejemplo {camera: {...}} o {camera: [...]}"
        )
    return rois

def px_from_centroid(val, W, H, assume_normalized=None):
    """
    val: [x, y] puede estar normalizado (0..1) o en píxeles.
    assume_normalized=None → heurística: si ambos <= 1.5 => normalizado.
    """
    x, y = float(val[0]), float(val[1])
    if assume_normalized is True or (assume_normalized is None and (x <= 1.5 and y <= 1.5)):
        # normalizado → píxeles
        px = max(0, min(int(round(x * W)), W - 1))
        py = max(0, min(int(round(y * H)), H - 1))
    else:
        # ya en píxeles
        px = max(0, min(int(round(x)), W - 1))
        py = max(0, min(int(round(y)), H - 1))
    return px, py

def draw_crosshair(img, x, y, color=(0,0,255), r=10, thickness=2):
    cv2.circle(img, (x, y), r, color, -1)
    cv2.line(img, (x - 2*r, y), (x + 2*r, y), color, thickness)
    cv2.line(img, (x, y - 2*r), (x, y + 2*r), color, thickness)

def iter_rois_for_camera(rois_cam):
    """
    Normaliza el formato de rois_cam para que siempre podamos iterar como:
    for obj_name, data in iter_rois_for_camera(rois_cam):
        ...

    Soporta:
    - Formato viejo (dict):
        columpioscam1:
          bench_1: {type: manual, centroid: [x, y]}
    - Formato nuevo (lista de dicts):
        columpioscam1:
          - name: bench_1
            x: 0.45
            y: 0.70
            type: manual
    """
    # Caso 1: dict {obj_name: {centroid: [...], ...}}
    if isinstance(rois_cam, dict):
        for obj_name, data in rois_cam.items():
            yield obj_name, data

    # Caso 2: lista de dicts [{"name": ..., "x": ..., "y": ...}, ...]
    elif isinstance(rois_cam, list):
        for i, obj in enumerate(rois_cam):
            if not isinstance(obj, dict):
                continue
            name = obj.get("name", f"obj_{i}")
            yield name, obj

    else:
        # Formato no soportado
        return

def get_centroid_from_data(data):
    """
    Extrae un centroide [x, y] a partir de un dict de objeto.
    Soporta:
    - {"centroid": [x, y], ...}
    - {"x": x, "y": y, ...}
    """
    if not isinstance(data, dict):
        return None

    if "centroid" in data:
        val = data["centroid"]
        if isinstance(val, (list, tuple)) and len(val) >= 2:
            return [val[0], val[1]]

    # Formato nuevo: x / y
    if "x" in data and "y" in data:
        return [data["x"], data["y"]]

    return None

def process_video(video_path, cam_name, rois_cam, args):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] No pude abrir: {video_path}")
        return

    # Writer opcional
    writer = None
    if args.out:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        base = Path(video_path).stem + "_centroides.mp4"
        out_path = out_dir / base
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
        H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (W, H))
        print(f"[OK] Guardando overlay → {out_path}")

    # Contar objetos válidos
    rois_list = list(iter_rois_for_camera(rois_cam))
    print(f"[INFO] ▶ {Path(video_path).name} (cámara={cam_name}, objs={len(rois_list)})")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        H, W = frame.shape[:2]

        # Dibuja centroides
        for obj_name, data in rois_list:
            centroid = get_centroid_from_data(data)
            if centroid is None:
                print(f"[WARN] {cam_name}.{obj_name} sin 'centroid' ni 'x/y' válido")
                continue

            cx, cy = px_from_centroid(
                centroid, W, H,
                assume_normalized=True if args.normalized else (False if args.pixels else None)
            )

            color = (0, 255, 255) if "bench" in obj_name.lower() or "banca" in obj_name.lower() else (0, 0, 255)
            draw_crosshair(frame, cx, cy, color=color, r=args.radius, thickness=args.thick)
            cv2.putText(
                frame,
                f"{obj_name} ({cx},{cy})",
                (cx + 12, cy - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (50, 255, 50),
                2,
                cv2.LINE_AA,
            )

        # Mostrar o escribir
        if writer:
            writer.write(frame)
        if not args.no_window:
            # Resize opcional conservando relación de aspecto
            if args.width > 0 and frame.shape[1] > args.width:
                scale = args.width / frame.shape[1]
                frame_small = cv2.resize(frame, (args.width, int(frame.shape[0]*scale)))
            else:
                frame_small = frame
            cv2.imshow("Validación de centroides (q para salir)", frame_small)
            if cv2.waitKey(args.delay) & 0xFF == ord('q'):
                break

    cap.release()
    if writer:
        writer.release()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", default="Downloads", help="Carpeta con .mp4")
    ap.add_argument("--config", default="configs/objects_prueba.yaml", help="YAML con centroides por cámara")
    ap.add_argument("--cam", default="", help="Filtrar por cámara (prefijo del filename, ej: columpioscam1)")
    ap.add_argument("--out", default="resultados_rey/centroides_manual", help="Si se indica, guarda MP4 con overlays en esta carpeta")
    ap.add_argument("--no-window", action="store_true", help="No abrir ventana (útil por SSH/headless)")
    ap.add_argument("--width", type=int, default=1280, help="Ancho de preview (0=original)")
    ap.add_argument("--delay", type=int, default=30, help="Delay de imshow en ms")
    ap.add_argument("--radius", type=int, default=10, help="Radio del marcador")
    ap.add_argument("--thick", type=int, default=2, help="Grosor del marcador")
    # Forzar interpretación de centroides:
    ap.add_argument("--normalized", action="store_true", help="Forzar a tratar centroides como normalizados 0..1")
    ap.add_argument("--pixels", action="store_true", help="Forzar a tratar centroides como píxeles")
    args = ap.parse_args()

    rois = load_rois(args.config)
    video_dir = Path(args.videos)
    if not video_dir.exists():
        print(f"[ERR] No existe la carpeta de videos: {video_dir}")
        return

    videos = [str(video_dir / f) for f in os.listdir(video_dir) if f.endswith(".mp4")]
    if not videos:
        print(f"[WARN] No hay .mp4 en {video_dir}")
        return

    print(f"[INFO] Cámaras en YAML: {list(rois.keys())}")

    for vid_path in sorted(videos):
        fname = os.path.basename(vid_path)
        cam_name = fname.split("-")[0]  # prefijo antes del primer '-'
        if args.cam and cam_name != args.cam:
            continue
        if cam_name not in rois:
            # No hay config para esta cámara
            continue
        process_video(vid_path, cam_name, rois[cam_name], args)

    if not args.no_window:
        cv2.destroyAllWindows()
    print("[OK] Revisión terminada.")

if __name__ == "__main__":
    main()
