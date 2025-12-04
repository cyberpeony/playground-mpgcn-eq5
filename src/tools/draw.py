# src/tools/draw.py
import os
import cv2
import argparse
from collections import deque

def pick_largest_moving_bbox(frame, fgbg, min_area, dilate_iters):
    """Devuelve (x,y,w,h) de la región en movimiento más grande o None."""
    fg = fgbg.apply(frame)
    fg = cv2.medianBlur(fg, 5)
    _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)
    fg = cv2.dilate(fg, None, iterations=dilate_iters)

    cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_area = None, 0
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area >= min_area and area > best_area:
            best, best_area = (x, y, w, h), area
    return best

def create_tracker(tracker_type="CSRT"):
    """Crea un tracker si OpenCV lo trae; si no, devuelve None (fallback a detección)."""
    tt = (tracker_type or "").upper()
    # OpenCV moderno expone trackers en cv2.legacy
    if hasattr(cv2, "legacy"):
        if tt == "CSRT" and hasattr(cv2.legacy, "TrackerCSRT_create"):
            return cv2.legacy.TrackerCSRT_create()
        if tt == "KCF" and hasattr(cv2.legacy, "TrackerKCF_create"):
            return cv2.legacy.TrackerKCF_create()
    # Algunos builds antiguos los exponen directo en cv2
    if tt == "CSRT" and hasattr(cv2, "TrackerCSRT_create"):
        return cv2.TrackerCSRT_create()
    if tt == "KCF" and hasattr(cv2, "TrackerKCF_create"):
        return cv2.TrackerKCF_create()
    return None  # sin contrib

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Ruta del video de entrada")
    ap.add_argument("--out", default="outputs/out_follow.mp4", help="Ruta de salida")
    # Dibujo
    ap.add_argument("--alpha", type=float, default=0.25, help="0=borde solo; >0 relleno translúcido")
    ap.add_argument("--thickness", type=int, default=2)
    ap.add_argument("--b", type=int, default=0)     # BGR
    ap.add_argument("--g", type=int, default=255)
    ap.add_argument("--r", type=int, default=0)
    # Detección movimiento
    ap.add_argument("--min-area", type=int, default=900, help="área mínima px para aceptar una región")
    ap.add_argument("--dilate", type=int, default=2, help="iteraciones de dilatación")
    ap.add_argument("--warmup", type=int, default=15, help="frames de calentamiento para MOG2")
    # Tracker & reinit
    ap.add_argument("--tracker", choices=["CSRT", "KCF"], default="CSRT")
    ap.add_argument("--reinit-interval", type=int, default=30, help="cada N frames intenta re-detectar y re-inicializar")
    ap.add_argument("--smooth", type=float, default=0.2, help="suavizado EMA 0–1 de la caja")
    # Inicialización manual opcional
    ap.add_argument("--x", type=int, default=-1)
    ap.add_argument("--y", type=int, default=-1)
    ap.add_argument("--w", type=int, default=-1)
    ap.add_argument("--h", type=int, default=-1)
    # Trazo de trayectoria
    ap.add_argument("--trail", type=int, default=32, help="tamaño de la cola de puntos del centro (0 para desactivar)")
    args = ap.parse_args()

    color = (args.b, args.g, args.r)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"No pude abrir {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    fps_out = max(1, round(fps))  # a veces ayuda con players
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Códec: intenta H.264/avc1 (más reproducible). Si te deja 0 bytes, cambia a "mp4v" o "XVID".
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # alternativa: "XVID" o "MJPG" (AVI)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    writer = cv2.VideoWriter(args.out, fourcc, fps_out, (W, H))

    # Sustractor de fondo
    fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=False)

    # Warmup MOG2
    frame_id = 0
    last_frame = None
    while frame_id < max(1, args.warmup):
        ok, frm = cap.read()
        if not ok: break
        last_frame = frm
        fgbg.apply(frm)
        frame_id += 1
    if last_frame is None:
        cap.release(); writer.release()
        raise SystemExit("Video vacío tras warmup")

    # Tracker (si existe en tu build)
    tracker = create_tracker(args.tracker)
    has_tracker = False

    # Suavizado EMA de la caja
    sm_x = sm_y = sm_w = sm_h = None

    # Trayectoria
    trail = deque(maxlen=max(0, args.trail))

    # Inicialización manual si viene bbox
    if args.w > 0 and args.h > 0 and args.x >= 0 and args.y >= 0:
        init_box = (args.x, args.y, args.w, args.h)
        if tracker is not None:
            has_tracker = tracker.init(last_frame, init_box)
        sm_x, sm_y, sm_w, sm_h = map(float, init_box)

    # Bucle principal
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_id += 1

        box_draw = None

        # Fallback: si no hay tracker en este OpenCV, detecta cada frame y suaviza
        if tracker is None or not has_tracker:
            det = pick_largest_moving_bbox(frame, fgbg, args.min_area, args.dilate)
            if det is not None:
                x, y, w, h = map(float, det)
                a = max(0.0, min(1.0, args.smooth))
                if sm_x is None:
                    sm_x, sm_y, sm_w, sm_h = x, y, w, h
                else:
                    sm_x = (1 - a) * sm_x + a * x
                    sm_y = (1 - a) * sm_y + a * y
                    sm_w = (1 - a) * sm_w + a * w
                    sm_h = (1 - a) * sm_h + a * h
                xi, yi, wi, hi = map(int, (sm_x, sm_y, sm_w, sm_h))
                box_draw = (xi, yi, wi, hi)
                if args.trail > 0:
                    cx = int(xi + wi/2); cy = int(yi + hi/2)
                    trail.append((cx, cy))

            # si existe tracker en tu build, intenta inicializar cuando haya detección
            if tracker is not None and det is not None and not has_tracker:
                has_tracker = tracker.init(frame, tuple(map(int, det)))

        else:
            # Hay tracker → actualizar y (opcional) re-inicializar periódicamente
            ok_tr, box = tracker.update(frame)
            if not ok_tr:
                has_tracker = False
            else:
                x, y, w, h = map(float, box)

                # Reinit cada N frames usando la mayor región en movimiento
                if args.reinit_interval > 0 and frame_id % args.reinit_interval == 0:
                    probe = pick_largest_moving_bbox(frame, fgbg, args.min_area, args.dilate)
                    if probe is not None:
                        px, py, pw, ph = map(float, probe)
                        if abs(px - x) + abs(py - y) > 0.25 * (w + h):
                            tracker = create_tracker(args.tracker)
                            has_tracker = tracker.init(frame, tuple(map(int, probe)))
                            x, y, w, h = px, py, pw, ph

                # Suavizado EMA
                a = max(0.0, min(1.0, args.smooth))
                if sm_x is None:
                    sm_x, sm_y, sm_w, sm_h = x, y, w, h
                else:
                    sm_x = (1 - a) * sm_x + a * x
                    sm_y = (1 - a) * sm_y + a * y
                    sm_w = (1 - a) * sm_w + a * w
                    sm_h = (1 - a) * sm_h + a * h

                xi, yi, wi, hi = map(int, (sm_x, sm_y, sm_w, sm_h))
                box_draw = (xi, yi, wi, hi)
                if args.trail > 0:
                    cx = int(xi + wi/2); cy = int(yi + hi/2)
                    trail.append((cx, cy))

        # Dibujo
        if box_draw:
            xi, yi, wi, hi = box_draw
            if args.alpha > 0:
                overlay = frame.copy()
                cv2.rectangle(overlay, (xi, yi), (xi + wi, yi + hi), color, -1)
                frame = cv2.addWeighted(overlay, args.alpha, frame, 1 - args.alpha, 0)
            cv2.rectangle(frame, (xi, yi), (xi + wi, yi + hi), color, args.thickness)

        # Trazo
        if args.trail > 0 and len(trail) > 1:
            for i in range(1, len(trail)):
                cv2.line(frame, trail[i-1], trail[i], (0, 255, 255), 2)

        writer.write(frame)
        if frame_id % 100 == 0:
            print(f"[info] frames: {frame_id}")

    cap.release()
    writer.release()
    print(f"[OK] Escribí {os.path.abspath(args.out)} ({W}x{H}@{fps_out:.0f}fps)")

if __name__ == "__main__":
    main()
