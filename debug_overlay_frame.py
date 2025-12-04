#ray
#generar de los keypoints normalizados coordenadas a pixeles

# debug_overlay_frame.py
# Script de DEBUG: NO toca skeletonExtractor ni genera CSVs nuevos.
# Solo toma un frame del video recortado, corre YOLO y dibuja
# el esqueleto + objetos definidos a mano en PIXELES.

from ultralytics import YOLO  # type: ignore
import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# CONFIGURACIÓN
# -------------------------------

# Nombre del clip (sin .mp4), igual al de trimmedClips/
SCENE_NAME = "columpioscam2-2024-12-14_17-55-00_0_Play_Object_Normal"

# Frame que quieres visualizar (mismo índice que en el video)
FRAME_TO_VIS = 8  # cámbialo al que quieras

# Rutas relativas desde la raíz del repo
VIDEO_PATH = f"video_processing/trimmedClips/{SCENE_NAME}.mp4"
MODEL_PATH = "video_processing/yolov8m-pose.pt"  # ajusta si usas otra

# Joints (igual que en grafoPanoramico)
NUM_JOINTS = 17
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
HAND_JOINTS = [9, 10]

# ⚠️ Objetos en PÍXELES para ESTE clip (ejemplo, cámbialos a los tuyos)
# Los números tipo (792,1176) son como en tu screenshot verde.
OBJECTS_PIX = [
    {"name": "seat_block_1", "x": 792, "y": 1176},
    {"name": "seat_block_2", "x": 1112, "y": 1176},
    {"name": "bench_1",      "x": 1154, "y": 1014},
    {"name": "bench_2",      "x": 1049, "y": 580},
]

# Radio de interacción mano-objeto en píxeles (ajústalo a ojo)
OBJ_RADIUS_PIX = 80

# -------------------------------
# CARGAR MODELO Y FRAME
# -------------------------------

print(f"[INFO] Cargando modelo desde {MODEL_PATH}...")
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"No se pudo abrir el video: {VIDEO_PATH}")

# Ir directo al frame deseado
cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_TO_VIS)
ret, frame_bgr = cap.read()
cap.release()

if not ret:
    raise RuntimeError(f"No se pudo leer el frame {FRAME_TO_VIS} del video {VIDEO_PATH}")

# Convertir a RGB para matplotlib
frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
h, w = frame_rgb.shape[:2]
print(f"[INFO] Frame {FRAME_TO_VIS} tamaño: {w}x{h}")

# -------------------------------
# CORRER YOLO SOLO EN ESTE FRAME
# -------------------------------

results = model.predict(frame_rgb, verbose=False)
if len(results) == 0 or results[0].keypoints is None:
    raise RuntimeError("YOLO no detectó pose en este frame.")

kpts = results[0].keypoints.xy.cpu().numpy()  # [num_personas, 17, 2]
num_personas = kpts.shape[0]
print(f"[INFO] Personas detectadas: {num_personas}")

# Por simplicidad, usamos la persona 0 (la más grande suele ser el sujeto principal)
person_id = 0
coords = kpts[person_id]  # [17, 2] en PÍXELES

# -------------------------------
# PLOT: IMAGEN REAL + ESQUELETO + OBJETOS
# -------------------------------

plt.figure(figsize=(10, 6))
plt.imshow(frame_rgb)
ax = plt.gca()

# 1) Esqueleto
for (j1, j2) in SKELETON_EDGES:
    x1, y1 = coords[j1]
    x2, y2 = coords[j2]
    plt.plot([x1, x2], [y1, y2], linewidth=2, color="cyan", alpha=0.8)

# 2) Joints
plt.scatter(coords[:, 0], coords[:, 1], s=25, color="yellow", edgecolors="black", label="joints")

# 3) Objetos + radios
for i, obj in enumerate(OBJECTS_PIX):
    ox = float(obj["x"])
    oy = float(obj["y"])
    name = obj.get("name", f"obj_{i}")

    # punto del objeto
    plt.scatter([ox], [oy], s=60, marker="s", color="red", edgecolors="black")
    plt.text(ox + 5, oy - 5, name, color="white", fontsize=9,
             bbox=dict(facecolor="black", alpha=0.4, pad=1))

    # círculo de radio
    circle = plt.Circle((ox, oy), OBJ_RADIUS_PIX,
                        fill=False, linestyle="--", edgecolor="white", alpha=0.6)
    ax.add_patch(circle)

# 4) Manos que entran al radio → línea verde
for j in HAND_JOINTS:
    hx, hy = coords[j]
    for obj in OBJECTS_PIX:
        ox = float(obj["x"])
        oy = float(obj["y"])
        dist_pix = np.linalg.norm([hx - ox, hy - oy])

        if dist_pix < OBJ_RADIUS_PIX:
            plt.plot([hx, ox], [hy, oy], linewidth=3, color="lime")
            plt.scatter([hx], [hy], s=60, color="lime", edgecolors="black")

plt.title(f"{SCENE_NAME} - frame {FRAME_TO_VIS}")
plt.axis("off")

out_path = f"{SCENE_NAME}_frame_{FRAME_TO_VIS}_overlay.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"[OK] Guardado {out_path}")
# Si tienes entorno gráfico local, también podrías usar:
# plt.show()
