import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importamos cosas del grafoPanoramico para reutilizar constantes
from grafoPanoramico import (
    CSV_PATH,
    SKELETON_EDGES,
    HAND_JOINTS,
    load_objects_from_yaml,
)

# -------------------------------------------------
# Configuración
# -------------------------------------------------

# Frame que quieres visualizar (usa uno de los que salen en tu CSV, ej. 0, 2, 4, ...)
FRAME_TO_VIS = 8

# Cámara para los objetos
CAMERA_ID = "columpioscam1"

# Umbral de distancia mano-objeto en el espacio normalizado
OBJ_DIST_THRESH = 0.15  # ajusta según veas

# -------------------------------------------------
# Cargar datos
# -------------------------------------------------

df = pd.read_csv(CSV_PATH)
objects = load_objects_from_yaml(camera_id=CAMERA_ID)

# Nos quedamos solo con el frame deseado
df_frame = df[df["frame"] == FRAME_TO_VIS]

if df_frame.empty:
    raise ValueError(f"No hay filas para frame={FRAME_TO_VIS} en {CSV_PATH}")

print(f"Filas en frame {FRAME_TO_VIS}:", len(df_frame))

# -------------------------------------------------
# Plot
# -------------------------------------------------

plt.figure(figsize=(6, 6))

# Dibujar objetos y círculos de umbral
for oid, obj in enumerate(objects):
    ox = float(obj["x"])
    oy = float(obj["y"])
    name = obj.get("name", f"obj_{oid}")

    # Punto del objeto
    plt.scatter([ox], [oy], s=80, marker="s", label="Objetos" if oid == 0 else "", zorder=3)

    # Círculo de radio = umbral
    circle = plt.Circle((ox, oy), OBJ_DIST_THRESH, fill=False, linestyle="--", alpha=0.5)
    plt.gca().add_patch(circle)

    # Etiqueta opcional del objeto
    plt.text(ox + 0.01, oy + 0.01, name, fontsize=8)

# Dibujar personas (pueden existir varias)
for _, row in df_frame.iterrows():
    pid = int(row["person_id"])

    # Coords de los joints de esta persona
    coords = []
    for j in range(17):  # NUM_JOINTS en tu script
        xj = float(row[f"x{j}"])
        yj = float(row[f"y{j}"])
        coords.append((xj, yj))

    coords = np.array(coords)

    # ---- Esqueleto (edges intra) ----
    for (j1, j2) in SKELETON_EDGES:
        x1, y1 = coords[j1]
        x2, y2 = coords[j2]
        plt.plot([x1, x2], [y1, y2], linewidth=1, alpha=0.6, color="C0")

    # Joints (puntos)
    plt.scatter(coords[:, 0], coords[:, 1], s=30, alpha=0.9, label="Joints humanos" if pid == 0 else "")

    # ---- Mano-objeto según umbral ----
    for j in HAND_JOINTS:
        hx, hy = coords[j]
        # Para cada objeto, checamos distancia
        for oid, obj in enumerate(objects):
            ox = float(obj["x"])
            oy = float(obj["y"])
            dist = np.linalg.norm([hx - ox, hy - oy])

            if dist < OBJ_DIST_THRESH:
                # Dibujar línea gruesa mano-objeto
                plt.plot(
                    [hx, ox],
                    [hy, oy],
                    linewidth=2,
                    alpha=0.9,
                    color="red",
                )

# Ajustes del plot
plt.gca().invert_yaxis()  # por si tu sistema original tiene y hacia abajo
plt.title(f"Frame {FRAME_TO_VIS} - esqueleto + objetos + umbral")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.tight_layout()

# Muestra en pantalla o guarda
plt.savefig(f"frame_{FRAME_TO_VIS}_esqueleto_objetos.png")
print(f"Guardado frame_{FRAME_TO_VIS}_esqueleto_objetos.png")
# Si tienes entorno gráfico, también puedes usar plt.show()
# plt.show()
