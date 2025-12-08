import torch
import numpy as np
import matplotlib.pyplot as plt

# Cargar el Data guardado (lo generaste tú, así que es seguro poner weights_only=False)
data = torch.load(
    "panoramic_graph_with_objects.pt",
    map_location="cpu",
    weights_only=False,
)

x = data.x.numpy()                   # [num_nodes, 6]
edge_index = data.edge_index.numpy() # [2, num_edges]
edge_type = data.edge_type.numpy()   # [num_edges]

xs = x[:, 0]     # coord x
ys = x[:, 1]     # coord y
f_norm = x[:, 2] # frame normalizado [0,1]
pid = x[:, 3]    # person_id
jid = x[:, 4]    # joint_id
is_obj = x[:, 5] > 0.5  # booleano

print("Número de nodos:", x.shape[0])
print("Número de aristas:", edge_index.shape[1])

# Frames únicos
unique_frames = np.unique(f_norm)
print("Frames normalizados únicos:", unique_frames)

# --- Elegimos cuántos frames guardar ---
# Puedes cambiar este número
num_frames_to_save = min(5, len(unique_frames))
print(f"Guardando visualizaciones de los primeros {num_frames_to_save} frames...")

for frame_idx in range(num_frames_to_save):
    f_sel = unique_frames[frame_idx]
    print(f"[Frame {frame_idx}] f_norm = {f_sel}")

    # Nodos de este frame
    mask_nodes = np.isclose(f_norm, f_sel, atol=1e-3)
    nodes_sel = np.where(mask_nodes)[0]
    print("  Nodos en este frame:", len(nodes_sel))

    # Humanos vs objetos en este frame
    nodes_obj = nodes_sel[is_obj[nodes_sel]]
    nodes_hum = nodes_sel[~is_obj[nodes_sel]]

    # Aristas que conectan solo nodos de este frame
    mask_edges = np.isin(edge_index[0], nodes_sel) & np.isin(edge_index[1], nodes_sel)
    edges_sel = edge_index[:, mask_edges]
    edge_types_sel = edge_type[mask_edges]

    # Como las aristas están duplicadas (u->v, v->u), nos quedamos con la mitad
    edges_sel = edges_sel[:, ::2]
    edge_types_sel = edge_types_sel[::2]

    # --- Plot ---
    plt.figure(figsize=(6, 6))

    # Joints humanos
    plt.scatter(xs[nodes_hum], ys[nodes_hum], s=30, alpha=0.8, label="Joints humanos")

    # Objetos (centroides)
    plt.scatter(xs[nodes_obj], ys[nodes_obj], s=80, marker="s", alpha=0.9, label="Objetos")

    # Dibujar aristas
    for (u, v), et in zip(edges_sel.T, edge_types_sel):
        x_u, y_u = xs[u], ys[u]
        x_v, y_v = xs[v], ys[v]

        # intra (esqueleto)
        if et == 0:
            plt.plot([x_u, x_v], [y_u, y_v], linewidth=1, alpha=0.4)
        # inter (pelvis-pelvis)
        elif et == 1:
            plt.plot([x_u, x_v], [y_u, y_v], linewidth=1, linestyle="--", alpha=0.5)
        # temporal
        elif et == 2:
            plt.plot([x_u, x_v], [y_u, y_v], linewidth=0.5, alpha=0.2)
        # obj (mano-objeto) → más gruesa
        elif et == 3:
            plt.plot([x_u, x_v], [y_u, y_v], linewidth=2, alpha=0.9)

    plt.gca().invert_yaxis()  # opcional por si tu sistema de coords va hacia abajo
    plt.legend()
    plt.title(f"Frame {frame_idx} (f_norm={f_sel:.3f})")
    plt.tight_layout()

    # En lugar de show, guardamos la imagen
    out_path = f"frame_{frame_idx:02d}_debug.png"
    plt.savefig(out_path)
    plt.close()
    print(f"  Guardado {out_path}")

print("Listo. Revisa los PNG generados en esta carpeta.")
