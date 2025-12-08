import re
from pathlib import Path

import numpy as np
import yaml

#centroides de obj (normalizados 0..1) para una cámara dada
#returns un np.ndarray de shape [n_obj, 2] con (x,y) p/obj
def load_objects_for_camera(objects_yaml_path, camera_id):
    objects_yaml_path = Path(objects_yaml_path)
    with objects_yaml_path.open("r") as f:
        data = yaml.safe_load(f)
    if camera_id not in data:
        raise KeyError(
            f"cam'{camera_id}' no está en {objects_yaml_path}. "
            f"available keys {list(data.keys())}"
        )
    obj_list = data[camera_id]
    coords = []
    for obj in obj_list:
        x = obj.get("x", None)
        y = obj.get("y", None)
        if x is None or y is None:
            raise ValueError(f"obj mal definido para cam {camera_id}: {obj}")
        coords.append([float(x), float(y)])
    if not coords:
        #sin objetos->array vacío
        return np.zeros((0, 2), dtype=np.float32)
    return np.array(coords, dtype=np.float32)

#returns un tensor panorámico [T, K_max, 17 + n_obj, 2] 
#los primeros 17 joints son humanos, los siguientes n_obj son objetos replicados para c/persona
def add_objects_to_skeleton_tensor(skel_tensor, object_coords):
    if skel_tensor.ndim != 4:
        raise ValueError(f"expected tensor , recibí {skel_tensor.shape}")
    T, K_max, V, C = skel_tensor.shape
    assert V == 17, f"expected 17 joints y V={V}"
    n_obj = object_coords.shape[0]
    V_prime = V + n_obj
    #out tensor
    pano = np.zeros((T, K_max, V_prime, C), dtype=np.float32)

    pano[:, :, :V, :] = skel_tensor#copiar 17 joints 
    if n_obj == 0:
        #sin obj, solo humanos
        return pano

    objs = object_coords.astype(np.float32)  #obj constantes en tiempo y en comun
    #expandir a [1,1,n_obj,2] y luego broadcast a [T,K_max,n_obj,2]
    objs_4d = objs.reshape(1, 1, n_obj, C)
    objs_4d = np.broadcast_to(objs_4d, (T, K_max, n_obj, C))
    pano[:, :, V:, :] = objs_4d
    return pano

#idx de joints
NOSE = 0
L_SHOULDER = 5
R_SHOULDER = 6
L_ELBOW = 7
R_ELBOW = 8
L_WRIST = 9
R_WRIST = 10
L_HIP = 11
R_HIP = 12
L_KNEE = 13
R_KNEE = 14
L_ANKLE = 15
R_ANKLE = 16

#pelvis = midpoint(L_HIP, R_HIP)

#bones humanos: pares (i, j) de joints conectados
HUMAN_BONES = [
    # torso aproximado: caderas <-> hombros <-> nariz
    (L_HIP, L_SHOULDER),
    (R_HIP, R_SHOULDER),
    (L_SHOULDER, NOSE),
    (R_SHOULDER, NOSE),
    # brazo izquierdo
    (L_SHOULDER, L_ELBOW),
    (L_ELBOW, L_WRIST),
    # brazo derecho
    (R_SHOULDER, R_ELBOW),
    (R_ELBOW, R_WRIST),
    # pierna izquierda
    (L_HIP, L_KNEE),
    (L_KNEE, L_ANKLE),
    # pierna derecha
    (R_HIP, R_KNEE),
    (R_KNEE, R_ANKLE),
]

#matrices A0, A_intra, A_inter para grafo con V_human joints y n_obj objetos
#returns A0, A_INTRA, A_INTER cada una de shape [V', V'] donde V' = V_human + n_obj
def build_adjacency_matrices(V_human=17, n_obj=0):
    V_prime = V_human + n_obj

    A0 = np.eye(V_prime, dtype=np.float32) #identidad
    A_intra = np.zeros((V_prime, V_prime), dtype=np.float32) #humano + obj <->manos

    #esqueleto
    for i, j in HUMAN_BONES:
        if i < V_human and j < V_human:
            A_intra[i, j] = 1.0
            A_intra[j, i] = 1.0
    #obj<->manos (para todos)
    #obj: idx [V_human .. V_prime-1]
    for v in range(V_human, V_prime):
        for hand in (L_WRIST, R_WRIST):
            if hand < V_human:
                A_intra[hand, v] = 1.0
                A_intra[v, hand] = 1.0
    # A_inter inter-persona (pelvis<->pelvis)
    # usando la dimensión M (personas) y el midpoint de hips
    A_inter = np.zeros((V_prime, V_prime), dtype=np.float32)
    return A0, A_intra, A_inter

#de tensor panorámico [T,M,V',2] a J stream [C,T,V',M]
#returns J: np.ndarray (C=2, T, V', M)
def pano_to_J(pano):
    if pano.ndim != 4:
        raise ValueError(f"expected pano [T,M,V',2] y recibe {pano.shape}")
    J = np.transpose(pano, (3, 0, 2, 1)) #reordenar ejes
    return J

#dif temporal a lo largo de T para un stream [C,T,V',M]
def temporal_diff_stream(X):
    dX = np.zeros_like(X)
    dX[:, 1:, :, :] = X[:, 1:, :, :] - X[:, :-1, :, :]
    return dX

#stream de bones B a partir de pano [T,M,V',2]
def compute_B_from_pano(pano, V_human=17):
    if pano.ndim != 4:
        raise ValueError(f"expected pano [T,M,V',2] y recibe {pano.shape}")

    T, M, V_prime, C = pano.shape
    if V_prime < V_human:
        raise ValueError(f"V' ({V_prime}) no puede ser < V_human ({V_human})")

    B = np.zeros_like(pano) #en ceros
    parent = np.full(V_human, -1, dtype=int) #mapeo padre->hijo, para c/joint, quién es su padre

    #padre=primer elemento del par (i, j) en HUMAN_BONES
    for i, j in HUMAN_BONES:
        if j < V_human:
            parent[j] = i

    #para c/joint con padre, calcular vector hijo - padre
    for j in range(V_human):
        p = parent[j]
        if p < 0:
            continue  #sin padre, se queda en 0
        #pano[..., joint, :] = (T, M, 2)
        B[:, :, j, :] = pano[:, :, j, :] - pano[:, :, p, :]

    #para nodos de obj (índices >= V_human), 0
    return B

# Construye los 4 streams J, B, JM, BM a partir de un tensor panorámico [T, M, V', 2]
def build_streams_from_pano(pano, V_human=17):
    J = pano_to_J(pano)  #joints 2, T, V', M)
    B_skel = compute_B_from_pano(pano, V_human=V_human)#bones en shape [T,M,V',2]
    B = np.transpose(B_skel, (3, 0, 2, 1)) #bones a [C,T,V',M]
    #dif temporales
    JM = temporal_diff_stream(J)
    BM = temporal_diff_stream(B)

    return J, B, JM, BM

#build de streams y matrices a partir de un pano [T,M,V',2]
#returns J, B, JM, BM, A0, A_intra, A_inter
def build_panoramic_graph_from_pano(pano, V_human=17):
    T, M, V_prime, C = pano.shape
    n_obj = V_prime - V_human
    A0, A_intra, A_inter = build_adjacency_matrices(V_human=V_human, n_obj=n_obj)
    J, B, JM, BM = build_streams_from_pano(pano, V_human=V_human)
    return J, B, JM, BM, A0, A_intra, A_inter

#del file name, infiere la cámara (cam1, cam2, etc) en YAML
def infer_camera_id_from_filename(npy_path: Path) -> str:
    name = npy_path.name
    m = re.search(r"cam(\d+)", name)
    if not m:
        raise ValueError(
            f"no se logra inferir cam del file: {name}"
        )
    cam_num = m.group(1)  
    camera_id = f"columpioscam{cam_num}"
    return camera_id

#dado el path a un .npy y al YAML de objetos, 
# infiere cam, carga centroides y devuelve el tensor [T,K,17+n_obj,2]
def build_panoramic_tensor(npy_path, objects_yaml_path):
    npy_path = Path(npy_path)
    skel = np.load(npy_path)  # [T,K,17,2]
    camera_id = infer_camera_id_from_filename(npy_path)
    print(f"cam para {npy_path.name}: {camera_id}")
    obj_coords = load_objects_for_camera(objects_yaml_path, camera_id)
    print(f"obj para {camera_id}: {obj_coords.shape[0]}")
    pano = add_objects_to_skeleton_tensor(skel, obj_coords)
    return pano

#recorre todos los .npy en npy_dir, construye el tensor panorámico
#correspondiente a cada uno y lo guarda en out_dir con el mismo nombre
def process_all_panoramic_tensors(
    npy_dir="../data/npy",
    out_dir="../data/panoramic_npy",
    objects_yaml_path="../configs/objects_manual.yaml",
):
    npy_dir = Path(npy_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    objects_yaml_path = Path(objects_yaml_path)
    npy_files = sorted(npy_dir.glob("*.npy"))
    print(f"found {len(npy_files)} .npy en {npy_dir.resolve()}")

    for npy_path in npy_files:
        print(f"\procesando {npy_path.name}")
        try:
            pano = build_panoramic_tensor(npy_path, objects_yaml_path)
        except Exception as e:
            print(f"error c/{npy_path.name}: {e}")
            continue

        out_path = out_dir / npy_path.name
        np.save(out_path, pano)
        print(f"guardado panorámico {out_path.resolve()}  shape={pano.shape}")

    print("\n .npy panorámicos generados.")


if __name__ == "__main__":
    process_all_panoramic_tensors()
