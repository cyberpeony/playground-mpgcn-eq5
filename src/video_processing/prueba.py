from pathlib import Path
import numpy as np
from panoramic_graph import build_panoramic_graph_from_pano

pano_path = Path("../data/panoramic_npy/columpios_cam4-2024-09-25 17:14:39_0_Social_People.npy")
pano = np.load(pano_path) 

print("pano shape:", pano.shape)

J, B, JM, BM, A0, A_intra, A_inter = build_panoramic_graph_from_pano(pano)

#(2, 48, V', 4)
print("J shape:", J.shape)  
print("B shape:", B.shape)    
print("JM shape:", JM.shape)  
print("BM shape:", BM.shape)  
#(V', V') 
print("A0 shape:", A0.shape)  
print("A_intra shape:", A_intra.shape) 
print("A_inter shape:", A_inter.shape)  
