import os
import glob
from tqdm import tqdm
import trimesh
import pymeshlab as ml
  
base_path = '/home/jazzie/ABO_RELEASE/3dmodels/original'
save_dir = '/home/jazzie/data/abo/3dmodels_lowres5k'
TARGET = 5000 # target number of verts

model_paths = glob.glob(os.path.join(base_path, '*', '*.glb'))
for model_path in tqdm(model_paths):
    asin = model_path.replace('.','/').split('/')[-2]

    ms = ml.MeshSet()
    ms.load_new_mesh(model_path)

    m = ms.current_mesh()
    print('input mesh has', m.vertex_number(), 'vertex and', m.face_number(), 'faces')

    # estimate number of faces to have 100+10000 vertex using Euler
    numFaces = 100 + 2*TARGET

    # simplify the mesh. Only first simplification will be agressive
    while (ms.current_mesh().vertex_number() > TARGET):
        ms.apply_filter('simplification_quadric_edge_collapse_decimation', targetfacenum=numFaces, preservenormal=True)
        print("Decimated to", numFaces, "faces mesh has", ms.current_mesh().vertex_number(), "vertex")
        #Refine our estimation to slowly converge to TARGET vertex number
        numFaces = numFaces - (ms.current_mesh().vertex_number() - TARGET)

    m = ms.current_mesh()
    print('output mesh has', m.vertex_number(), 'vertex and', m.face_number(), 'faces')

    verts = m.vertex_matrix()
    faces = m.face_matrix()

    trimesh_mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    save_path = os.path.join(save_dir,'%s.glb'%asin)
    print(save_path)
    trimesh_mesh.export(save_path)



