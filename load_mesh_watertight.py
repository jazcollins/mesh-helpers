import os
import glob
import trimesh

use_obj = True
write_pix3d = True

if use_obj:
    path_to_objs = '/home/jazzie/data/pix3d/model/*/*/'
    # path_to_objs = '/home/jazzie/AMAZON3D146K/3dmodels_obj'
    model_paths = glob.glob(os.path.join(path_to_objs, 'model.obj'))
else:
    path_to_glbs = '/home/jazzie/ABO_RELEASE/3dmodels/original'
    model_paths = glob.glob(os.path.join(path_to_glbs, '*/*.glb'))

n_watertights = 0
for i, model_path in enumerate(model_paths):
    scene_or_mesh = trimesh.load(model_path, force='mesh') # process=False) to avoid weird watertightness issues
    if type(scene_or_mesh) == trimesh.base.Trimesh:
        mesh = scene_or_mesh
    else:
        key = list(scene_or_mesh.geometry.keys())
        if len(key) > 1:
            print(model_path, 'has too many keys:', key)
        mesh = scene_or_mesh.geometry[key[0]]
#     print('is_wateright:', mesh.is_watertight)
#     print('fill_holes:', mesh.fill_holes())
    mesh.fill_holes()
    if mesh.is_watertight:
        print(model_path)
        n_watertights += 1
        print(n_watertights, 'watertight meshes out of', i, 'meshes seen') 
        if write_pix3d:
            model_cat = model_path.split('/')[-3]
            model_name = model_path.split('/')[-2] 
            path = os.path.join('/home/jazzie/code/occupancy_networks/data/Pix3D', model_cat, model_name)
            if not os.path.exists(path):
                os.makedirs(path)

print(len(model_paths), 'model paths')
print(n_watertights, 'are watertight')
print('done')
