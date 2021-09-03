import os
import glob
import pymesh

# DOESN'T WORK BC WRONG PYMESH
path_to_objs = '/home/jazzie/AMAZON3D146K/3dmodels_obj'
objs = glob.glob(os.path.join(path_to_objs, '*.obj'))

for obj in objs:
    mesh = pymesh.load_mesh(obj)
    import pdb;pdb.set_trace()

print(len(objs))
print('done')
