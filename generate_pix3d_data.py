import os
import glob
import json

base_path = '/home/jazzie/code/occupancy_networks/data/Pix3D'
pix3d_model_paths = '/home/jazzie/data/pix3d/model'
pix3d_base_path = '/home/jazzie/data/pix3d'
json_path = '/home/jazzie/data/pix3d/pix3d.json'
model_paths = glob.glob(os.path.join(base_path, '*/*'))

with open(json_path, 'r') as f:
    data = json.load(f)

# PUT PIX3D IMAGES IN THEIR SPOTS
catmodel_to_imgs = {}
for pix3d_data in data:
    cat_model = '/'.join(pix3d_data['model'].split('/')[1:3])
    if cat_model in catmodel_to_imgs.keys():
        catmodel_to_imgs[cat_model].append(pix3d_data['img'])
    else:
        catmodel_to_imgs[cat_model] = [pix3d_data['img']]

for path in model_paths:
    cat_model = '/'.join(path.split('/')[-2:])
    imgs = catmodel_to_imgs[cat_model]
    save_path = os.path.join(path, 'images')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for img in imgs:
        img = 'preprocessed_' + img
        img_path = os.path.join(pix3d_base_path, img)

        run_string = 'cp %s %s'%(img_path, save_path)
        print(run_string)

'''
# GENERATE BINVOX AND POINTCLOUDS IN THEIR SPOTS
for path in model_paths:
    cat_model = '/'.join(path.split('/')[-2:])
    obj_path = os.path.join(pix3d_model_paths, cat_model)
    
    # uncomment to print out string that makes voxels and pointclouds
    # run_string = 'python sample_mesh.py %s --voxels_folder %s --points_folder %s'%(obj_path, path, path)

    print(run_string)
'''

print('done')

