# A simple script that uses blender to render views of a single object by rotation the camera around it.  Works with blender version 2.79b.
# Also produces normal map at the same time.
# Adapted from stanford-shapenet-renderer
# https://github.com/panmari/stanford-shapenet-renderer/blob/f77a7932a644f7acaee93176ae21bdd87c13e765/render_blender.py
# Example:
# srun --gres gpu:1 --qos low --pty /home/jazzie/blender-2.79b-linux-glibc219-x86_64/blender -b --python pix3d_shapenet_blender_render.py -- --views 10
#

import argparse, sys, os, random
import bpy
import json
import numpy as np
from math import radians

parser = argparse.ArgumentParser(description='Renders given obj file by rotation a camera around it.')
parser.add_argument('--views', type=int, default=30,
                    help='number of views to be rendered')
parser.add_argument('--cls_idx', type=int, default=1,
                    help='which class to render')
parser.add_argument('--output_folder', type=str, default='/tmp',
                    help='The path the output will be dumped to.')
parser.add_argument('--scale', type=float, default=1,
                    help='Scaling factor applied to model. Depends on size of mesh.')
parser.add_argument('--remove_doubles', type=bool, default=True,
                    help='Remove double vertices to improve mesh quality.')
parser.add_argument('--edge_split', type=bool, default=True,
                    help='Adds edge split filter.')
parser.add_argument('--depth_scale', type=float, default=1.4,
                    help='Scaling that is applied to depth. Depends on size of mesh. Try out various values until you get a good result. Ignored if format is OPEN_EXR.')
parser.add_argument('--color_depth', type=str, default='8',
                    help='Number of bit per channel used for output. Either 8 or 16.')
parser.add_argument('--format', type=str, default='PNG',
                    help='Format of files generated. Either PNG or OPEN_EXR')

argv = sys.argv[sys.argv.index("--") + 1:]
args = parser.parse_args(argv)

# globals
DATASET_DIR =           '/home/jazzie/data/pix3d/'
RESULTS_DIR =           '/home/jazzie/data/pix3d/v2/renders'
RESOLUTION =            512
RENDER_NORMALS =        True
COLOR_DEPTH =           16
NORMAL_FORMAT =         'PNG'
CAMERA_FOV_RANGE =      [20, 50]
LIGHT_NUM =             6
LIGHT_ENERGY =          12

def parent_obj_to_camera(b_camera):
    origin = (0, 0, 0)
    b_empty = bpy.data.objects.new("Empty", None)
    b_empty.location = origin
    b_camera.parent = b_empty  # setup parenting

    scn = bpy.context.scene
    scn.objects.link(b_empty)
    scn.objects.active = b_empty
    return b_empty

def create_random_point_lights(number, radius, energy=10):
    lights = []

    for i in range(number):
        # create light datablock, set attributes
        light_data = bpy.data.lamps.new(name='ptlight%d'%i, type='POINT')
        light_data.energy = energy

        # create new object with our light datablock
        light_object = bpy.data.objects.new(name='ptlight%d'%i, object_data=light_data)

        #change location
        light_object.location = np.random.uniform(-1., 1., size=3)
        light_object.location *= radius / np.linalg.norm(light_object.location)

        lights.append(light_object)

    for light in lights:
        # link light object
        # bpy.context.collection.objects.link(light)
        bpy.context.scene.objects.link(light)

    return lights

def render_multiple(OBJ_PATH, OUTPUT_DIR, DATA_DICT):

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # Set up rendering of normal map.
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links

    # Add passes for additionally dumping albedo and normals.
    bpy.context.scene.render.layers["RenderLayer"].use_pass_normal = True
    bpy.context.scene.render.layers["RenderLayer"].use_pass_color = True
    bpy.context.scene.render.image_settings.file_format = args.format
    bpy.context.scene.render.image_settings.color_depth = args.color_depth

    # Clear default nodes
    for n in tree.nodes:
        tree.nodes.remove(n)

    # Create input render layer node.
    render_layers = tree.nodes.new('CompositorNodeRLayers')

    depth_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    depth_file_output.label = 'Depth Output'
    map = tree.nodes.new(type='CompositorNodeMapValue')
    map.offset = [-0.7]
    map.size = [args.depth_scale]
    map.use_min = True
    map.min = [0]
    links.new(render_layers.outputs['Depth'], map.inputs[0])
    links.new(map.outputs[0], depth_file_output.inputs[0])

    normal_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    normal_file_output.label = 'Normal Output'
    links.new(render_layers.outputs['Normal'], normal_file_output.inputs[0])

    bpy.ops.import_scene.obj(filepath=OBJ_PATH)
    obj = bpy.context.selected_objects[0]
    obj.rotation_euler = 0,0,0      # clear default rotation
    obj.location = 0,0,0            # clear default translation
    verts = np.array([tuple(obj.matrix_world * v.co) for v in obj.data.vertices])
    vmin = verts.min(axis=0)
    vmax = verts.max(axis=0)
    vcen = (vmin+vmax)/2
    obj_size = np.abs(verts - vcen).max()

    # add lights
    light_objects = create_random_point_lights(LIGHT_NUM, 6*obj_size, energy=LIGHT_ENERGY)

    scene = bpy.context.scene
    scene.render.resolution_x = RESOLUTION
    scene.render.resolution_y = RESOLUTION
    scene.render.resolution_percentage = 100
    scene.render.alpha_mode = 'TRANSPARENT'
    
    bpy.ops.object.camera_add(location=(0, 0, 1))
    cam = bpy.context.object
    cam.location = (0, 1, 0.6)
    cam_constraint = cam.constraints.new(type='TRACK_TO')
    cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    cam_constraint.up_axis = 'UP_Y'
    b_empty = parent_obj_to_camera(cam)
    cam_constraint.target = b_empty
    scene.camera = cam

    model_identifier = os.path.split(os.path.split(OBJ_PATH)[0])[1]
    fp = OUTPUT_DIR
    scene.render.image_settings.file_format = 'PNG'  # set output format to .png

    stepsize = 360.0 / args.views
    rotation_mode = 'XYZ'

    for output_node in [normal_file_output]:
        output_node.base_path = ''

    for i in range(0, args.views):
        print("Rotation {}, {}".format((stepsize * i), radians(stepsize * i)))

        scene.render.filepath = fp + '_r_{0:03d}'.format(int(i * stepsize))
        normal_file_output.file_slots[0].path = scene.render.filepath + "_normal.png"

        bpy.ops.render.render(write_still=True)  # render still

        b_empty.rotation_euler[2] += radians(stepsize)

if __name__ == "__main__":

    # random seed
    np.random.seed(0)

    data_list = json.load(open('/home/jazzie/data/pix3d/pix3d.json'))

    classes = ['bed', 'bookcase', 'chair', 'desk', 'misc', 'sofa', 'table', 'tool', 'wardrobe']
    CLASS = classes[args.cls_idx]
    # random.shuffle(data_list)
    for data_dict in data_list:
        model_path = os.path.join(DATASET_DIR, data_dict['model'])
        model_name = model_path.split('/')[-2]
        model_class = model_path.split('/')[-3]

        if model_class != CLASS:
            continue
        
        # setup
        bpy.ops.wm.read_homefile(use_empty=True)
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.cycles.device = 'GPU'
        for scene in bpy.data.scenes:
            scene.cycles.device = 'GPU'
        bpy.context.user_preferences.addons["cycles"].preferences.compute_device_type = "CUDA"

        output_dir = '%s/%s'%(RESULTS_DIR, model_class)
        img_name = data_dict['img'].replace('.','/').split('/')[-2]

        # if len(glob.glob(os.path.join(OUTPUT_DIR,'r_' + img_name + '*'))) >= 2:
        #     print('already finished - continuing!')
        #     continue

        render_multiple(model_path, output_dir, data_dict)

        import pdb;pdb.set_trace()
