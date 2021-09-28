"""
Run as: srun --gres gpu:1 --qos high2 --pty /home/jazzie/blender-2.82-linux64/blender -b --python pix3d_posed_blender_render.py -- regexp

Renders all models s.t.  model_name matches regexp
"""
import os
import os.path as osp
import sys
from shutil import copyfile
import numpy as np
import json
import re
import sys, traceback
import bpy
import glob
from math import radians
import argparse
from mathutils import Matrix

# Add this folder to path
sys.path.append(osp.dirname(osp.abspath(__file__)))
import utils

# parser = argparse.ArgumentParser()
# parser.add_argument('--cls_idx', type=int, help='index of class to render')
# args = parser.parse_args()
cls_idx = 2
DATASET_DIR =          '/home/jazzie/data/pix3d/'
RESULTS_DIR =          '/home/jazzie/data/pix3d/renders_posed_viewspace_test'
RESOLUTION =            512
RENDER_DEPTH =          False # True
RENDER_NORMALS =        True
COLOR_DEPTH =           16
DEPTH_FORMAT =          'OPEN_EXR'
COLOR_FORMAT =          'PNG'
NORMAL_FORMAT =         'OPEN_EXR'
CAMERA_FOV_RANGE =      [20, 50] # [20, 50] # [40, 40]
# CAMERA_FOC_RANGE =      [25, 60] # [20, 50] # [40, 40]
LIGHT_NUM =             6 # 6
LIGHT_ENERGY =          12 # 8

def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def parent_obj_to_camera(b_camera):
    origin = (0, 0, 0)
    b_empty = bpy.data.objects.new("Empty", None)
    b_empty.location = origin
    b_camera.parent = b_empty  # setup parenting
    bpy.context.scene.collection.objects.link(b_empty)
    bpy.context.view_layer.objects.active = b_empty
    return b_empty

def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        matrix_list.append(list(row))
    return matrix_list

def import_obj(obj_path) -> bpy.types.Object:
    status = bpy.ops.import_scene.obj(filepath=obj_path, axis_forward='-Z', axis_up='Y')
    assert('FINISHED' in status)
    obj = bpy.context.selected_objects[0]
    obj.rotation_euler = 0,0,0      # clear default rotation
    obj.location = 0,0,0            # clear default translation
    bpy.context.view_layer.update()
    return obj

def setup_nodegraph(scene):
    # Render Optimizations
    scene.render.use_persistent_data = True

    # Set up rendering of depth map.
    scene.use_nodes = True
    tree = scene.node_tree
    links = tree.links

    # Add passes for additionally dumping albedo and normals.
    scene.view_layers["View Layer"].use_pass_normal = True

    # Create input render layer node.
    render_layers = tree.nodes.new('CompositorNodeRLayers')
    if RENDER_DEPTH:
        depth_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
        depth_file_output.label = 'Depth Output'
        links.new(render_layers.outputs['Depth'], depth_file_output.inputs[0])
        depth_file_output.format.file_format = str(DEPTH_FORMAT)
        depth_file_output.base_path = ''
    else:
        depth_file_output = None
    
    if RENDER_NORMALS:
        '''
        scale_normal = tree.nodes.new(type='CompositorNodeMixRGB')
        scale_normal.blend_type = 'MULTIPLY'
        # scale_normal.inputs[2].default_value = (0.5, 0.5, 0.5, 1)
        scale_normal.inputs[2].default_value = (1.0, 1.0, 1.0, 1)
        links.new(render_layers.outputs['Normal'], scale_normal.inputs[1])

        bias_normal = tree.nodes.new(type="CompositorNodeMixRGB")
        bias_normal.blend_type = 'ADD'
        # bias_normal.inputs[2].default_value = (0.5, 0.5, 0.5, 0)
        bias_normal.inputs[2].default_value = (0.1, 0.1, 0.1, 0)
        links.new(scale_normal.outputs[0], bias_normal.inputs[1])
        normal_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
        normal_file_output.label = 'Normal Output'
        links.new(bias_normal.outputs[0], normal_file_output.inputs[0])
        '''
        normal_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
        normal_file_output.label = 'Normal Output'
        links.new(render_layers.outputs['Normal'], normal_file_output.inputs[0])
        
        normal_file_output.format.file_format = str(NORMAL_FORMAT)
        normal_file_output.base_path = ''
    else:
        normal_file_output = None

    return depth_file_output, normal_file_output

def create_random_point_lights(number, radius, energy=10):
    lights = []

    for i in range(number):
        # create light datablock, set attributes
        light_data = bpy.data.lights.new(name=f'ptlight{i}', type='POINT')
        light_data.energy = energy

        # create new object with our light datablock
        light_object = bpy.data.objects.new(name=f'ptlight{i}', object_data=light_data)

        #change location
        light_object.location = np.random.uniform(-1., 1., size=3)
        light_object.location *= radius / np.linalg.norm(light_object.location)

        lights.append(light_object)

    for light in lights:
        # link light object
        bpy.context.collection.objects.link(light)

    return lights

def render_multiple(obj_path, output_dir, data_dict, resolution, depth=True, normals=True, color_depth=16):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Clear scene
    utils.clean_objects()

    # Import obj
    obj_object = import_obj(obj_path)
    print('Imported name: ', obj_object.name)
    verts = np.array([tuple(obj_object.matrix_world @ v.co) for v in obj_object.data.vertices])
    vmin = verts.min(axis=0)
    vmax = verts.max(axis=0)
    vcen = (vmin+vmax)/2
    obj_size = np.abs(verts - vcen).max()

    scene = bpy.context.scene

    # Setup Node graph for rendering rgbs,depth,normals
    (depth_file_output, normal_file_output) = setup_nodegraph(scene)

    # Add random lighting
    light_objects = create_random_point_lights(LIGHT_NUM, 6*obj_size, energy=LIGHT_ENERGY)

    # Create collection for objects not to render with background
    objs = [ob for ob in scene.objects if ob.type in ('EMPTY') and 'Empty' in ob.name]
    bpy.ops.object.delete({"selected_objects": objs})

    # delete material if it exists
    obj_object.data.materials.clear()

    # Setup camera, constraint to empty object
    cam = utils.create_camera(location=(0, 0, 1))
    cam.data.sensor_fit = 'HORIZONTAL'
    cam.data.sensor_width = 36.0
    cam.data.sensor_height = 36.0
    b_empty = parent_obj_to_camera(cam)
    
    # utils.add_track_to_constraint(cam, b_empty)
    constraint = cam.constraints.new(type='TRACK_TO')
    constraint.target = b_empty
    constraint.track_axis = 'TRACK_NEGATIVE_Z'
    constraint.up_axis = 'UP_Z'

    # Move everything to be centered at vcen
    b_empty.location = vcen
    for light in light_objects:
        light.location += b_empty.location

    # Image settings
    scene.camera = cam
    scene.render.engine = 'CYCLES'
    scene.render.image_settings.file_format = str(COLOR_FORMAT)
    scene.render.image_settings.color_depth = str(COLOR_DEPTH)
    scene.render.resolution_x = data_dict['img_size'][0] #  RESOLUTION
    scene.render.resolution_y = data_dict['img_size'][1] # RESOLUTION
    scene.render.resolution_percentage = 100
    scene.render.dither_intensity = 0.0
    scene.render.film_transparent = True
    scene.view_layers[0].cycles.use_denoising = True
    scene.cycles.samples = 128

    
    # TRYING
    # this turns of gamma correction and makes nomral 'image' more accurate
    # however makes rendered image very dark
    # scene.view_settings.view_transform = 'Raw'

    out_data = {
        'obj_path':remove_prefix(obj_path, DATASET_DIR),
    }
    out_data['frames'] = []


    # remove model name to better match image dir structure
    # output_dir = '/'.join(output_dir.split('/')[0:-1])
    img_name = data_dict['img'].replace('.','/').split('/')[-2]
    scene.render.filepath = output_dir + '/r_' + str(img_name)
    
    # TODO should this go in a list?
    rot_mat = data_dict['rot_mat']
    trans_vec = data_dict['trans_mat']
    scale = 1
    obj = obj_object

    trans_4x4 = Matrix.Translation(trans_vec)
    rot_4x4 = Matrix(rot_mat).to_4x4()
    scale_4x4 = Matrix(np.eye(4)) # don't scale here
    obj.matrix_world = trans_4x4 @ rot_4x4 @ scale_4x4
    obj.scale = (scale, scale, scale)
    
    f = data_dict['focal_length']
    proj_model = 'PERSP'
    sensor_fit = 'HORIZONTAL'
    sensor_width = 32
    sensor_height = 18
    xyz = (0, 0, 0)
    rot_vec_rad = (0, np.pi, 0)

    bpy.context.view_layer.update()
    cam.location = xyz
    cam.rotation_euler = rot_vec_rad

    cam.data.type = proj_model
    cam.data.lens = f
    cam.data.sensor_fit = sensor_fit
    cam.data.sensor_width = sensor_width
    cam.data.sensor_height = sensor_height
    bpy.context.view_layer.update()

    if RENDER_DEPTH:
        depth_file_output.file_slots[0].path = scene.render.filepath + "_depth_"
    if RENDER_NORMALS:
        normal_file_output.file_slots[0].path = scene.render.filepath + "_normal_"

    bpy.ops.render.render(write_still=True)  # render still
    bpy.context.view_layer.update()

if __name__ == "__main__":

    # RANDOM SEED
    np.random.seed(0)

    # Check if only part of data has to be rerun
    regexp = ".*"
    pattern = re.compile(regexp)

    
    # LOAD IMAGES INSTEAD...
    data_list = json.load(open('/home/jazzie/data/pix3d/pix3d.json'))
    classes = ['bed', 'bookcase', 'chair', 'desk', 'misc', 'sofa', 'table', 'tool', 'wardrobe']
    CLASS = classes[cls_idx]
    import random
    # random.shuffle(data_list)
    for data_dict in data_list:
        model_path = os.path.join(DATASET_DIR, data_dict['model'])
        model_name = model_path.split('/')[-2]
        model_class = model_path.split('/')[-3]
        
        if model_class != CLASS:
            continue

        OBJ_PATH = model_path
        VIEWS = 1
        
        # reset scene
        bpy.ops.wm.read_homefile(use_empty=True)

        # FROM https://gist.github.com/S1U/13b8efe2c616a25d99de3d2ac4b34e86
        # Mark all scene devices as GPU for cycles
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.cycles.device = 'GPU'

        for scene in bpy.data.scenes:
            scene.cycles.device = 'GPU'

        # Enable CUDA
        bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'

        # Enable and list all devices, or optionally disable CPU
        for devices in bpy.context.preferences.addons['cycles'].preferences.get_devices():
            for d in devices:
                d.use = True
                if d.type == 'CPU':
                    d.use = False

        try:
            OUTPUT_DIR = f'{RESULTS_DIR}/{model_class}'
            img_name = data_dict['img'].replace('.','/').split('/')[-2]
            # if img_name != '0004':
            #     continue
            
            ## CHANGE BACK BEFORE DECLARING DONE
            # if len(glob.glob(os.path.join(OUTPUT_DIR,'r_' + img_name + '*'))) >= 2:
            #     print('already finished - continuing!')
            #     continue
            
            render_multiple(
                OBJ_PATH,
                OUTPUT_DIR,
                data_dict,
                RESOLUTION,
                depth=RENDER_DEPTH,
                normals=RENDER_NORMALS,
            )
        except:
            eprint("*** failed", model_name)
            exc_type, exc_value, exc_traceback = sys.exc_info()
            eprint("*** print_tb:")
            traceback.print_tb(exc_traceback, limit=1, file=sys.stderr)
            eprint("*** print_exception:")
            # exc_type below is ignored on 3.5 and later
            traceback.print_exception(exc_type, exc_value, exc_traceback,
                                    limit=2, file=sys.stderr)
#         import pdb;pdb.set_trace()
