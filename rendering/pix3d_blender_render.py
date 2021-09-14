"""
Run as: srun --gres gpu:1 --qos high2 --pty /home/jazzie/blender-2.82-linux64/blender -b --python pix3d_blender_render.py -- regexp

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

# Add this folder to path
sys.path.append(osp.dirname(osp.abspath(__file__)))
import utils

# parser = argparse.ArgumentParser()
# parser.add_argument('--cls_idx', type=int, help='index of class to render')
# args = parser.parse_args()
cls_idx = 2
DATASET_DIR =          '/home/jazzie/data/pix3d/model'
RESULTS_DIR =          '/home/jazzie/data/pix3d/renders'
VIEWS =                 24
RESOLUTION =            512
RENDER_DEPTH =          True
RENDER_NORMALS =        True
COLOR_DEPTH =           16
DEPTH_FORMAT =          'OPEN_EXR'
COLOR_FORMAT =          'PNG'
NORMAL_FORMAT =         'PNG'
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
    # scene.view_layers["View Layer"].use_pass_color = True

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
        normal_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
        normal_file_output.label = 'Normal Output'
        links.new(render_layers.outputs['Normal'], normal_file_output.inputs[0])
        normal_file_output.format.file_format = str(NORMAL_FORMAT)
        normal_file_output.base_path = ''
    else:
        normal_file_output = None

    # albedo_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    # albedo_file_output.label = 'Albedo Output'
    # links.new(render_layers.outputs['Color'], albedo_file_output.inputs[0])
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

def render_multiple(obj_path, output_dir, views, resolution, depth=True, normals=True, color_depth=16):

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
    scene.render.resolution_x = RESOLUTION
    scene.render.resolution_y = RESOLUTION
    scene.render.resolution_percentage = 100
    scene.render.dither_intensity = 0.0
    scene.render.film_transparent = True
    scene.view_layers[0].cycles.use_denoising = True
    scene.cycles.samples = 128

    out_data = {
        'obj_path':remove_prefix(obj_path, DATASET_DIR),
    }
    out_data['frames'] = []

    eles = np.random.uniform(-10.0, 60.0, VIEWS)
    azis = np.random.uniform(0, 360.0, VIEWS)
    # for debugging
    stepsize = 360.0 / VIEWS

    for i in range(0, VIEWS):
        scene.render.filepath = output_dir + '/r_' + str(i)
        
        # random rotation
        # b_empty.rotation_euler = np.random.uniform(0, 2*np.pi, size=3)
        
        b_empty.rotation_euler[0] = radians(-1 * eles[i])
        b_empty.rotation_euler[1] += radians(stepsize)
        # print(b_empty.rotation_euler)
        # import pdb;pdb.set_trace()
        
        # Update camera location and angle
        bpy.context.view_layer.update()
        # cam = scene.camera
        # cam.data.lens = np.random.uniform(CAMERA_FOV_RANGE[0],CAMERA_FOV_RANGE[1])
        cam.data.angle = np.random.uniform(CAMERA_FOV_RANGE[0],CAMERA_FOV_RANGE[1]) * np.pi/180
        # cam.data.angle = cam.data.lens * np.pi/180
        # cam.data.angle = 40 * np.pi/180
        cam.location =  (0, 0, 1.8 * obj_size/np.tan(cam.data.angle/2))
        bpy.context.view_layer.update()

        if RENDER_DEPTH:
            depth_file_output.file_slots[0].path = scene.render.filepath + "_depth_"
        if RENDER_NORMALS:
            normal_file_output.file_slots[0].path = scene.render.filepath + "_normal_"

        bpy.ops.render.render(write_still=True)  # render still

        bpy.context.view_layer.update()
        frame_data = {
            'file_path': remove_prefix(scene.render.filepath, DATASET_DIR),
            'transform_matrix': listify_matrix(cam.matrix_world),

            # Independent components that make up transformation matrix
            'camera':{
                'angle_x': cam.data.angle_x,
                'angle_y': cam.data.angle_y,
                'shift_x': cam.data.shift_x,
                'shift_y': cam.data.shift_y,
                'sensor_height': cam.data.sensor_height,
                'sensor_width': cam.data.sensor_width,
                'sensor_fit': cam.data.sensor_fit,
                # 'focal': cam.data.lens,
                'location': list(cam.location),
                # 'scale': list(cam.scale),
                # 'rotation_quaternion': list(cam.rotation_quaternion),
                # 'be_location': list(b_empty.location),
                # 'be_scale': list(b_empty.scale),
                # 'be_rotation_euler': list(b_empty.rotation_euler),
                # 'be_rotation_matrix': listify_matrix(b_empty.matrix_world),
            }
        }
        out_data['frames'].append(frame_data)

    with open(output_dir + '/' + 'transforms.json', 'w') as out_file:
        json.dump(out_data, out_file, indent=4)


if __name__ == "__main__":

    # FROM https://gist.github.com/S1U/13b8efe2c616a25d99de3d2ac4b34e86
    # Mark all scene devices as GPU for cycles
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'GPU'

    eprint("---------------   SCENE LIST   ---------------")
    for scene in bpy.data.scenes:
        eprint(scene.name)
        scene.cycles.device = 'GPU'

    # Enable CUDA
    bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'

    # Enable and list all devices, or optionally disable CPU
    eprint("----------------------------------------------")
    eprint(bpy.context.preferences.addons['cycles'].preferences.get_devices())
    eprint("----------------------------------------------")
    print("----------------------------------------------")
    for devices in bpy.context.preferences.addons['cycles'].preferences.get_devices():
        eprint(devices)
        for d in devices:
            d.use = True
            if d.type == 'CPU':
                d.use = False
            eprint("Device '{}' type {} : {}" . format(d.name, d.type, d.use))
            print("Device '{}' type {} : {}" . format(d.name, d.type, d.use))
    eprint("----------------------------------------------")
    print("----------------------------------------------")

    # RANDOM SEED
    np.random.seed(0)

    # Check if only part of data has to be rerun
    regexp = ".*"
    pattern = re.compile(regexp)

#     model_paths = glob.glob(f'{DATASET_DIR}/*/*/*.obj') # /class/obj_name/model.obj
    
    classes = ['bed', 'bookcase', 'chair', 'desk', 'misc', 'sofa', 'table', 'tool', 'wardrobe']
    CLASS = classes[cls_idx]
    model_paths = glob.glob(f'{DATASET_DIR}/{CLASS}/*/*.obj') # /class/obj_name/model.obj
    for _mi, model_path in enumerate(model_paths):
        model_name = model_path.split('/')[-2]
        model_class = model_path.split('/')[-3]
        
        print(f'{_mi:04d}: {model_name}')
        eprint(f'{_mi:04d}: {model_name}')
        OBJ_PATH = model_path
        
        try:
            OUTPUT_DIR = f'{RESULTS_DIR}/{model_class}/{model_name}'
            if len(glob.glob(osp.join(OUTPUT_DIR,'*.png'))) >= 48:
                if len(glob.glob(osp.join(OUTPUT_DIR,'*.png'))) != 48:
                    print('too many images in', OUTPUT_DIR)
                    import pdb;pdb.set_trace()
                print('already finished - continuing!')
                continue
            render_multiple(
                OBJ_PATH,
                OUTPUT_DIR,
                VIEWS,
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
