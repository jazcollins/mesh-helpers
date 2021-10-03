"""
Run as: srun --gres gpu:1 --qos high2 --pty /home/jazzie/blender-2.82-linux64/blender -b --python blender_abo_lighting.py -- --cls_idx=0 --dset=pix3d

"""
import random
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
import mathutils
from mathutils import Matrix
# Add this folder to path
sys.path.append(osp.dirname(osp.abspath(__file__)))
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--cls_idx', type=int, help='index of class to render')
parser.add_argument('--dset', type=str, help='dataset to render')
argv = sys.argv[sys.argv.index("--") + 1:]
args = parser.parse_args(argv)

# cls_idx = 0
DATASET_DIR =          '/home/jazzie/data/pix3d/model'
RESULTS_DIR =          '/home/jazzie/data/abo/lighting/' # TODO not sure if this write to the right place??
VIEWS =                 12
RESOLUTION =            256 # 512
RENDER_DEPTH =          False # True
RENDER_NORMALS =        False
COLOR_DEPTH =           16
DEPTH_FORMAT =          'OPEN_EXR'
COLOR_FORMAT =          'PNG'
# NORMAL_FORMAT =         'PNG'
NORMAL_FORMAT =         'OPEN_EXR'
CAMERA_FOV_RANGE =      [20, 50] # [20, 50] # [40, 40]
# CAMERA_FOC_RANGE =      [25, 60] # [20, 50] # [40, 40]

if args.dset == 'abo':
    LIGHT_NUM =             8
    LIGHT_ENERGY =          20 
    RAD_MULT =              4
if args.dset == 'pix3d':
    LIGHT_NUM =             6
    LIGHT_ENERGY =          12
    RAD_MULT =              6

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

def import_glb(glb_path) -> bpy.types.Object:
    """
        Import GLB at glb_path, return corresponding mesh object
        Assumes the scene is empty
    """
    status = bpy.ops.import_scene.gltf(filepath=glb_path)
    assert('FINISHED' in status)
    bpy.ops.object.select_all(action='SELECT')
    objects = bpy.context.selected_objects[:]
    obj = [o for o in objects if o.type=='MESH'][0]
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
        scale_normal.inputs[2].default_value = (0.5, 0.5, 0.5, 1)
        links.new(render_layers.outputs['Normal'], scale_normal.inputs[1])

        bias_normal = tree.nodes.new(type='CompositorNodeMixRGB')
        bias_normal.blend_type = 'ADD'
        bias_normal.inputs[2].default_value = (0.5, 0.5, 0.5, 0)
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

    # albedo_file_output = tree.nodes.new(type="CompositorNodeOutputFile")
    # albedo_file_output.label = 'Albedo Output'
    # links.new(render_layers.outputs['Color'], albedo_file_output.inputs[0])
    return depth_file_output, normal_file_output

def add_environment_lighting(scene):
    world = scene.world
    world.use_nodes = True

    enode = world.node_tree.nodes.new('ShaderNodeTexEnvironment')
    enode.image = bpy.data.images.load(ENV_LIGHTING_PATH)

    node_tree = world.node_tree
    node_tree.links.new(enode.outputs['Color'], node_tree.nodes['Background'].inputs['Color'])

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

def render_multiple(obj_path, output_dir, views, resolution, depth=True, normals=True, color_depth=16, data_dict=None):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Clear scene
    utils.clean_objects()

    # Import object
    ext = obj_path.split('.')[-1]
    if ext == 'glb':
        obj_object = import_glb(obj_path)
    elif ext == 'obj':
        obj_object = import_obj(obj_path)
    else:
        assert False, 'unrecognized ext type!'

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
    light_objects = create_random_point_lights(LIGHT_NUM, RAD_MULT*obj_size, energy=LIGHT_ENERGY)
    # Create collection for objects not to render with background
    objs = [ob for ob in scene.objects if ob.type in ('EMPTY') and 'Empty' in ob.name]
    bpy.ops.object.delete({"selected_objects": objs})

    # delete material if it exists
    if args.dset == 'pix3d':
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

    # set up data_dict specific things if relevant
    if data_dict is not None:
        views = 1
        RESOLUTION = data_dict['img_size'][0]
        RESOLUTION = data_dict['img_size'][1]
        # img_names = ['posed_' + data_dict['img'].replace('.','/').split('/')[-2]]
        img_names = [data_dict['img'].replace('.','/').split('/')[-2]]
        trans_4x4 = Matrix.Translation(data_dict['trans_mat'])
        rot_4x4 = Matrix(data_dict['rot_mat']).to_4x4()
        scale_4x4 = Matrix(np.eye(4)) # no scale
        matrix_world = trans_4x4 @ rot_4x4 @ scale_4x4
        poses = [matrix_world]
        focals = [data_dict['focal_length']]
        cam.location = (0, 0, 0)
        cam.rotation_euler = (0, np.pi, 0)
    else:
        RESOLUTION = 512
        RESOLUTION = 512
        img_names = ['r_' + str(i) for i in range(views)]
        stepsize = 360.0 / views
        poses = []
        focals = []
        # translation
        # cam.location =  (0, 0, 1.8 * obj_size/np.tan(cam.data.angle/2))
        for i in range(views):
            # trans_4x4 = Matrix.Translation((0, 0, 1.8*obj_size))
            trans_4x4 = Matrix.Translation((0, 0, 4.8*obj_size))
            rot_4x4 = mathutils.Matrix.Rotation(radians(2*stepsize), 4, 'Y') # @ mathutils.Matrix.Rotation(radians(10.0), 4, 'X')
            scale_4x4 = Matrix(np.eye(4)) # no scale
            matrix_world =  trans_4x4 @ rot_4x4 @ scale_4x4
            poses.append(matrix_world)
            
            # TODO note  i think this was on for pix3d renders
            # cam.location = (0, 0, 0)

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

    # eles = np.random.uniform(-10.0, 60.0, VIEWS)
    # azis = np.random.uniform(0, 360.0, VIEWS)

    for i in range(0, views):
        
        # delete lights and set up new ones
        bpy.ops.object.select_all(action='DESELECT')
        bpy.ops.object.select_by_type(type='LIGHT')
        bpy.ops.object.delete()

        light_objects = create_random_point_lights(LIGHT_NUM, RAD_MULT*obj_size, energy=LIGHT_ENERGY)
        for light in light_objects:
            light.location += b_empty.location

        scene.render.filepath = os.path.join(output_dir, img_names[i])
        
        obj_object.matrix_world = poses[i]
        obj_object.scale = (1, 1, 1)
        
        # Update camera location and angle
        bpy.context.view_layer.update()
       
        if args.dset == 'pix3d':
            if not POSED:
                cam.data.angle = np.random.uniform(CAMERA_FOV_RANGE[0],CAMERA_FOV_RANGE[1]) * np.pi/180
                cam.location =  (0, 0, 1.8 * obj_size/np.tan(cam.data.angle/2))
            else:
                # TODO
                # use foclas!
                pass
        else:
            # cam.data.angle = np.random.uniform(CAMERA_FOV_RANGE[0],CAMERA_FOV_RANGE[1]) * np.pi/180
            cam.data.angle = 20 * np.pi/180
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
            }
        }
        out_data['frames'].append(frame_data)

    with open(output_dir + '/' + 'transforms.json', 'w') as out_file:
        json.dump(out_data, out_file, indent=4)

def reset_and_set_gpu():
    # reset scene
    bpy.ops.wm.read_homefile(use_empty=True)
    
    # Mark all scene devices as GPU for cycles
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'GPU'

    for scene in bpy.data.scenes:
        scene.cycles.device = 'GPU'

    # Enable CUDA
    bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'

    for devices in bpy.context.preferences.addons['cycles'].preferences.get_devices():
        eprint(devices)
        for d in devices:
            d.use = True
            if d.type == 'CPU':
                d.use = False

if __name__ == "__main__":

    # RANDOM SEED
    np.random.seed(0)

    if args.dset == 'pix3d':
        data_list = json.load(open('/home/jazzie/data/pix3d/pix3d.json'))

        classes = ['bed', 'bookcase', 'chair', 'desk', 'misc', 'sofa', 'table', 'tool', 'wardrobe']
        CLASS = classes[args.cls_idx]
        POSED = False 
        model_paths = glob.glob(f'{DATASET_DIR}/{CLASS}/*/*.obj') # /class/obj_name/model.obj
        for _mi, model_path in enumerate(model_paths):
            
            model_name = model_path.split('/')[-2]
            model_class = model_path.split('/')[-3]
            
            if POSED:
                # collect all data dicts that correspond to this certain model
                # TODO
                # for now just collect one
                model_ddicts = []
                for ddict in data_list:
                    if ddict['model'].split('/')[-2] == model_name:
                        model_ddicts.append(ddict)
                data_dict = model_ddicts[0]
                save_dir = os.path.join(RESULTS_DIR, 'posed')
                import pdb;pdb.set_trace()
            else:
                save_dir = os.path.join(RESULTS_DIR, 'renders')
                data_dict = None
            
            print(f'{_mi:04d}: {model_name}')
            OBJ_PATH = model_path


           
            OUTPUT_DIR = f'{save_dir}/{model_class}/{model_name}'
            if len(glob.glob(osp.join(OUTPUT_DIR,'*.exr'))) >= VIEWS:
                print('already finished - continuing!')
                continue
            reset_and_set_gpu()
            render_multiple(
                OBJ_PATH,
                OUTPUT_DIR,
                VIEWS,
                RESOLUTION,
                depth=RENDER_DEPTH,
                normals=RENDER_NORMALS,
                data_dict=data_dict
            )
            import pdb;pdb.set_trace()
    elif args.dset == 'abo':
        model_paths = glob.glob('/home/jazzie/ABO_RELEASE/3dmodels/original/*/*.glb')

        model_paths = [model_path for model_path in model_paths if 'B00EUL2B16' in model_path]
        for _mi, model_path in enumerate(model_paths):
            model_name = model_path.split('/')[-1][0:-4] # remove ext

            save_dir = os.path.join(RESULTS_DIR, 'renders')
            OUTPUT_DIR = '%s/%s'%(save_dir, model_name)
            
            reset_and_set_gpu()
            render_multiple(model_path, OUTPUT_DIR, VIEWS, RESOLUTION, depth=RENDER_DEPTH, normals=RENDER_NORMALS)
            # import pdb;pdb.set_trace()
    else:
        assert False, 'unrecognized dset!'
    
    print('done')
