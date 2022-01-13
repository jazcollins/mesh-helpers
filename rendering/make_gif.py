import glob
import os
import subprocess

data_dir = '/home/jazzie/code/object-triplets/data/ShapeNet.v1/renders'
save_dir = '/home/jazzie/code/object-triplets/data/ShapeNet.v1/gifs'

model_dirs = glob.glob(os.path.join(data_dir, '*', '*'))

for model_dir in model_dirs:
    cls = model_dir.split('/')[-2]
    model_id = model_dir.split('/')[-1]
    model_save_dir = os.path.join(save_dir, cls)
    os.makedirs(model_save_dir, exist_ok=True)
   
    mp4_save_name = os.path.join(model_save_dir, '%s_%s.mp4'%(cls, model_id))
    movie_cmd = 'ffmpeg -i %s -filter_complex "[0]split=2[bg][fg];[bg]drawbox=c=white@1:replace=1:t=fill[bg];[bg][fg]overlay=format=auto" -c:a copy %s'%(os.path.join(model_dir,'r_%03d.png'), mp4_save_name)

    os.system(movie_cmd)
#     process = subprocess.Popen(movie_cmd.split(), stdout=subprocess.PIPE)
#     output, error = process.communicate()

    gif_save_name = os.path.join(model_save_dir, '%s_%s.gif'%(cls, model_id))
    gif_cmd = 'ffmpeg -ss 0 -t 4.2 -i %s -vf "fps=16,scale=320:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" -loop 0 %s'%(mp4_save_name, gif_save_name)
    os.system(gif_cmd)
    # process = subprocess.Popen(movie_cmd.split(), stdout=subprocess.PIPE)
    # output, error = process.communicate()
    os.system('rm %s'%mp4_save_name)
