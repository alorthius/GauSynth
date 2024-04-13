import os


def split_video(fps, vid_path, output_path, resolution):
    os.system(f"ffmpeg -i {vid_path} -vf fps={fps},scale={resolution}x{resolution} -start_number 0 {output_path}/%2d.png")


def form_video(fps, frames_path, output_vid):
    os.system(f"ffmpeg -framerate {fps} -pattern_type glob -i '{frames_path}/*.png' -c:v libx264 -pix_fmt yuv420p -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' {output_vid}")
