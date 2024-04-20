import os


def split_video(fps, vid_path, output_path, resolution):
    os.system(f"ffmpeg -i {vid_path} -vf fps={fps},scale={resolution}x{resolution} -start_number 0 {output_path}/%5d.png")


def form_video(fps, frames_path, output_vid):
    os.system(f"ffmpeg -y -framerate {fps} -pattern_type glob -i '{frames_path}/*.png' -c:v libx264 -pix_fmt yuv420p -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2' {output_vid}")


def form_colmap_video(fps, frames_path, output_vid):
    os.system(f"ffmpeg -y -framerate {fps} -pattern_type glob -i '{frames_path}/*.png' -c:v libx264 -pix_fmt yuv420p -vf 'pad=ceil(iw/2)*2:ceil(ih/2)*2'  -filter:v 'vflip' {output_vid}")


def calc_new_fps(orig_fps, orig_frames, new_frames):
    new_fps = int(orig_fps * (new_frames / orig_frames))
    return new_fps


if __name__ == "__main__":
    form_colmap_video(30, "data/r", "data/colmap.mp4")
