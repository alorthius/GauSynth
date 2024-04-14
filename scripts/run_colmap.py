import os
import sys
sys.path.append("colmap/scripts/python/")

from colmap.scripts.python.visualize_model import Model


class ColmapScreenCapturer(Model):
    def __init__(self, input_path):
        super().__init__()

        self.read_model(input_path, ext=".bin")

        print("num_cameras:", len(self.cameras))
        print("num_images:", len(self.images))
        print("num_points3D:", len(self.points3D))

        self.create_window()
        self.add_points()
        self.add_cameras(scale=0.25)

    def flip(self):
        params = self._Model__vis.get_view_control().convert_to_pinhole_camera_parameters()
        params.extrinsic[1, :] = -params.extrinsic[1, :]  # flip
        self._Model__vis.get_view_control().convert_from_pinhole_camera_parameters(params)

    def capture_multiple_views(self, output_dir, num_views=50):
        ctr = self._Model__vis.get_view_control()
        ctr.set_zoom(0.5)

        for i in range(num_views):
            ctr.rotate(4 * 360 / num_views, 0.0)  # rotate the view horizontally
            # self.flip()
            self._Model__vis.update_renderer()
            self._Model__vis.poll_events()
            self._Model__vis.capture_screen_image(f"{output_dir}/{str(i).zfill(2)}.png")

        self._Model__vis.destroy_window()


def visualize_colmap(input_path, output_path):
    c = ColmapScreenCapturer(input_path)
    c.capture_multiple_views(output_path)
    return (len(l) for l in [c.cameras, c.images, c.points3D])


def run_colmap(workdir, images):
    os.system(f"colmap automatic_reconstructor --workspace_path {workdir} --image_path {images} --data_type video --quality extreme --single_camera 1 --dense 0")
