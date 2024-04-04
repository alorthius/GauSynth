import os

import numpy as np
import cv2
import torch

from external.swin2sr.models.network_swin2sr import Swin2SR as net


THIS_FILE_DIR = os.path.dirname(os.path.realpath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(THIS_FILE_DIR, ".."))


class SwinWrapper:

    def __init__(self, weights):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = net(
            upscale=4, in_chans=3, img_size=64, window_size=8,
            img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2, upsampler='nearest+conv', resi_connection='1conv'
        )
        param_key_g = 'params_ema'
        pretrained_model = torch.load(weights)
        self.model.load_state_dict(
            pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model,
            strict=True
        )

        self.model.eval()
        self.model = self.model.to(self.device)

        self.border = 0
        self.window_size = 8

        self.tile = 512
        self.tile_overlap = 32
        self.scale = 4

    def to_device(self, device):
        self.device = device
        self.model = self.model.to(self.device)

    def inference(self, image, tile_size=512, tile_overlap=32, keep_dim=True):
        self.tile = tile_size
        self.tile_overlap = tile_overlap

        img_lq = image.astype(np.float32) / 255.
        if img_lq.shape[2] == 4:
            img_lq = img_lq[:, :, :3]

        img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))  # HCW-BGR to CHW-RGB
        img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(self.device)  # CHW-RGB to NCHW-RGB

        with torch.no_grad():
            # pad input image to be a multiple of window_size
            _, _, h_old, w_old = img_lq.size()
            h_pad = (h_old // self.window_size + 1) * self.window_size - h_old
            w_pad = (w_old // self.window_size + 1) * self.window_size - w_old
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]

            if image.shape[0] * image.shape[1] <= 512 * 512:
                output = self.model(img_lq)
                print("Swin2SR Tile 1/1")
            else:  # tiles
                b, c, h, w = img_lq.size()
                tile = min(self.tile, h, w)

                stride = tile - self.tile_overlap
                h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
                w_idx_list = list(range(0, w - tile, stride)) + [w - tile]
                E = torch.zeros(b, c, h * self.scale, w * self.scale).type_as(img_lq)
                W = torch.zeros_like(E)

                tiles = len(h_idx_list) * len(w_idx_list)
                t = 1
                for h_idx in h_idx_list:
                    for w_idx in w_idx_list:
                        print(f"Swin2SR Tile {t}/{tiles}")
                        t += 1
                        in_patch = img_lq[..., h_idx:h_idx + tile, w_idx:w_idx + tile]
                        out_patch = self.model(in_patch)
                        out_patch_mask = torch.ones_like(out_patch)
                        h_tile_min, w_tile_min = h_idx * self.scale, w_idx * self.scale
                        h_tile_max, w_tile_max = (h_idx + tile) * self.scale, (w_idx + tile) * self.scale
                        E[..., h_tile_min:h_tile_max, w_tile_min:w_tile_max].add_(out_patch)
                        W[..., h_tile_min:h_tile_max, w_tile_min:w_tile_max].add_(out_patch_mask)
                output = E.div_(W)

            output = output[..., :h_old * 4, :w_old * 4]
            output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            if output.ndim == 3:
                output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
            output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8

            if keep_dim:
                output = cv2.resize(output, (w_old, h_old))  # resize back to original resolution
            return output


if __name__ == "__main__":
    from pathlib import Path
    dir_path = Path(PROJECT_DIR) / "superresolution"
    paths = {
        "weights": dir_path / "weights" / "Swin2SR_RealworldSR_X4_64_BSRGAN_PSNR.pth",
    }
    swin = SwinWrapper(**paths)
    image = cv2.imread("textsr/test_image.png")
    swin_upscaled = swin.inference(image)
    cv2.imwrite("textsr/test_res.png", swin_upscaled)
    print("Ok.")
