import itertools

import numpy as np
import cv2
import torch

from swin2sr.models.network_swin2sr import Swin2SR as net


swin_init_params = {
    "upscale": 4,
    "in_chans": 3,
    "img_size": 64,
    "window_size": 8,
    "img_range": 1.,
    "depths": [6, 6, 6, 6, 6, 6],
    "embed_dim": 180,
    "num_heads": [6, 6, 6, 6, 6, 6],
    "mlp_ratio": 2,
    "upsampler": "nearest+conv",
    "resi_connection": "1conv",
}


class Swin2SrUpscaler:
    def __init__(self, weights: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # load model
        self.model = net(**swin_init_params)

        pretrained_model = torch.load(weights)
        self.model.load_state_dict(
            pretrained_model["params_ema"] if "params_ema" in pretrained_model.keys() else pretrained_model,
            strict=True
        )
        self.model.eval()
        self.model = self.model.to(self.device)

        self.window_size = 8
        self.tile = 512
        self.tile_overlap = 32
        self.scale = 4

    def _pad(self, dim):
        pad = (dim // self.window_size + 1) * self.window_size - dim
        return pad

    def _process_tiles(self, img_lq, h_orig, w_orig):
        if h_orig * w_orig <= 512 * 512:  # single tile
            print("Swin2SR Tile 1/1")
            return self.model(img_lq)

        # tile processing
        b, c, h, w = img_lq.size()
        tile = min(self.tile, h, w)
        stride = tile - self.tile_overlap

        h_idx_list = list(range(0, h - tile, stride)) + [h - tile]
        w_idx_list = list(range(0, w - tile, stride)) + [w - tile]

        E = torch.zeros(b, c, h * self.scale, w * self.scale).type_as(img_lq)
        W = torch.zeros_like(E)

        for t, (h_i, w_i) in enumerate(itertools.product(h_idx_list, w_idx_list), start=1):
            print(f"Swin2SR Tile {t}/{len(h_idx_list) * len(w_idx_list)}")

            in_patch = img_lq[..., h_i:h_i + tile, w_i:w_i + tile]
            out_patch = self.model(in_patch)
            out_patch_mask = torch.ones_like(out_patch)

            E[..., h_i * self.scale:(h_i + tile) * self.scale,
            w_i * self.scale:(w_i + tile) * self.scale] += out_patch

            W[..., h_i * self.scale:(h_i + tile) * self.scale,
            w_i * self.scale:(w_i + tile) * self.scale] += out_patch_mask

        return E.div_(W)

    def inference(self, image: np.array, keep_dim=True) -> np.array:
        img_lq = image.astype(np.float32) / 255.
        if img_lq.shape[2] == 4:
            img_lq = img_lq[:, :, :3]

        # convert HCW-BGR to BCHW-RGB
        img_lq = np.transpose(img_lq[:, :, [2, 1, 0]], (2, 0, 1))
        img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            _, _, h_orig, w_orig = img_lq.size()
            h_pad, w_pad = self._pad(h_orig), self._pad(w_orig)

            # pad input image to be a multiple of window_size
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_orig + h_pad, :]
            img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_orig + w_pad]

            output = self._process_tiles(img_lq, h_orig, w_orig)

            # post-process output
            output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
            output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
            output = (output * 255.0).round().astype(np.uint8)
            output = output[0:self.scale * h_orig, 0:self.scale * w_orig, :]

            if keep_dim:  # resize back to original resolution
                output = cv2.resize(output, (w_orig, h_orig))
            return output


if __name__ == "__main__":
    weights = "checkpoints/Swin2SR_RealworldSR_X4_64_BSRGAN_PSNR.pth"
    image = cv2.imread("test_input.png")

    swin2sr = Swin2SrUpscaler(weights)
    image = swin2sr.inference(image, keep_dim=False)

    cv2.imwrite("test_swin2sr.png", image)
