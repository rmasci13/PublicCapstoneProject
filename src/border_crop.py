"""Removes the black field-of-view border that many dermoscopy images have
(circular vignette, or plain black bars/margins) so the model can't use it as a
cue. Drop it in as the FIRST step of both the train and eval transforms.

    from border_crop import CropBlackBorder
    transforms.Compose([CropBlackBorder(), transforms.Resize(size), transforms.CenterCrop(size), ...])

Pure PIL/numpy, so it also works outside PyTorch (e.g. for the Grad-CAM script
or for checking how many images have a border with has_border()).
"""

import numpy as np


class CropBlackBorder:
    """Crops away black borders; images without a border are returned unchanged.

    1. Crop to the bounding box of the non-dark pixels (removes black bars/margins).
    2. If the corners are still dark (circular vignette), shrink to the largest
       centred square whose four corners are all clean skin. For a full circular
       field of view that is the inscribed square (~64% of the circle's area).
    If the crop would throw away most of the image, the original is returned.
    """

    def __init__(self, dark_thresh=25, corner_frac=0.08, clean_frac=0.98,
                 margin=0.03, min_keep=0.4):
        self.dark_thresh = dark_thresh  # gray level below which a pixel counts as "black border"
        self.corner_frac = corner_frac  # corner patch size, as a fraction of the shorter side
        self.clean_frac = clean_frac    # share of each corner patch that must be non-dark
        self.margin = margin            # extra shrink so the anti-aliased circle rim is excluded
        self.min_keep = min_keep        # never return a crop smaller than this share of the shorter side

    def _bright(self, img):
        return np.asarray(img.convert("L")) > self.dark_thresh

    def _corners_clean(self, mask):
        h, w = mask.shape
        k = max(2, int(round(self.corner_frac * min(h, w))))
        corners = (mask[:k, :k], mask[:k, -k:], mask[-k:, :k], mask[-k:, -k:])
        return min(c.mean() for c in corners) >= self.clean_frac

    def has_border(self, img):
        """True if the image has a black border/vignette (dark corners)."""
        return not self._corners_clean(self._bright(img))

    def __call__(self, img):
        mask = self._bright(img)
        if self._corners_clean(mask):
            return img  # no border

        # 1. bounding box of the bright content
        rows = np.where(mask.mean(axis=1) > 0.05)[0]
        cols = np.where(mask.mean(axis=0) > 0.05)[0]
        if len(rows) == 0 or len(cols) == 0:
            return img
        y0, y1, x0, x1 = int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
        box = (x0, y0, x1, y1)

        # 2. circular vignette: largest centred square with clean corners
        sub_mask = mask[y0:y1, x0:x1]
        if not self._corners_clean(sub_mask):
            ch, cw = sub_mask.shape
            cy, cx = ch // 2, cw // 2
            max_half = min(ch, cw) // 2
            for half in np.linspace(max_half, 0.2 * max_half, 60):
                half = int(half)
                if half < 2:
                    break
                if self._corners_clean(sub_mask[cy - half:cy + half, cx - half:cx + half]):
                    half = int(half * (1 - self.margin))
                    box = (x0 + cx - half, y0 + cy - half, x0 + cx + half, y0 + cy + half)
                    break
            else:
                return img  # no clean square found; leave the image alone

        W, H = img.size
        if (box[2] - box[0]) < self.min_keep * min(W, H) or (box[3] - box[1]) < self.min_keep * min(W, H):
            return img
        return img.crop(box)

    def __repr__(self):
        return f"{self.__class__.__name__}(dark_thresh={self.dark_thresh}, margin={self.margin})"