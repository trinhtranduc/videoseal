# Tiled embedding and detection with a short, repeated code.
#
# The stock models resize every image to 256x256 and pool the message over the whole
# image, so a small crop of a large image cannot be read back. Here the same message is
# embedded into every tile at native resolution, and detection slides windows over the
# (possibly cropped or resized) image and adds up the evidence.
#
# The models carry a fixed number of bits (256 for PixelSeal). A short code (e.g. 48 bits)
# is repeated to fill them, which gives error correction for free when reading.

import torch

SEED = 20261119


def _perm(nbits_model: int) -> torch.Tensor:
    """Fixed interleaving so the repeats of one code bit are spread over the message."""
    gen = torch.Generator().manual_seed(SEED)
    return torch.randperm(nbits_model, generator=gen)


def expand_code(code: torch.Tensor, nbits_model: int = 256) -> torch.Tensor:
    """code: 1 x K tensor of 0/1. Returns the 1 x nbits_model message that repeats it."""
    k = code.shape[-1]
    msg = torch.empty(1, nbits_model)
    msg[0, _perm(nbits_model)] = code[0, torch.arange(nbits_model) % k].float()
    return msg


def collapse_logits(logits: torch.Tensor, k: int) -> torch.Tensor:
    """logits: nbits_model values (positive = bit 1). Returns K summed soft bits."""
    n = logits.shape[-1]
    soft = torch.zeros(k)
    soft.index_add_(0, torch.arange(n) % k, logits[_perm(n)].float().cpu())
    return soft


def _starts(size: int, tile: int, stride: int) -> list:
    if size <= tile:
        return [0]
    starts = list(range(0, size - tile + 1, stride))
    if starts[-1] + tile < size:
        starts.append(size - tile)  # last tile is aligned to the far edge
    return starts


@torch.no_grad()
def embed_tiled(model, img: torch.Tensor, msg: torch.Tensor, tile: int = 256, batch: int = 16) -> torch.Tensor:
    """img: 1 x C x H x W in [0, 1]. Embeds `msg` (1 x nbits) into every tile. Returns the marked image."""
    _, _, h, w = img.shape
    th, tw = min(tile, h), min(tile, w)
    boxes = [(y, x) for y in _starts(h, th, th) for x in _starts(w, tw, tw)]
    out = img.clone()
    for i in range(0, len(boxes), batch):
        part = boxes[i:i + batch]
        tiles = torch.stack([img[0, :, y:y + th, x:x + tw] for y, x in part])
        marked = model.embed(tiles, msgs=msg.repeat(len(part), 1), is_video=False)["imgs_w"]
        for (y, x), m in zip(part, marked):
            out[0, :, y:y + th, x:x + tw] = m
    return out


@torch.no_grad()
def detect_tiled(model, img: torch.Tensor, k: int,
                 windows=(96, 128, 160, 200, 256, 320, 400, 512, 640, 800, 1024),
                 max_windows: int = 48, batch: int = 32) -> dict:
    """Slide windows of several sizes over img (1 x C x H x W) and add up the message logits.

    The leaked copy may have been resized, so the size of a tile in it is unknown. The window
    size that matches the scale of the embedded tiles gives consistent evidence; the others
    give noise. Returns the K soft bits of the most self-consistent window size.
    At most `max_windows` evenly spread windows are read per size.
    """
    _, _, h, w = img.shape
    best = None
    for win in windows:
        wh, ww = min(win, h), min(win, w)
        boxes = [(y, x) for y in _starts(h, wh, max(1, wh // 2)) for x in _starts(w, ww, max(1, ww // 2))]
        if len(boxes) > max_windows:
            picks = torch.linspace(0, len(boxes) - 1, max_windows).round().long().tolist()
            boxes = [boxes[i] for i in picks]
        total = None
        for i in range(0, len(boxes), batch):
            crops = torch.stack([img[0, :, y:y + wh, x:x + ww] for y, x in boxes[i:i + batch]])
            logits = model.detect(crops, is_video=False)["preds"][:, 1:].float().cpu().sum(dim=0)
            total = logits if total is None else total + logits
        soft = collapse_logits(total, k)
        # Noise adds up like sqrt(n); a real mark adds up like n.
        score = float(soft.abs().mean() / (len(boxes) ** 0.5))
        if best is None or score > best["score"]:
            best = {"soft": soft, "bits": (soft > 0).float(), "score": score, "window": win, "n_windows": len(boxes)}
        if (wh, ww) == (h, w):
            break  # larger windows would be the same whole-image crop
    return best


@torch.no_grad()
def embed_dual(model, img: torch.Tensor, msg: torch.Tensor, tile: int = 512, batch: int = 16) -> torch.Tensor:
    """Two layers carrying the same message: one over the whole image and one per tile.

    The whole-image layer is coarse (the model works at 256x256 and upsamples), so it
    survives noise and re-photographing; the tiled layer is fine and survives cropping.
    Both residuals are computed from the clean image and added.
    """
    whole = model.embed(img, msgs=msg, is_video=False)["imgs_w"] - img
    tiled = embed_tiled(model, img, msg, tile=tile, batch=batch) - img
    return (img + whole + tiled).clamp(0, 1)


@torch.no_grad()
def detect_dual(model, img: torch.Tensor, k: int, **kwargs) -> list:
    """Reads both layers. Returns two candidates (whole image, tiled); the caller matches
    each against its registry and must count them as two tries in its significance test."""
    soft = collapse_logits(model.detect(img, is_video=False)["preds"][0, 1:], k)
    whole = {"soft": soft, "bits": (soft > 0).float(), "score": float(soft.abs().mean()), "window": 0, "n_windows": 1}
    return [whole, detect_tiled(model, img, k, **kwargs)]
