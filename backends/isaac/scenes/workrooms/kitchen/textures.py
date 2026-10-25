"""Deterministic original oak albedo; generated only by a frozen asset builder.

The raster carries fine fibre detail independently of mesh tessellation. It is
an appearance map, not measured wood or a change to physical contact material.
"""
from pathlib import Path

import numpy as np
from PIL import Image


def oak_albedo(size=1024):
    """Return a seamless, seeded RGB texture covering 400 by 280 mm of wood."""
    rng = np.random.default_rng(20261019)
    y, x = np.mgrid[:size, :size]/float(size)

    def noise(nx, ny, u=x, v=y):
        grid = rng.uniform(-1., 1., (ny, nx))
        sx, sy = (u % 1.)*nx, (v % 1.)*ny
        ix, iy = np.floor(sx).astype(int), np.floor(sy).astype(int)
        tx, ty = sx-ix, sy-iy
        tx, ty = tx*tx*(3-2*tx), ty*ty*(3-2*ty)
        low = grid[iy % ny, ix % nx]*(1-tx)+grid[iy % ny, (ix+1) % nx]*tx
        high = grid[(iy+1) % ny, ix % nx]*(1-tx)+grid[(iy+1) % ny, (ix+1) % nx]*tx
        return low*(1-ty)+high*ty

    # Distorted growth bands and long fibres. Every warp is periodic so UV
    # repeats have no abrupt boundary, including rounded tray walls.
    warp = .008*np.sin(2*np.pi*x)+.004*np.sin(4*np.pi*x+2*np.sin(2*np.pi*y))
    warp += .007*noise(4, 5)
    grain_y = y+warp
    broad = noise(3, 15, x, grain_y)
    fibre = noise(6, 180, x, grain_y)
    fine = noise(32, 440, x, grain_y)
    pores = np.maximum(noise(56, 330, x, grain_y)-.55, 0.)
    narrow_rings = np.maximum(noise(3, 75, x, grain_y)-.30, 0.)
    variation = 9*broad+4.5*fibre+1.6*fine-25*pores-13*narrow_rings
    cloud = 3*noise(8, 7)
    rgb = np.array([183., 140., 94.])[None, None, :]
    rgb = rgb+variation[..., None]*np.array([1., .88, .66])+cloud[..., None]
    return np.uint8(np.clip(np.rint(rgb), 0, 255))


def write_task_textures(output):
    """Write into a fresh versioned asset output and return relative references.

    The caller owns frozen-source verification and immutable publication. No
    files are written by the USD author itself, including in-memory tests.
    """
    output = Path(output)
    relative = 'textures/oak_albedo.png'
    target = output/relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError('Kitchen texture already exists: '+str(target))
    Image.fromarray(oak_albedo(), mode='RGB').save(target, optimize=False, compress_level=9)
    return {'wood_albedo': relative}
