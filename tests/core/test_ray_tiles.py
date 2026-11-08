"""Ray tiles name every ray's echo exactly: scan.u16, pulse.u32 and echo.u8 join each cloud row to its Shots echo."""

import numpy as np
import pytest

from sylva.io import tile_ray_clouds
from sylva.shots import Shots


def _scan(rng, origin, n):
    d = rng.normal(size=(n, 3))
    d[:, 2] = np.abs(d[:, 2]) * 0.3
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    count = rng.integers(0, 4, n)  # misses and multi-echo pulses
    start = np.concatenate([[0], np.cumsum(count)[:-1]])
    rng_ = np.concatenate([np.sort(rng.uniform(1.0, 9.0, c)) for c in count])
    refl = rng.uniform(-25, -5, len(rng_)).astype(np.float32)
    return Shots(np.tile(origin, (n, 1)), d, start, count, rng_, {"reflectance": refl})


def _cloud(path):
    raw = path.read_bytes()
    head = raw.index(b"end_header\n") + len(b"end_header\n")
    dt = np.dtype([("x", "<f8"), ("y", "<f8"), ("z", "<f8"), ("time", "<f8"), ("nx", "<f4"), ("ny", "<f4"),
                   ("nz", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1"), ("a", "u1")])
    return np.frombuffer(raw[head:], dt)


@pytest.mark.parametrize("voxel", [0.0, 0.5])
def test_sidecars_name_each_rays_echo(tmp_path, voxel):
    rng = np.random.default_rng(3)
    scans = [_scan(rng, [0.0, 0.0, 1.5], 400), _scan(rng, [4.0, -3.0, 1.5], 300)]
    files = []
    for i, s in enumerate(scans):
        f = tmp_path / f"scan{i}.parquet"
        s.save(f, double=True, row_group_size=7)  # pulse rows must run on across row groups
        files.append(f)
    tiles, _ = tile_ray_clouds(files, tmp_path / "out", width=50.0, buffer=1.0, voxel=voxel, terrain_voxel=0.5,
                               min_rays=1, threads=1)
    loaded = [Shots.load(f) for f in files]
    checked = 0
    for t in tiles:
        if not t["written"]:
            continue
        d = tmp_path / "out" / "tiles" / t["tile"]
        cloud = _cloud(d / "cloud.ply")
        scan = np.fromfile(d / "scan.u16", np.uint16)
        pulse = np.fromfile(d / "pulse.u32", np.uint32)
        echo = np.fromfile(d / "echo.u8", np.uint8)
        assert len(scan) == len(pulse) == len(echo) == len(cloud)
        for k in range(len(cloud)):
            s = loaded[scan[k]]
            assert echo[k] < s.echo_count[pulse[k]]
            e = s.echo_start[pulse[k]] + echo[k]
            xyz = s.origin[pulse[k]] + s.direction[pulse[k]] * s.echo_range[e]
            np.testing.assert_allclose([cloud["x"][k], cloud["y"][k], cloud["z"][k]], xyz, atol=1e-9)
        checked += len(cloud)
    n_echoes = sum(s.n_echoes for s in scans)
    assert checked == n_echoes if voxel == 0 else 0 < checked < n_echoes
