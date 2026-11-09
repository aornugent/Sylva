import numpy as np
import pytest

from sylva.qsm import rebase

from test_qsm_support_tree import PARAMS, scanned


@pytest.fixture(scope="module")
def scans(tmp_path_factory):
    """An 8 cm trunk at the origin, a 1.2 cm stem 30 cm off it, and a bush of seven 8 mm stems in a 5 cm ring at
    (2, 0), scanned from three sides."""
    a, b, r = [[0, 0, 0], [0.3, 0, 0]], [[0, 0, 2.5], [0.3, 0, 2.2]], [0.08, 0.012]
    for k in range(7):
        t = 2 * np.pi * k / 7
        a.append([2 + 0.05 * np.cos(t), 0.05 * np.sin(t), 0])
        b.append([2 + 0.05 * np.cos(t), 0.05 * np.sin(t), 2.0])
        r.append(0.008)
    return scanned(tmp_path_factory.mktemp("rebase"), np.array(a, float), np.array(b, float), r,
                   [(1, -6.0), (6.5, 3.0), (-5.0, 4.5)], zenith=(75.0, 110.0))


def plant(members):
    """Chains of points (one per member, with its radius) as one plant's rows: a, b, r_prior, parent, member."""
    a, b, r, p, m = [], [], [], [], []
    for k, (pts, rr) in enumerate(members):
        pts, off = np.asarray(pts, float), sum(len(x) for x in a)
        a.append(pts[:-1])
        b.append(pts[1:])
        r.append(np.full(len(pts) - 1, rr))
        p.append(np.r_[-1, np.arange(len(pts) - 2) + off])
        m.append(np.full(len(pts) - 1, k))
    return [np.concatenate(x) for x in (a, b, r, p, m)]


def rows_of(f, k):
    return np.flatnonzero(f["member"] == k)


Z = np.linspace(0, 2.5, 6)
BESIDE = np.c_[np.zeros(6), np.r_[np.full(4, 0.12), 0, 0], Z]  # the trunk drawn 12 cm beside its wood under 1.5 m


def test_a_base_drawn_on_its_wood_is_kept_and_one_beside_it_is_re_sectioned(scans):
    on = plant([(np.c_[np.zeros(6), np.zeros(6), Z], 0.08)])
    f = rebase(scans, *on, [0, 0, 0], params=PARAMS, reach=0.7, threads=2)
    assert not f["columns"]["chosen"].any() and np.array_equal(f["a"], on[0]) and np.array_equal(f["b"], on[1])
    f = rebase(scans, *plant([(BESIDE, 0.06)]), [0, 0, 0], params=PARAMS, reach=0.7, threads=2)
    c = f["columns"]
    assert c["chosen"].sum() == 1 and c["owner"][c["chosen"]][0] == 0 and f["counts"]["joined"] == 1
    # its base is now the column: on the trunk, about its radius, from the ground
    col = f["column"] >= 0
    mid = (f["a"][col] + f["b"][col]) / 2
    assert np.abs(mid[:, :2]).max() < 0.02 and np.abs(np.log(f["r_prior"][col] / 0.08)).max() < 0.1
    assert f["a"][f["parent"] < 0].tolist() == [[pytest.approx(0, abs=0.02), pytest.approx(0, abs=0.02), 0.0]]
    top = f["b"][col, 2].max()  # the column's top, where RCT's trunk is cut: nothing of it is left under there
    assert top > 1.3 and (np.maximum(f["a"][~col, 2], f["b"][~col, 2]) > top - 1e-9).all()


def test_a_fan_through_air_joins_its_column_and_a_thin_stem_on_its_wood_keeps_its_rows(scans):
    root = [0.15, -0.3, 0]  # a root fan: three RCT trees from one point through air onto the trunk
    fan = [([root, [0.1, -0.2, 0.8], [0.02, 0, 1.6], [0, 0, 2.5]], 0.05),
           ([root, [0.0, -0.4, 0.9], [-0.05, -0.05, 1.7], [-0.05, -0.1, 3.5]], 0.05),
           ([root, [-0.1, -0.3, 0.7], [-0.05, 0.08, 1.55], [-0.1, 0.1, 2.3]], 0.03),
           (np.c_[np.full(6, 0.3), np.zeros(6), np.linspace(0, 2.2, 6)], 0.012)]  # the thin stem, drawn on it
    f = rebase(scans, *plant(fan), [0.15, -0.15, 0], params=PARAMS, reach=0.8, threads=2)
    c = f["columns"]
    trunk = np.flatnonzero(c["chosen"] & (c["r"] > 0.05))
    assert len(trunk) == 1 and c["owner"][trunk[0]] == 1  # the biggest crown's
    # the other fan members start at their cuts at the column's top; the thin stem keeps its rows from the ground
    top = f["b"][f["column"] == trunk[0], 2].max()
    for k in (0, 2):
        assert f["a"][rows_of(f, k)][f["parent"][rows_of(f, k)] < 0][0, 2] == pytest.approx(top)
    thin = rows_of(f, 3)
    assert len(thin) == 5 and (f["column"][thin] < 0).all() and f["a"][thin[0], 2] == 0
    for i in range(len(f["a"])):  # every member hangs from one base, parents first
        assert f["parent"][i] < i and (f["parent"][i] < 0 or f["member"][f["parent"][i]] == f["member"][i])


def test_a_bush_of_thin_stems_read_as_one_column_keeps_its_stems(scans):
    pts = [(np.c_[np.full(6, 2 + 0.05 * np.cos(t)), np.full(6, 0.05 * np.sin(t)), np.linspace(0, 2.0, 6)], 0.012)
           for t in 2 * np.pi * np.arange(7) / 7]
    rows = plant(pts)
    f = rebase(scans, *rows, [2, 0, 0], params=PARAMS, reach=0.7, threads=2)
    assert not f["columns"]["chosen"].any() and f["counts"]["kept"] == 7
    assert np.array_equal(f["a"], rows[0]) and np.array_equal(f["member"], rows[4])
