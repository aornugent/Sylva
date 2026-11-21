"""The scene bindings (`_core.Scene`, `_core.RangeImage`, `_core.settings_defaults`): settings dicts, bad input
raising, an empty scene costing nothing, and a pole's rows and information."""
import numpy as np
import pytest

from sylva import _core


def pole_image():
    """A small scan of a vertical pole (R 3 cm, 4 m away on +y) in front of a wall 8 m away, first echoes only."""
    line, k = np.meshgrid(np.arange(200), np.arange(400), indexing='ij')
    zen, az = np.radians(84.0 + 0.03 * k.ravel()), np.radians(-3.0 + 0.03 * line.ravel())
    w = np.stack([np.sin(zen) * np.sin(az), np.sin(zen) * np.cos(az), np.cos(zen)], 1)
    h = np.hypot(w[:, 0], w[:, 1])
    miss = 4.0 * w[:, 0] / h
    hit = np.abs(miss) < 0.03
    r = np.where(hit, (4.0 * w[:, 1] - np.sqrt(np.clip(0.03 ** 2 - miss ** 2, 0, None))) / h, 8.0 / w[:, 1])
    t = line.ravel() * 6.6e-3 + k.ravel() * 0.825e-6
    return _core.RangeImage(t, np.ascontiguousarray(w), r.astype(np.float32), np.eye(3), (0.0, 0.0, 0.0), 0)


def pole(r=0.03, **kw):
    return dict(a=np.array([[0.0, 4.0, -0.1]]), b=np.array([[0.0, 4.0, 0.1]]), r=np.array([r]), **kw)


def test_a_saved_image_loads_as_the_same_image(tmp_path):
    v = pole_image()
    v.save(tmp_path / 'pole')
    w = _core.RangeImage.load(tmp_path / 'pole')
    a, b = v.arrays(), w.arrays()
    assert w.shape == v.shape and w.step == v.step and a.keys() == b.keys()
    assert all(np.array_equal(a[k], b[k], equal_nan=a[k].dtype.kind == 'f') for k in a)
    assert _core.Scene([w], pole()).cost() == _core.Scene([v], pole()).cost()
    (tmp_path / 'other').write_bytes(b'not an image')
    with pytest.raises(ValueError):
        _core.RangeImage.load(tmp_path / 'other')


def test_settings_round_trip_and_unknown_keys_raise():
    d = _core.settings_defaults()
    assert d['discount'] > 0 and 'beam_exit' in d and 'refl_leaf' in d
    v = pole_image()
    _core.Scene([v], pole(), d)
    with pytest.raises(ValueError):
        _core.Scene([v], pole(), {'dispersion': 4.0})
    with pytest.raises(ValueError):
        _core.Scene([v], pole(misfit=np.array([0.002])))


def test_bad_input_raises_instead_of_panicking():
    v = pole_image()
    sc = _core.Scene([v], pole())
    for bad in (pole(r=0.0), pole(r=-1.0), pole(opacity=np.array([1.5])),
                dict(a=np.zeros((1, 3)), b=np.zeros((1, 3)), r=np.array([0.03]))):
        with pytest.raises(ValueError):
            sc.keep(bad)
        with pytest.raises(ValueError):
            sc.score([bad])
    for call in (lambda: sc.keep(take_out=[7]), lambda: sc.keep(pole(), [7]), lambda: sc.predict(pole(), 3),
                 lambda: sc.information([7]), lambda: sc.keep(take_out=[0, 0]), lambda: sc.score([pole()], [[7]]),
                 lambda: sc.information([0], take_out=[0]), lambda: sc.rows(take_out=[3]), lambda: sc.score([pole()], [[], []])):
        with pytest.raises(ValueError):
            call()


def test_an_empty_scene_costs_nothing_and_a_pole_is_pinned_down():
    v = pole_image()
    assert _core.Scene([v], dict(a=np.zeros((0, 3)), b=np.zeros((0, 3)), r=np.zeros(0))).cost() == 0.0
    sc = _core.Scene([v], pole())
    rows = sc.rows()
    parts = sum(rows[k] for k in ('ray_on', 'ray_empty', 'ray_in_front', 'ray_behind', 'alignment'))
    assert np.allclose(parts, rows['cost']) and rows['cost'][0] < 0
    info = sc.information([0])
    assert info['slope'].shape == (1, 5) and np.isfinite(info['slope']).all()
    assert info['expected'].shape == (1, 5, 5) and np.linalg.eigvalsh(info['expected'][0]).min() > -1e-6 * np.abs(info['expected'][0]).max()
    assert np.diag(info['expected'][0]).min() > 0
    j = info['gradient_variance'][0]
    assert j.shape == (5, 5) and np.linalg.eigvalsh(j).min() > -1e-6 * np.abs(j).max() and j[2, 2] < info['expected'][0][2, 2]
    assert 0 < rows['arc'][0] <= np.pi
    assert np.isclose(sc.score([pole(r=0.02)]).sum(), sc.score([pole(r=0.02)])[0, 0])


def test_reads_for_a_proposal_are_the_reads_after_keeping_it():
    """Research 130 slice-s1: rows and information for a proposal (put in, or swapped for a live element), nothing
    kept, equal the rows and information after keeping it; its elements are numbered from `next_id()`, and keys the
    caller gives ride along with them."""
    v = pole_image()
    sc = _core.Scene([v], pole())
    wide = pole(r=0.033, key=np.array([12]))
    me = sc.next_id()
    rows, info = sc.rows(wide, [0]), sc.information([me], wide, [0])
    cost = sc.score([wide], [[0]]).sum()
    assert sc.next_id() == me and np.isclose(sc.cost(), _core.Scene([v], pole()).cost(), rtol=0, atol=0)
    ids, d = sc.keep(wide, [0])
    assert list(ids) == [me] and d == cost and list(sc.elements(ids)['key']) == [12]
    after = sc.rows()
    assert all(np.array_equal(rows[k], after[k]) for k in after)
    assert np.array_equal(info['expected'], sc.information([me])['expected'])



def test_located_edges_cross_the_binding():
    """Rows carry the reflectance level, predictions the echo's predicted and observed reflectance, and the image its
    first echoes' reflectance (NaN here: the image has none)."""
    v = pole_image()
    sc = _core.Scene([v], pole())
    rows, p = sc.rows(), sc.predict(pole(), 0)
    assert {'refl_level', 'refl_level_sd'} <= set(rows) and rows['refl_level_sd'][0] == 0.0
    assert {'refl', 'cos', 'obs_refl'} <= set(p) and np.isfinite(p['refl'][p['fill'] > 0.5]).all() and np.isnan(p['obs_refl']).all()
    assert np.isnan(v.arrays()['refl']).all()


def test_the_held_alignment_crosses_the_binding():
    """Moves hold the alignment: a move's cost is near the exact one, and `refit_alignment` re-solves it, after which
    the cost is a fresh scene's of the same elements."""
    v = pole_image()
    sc = _core.Scene([v], pole())
    sc.keep(pole(r=0.02))
    both = {k: np.concatenate([pole()[k], pole(r=0.02)[k]]) for k in pole()}
    fresh = _core.Scene([v], both).cost()
    assert np.isclose(sc.cost(), fresh, rtol=0.01, atol=0.05)
    sc.refit_alignment()
    assert np.isclose(sc.cost(), fresh, rtol=1e-12)


def test_a_fisher_step_and_the_clear_range_cross_the_binding():
    """Research 130 slice-derivatives: `information`'s slope and expected information make a Fisher-scoring step that
    moves a stick too wide toward the pole's radius; the image gives each cell's free space (its first echo less the
    receiver's resolution, the longest range for a miss) under the settings."""
    v = pole_image()
    sc = _core.Scene([v], pole(r=0.036))
    info = sc.information([0])
    step = -np.linalg.solve(info['expected'].sum(0) + np.eye(5), info['slope'].sum(0))
    assert np.log(0.03 / 0.036) < step[0] < 0
    d = _core.settings_defaults()
    A = v.arrays(d)
    echo = A['cell'] == 0
    reach = -d['res_a'] / d['res_b']
    assert np.allclose(A['free_to'][echo], np.maximum(A['range'][echo] - reach, 0), atol=1e-4)
    assert (A['free_to'][A['cell'] == 1] == 80.0).all()


def test_the_depth_pyramid_crosses_the_binding():
    """Research 130 slice-pyramid: per voxel, off the image's depth pyramid, the pulses covering it, those clear past it
    and those ending in it: in front of the pole all are clear, behind it some are stopped; bad boxes raise."""
    v = pole_image()
    fs = v.carve([-0.1, 1.0, -0.1], [0.1, 8.6, 0.1], 0.2)
    assert all(fs[k].shape == (1, 38, 1) for k in ('pulses', 'clear', 'ending'))
    n, c, e = (fs[k][0, :, 0] for k in ('pulses', 'clear', 'ending'))
    assert (c <= n).all() and (e <= n).all()
    assert n[5] > 0 and c[5] == n[5]
    assert c[20] < n[20]
    with pytest.raises(ValueError):
        v.carve([0.0, 0.0, 0.0], [0.0, 1.0, 1.0], 0.2)


def test_a_chain_crosses_the_binding():
    """Research 130 slice 6: a `group` column makes elements one term (one row, its group named)."""
    v = pole_image()
    halves = dict(a=np.array([[0.0, 4.0, -0.1], [0.0, 4.0, 0.0]]), b=np.array([[0.0, 4.0, 0.0], [0.0, 4.0, 0.1]]), r=np.array([0.03, 0.03]))
    rows = _core.Scene([v], dict(halves, group=np.array([4, 4]))).rows()
    assert list(rows['group']) == [4] and list(rows['elem']) == [0]
    assert list(_core.Scene([v], halves).rows()['group']) == [-1, -1]


def test_the_grower_and_the_coupling_cross_the_binding():
    """Research 130 slice-grower: a seed on the pole grows into a chain along it inside the region a Python test
    gives, read exactly; the coupling row of its sticks is pure geometry."""
    v = pole_image()
    seed = dict(a=np.array([[0.0, 4.0, -0.02]]), b=np.array([[0.0, 4.0, 0.02]]), r=np.array([0.03]))
    (g,) = _core.Scene([v], dict(a=np.zeros((0, 3)), b=np.zeros((0, 3)), r=np.zeros(0))).grow(seed, lambda p: np.abs(p[:, 2]) < 0.2)
    assert g['sticks'].shape[1] == 7 and g['length'] > 0.2 and g['cost'] < g['seed_cost'] < 0
    assert np.isclose(g['scan'].sum(), g['cost']) and np.allclose(g['sticks'][:, 1], 4.0)
    row = _core.coupling(g['sticks'])
    assert np.isclose(row['length'], g['length']) and row['gap'] < 1e-9


def test_the_collapsed_field_crosses_the_binding():
    """With a foliage field (its prior the settings'), `score` carries the field's part last and sums to keeping it,
    the field's counts for a proposal are those after keeping it, the rows carry a `field` part beside the cost, and
    each row's parts (the plug-in of the hits it takes in its ray parts) sum to its cost."""
    v = pole_image()
    box = dict(lo=[-0.5, 2.9, -0.3], hi=[0.5, 8.9, 0.3], voxel=0.1)
    side = dict(a=np.array([[0.2, 4.0, -0.1]]), b=np.array([[0.2, 4.0, 0.1]]), r=np.array([0.02]))
    sc = _core.Scene([v], side, None, None, box)
    c = sc.score([pole()])
    assert c.shape == (1, 2) and c[0, 1] != 0.0
    counts = sc.foliage_voxels(pole())
    _, d = sc.keep(pole())
    assert c.sum() == d and all(np.array_equal(counts[k], sc.foliage_voxels()[k]) for k in counts)
    rows = sc.rows()
    assert rows['field'].shape == rows['cost'].shape
    parts = sum(rows[k] for k in ('ray_on', 'ray_empty', 'ray_in_front', 'ray_behind', 'alignment'))
    assert np.allclose(parts, rows['cost'])


def test_the_rules_consumers_read_beside_the_cost_cross_the_binding():
    """The footprint, the detection floor, a fill with its slope in ln r and the field's marginal per voxel, the
    scene's own rules, under the settings given (the defaults without)."""
    from math import lgamma, log
    s = _core.settings_defaults()
    t = np.array([4.0, 30.0])
    assert np.allclose(_core.footprint(t), s['beam_exit'] + s['beam_divergence'] * t)
    assert np.allclose(_core.floor(t, dict(floor_db=-40.0)), -40.0 + s['floor_slope'] * np.log10(np.maximum(t, s['floor_ref']) / s['floor_ref']))
    f, g = _core.beam_fill(np.array([0.0, 0.03, 0.1]), 0.03, np.full(3, 0.004))
    assert np.allclose(f, [1.0, 0.5, 0.0]) and g[0] == 0.0 and g[1] > 0.0
    h = 1e-6
    up, _ = _core.beam_fill(np.array([0.03]), 0.03 * np.exp(h), np.array([0.004]))
    assert abs((up[0] - f[1]) / h - g[1]) < 1e-4 * g[1]
    a0, b0 = 0.05 * 0.4, 0.4
    nb = _core.field_ln_nb(np.array([2.0, 0.0]), np.array([3.0, 0.0]), 0.05, 0.4)
    assert np.allclose(nb, [lgamma(a0 + 2) - lgamma(a0) + a0 * log(b0) - (a0 + 2) * log(b0 + 3), 0.0])
    from math import erfc, sqrt
    floor = _core.floor(np.array([4.0]))[0]
    p = _core.p_detect(np.array([-15.0]), np.array([0.5]), np.array([4.0]))
    assert np.isclose(p[0], 0.5 * erfc(-((-15.0 + 10 * np.log10(0.5) - floor) / s['detect_sd']) / sqrt(2)))
    assert np.allclose(_core.mismatch_sd(np.array([0.01, 0.1])), s['mismatch_base'] + s['mismatch_slope'] * np.array([0.01, 0.1]))
    keys, w = _core.alignment_hats(np.array([[0.3, 4.1, -0.7], [2.0, 2.0, 2.0]]))
    assert keys.shape == (2, 8, 3) and np.allclose((w ** 2).sum(1), 1.0) and (w[1] > 0).sum() == 1


def test_a_chain_kept_together():
    """Research 130 slice-consolidate: `keep` of a list is one move, the cost as keeping one by one, with a field."""
    v = pole_image()
    box = dict(lo=[-0.5, 2.9, -0.3], hi=[0.5, 8.9, 0.3], voxel=0.1)
    halves = dict(a=np.array([[0.0, 4.0, -0.1], [0.0, 4.0, 0.0]]), b=np.array([[0.0, 4.0, 0.0], [0.0, 4.0, 0.1]]), r=np.array([0.03, 0.03]),
                  group=np.array([4, 4]))
    one, many = _core.Scene([v], pole(), None, None, box), _core.Scene([v], pole(), None, None, box)
    for k in range(2):
        one.keep({key: x[k:k + 1] for key, x in halves.items()})
    ids, _ = many.keep(halves)
    assert list(ids) == [1, 2] and one.cost() == many.cost()
