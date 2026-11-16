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
            sc.add(bad)
    for call in (lambda: sc.remove([7]), lambda: sc.change(7, pole()), lambda: sc.predict(pole(), 3),
                 lambda: sc.information([7]), lambda: sc.remove([0, 0])):
        with pytest.raises(ValueError):
            call()


def test_an_empty_scene_costs_nothing_and_a_pole_is_pinned_down():
    v = pole_image()
    assert _core.Scene([v], dict(a=np.zeros((0, 3)), b=np.zeros((0, 3)), r=np.zeros(0))).cost() == 0.0
    sc = _core.Scene([v], pole())
    rows = sc.rows()
    parts = sum(rows[k] for k in ('ray_on', 'ray_empty', 'ray_in_front', 'ray_behind', 'class', 'alignment'))
    assert np.allclose(parts, rows['cost']) and rows['cost'][0] < 0
    info = sc.information([0])
    assert info['hessian'].shape == (1, 5, 5) and np.isfinite(info['hessian']).all() and np.diag(info['hessian'][0]).max() > 0
    assert info['expected'].shape == (1, 5, 5) and np.linalg.eigvalsh(info['expected'][0]).min() > -1e-6 * np.abs(info['expected'][0]).max()
    j = info['gradient_variance'][0]
    assert j.shape == (5, 5) and np.linalg.eigvalsh(j).min() > -1e-6 * np.abs(j).max() and j[2, 2] < info['expected'][0][2, 2]
    assert 0 < rows['arc'][0] <= np.pi
    assert np.isclose(sc.costs_to_add([pole(r=0.02)]).sum(), sc.costs_to_add([pole(r=0.02)])[0, 0])



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
    """Moves hold the alignment: `exact_cost` re-solves it without changing the scene, `refit_alignment` makes `cost`
    the exact one, `costs_to_remove` equals removing, and `interactions` names hypotheses sharing cells and nodes."""
    v = pole_image()
    sc = _core.Scene([v], pole())
    assert sc.cost() == sc.exact_cost()
    ids, _ = sc.add(pole(r=0.02))
    held = sc.cost()
    assert np.isclose(held, sc.exact_cost(), rtol=0.01, atol=0.05) and sc.cost() == held
    sc.refit_alignment()
    assert sc.cost() == sc.exact_cost()
    out = sc.costs_to_remove(ids)
    assert out.shape == (1, 1) and np.isclose(out[0, 0], sc.remove(ids))
    pairs = sc.interactions([pole(), pole(r=0.02)])
    assert list(pairs['i']) == [0] and list(pairs['j']) == [1] and pairs['cells'][0] and pairs['nodes'][0]


def test_expected_information_alone_and_free_space_cross_the_binding():
    """Research 130 step 3c: `information` without the observed stencil gives the same expected information and NaN
    for the observed part; the image gives each cell's free space (its first echo less the receiver's resolution, the
    longest range for a miss) under the settings."""
    v = pole_image()
    sc = _core.Scene([v], pole())
    full, alone = sc.information([0]), sc.information([0], observed=False)
    assert np.allclose(full['expected'], alone['expected']) and np.isnan(alone['hessian']).all() and np.isnan(alone['gradient']).all()
    d = _core.settings_defaults()
    A = v.arrays(d)
    echo = A['cell'] == 0
    reach = -d['res_a'] / d['res_b']
    assert np.allclose(A['free_to'][echo], np.maximum(A['range'][echo] - reach, 0), atol=1e-4)
    assert (A['free_to'][A['cell'] == 1] == 80.0).all()


def test_a_chain_crosses_the_binding():
    """Research 130 slice 6: a `group` column makes elements one term (one row, its group named), and `coupling` is
    pure geometry beside the cost."""
    v = pole_image()
    halves = dict(a=np.array([[0.0, 4.0, -0.1], [0.0, 4.0, 0.0]]), b=np.array([[0.0, 4.0, 0.0], [0.0, 4.0, 0.1]]), r=np.array([0.03, 0.03]))
    rows = _core.Scene([v], dict(halves, group=np.array([4, 4]))).rows()
    assert list(rows['group']) == [4] and list(rows['elem']) == [0]
    assert list(_core.Scene([v], halves).rows()['group']) == [-1, -1]
    c = _core.coupling(halves)
    assert np.isclose(c['length'], 0.2) and c['gap'] == 0.0 and np.isfinite(c['radius']) and np.isfinite(c['bend'])
