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
                 lambda: sc.information(7), lambda: sc.remove([0, 0])):
        with pytest.raises(ValueError):
            call()


def test_an_empty_scene_costs_nothing_and_a_pole_is_pinned_down():
    v = pole_image()
    assert _core.Scene([v], dict(a=np.zeros((0, 3)), b=np.zeros((0, 3)), r=np.zeros(0))).cost() == 0.0
    sc = _core.Scene([v], pole())
    rows = sc.rows()
    parts = sum(rows[k] for k in ('ray_on', 'ray_empty', 'ray_in_front', 'ray_behind', 'class', 'alignment'))
    assert np.allclose(parts, rows['cost']) and rows['cost'][0] < 0
    info = sc.information(0)
    assert info['hessian'].shape == (1, 5, 5) and np.isfinite(info['hessian']).all() and np.diag(info['hessian'][0]).max() > 0
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
