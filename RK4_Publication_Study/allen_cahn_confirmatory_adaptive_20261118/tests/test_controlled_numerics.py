"""CPU checks of the actual frozen training-loss and update paths."""
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys

import jax.numpy as jnp
import numpy as np
import pytest

SOURCE = Path(__file__).resolve().parents[1] / 'source_snapshot'
sys.path.insert(0, str(SOURCE))
spec = importlib.util.spec_from_file_location('controlled_ac_solver', SOURCE/'GPT-code-invloss.py')
solver = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = solver
spec.loader.exec_module(solver)
import controlled_runner as runner


class ScalarTrajectory:
    def apply(self, params, coordinates):
        return params['a'] + params['b'] * coordinates[..., 0]


def fixture(integrator='rk4', normalize=False, method='fd_rk4', clip=0.):
    cfg = replace(solver.Config(), temporal_integrator=integrator,
                  normalize_rk4_defect=normalize, method=method,
                  hard_initial_condition=False, rk4_substeps=1,
                  residual_weight=1., rk4_weight=.03 if method=='fd_rk4' else 0.,
                  initial_weight=1., boundary_weight=0., previous_weight=0.,
                  data_weight=0., adaptive_weights=False, gradient_clip_norm=clip)
    params = {'network':{'a':jnp.asarray(.2),'b':jnp.asarray(.1)},
              'physics_raw':jnp.log(jnp.asarray([5.,5.,1e-4]))}
    x = jnp.linspace(-1.,1.,8,endpoint=False)
    initial = jnp.full((8,),.2)
    api = solver.make_update(ScalarTrajectory(),cfg,x,initial,jnp.asarray(0.),params)
    args = (jnp.asarray([[0.,.02]]),jnp.asarray(0.),initial,
            jnp.empty((0,)),jnp.empty((0,8)),jnp.zeros((1,)),jnp.zeros((1,)),jnp.zeros((1,)))
    return cfg,params,api,args


def step(u,dt,kind):
    rhs = lambda v:-5*v**3+5*v
    k1 = rhs(u)
    if kind == 'heun':
        return u+dt*(k1+rhs(u+dt*k1))/2
    k2=rhs(u+dt*k1/2); k3=rhs(u+dt*k2/2); k4=rhs(u+dt*k3)
    return u+dt*(k1+2*k2+2*k3+k4)/6


@pytest.mark.parametrize('kind',['rk4','heun'])
def test_actual_temporal_loss_matches_whole_state_stages(kind):
    cfg,params,api,args=fixture(kind)
    terms,_=api[1](params,*args)
    expected=(.202-step(.2,.02,kind))**2
    assert np.isclose(float(terms[1]),expected,rtol=5e-5,atol=1e-10)


def test_rate_normalization_changes_loss_by_duration_squared():
    _,params,raw,args=fixture(normalize=False)
    _,_,rate,_=fixture(normalize=True)
    raw_terms,_=raw[1](params,*args);rate_terms,_=rate[1](params,*args)
    assert np.isclose(float(rate_terms[1]/raw_terms[1]),1/.02**2,rtol=2e-5)


def test_fd_path_skips_temporal_computation(monkeypatch):
    def forbidden(*args):
        raise AssertionError('Disabled temporal penalty evaluated')
    monkeypatch.setattr(solver,'normalized_rk4_mse',forbidden)
    _,params,api,args=fixture(normalize=True,method='fd')
    terms,_=api[1](params,*args)
    assert float(terms[1]) == 0.
    _,rms,_,_=api[2](params,*args)
    assert float(rms[1]) == 0.


def test_update_reports_clipping_and_rejects_nonfinite_proposal():
    cfg,params,api,args=fixture(clip=1e-4)
    state=api[6].init(params)
    result=api[0](params,state,*args,jnp.asarray([1.,.03,1.,0.,0.,0.]))
    norm,clipped,finite=result[-1]
    assert float(norm)>cfg.gradient_clip_norm and bool(clipped) and bool(finite)
    broken={**params,'network':{'a':jnp.asarray(1e20),'b':jnp.asarray(.1)}}
    result=api[0](broken,api[6].init(broken),*args,jnp.asarray([1.,.03,1.,0.,0.,0.]))
    assert not bool(result[-1][2])


def test_temporal_gradient_guard_caps_ratio_and_preserves_budget():
    cfg = replace(
        solver.Config(), adaptive_max_temporal_gradient_ratio=3.0,
        adaptive_min_multiplier=0.01, adaptive_max_multiplier=100.0,
    )
    weights = np.asarray([0.5, 1.5, 100.0, 1.0, 1.0, 0.0])
    base = np.asarray([1.0, 1.0, 100.0, 1.0, 1.0, 0.0])
    gradient_rms = np.asarray([1.0, 10.0, 1.0, 1.0, 1.0, 0.0])
    guarded, record = runner.apply_temporal_gradient_guard(
        weights, base, gradient_rms, cfg)
    ratio = guarded[1] * gradient_rms[1] / (guarded[0] * gradient_rms[0])
    assert record['triggered']
    assert np.isclose(ratio, 3.0)
    assert np.isclose(guarded[0] + guarded[1], weights[0] + weights[1])
    assert np.allclose(guarded[2:], weights[2:])


def test_temporal_gradient_guard_is_inactive_at_large_sentinel():
    cfg = replace(solver.Config(), adaptive_max_temporal_gradient_ratio=1.0e30)
    weights = np.asarray([0.5, 1.5, 100.0, 1.0, 1.0, 0.0])
    guarded, record = runner.apply_temporal_gradient_guard(
        weights, weights, np.ones_like(weights), cfg)
    assert not record['enabled']
    assert not record['triggered']
    assert np.array_equal(guarded, weights)


def test_output_guard_preserves_existing_artifact(tmp_path):
    out=tmp_path/'old';out.mkdir();sentinel=out/'keep.txt';sentinel.write_text('keep')
    cfg=replace(solver.Config(),output_dir=str(out),checkpoint_dir=str(tmp_path/'checkpoints'))
    with pytest.raises(FileExistsError):
        runner.reserve_paths(cfg)
    assert sentinel.read_text()=='keep'


def test_initial_only_guard_rejects_future_truth(tmp_path):
    import scipy.io
    path=tmp_path/'input.mat'
    x=np.linspace(-1,1,9);t=np.asarray([0.,.8,1.2]);u=np.zeros((9,3));u[:,0]=x*x*np.cos(np.pi*x)
    cfg=replace(solver.Config(),data_path=str(path),spatial_points=8,train_end_time=.8,forecast_end_time=1.2)
    payload={'x':x,'t':t,'usol':u,'noninitial_values_are_placeholders':1}
    scipy.io.savemat(path,payload)
    runner.validate_initial_only_input(path,cfg)
    u[4,1]=.1;scipy.io.savemat(path,payload)
    with pytest.raises(ValueError,match='initial-only'):
        runner.validate_initial_only_input(path,cfg)
