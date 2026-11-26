"""Matched fixed-update Allen--Cahn runner; imports only the frozen solver API.

Inputs contain exact initial values and zero placeholders at every later time.
No reference error is evaluated here. All checkpoints retain the final iterate.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import jax
import jax.numpy as jnp
import numpy as np
import scipy.io


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree_sha256(tree):
    digest = hashlib.sha256()
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        value = np.ascontiguousarray(jax.device_get(leaf))
        digest.update(str(path).encode())
        digest.update(str(value.dtype).encode())
        digest.update(json.dumps(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def array_sha256(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def apply_temporal_gradient_guard(weights, base, gradient_rms, cfg):
    """Cap RK4 gradient dominance while preserving the physics-group budget.

    The grouped controller allocates a fixed budget between the PDE residual
    and the rate-normalised RK4 defect.  This guard only acts when the current
    weighted RK4-to-residual network-gradient ratio exceeds the predeclared
    cap.  Any removed RK4 coefficient is returned to the residual term.
    """
    weights = np.asarray(weights, dtype=np.float64).copy()
    base = np.asarray(base, dtype=np.float64)
    gradient_rms = np.asarray(gradient_rms, dtype=np.float64)
    cap = float(cfg.adaptive_max_temporal_gradient_ratio)
    enabled = bool(np.isfinite(cap) and cap < 1.0e20)
    record = {
        'enabled': enabled, 'triggered': False,
        'cap': cap if enabled else None,
        'ratio_before': None, 'ratio_after': None,
    }
    if not enabled:
        return weights, record
    residual_index, rk4_index = 0, 1
    residual_gradient = gradient_rms[residual_index]
    rk4_gradient = gradient_rms[rk4_index]
    epsilon = float(cfg.adaptive_gradient_epsilon)
    if (not np.isfinite(residual_gradient) or not np.isfinite(rk4_gradient)
            or residual_gradient <= epsilon or rk4_gradient <= epsilon
            or weights[residual_index] <= 0.0 or weights[rk4_index] <= 0.0):
        return weights, record
    ratio_before = float(
        weights[rk4_index] * rk4_gradient
        / (weights[residual_index] * residual_gradient)
    )
    record['ratio_before'] = ratio_before
    if ratio_before <= cap:
        record['ratio_after'] = ratio_before
        return weights, record

    budget = float(weights[residual_index] + weights[rk4_index])
    target_rk4 = budget * cap * residual_gradient / (
        rk4_gradient + cap * residual_gradient
    )
    lower_rk4 = max(
        base[rk4_index] * cfg.adaptive_min_multiplier,
        budget - base[residual_index] * cfg.adaptive_max_multiplier,
    )
    upper_rk4 = min(
        base[rk4_index] * cfg.adaptive_max_multiplier,
        budget - base[residual_index] * cfg.adaptive_min_multiplier,
    )
    target_rk4 = float(np.clip(target_rk4, lower_rk4, upper_rk4))
    weights[rk4_index] = target_rk4
    weights[residual_index] = budget - target_rk4
    ratio_after = float(
        weights[rk4_index] * rk4_gradient
        / (weights[residual_index] * residual_gradient)
    )
    record.update(
        triggered=True,
        ratio_after=ratio_after,
        weights_after={
            'residual': float(weights[residual_index]),
            'rk4': float(weights[rk4_index]),
        },
    )
    return weights, record


def validate_initial_only_input(path, cfg):
    """Validate inclusive periodic coordinates before creating any output."""
    mat = scipy.io.loadmat(path)
    x = np.asarray(mat['x'], dtype=np.float64).ravel()
    t = np.asarray(mat['t'], dtype=np.float64).ravel()
    values = np.asarray(mat['usol'], dtype=np.float64)
    if values.shape == (t.size, x.size):
        values = values.T
    if values.shape != (x.size, t.size):
        raise ValueError('Expected usol shape (space,time) or (time,space)')
    if x.size < 4 or t.size < 3 or not all(np.all(np.isfinite(v)) for v in (x,t,values)):
        raise ValueError('Finite, nonempty spatial/time arrays are required')
    if np.any(np.diff(x) <= 0) or np.any(np.diff(t) <= 0):
        raise ValueError('Spatial and temporal coordinates must be strictly increasing')
    if not np.allclose(np.diff(x), np.diff(x)[0], rtol=1e-6, atol=1e-10):
        raise ValueError('A uniform periodic reference-coordinate grid is required')
    if not np.isclose(x[0], -1.) or not np.isclose(x[-1], 1.):
        raise ValueError('This source snapshot requires inclusive periodic endpoints [-1,1]')
    if not np.isclose(values[0,0], values[-1,0], atol=1e-7):
        raise ValueError('The two duplicate periodic endpoints must have equal initial values')
    if t[0] != 0 or not np.any(np.isclose(t, cfg.train_end_time, atol=1e-8, rtol=0)):
        raise ValueError('The training endpoint must be an existing time-grid coordinate, starting at 0')
    if not np.any(np.isclose(t, cfg.forecast_end_time, atol=1e-8, rtol=0)):
        raise ValueError('The forecast endpoint must be an existing time-grid coordinate')
    if not (t[0] < cfg.train_end_time < cfg.forecast_end_time <= t[-1] + 1e-8):
        raise ValueError('Invalid visible/forecast intervals')
    if cfg.spatial_points < 4 or cfg.spatial_points > x.size-1:
        raise ValueError('spatial_points must be >=4 and <= the number of unique input-grid nodes')
    if np.any(values[:,1:] != 0):
        raise ValueError('Controlled trainer accepts initial-only data; every later field must be zero')
    if int(np.asarray(mat.get('noninitial_values_are_placeholders', [[0]])).squeeze()) != 1:
        raise ValueError('Input must explicitly mark noninitial_values_are_placeholders=1')
    # Later entries are placeholders, not truths, and are discarded here.
    return x, t[t <= cfg.forecast_end_time + 1e-8], values[:,0]


def reserve_paths(cfg):
    out, checkpoints = Path(cfg.output_dir).resolve(), Path(cfg.checkpoint_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Refusing to overwrite nonempty output: {out}; resume into a fresh directory')
    if checkpoints.exists() and any(checkpoints.iterdir()):
        raise FileExistsError(f'Refusing to overwrite checkpoints: {checkpoints}')
    if out == checkpoints:
        raise ValueError('Output and checkpoint directories must be distinct')
    out.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    # Exclusive creation also prevents two workers claiming the same empty output.
    with (out / 'run.lock').open('x') as stream:
        stream.write(f'pid={os.getpid()}\n')
    return out, checkpoints


def run(cfg, solver):
    started = time.perf_counter()
    x, t, initial_profile = validate_initial_only_input(cfg.data_path, cfg)
    out, checkpoint_dir = reserve_paths(cfg)
    diagnostics = {
        'schema_version': 1, 'status': 'running', 'config': cfg.__dict__,
        'protocol': 'fixed_updates_dynamic_weights_final_iterate_v2',
        'checkpoint_selection': 'final_iterate',
        'reference_errors_eligible': False, 'reference_values_used': 'initial_only',
        'physics_engagement': ('adaptive after a fixed warmup' if cfg.adaptive_weights
                               else 'all enabled coefficients constant from update 1'),
        'temporal_active': cfg.method == 'fd_rk4' and cfg.rk4_weight > 0,
        'defect_normalization': 'rate' if cfg.normalize_rk4_defect else 'raw',
        'gradient_clipping_threshold': cfg.gradient_clip_norm,
        'data_sha256': file_sha256(cfg.data_path),
        'source_sha256': {p.name: file_sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
        'environment': {'python': platform.python_version(), 'jax': jax.__version__,
                        'devices': [str(d) for d in jax.devices()]},
        'completed_updates': 0, 'nonfinite_updates': 0, 'clipped_gradient_updates': 0,
        'block_records': [], 'gradient_diagnostics': [], 'adaptive_updates': [],
        'safeguard_event_count': 0, 'resume_from': cfg.resume_checkpoint,
    }
    atomic_json(out/'diagnostics.json', diagnostics)
    try:
        result = _train(cfg, solver, x, t, initial_profile, out, checkpoint_dir, diagnostics)
        np.savez_compressed(out/'predictions.npz', **result)
        # Compatibility name used by the historical family of launchers.
        os.link(out/'predictions.npz', out/'allen_cahn_fd_rk4_pinn.npz')
        diagnostics['status'] = 'complete'
        diagnostics['wall_seconds_this_invocation'] = time.perf_counter() - started
        diagnostics['prediction_sha256'] = file_sha256(out/'predictions.npz')
        memory = [d.memory_stats() for d in jax.devices()]
        diagnostics['device_memory_stats_after_run'] = memory
        import resource
        diagnostics['process_peak_rss_kib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        atomic_json(out/'diagnostics.json', diagnostics)
        atomic_json(out/'controlled_run.json', diagnostics)
        atomic_json(out/'completion.json', {
            'status': 'complete', 'completed_updates': diagnostics['completed_updates'],
            'expected_updates': cfg.slabs*cfg.epochs,
            'initial_parameters_sha256': diagnostics['initial_parameters_sha256'],
            'pair_schedule_sha256': diagnostics['pair_schedule_sha256'],
            'prediction_sha256': diagnostics['prediction_sha256'],
            'reference_errors_eligible': False,
        })
        print(json.dumps({'status':'complete','output':str(out),
                          'completed_updates':diagnostics['completed_updates']}))
    except BaseException as error:
        diagnostics['status'] = 'failed'
        diagnostics['error'] = f'{type(error).__name__}: {error}'
        diagnostics['wall_seconds_this_invocation'] = time.perf_counter() - started
        atomic_json(out/'diagnostics.json', diagnostics)
        atomic_json(out/'controlled_run.json', diagnostics)
        raise


def _train(cfg, solver, x, t, initial_profile, out, checkpoint_dir, diagnostics):
    train_stop = int(np.flatnonzero(np.isclose(t,cfg.train_end_time,atol=1e-8,rtol=0))[0])
    t_train = t[:train_stop+1]
    edges = np.linspace(0, train_stop, cfg.slabs+1, dtype=int)
    if np.any(np.diff(edges) <= 0):
        raise ValueError('Every block needs at least one distinct adjacent time pair')
    x_grid_np = np.linspace(-1,1,cfg.spatial_points,endpoint=False,dtype=np.float32)
    dx = 2./cfg.spatial_points
    if not np.allclose(np.diff(x_grid_np), dx, rtol=2e-5, atol=1e-7):
        raise ValueError('Training FD grid spacing validation failed')
    x_grid = jnp.asarray(x_grid_np)
    initial_exact = jnp.asarray(np.interp(x_grid_np,x,initial_profile),dtype=jnp.float32)
    model = solver.build_model(cfg,(float(t_train[0]),-1.),(float(t_train[-1]),1.))
    params = {'network': model.init(jax.random.PRNGKey(cfg.seed),jnp.zeros((1,2),dtype=jnp.float32)),
              'physics_raw': jnp.log(jnp.asarray((cfg.initial_alpha,cfg.initial_beta,cfg.initial_nu),dtype=jnp.float32))}
    diagnostics['initial_parameters_sha256'] = tree_sha256(params)
    network_count = sum(np.asarray(leaf).size for leaf in jax.tree_util.tree_leaves(params['network']))
    diagnostics['network_parameter_count'] = int(network_count)
    diagnostics['grid'] = {'space_nodes':cfg.spatial_points,'dx':dx,'period':2.,
                           'periodic_duplicate_excluded':True,'training_endpoint':float(t_train[-1])}
    # Precompute the complete schedule before any optimisation. All methods use
    # these identical pairs; diagnostics do not consume the random stream.
    block_seeds = np.random.SeedSequence(cfg.seed).spawn(cfg.slabs)
    diagnostics['sampling_scheme'] = 'independent_per_block_SeedSequence_spawn'
    schedules, schedule_indices = [], []
    for block in range(cfg.slabs):
        rng = np.random.default_rng(block_seeds[block])
        block_t = t_train[edges[block]:edges[block+1]+1]
        indices = rng.integers(0,block_t.size-1,size=(cfg.epochs,cfg.pairs_per_step))
        schedule_indices.append(indices.astype(np.int32))
        schedules.append(np.stack((block_t[:-1][indices],np.diff(block_t)[indices]),axis=-1).astype(np.float32))
    schedule = np.stack(schedules)
    diagnostics['pair_schedule_sha256'] = array_sha256(schedule)
    diagnostics['sampled_pairs_sha256'] = diagnostics['pair_schedule_sha256']
    diagnostics['block_pair_schedule_sha256'] = [array_sha256(v) for v in schedule]
    np.savez_compressed(out/'pair_schedule.npz',pairs=schedule,indices=np.stack(schedule_indices),block_edges=edges)
    update,evaluate,gradient_stats,grid_prediction,net,coefficients,optimizer,_ = solver.make_update(
        model,cfg,x_grid,initial_exact,jnp.asarray(t_train[0],dtype=jnp.float32),params)
    opt_state = optimizer.init(params)
    past_t = jnp.asarray(t_train[:1],dtype=jnp.float32)
    past_values = initial_exact[None,:]
    history, parameter_history, weight_history, adaptive_trace = [],[],[],[]
    start_block = 0
    if cfg.resume_checkpoint:
        restored = solver.load_checkpoint(cfg.resume_checkpoint,cfg)
        audit = restored.get('controlled_state')
        if not audit:
            raise ValueError('Cannot resume a historical best-checkpoint run as controlled final-iterate training')
        for key in ('initial_parameters_sha256','pair_schedule_sha256','data_sha256','source_sha256'):
            if audit[key] != diagnostics[key]:
                raise ValueError(f'Resume provenance mismatch: {key}')
        params,opt_state = restored['params'],restored['opt_state']
        past_t,past_values = jnp.asarray(restored['past_t']),jnp.asarray(restored['past_values'])
        history = list(restored['history']); parameter_history = list(restored['parameter_history'])
        weight_history = list(restored['weight_history'])
        adaptive_trace = list(np.asarray(restored.get('adaptive_trace', [])))
        start_block = int(restored['next_slab'])
        for key in ('block_records','gradient_diagnostics','adaptive_updates',
                    'safeguard_event_count','completed_updates','clipped_gradient_updates'):
            diagnostics[key] = audit.get(key, diagnostics[key])
        if not 0 <= start_block <= cfg.slabs:
            raise ValueError('Invalid next block in resume checkpoint')
    atomic_json(out/'diagnostics.json',diagnostics)
    with (out/'update_diagnostics.csv').open('x',newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['block','update_in_block','global_update','loss_before_update',
                         'gradient_global_norm_before_clip','gradient_clipped','finite_update',
                         'update_wall_seconds'])
        for block in range(start_block,cfg.slabs):
            block_started = time.perf_counter()
            block_t = t_train[edges[block]:edges[block+1]+1]
            initial_values = initial_exact if block == 0 else past_values[-1]
            replay_t,replay_values = solver.select_replay_history(past_t,past_values,cfg.deduplicate_replay)
            base_weights = np.asarray(solver.base_loss_weights(cfg,block),dtype=np.float64)
            adaptive_state = solver.initialize_adaptive_state(base_weights)
            weights = adaptive_state['weights'].copy()
            warmup = (cfg.adaptive_first_slab_warmup if block == 0
                      else cfg.adaptive_later_slab_warmup)
            indices = np.linspace(0,block_t.size-2,min(cfg.pairs_per_step,block_t.size-1),dtype=int)
            monitor_pairs = jnp.asarray(np.column_stack((block_t[:-1][indices],np.diff(block_t)[indices])),dtype=jnp.float32)
            # Fixed forward-only dummy observation arrays never contain truth.
            observations = (jnp.zeros((1,),dtype=jnp.float32),)*3
            args = (jnp.asarray(block_t[0],dtype=jnp.float32),initial_values,replay_t,replay_values,*observations)
            entry_before = np.asarray(grid_prediction(params,args[0])) - np.asarray(initial_values)
            block_record = {'block':block+1,'time_interval':[float(block_t[0]),float(block_t[-1])],
                            'updates':cfg.epochs,'entry_grid_rms_before':float(np.sqrt(np.mean(entry_before**2))),
                            'configured_weights':weights.tolist(),'clipped_gradient_updates':0,
                            'adaptive_updates':0,'safeguard_events':0,
                            'max_gradient_global_norm':0.,'update_wall_seconds':0.,
                            'gradient_diagnostic_wall_seconds':0.,'monitor_wall_seconds':0.}
            def diagnose(completed):
                tick = time.perf_counter()
                terms,rms,physics_rms,cosine = gradient_stats(params,monitor_pairs,*args)
                terms,rms,physics_rms = (np.asarray(v,dtype=float) for v in (terms,rms,physics_rms))
                cosine = float(cosine)
                if not all(np.all(np.isfinite(v)) for v in (terms,rms,physics_rms,cosine)):
                    raise FloatingPointError(f'Nonfinite gradient diagnostic in block {block+1}, update {completed}')
                norms = rms*np.sqrt(network_count)
                weighted = norms*weights
                ratio = float(weighted[1]/weighted[0]) if weighted[0]>0 else None
                diagnostics['gradient_diagnostics'].append({
                    'block':block+1,'completed_updates_in_block':completed,
                    'losses':dict(zip(solver.LOSS_NAMES,terms.tolist())),
                    'weights':dict(zip(solver.LOSS_NAMES,weights.astype(float).tolist())),
                    'network_gradient_rms':dict(zip(solver.LOSS_NAMES,rms.tolist())),
                    'network_gradient_norm':dict(zip(solver.LOSS_NAMES,norms.tolist())),
                    'weighted_network_gradient_norm':dict(zip(solver.LOSS_NAMES,weighted.tolist())),
                    'weighted_temporal_to_pde_gradient_ratio':ratio,
                    'temporal_pde_gradient_cosine':cosine if diagnostics['temporal_active'] else None,
                    'temporal_active':diagnostics['temporal_active']})
                block_record['gradient_diagnostic_wall_seconds'] += time.perf_counter()-tick
            diagnose(0)
            for epoch in range(cfg.epochs):
                if (cfg.adaptive_weights and epoch >= warmup
                        and (epoch - warmup) % cfg.adaptive_update_every == 0):
                    sampled_terms, sampled_rms, physics_rms, cosine = gradient_stats(
                        params, monitor_pairs, *args)
                    sampled_terms = np.asarray(sampled_terms, dtype=np.float64)
                    sampled_rms = np.asarray(sampled_rms, dtype=np.float64)
                    physics_rms = np.asarray(physics_rms, dtype=np.float64)
                    cosine = float(np.asarray(cosine))
                    if not all(np.all(np.isfinite(value)) for value in
                               (sampled_terms, sampled_rms, physics_rms, cosine)):
                        raise FloatingPointError(
                            f'Nonfinite adaptive statistic in block {block+1}, update {epoch+1}')
                    adaptive_state = solver.update_adaptive_state(
                        adaptive_state, base_weights, sampled_terms, sampled_rms, cfg)
                    weights = adaptive_state['weights'].copy()
                    weights, guard = apply_temporal_gradient_guard(
                        weights, base_weights, sampled_rms, cfg)
                    adaptive_state['weights'] = weights.copy()
                    active = base_weights > 0.0
                    adaptive_state['log_multipliers'][active] = np.log(
                        weights[active] / base_weights[active])
                    weighted_ratio = None
                    if (sampled_rms[0] > cfg.adaptive_gradient_epsilon
                            and weights[0] > 0.0):
                        weighted_ratio = float(
                            weights[1] * sampled_rms[1] / (weights[0] * sampled_rms[0]))
                    update_record = {
                        'block': block + 1, 'update_in_block': epoch + 1,
                        'weights': dict(zip(solver.LOSS_NAMES, weights.tolist())),
                        'losses': dict(zip(solver.LOSS_NAMES, sampled_terms.tolist())),
                        'network_gradient_rms': dict(zip(solver.LOSS_NAMES, sampled_rms.tolist())),
                        'weighted_temporal_to_pde_gradient_ratio': weighted_ratio,
                        'temporal_pde_gradient_cosine': cosine,
                        'guard': guard,
                    }
                    diagnostics['adaptive_updates'].append(update_record)
                    block_record['adaptive_updates'] += 1
                    if guard['triggered']:
                        diagnostics['safeguard_event_count'] += 1
                        block_record['safeguard_events'] += 1
                    adaptive_trace.append(np.concatenate((
                        np.asarray((block + 1, epoch + 1), dtype=np.float64),
                        weights, sampled_terms, sampled_rms,
                        np.asarray((guard['ratio_before'] if guard['ratio_before'] is not None else np.nan,
                                    guard['ratio_after'] if guard['ratio_after'] is not None else np.nan,
                                    float(guard['triggered'])), dtype=np.float64),
                    )))
                tick = time.perf_counter()
                candidate,candidate_opt,loss,terms,_,stats = update(
                    params,opt_state,jnp.asarray(schedule[block,epoch]),*args,jnp.asarray(weights))
                norm,clipped,finite = (np.asarray(v).item() for v in stats)
                elapsed = time.perf_counter()-tick
                block_record['update_wall_seconds'] += elapsed
                if not finite:
                    diagnostics['nonfinite_updates'] += 1
                    diagnostics['failed_block'] = block+1
                    diagnostics['failed_update_in_block'] = epoch+1
                    writer.writerow([block+1,epoch+1,diagnostics['completed_updates']+1,
                                     float(loss),norm,bool(clipped),False,elapsed]);stream.flush()
                    raise FloatingPointError('Nonfinite loss/gradient/parameter/optimiser state; update rejected')
                params,opt_state = candidate,candidate_opt
                diagnostics['completed_updates'] += 1
                diagnostics['clipped_gradient_updates'] += int(clipped)
                block_record['clipped_gradient_updates'] += int(clipped)
                block_record['max_gradient_global_norm'] = max(block_record['max_gradient_global_norm'],float(norm))
                writer.writerow([block+1,epoch+1,diagnostics['completed_updates'],float(loss),float(norm),bool(clipped),True,elapsed])
                if (epoch+1)%cfg.diagnostic_every == 0 or epoch+1 == cfg.epochs:
                    diagnose(epoch+1)
                    stream.flush()
                    print(f'block={block+1}/{cfg.slabs} updates={epoch+1}/{cfg.epochs} loss={float(loss):.6g}',flush=True)
            tick = time.perf_counter()
            final_terms,physics = evaluate(params,monitor_pairs,*args)
            final_terms,physics = np.asarray(final_terms,dtype=float),np.asarray(physics,dtype=float)
            block_record['monitor_wall_seconds'] += time.perf_counter()-tick
            if not np.all(np.isfinite(final_terms)):
                raise FloatingPointError('Nonfinite final monitored losses')
            entry_after = np.asarray(grid_prediction(params,args[0]))-np.asarray(initial_values)
            block_record['entry_grid_rms_after'] = float(np.sqrt(np.mean(entry_after**2)))
            block_record['final_losses'] = dict(zip(solver.LOSS_NAMES,final_terms.tolist()))
            block_record['final_weighted_loss'] = float(np.dot(weights,final_terms))
            block_record['final_parameters_sha256'] = tree_sha256(params)
            current_t = jnp.asarray(block_t,dtype=jnp.float32)
            current_values = jax.vmap(lambda ti:grid_prediction(params,ti))(current_t)
            if block == 0:
                past_t,past_values = current_t,current_values
            else:
                past_t = jnp.concatenate((past_t,current_t[1:]))
                past_values = jnp.concatenate((past_values,current_values[1:]))
            history.append(block_record['final_weighted_loss'])
            parameter_history.append(physics.tolist());weight_history.append(weights.tolist())
            # Force accepted history ready before the block timing is captured.
            np.asarray(past_values)
            block_record['wall_seconds_before_checkpoint'] = time.perf_counter()-block_started
            diagnostics['block_records'].append(block_record)
            audit = {key:diagnostics[key] for key in ('initial_parameters_sha256','pair_schedule_sha256',
                     'data_sha256','source_sha256','block_records','gradient_diagnostics',
                     'adaptive_updates','safeguard_event_count','completed_updates',
                     'clipped_gradient_updates')}
            checkpoint_path = checkpoint_dir/f'checkpoint_slab_{block+1:03d}.pkl'
            if checkpoint_path.exists():
                raise FileExistsError(f'Refusing to overwrite immutable checkpoint {checkpoint_path}')
            tick = time.perf_counter()
            solver.save_checkpoint(checkpoint_path,params=params,opt_state=opt_state,next_slab=block+1,
                past_t=past_t,past_values=past_values,history=history,parameter_history=parameter_history,
                weight_history=weight_history,adaptive_trace=adaptive_trace,
                rng_state=rng.bit_generator.state,
                cfg=cfg,controlled_state=audit)
            block_record['checkpoint_wall_seconds'] = time.perf_counter()-tick
            block_record['wall_seconds'] = time.perf_counter()-block_started
            atomic_json(out/'diagnostics.json',diagnostics)
    eval_started = time.perf_counter()
    eval_x = x[:-1].astype(np.float32)
    # Evaluate each time on one vector grid: stable memory even for the full 1024 grid.
    evaluate_grid = jax.jit(lambda ti:net(params,jnp.full_like(jnp.asarray(eval_x),ti),jnp.asarray(eval_x)))
    prediction = np.stack([np.asarray(evaluate_grid(jnp.asarray(ti,dtype=np.float32))) for ti in t],axis=1)
    accepted = solver.periodic_linear_resample(np.asarray(past_values),x_grid_np,eval_x).T
    if not np.all(np.isfinite(prediction)):
        raise FloatingPointError('Nonfinite direct predictions')
    final_history = np.stack([np.asarray(grid_prediction(params,ti)) for ti in past_t])
    diagnostics['final_network_history_difference_rms'] = float(np.sqrt(np.mean((final_history-np.asarray(past_values))**2)))
    diagnostics['final_parameters_sha256'] = tree_sha256(params)
    diagnostics['evaluation_wall_seconds'] = time.perf_counter()-eval_started
    diagnostics['training_update_wall_seconds'] = sum(v['update_wall_seconds'] for v in diagnostics['block_records'])
    diagnostics['gradient_diagnostic_wall_seconds'] = sum(v['gradient_diagnostic_wall_seconds'] for v in diagnostics['block_records'])
    diagnostics['monitor_wall_seconds'] = sum(v['monitor_wall_seconds'] for v in diagnostics['block_records'])
    return {'x':eval_x,'t':t,'prediction':prediction,'direct_prediction':prediction,
            'accepted_t':np.asarray(past_t),'accepted_prediction':accepted,
            'terminal_state':prediction[:,train_stop],'terminal_time':cfg.train_end_time,
            'fd_x':x_grid_np,'fd_terminal_state':np.asarray(grid_prediction(params,jnp.asarray(cfg.train_end_time,dtype=jnp.float32))),
            'training_mask':t<=cfg.train_end_time+1e-8,'forecast_mask':t>cfg.train_end_time+1e-8,
            'prediction_mode':np.asarray('direct'),'reference_values_used':np.asarray('initial_only'),
            'temporal_integrator':np.asarray(cfg.temporal_integrator),
            'alpha':cfg.alpha,'beta':cfg.beta,'nu':cfg.nu,
            'seed':cfg.seed,'model_version':solver.MODEL_VERSION,
            'slab_loss':np.asarray(history),'slab_final_loss_weights':np.asarray(weight_history),
            'adaptive_trace':np.asarray(adaptive_trace,dtype=np.float64),
            'adaptive_trace_columns':np.asarray(
                ('block','update_in_block',
                 *(f'weight_{name}' for name in solver.LOSS_NAMES),
                 *(f'loss_{name}' for name in solver.LOSS_NAMES),
                 *(f'gradient_rms_{name}' for name in solver.LOSS_NAMES),
                 'guard_ratio_before','guard_ratio_after','guard_triggered'))}
