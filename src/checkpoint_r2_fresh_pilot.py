"""Bounded fresh-only R2 pilot. No production extraction, fitted heads or E route."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path
import numpy as np
from src.checkpoint_r2_fresh_inputs import (ROOT, build_manifest, canonical, file_hash, hash_value,
                                           require, text_hash)
from src.checkpoint_r2_fresh_store import (fsync_dir, load_features, pin, storage_probe,
                                         write_json, write_shard)

CONFIG = 'config/checkpoint_r2/fresh_pilot_v1.json'
CONFIG_SHA256 = 'c64b523a4a56e2f1c5e1dc77036ec2f97a7547e17c4d488088809dc307cdfe00'
MANIFEST = 'data/checkpoint_r2_fresh_pilot_v1/manifest.json'


def plan(root=ROOT, rebuild=True):
    require(file_hash(root / CONFIG) == CONFIG_SHA256, 'fresh contract hash mismatch')
    cfg = json.loads((root / CONFIG).read_text())
    for name, expected in cfg['frozen_sha256'].items():
        require(file_hash(root / name) == expected, 'frozen input changed: ' + name)
    require(file_hash(root / MANIFEST) == cfg['manifest_sha256'], 'pilot manifest changed')
    manifest = json.loads((root / MANIFEST).read_text())
    if rebuild:
        require(manifest == build_manifest(root), 'selection IDs/text/bindings differ from deterministic rebuild')
    require(len(manifest['stages']['calibration']) == 80 and len(manifest['stages']['verification']) == 240,
            'pilot bounded counts')
    allrows = sum(manifest['stages'].values(), [])
    require(len({r['id'] for r in allrows}) == 320 and all(r['split'] in ('train', 'validation') for r in allrows),
            'unique train/development only')
    require(all(r['split'] == 'train' for r in manifest['stages']['calibration']), 'train-only calibration')
    require(cfg['batch_grid'] == [1, 2, 4, 8] and cfg['timing_repeats'] == 3, 'bounded comparison')
    return cfg, manifest


def semantic_view(packet, rows):
    require(len(packet) == len(rows), 'missing token rows')
    logical = []
    for p, row in zip(packet, rows):
        require(p['id'] == row['id'] and p['statement_sha256'] == text_hash(row['statement']), 'ordered IDs/text')
        tokens, mask, pos = p['token_ids'], p['attention_mask'], p['position_ids']
        require(len(tokens) == len(mask) == len(pos) and set(mask) <= {0, 1} and sum(mask) > 0, 'mask dimensions')
        n = sum(mask)
        require(mask == [1] * n + [0] * (len(mask) - n), 'right padding mask')
        require(tokens[:n] == p['unpadded_token_ids'] and all(t == p['pad_token_id'] for t in tokens[n:]), 'tokens/pads')
        specials = p['unpadded_special_tokens_mask']
        require(len(specials) == n and set(specials) <= {0, 1}
                and [t for t, special in zip(tokens[:n], specials) if not special] == p['raw_token_ids'], 'special-token sequence')
        require(pos == list(range(n)) + [0] * (len(mask) - n), 'semantic positions')
        require(p['readout_index'] == n - 1 == p['semantic_readout_position'], 'final nonpadding readout')
        require(p['add_special_tokens'] and not p['truncation'] and not p['chat_template'], 'input policy')
        logical.append(dict(id=p['id'], text_sha256=p['statement_sha256'], tokens=tokens[:n],
                            attention_mask=mask[:n], position_ids=pos[:n], readout_position=n - 1,
                            special_tokens_mask=specials, raw_token_ids=p['raw_token_ids']))
    return logical


def numerical(reference, observed):
    require(reference.shape == observed.shape and reference.ndim == 3, 'comparison layer dimensions')
    require(np.isfinite(reference).all() and np.isfinite(observed).all(), 'comparison nonfinite')
    a, b = reference.astype(np.float64), observed.astype(np.float64)
    delta = b - a
    scale = np.maximum(np.sqrt(np.mean(a * a, axis=(0, 2))), 1e-6)
    rms = np.sqrt(np.mean(delta * delta, axis=(0, 2)))
    maximum = np.max(np.abs(delta), axis=(0, 2))
    return dict(max_absolute=maximum.tolist(), rms_absolute=rms.tolist(),
                relative_rms=(rms / scale).tolist(), scaled_max_absolute=(maximum / scale).tolist(),
                reference_rms=scale.tolist(), relative_definition='per-layer RMS(delta)/max(RMS(reference),1e-6)')


def differences(reference, observed):
    # Distinguish BF16 computation variation from subsequent FP16 representation error.
    ra, ob = reference.astype(np.float16).astype(np.float32), observed.astype(np.float16).astype(np.float32)
    return dict(computation=numerical(reference, observed), stored_fp16=numerical(ra, ob),
                reference_storage_effect=numerical(reference, ra), observed_storage_effect=numerical(observed, ob))


def calibrate(report, acceptance):
    limits = {}
    for kind in ('computation', 'stored_fp16'):
        values = report[kind]
        require(max(values['relative_rms']) <= acceptance['hard_relative_rms_ceiling']
                and max(values['scaled_max_absolute']) <= acceptance['hard_scaled_max_ceiling'], 'calibration exceeds hard ceilings')
        limits[kind] = {}
        for name, floor in [('max_absolute', acceptance['max_floor_scale']),
                            ('rms_absolute', acceptance['rms_floor_scale'])]:
            limits[kind][name] = np.maximum(acceptance['multiplier'] * np.array(values[name]),
                                           floor * np.array(values['reference_rms'])).tolist()
        limits[kind]['relative_rms'] = np.maximum(acceptance['multiplier'] * np.array(values['relative_rms']),
                                                 acceptance['rms_floor_scale']).tolist()
    return limits


def accepted(report, limits, acceptance):
    for kind in ('computation', 'stored_fp16'):
        values = report[kind]
        if max(values['relative_rms']) > acceptance['hard_relative_rms_ceiling'] or max(values['scaled_max_absolute']) > acceptance['hard_scaled_max_ceiling']:
            return False
        for name in ('max_absolute', 'rms_absolute', 'relative_rms'):
            if np.any(np.array(values[name]) > np.array(limits[kind][name])):
                return False
    return True


def atomic_pointer(path, value):
    tmp = path.with_suffix('.updating')
    with tmp.open('wb') as f:
        f.write(canonical(value))
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    fsync_dir(path.parent)


def exceeded(state, now):
    elapsed = max(0, now - state['as_of_monotonic'])
    if state['active_seconds'] + (elapsed if state['phase'] == 'active' else 0) >= 7200:
        return 'two_active_hours'
    if state['phase'] == 'setup' and state['phase_seconds'] + elapsed >= 3600:
        return 'setup_one_hour_timeout'
    for model, seconds in state['gpu_seconds'].items():
        if seconds + (elapsed if state['gpu_inflight'] == model else 0) >= 3600:
            return 'one_gpu_hour_' + model
    return None


class Budget:
    def __init__(self, output, clock=time.monotonic):
        self.output, self.clock = output, clock
        self.started = self.phase_started = clock()
        self.phase = 'setup'
        self.active = self.setup = 0.
        self.gpu = {'qwen': 0., 'llama': 0.}
        self.inflight = None
        self.gpu_started = None
        prior = output / 'budget.json'
        if prior.exists():
            state = json.loads(prior.read_text())
            require(state['phase'] == 'setup' and state['gpu_inflight'] is None, 'refuse budget reset after execution')
            require(not any(state['gpu_seconds'].values()) and state['active_seconds'] == 0, 'refuse spent campaign reset')
            now = clock()
            inherited = state['setup_seconds'] + max(0., now - state['as_of_monotonic'])
            self.setup = inherited
            self.started = now - (state['total_billable_wall_seconds'] + max(0., now - state['as_of_monotonic']))
        self.watch()

    def snapshot(self):
        now = self.clock()
        elapsed = now - self.phase_started
        gpu = dict(self.gpu)
        if self.inflight:
            gpu[self.inflight] += now - self.gpu_started
        return dict(phase=self.phase, phase_seconds=elapsed, as_of_monotonic=now,
                    active_seconds=self.active + (elapsed if self.phase == 'active' else 0),
                    setup_seconds=self.setup + (elapsed if self.phase == 'setup' else 0),
                    gpu_seconds=gpu, gpu_inflight=self.inflight, total_billable_wall_seconds=now - self.started,
                    accounting='GPU conservative wall from load start through teardown, including host/writing; '
                               'billable wall includes setup/download; provisioning/queue additional')

    def watch(self):
        atomic_pointer(self.output / 'budget.json', self.snapshot())
        reason = exceeded(self.snapshot(), self.clock())
        if reason:
            raise TimeoutError(reason)

    def transition(self, phase):
        now = self.clock()
        if self.phase == 'active':
            self.active += now - self.phase_started
        else:
            self.setup += now - self.phase_started
        self.phase, self.phase_started = phase, now
        self.watch()

    def begin_model(self, model):
        require(self.inflight is None, 'models must execute sequentially')
        self.inflight, self.gpu_started = model, self.clock()
        self.watch()

    def end_model(self):
        self.gpu[self.inflight] += self.clock() - self.gpu_started
        self.inflight = None
        self.watch()


def stage(output, name, rows, batch_size, backend, spec, provenance, budget):
    require(0 < len(rows) <= 320 and len({r['id'] for r in rows}) == len(rows), 'bounded unique shard inputs')
    arrays, tokens, durations = [], [], []
    start = time.monotonic()
    for i in range(0, len(rows), batch_size):
        budget.watch()
        selected = rows[i:i + batch_size]
        begin = time.monotonic()
        compute, packet = backend.forward(selected, batch_size)
        semantic_view(packet, selected)
        require(compute.dtype == np.float32 and compute.shape == (len(selected), spec['layers'], spec['width']), 'compute axes/dtype')
        arrays.append(compute)
        tokens.extend(packet)
        durations.append(dict(inputs=len(selected), tokenization_forward_transfer_seconds=time.monotonic() - begin))
        budget.watch()
    values = np.concatenate(arrays)
    write_started = time.monotonic()
    # The intended production path uses bounded contiguous shards of up to 320 rows,
    # independent of GPU batch size. Includes exclusive publication and strict reopen.
    shard = write_shard(output / name / '0000', rows, values, spec,
                        dict(provenance, stage=name, timing_repeat=name.startswith('timing-'),
                             batch_size=batch_size, padding='none' if batch_size == 1 else 'right'), tokens)
    write_seconds = time.monotonic() - write_started
    budget.watch()
    wall = time.monotonic() - start
    receipt = dict(stage=name, unique_input_count=len(rows), timing_repeat=name.startswith('timing-'),
                   batch_size=batch_size, end_to_end_seconds=wall, inputs_per_second=len(rows) / wall,
                   batches=durations, shards=[shard], serialization_publication_reopen_checks_seconds=write_seconds,
                   output_bytes=shard['output_bytes'], payload_bytes=values.size * 2,
                   ordered_ids_sha256=hash_value([r['id'] for r in rows]))
    write_json(output / (name + '.json'), receipt)
    return values, tokens, receipt


def projection(spec, timing, cfg, snapshot, hourly_price, model_setup_seconds=0.):
    hours = [35843 / t['inputs_per_second'] / 3600 for t in timing]
    payload = 35843 * spec['layers'] * spec['width'] * 2
    return dict(raw_unique_inputs=35843, payload_bytes=payload,
                measured_repeat_inputs_per_second=[t['inputs_per_second'] for t in timing],
                projected_raw_hours_mean=float(np.mean(hours)), projected_raw_hours_observed_range=[min(hours), max(hours)],
                projection_scope='representative pilot extrapolation; length/composition variation beyond pilot unmeasured; '
                                 'timing includes writing/reopen/hash checks; repeat inputs are not new observations',
                setup_download_hours=snapshot['download_and_hash_seconds'] / 3600,
                model_load_and_integrity_hours=model_setup_seconds / 3600,
                weights_and_tokenizer_bytes=snapshot['downloaded_bytes'],
                production_storage=dict(one_tensor_copy_bytes=payload, retained_second_copy_bytes=payload,
                                        shard_metadata_extra='measure from pilot; not included in tensor inventory',
                                        temporary_shard_bytes=cfg['production_shard_rows'] * spec['layers'] * spec['width'] * 2,
                                        temporary_weight_download_bytes=max((f['bytes'] for f in snapshot.get('files', {}).values()), default=0),
                                        pilot_retained_outputs_extra='sum actual pilot file sizes from checksummed inventory',
                                        verification_read_bytes=payload, transfer_bytes=payload,
                                        transfer_time='unmeasured until actual transfer; bytes / measured sustained bytes per second',
                                        usable_full_quota='requires separate real allocation/quota evidence before full extraction'),
                hourly_price=hourly_price, dollar_raw_range=None if hourly_price is None else [min(hours) * hourly_price, max(hours) * hourly_price],
                cost_formula='hourly_price * (raw_hours + setup/download_hours + transfer/retained-instance_hours + provisioning_hours)',
                behavior_chat_included=False)


def run_model(output, model, cfg, manifest, snapshot, runtime, budget, backend_factory, hourly_price):
    spec = cfg['models'][model]
    destination = output / model
    destination.mkdir()
    budget.transition('setup')
    budget.begin_model(model)
    backend = None
    try:
        setup_started = time.monotonic()
        backend = backend_factory(spec, snapshot, cfg['execution'])
        model_setup_seconds = time.monotonic() - setup_started
        # Lock actual runtime, files and frozen manifest before FIRST inference.
        write_json(destination / 'runtime-before-inference.json', dict(runtime, backend=backend.details,
                   model_snapshot=snapshot, contract_sha256=CONFIG_SHA256, manifest_sha256=cfg['manifest_sha256']))
        provenance = dict(commit=json.loads((output / 'launch.json').read_text())['commit'], model=model,
                          contract_sha256=CONFIG_SHA256, manifest_sha256=cfg['manifest_sha256'],
                          runtime_receipt_sha256=file_hash(destination / 'runtime-before-inference.json'))
        budget.transition('active')
        cal, ver = manifest['stages']['calibration'], manifest['stages']['verification']
        warm = []
        for i in range(3):
            _, _, receipt = stage(destination, f'warmup-{i}', cal[:8], 1, backend, spec, provenance, budget)
            warm.append(receipt)
        reference, ref_tokens, _ = stage(destination, 'calibration-reference', cal, 1, backend, spec, provenance, budget)
        calibration, limits, usable = {}, {}, []
        for batch in cfg['batch_grid']:
            try:
                observed, tokens, _ = stage(destination, f'calibration-b{batch}', cal, batch, backend, spec, provenance, budget)
            except Exception as exc:
                if type(exc).__name__ != 'OutOfMemoryError' or batch == 1:
                    raise
                backend.torch.cuda.empty_cache()
                calibration[str(batch)] = dict(accepted=False, reason='bounded grid OOM; no retry')
                continue
            require(semantic_view(tokens, cal) == semantic_view(ref_tokens, cal), 'batch token semantics differ')
            report = differences(reference, observed)
            try:
                limit = calibrate(report, cfg['numerical_acceptance'])
            except ValueError:
                calibration[str(batch)] = dict(accepted=False, differences=report, reason='hard calibration ceiling')
                require(batch != 1, 'batch1 repeatability failed calibration')
                continue
            limits[str(batch)] = limit
            calibration[str(batch)] = dict(accepted=True, differences=report, limits=limit)
            usable.append(batch)
        # This immutable receipt is published BEFORE independent reference or verification inference.
        write_json(destination / 'locked-calibration.json', dict(calibration=calibration, limits=limits,
                   procedure=cfg['numerical_acceptance'], calibration_ids=[r['id'] for r in cal],
                   verification_ids=[r['id'] for r in ver], tolerance_widening_allowed=False))
        lock_hash = file_hash(destination / 'locked-calibration.json')
        ref, tok, _ = stage(destination, 'verification-reference', ver, 1, backend, spec, provenance, budget)
        verification = {}
        for batch in usable:
            try:
                observed, packet, _ = stage(destination, f'verification-b{batch}', ver, batch, backend, spec, provenance, budget)
            except Exception as exc:
                if type(exc).__name__ != 'OutOfMemoryError' or batch == 1:
                    raise
                backend.torch.cuda.empty_cache()
                verification[str(batch)] = dict(accepted=False, reason='verification OOM; fallback batch1')
                continue
            require(semantic_view(packet, ver) == semantic_view(tok, ver), 'verification token semantics differ')
            report = differences(ref, observed)
            verification[str(batch)] = dict(accepted=accepted(report, limits[str(batch)], cfg['numerical_acceptance']), differences=report)
        require(verification['1']['accepted'], 'independent batch1 numerical failure; fix correctness before extraction')
        require(file_hash(destination / 'locked-calibration.json') == lock_hash, 'locked tolerances changed')
        fallback = any(not r['accepted'] for r in verification.values())
        batch = 1 if fallback else max(usable)
        policy = dict(batch_size=batch, padding='none' if batch == 1 else 'right', ordering='frozen manifest order; contiguous batches',
                      fallback=fallback, fallback_reason='independent batching failure' if fallback else None,
                      locked_calibration_sha256=lock_hash, verification=verification)
        write_json(destination / 'accepted-policy.json', policy)
        # Full real path with fixed repeat count; 320 observations, never 960 observations.
        rows = cal + ver
        timing = [stage(destination, f'timing-{i}', rows, batch, backend, spec, provenance, budget)[2] for i in range(cfg['timing_repeats'])]
        # Eight real rows form a portable shard using the same loader and format.
        transfer = np.concatenate([np.asarray(load_features(Path(s['path']))[0]) for s in timing[0]['shards']])[:8].astype(np.float32)
        transfer_rows = rows[:8]
        source_tokens = []
        for shard in timing[0]['shards']:
            source_tokens.extend(json.loads((Path(shard['path']) / 'tokens.json').read_text()))
        small = write_shard(destination / 'transfer-ready', transfer_rows, transfer, spec,
                            dict(provenance, purpose='mac2-independent-loader-check', stage='transfer-ready',
                                 batch_size=batch, padding='none' if batch == 1 else 'right'), source_tokens[:8])
        result = dict(model=model, gpu=runtime['gpu'], runtime=runtime['packages'], input_count=320,
                      calibration_count=80, verification_count=240, correctness='passed fresh semantics and independent numerical verification',
                      historical_compatibility='not assessed', research_result=False, accepted_policy=policy,
                      warmup_seconds=[t['end_to_end_seconds'] for t in warm],
                      steady_state_inputs_per_second=[t['inputs_per_second'] for t in timing],
                      throughput_mean=float(np.mean([t['inputs_per_second'] for t in timing])),
                      throughput_sample_std=float(np.std([t['inputs_per_second'] for t in timing], ddof=1)),
                      model_setup_seconds=model_setup_seconds, peak_vram_bytes=backend.peak(), timing_output_bytes=[t['output_bytes'] for t in timing],
                      unique_tensor_payload_bytes=320 * spec['layers'] * spec['width'] * 2,
                      transfer_ready=small, projection=projection(spec, timing, cfg, snapshot, hourly_price, model_setup_seconds),
                      recommendation='proceed to separate complete-fresh-extraction authorization review' if not fallback else
                                     'proceed to separate review using verified batch1 fallback; batching failed')
        write_json(destination / 'result.json', result)
        return result
    finally:
        if backend is not None:
            backend.close()
        budget.end_model()


def worker(output, cache, hourly_price):
    cfg, manifest = plan(rebuild=False)
    budget = Budget(output)
    results = {}
    try:
        from src.checkpoint_r2_fresh_backend import FreshBackend, prepare_snapshots, runtime_receipt
        runtime = runtime_receipt(cfg['runtime_lock'])
        write_json(output / 'host-runtime.json', runtime)
        reserve = cfg['pilot_output_reserve_bytes']
        write_json(output / 'storage-before-inference.json', storage_probe(output.parent, reserve))
        snapshots = prepare_snapshots(cfg, cache, output)
        budget.watch()
        write_json(output / 'manifest-before-inference.json', manifest)
        for model in ('qwen', 'llama'):
            results[model] = run_model(output, model, cfg, manifest, snapshots[model], runtime, budget, FreshBackend, hourly_price)
        # Bind every published pilot artifact, including receipts, for transfer and handoff.
        inventory = {str(p.relative_to(output)): pin(p) for p in sorted(output.rglob('*'))
                     if p.is_file() and p.name not in ('budget.json', 'budget.updating') and not p.name.endswith('.partial')}
        write_json(output / 'artifact-inventory.json', inventory)
        from datetime import datetime, timezone
        provision_started = output.parent / 'provision-started-utc.txt'
        provision_ended = output.parent / 'provision-completed-utc.txt'
        provisioning = None
        if provision_started.exists() and provision_ended.exists():
            start = datetime.fromisoformat(provision_started.read_text().strip().replace('Z', '+00:00'))
            end = datetime.fromisoformat(provision_ended.read_text().strip().replace('Z', '+00:00'))
            provisioning = dict(seconds=(end-start).total_seconds(), started_utc=start.isoformat(),
                                completed_utc=end.isoformat(),
                                total_billable_since_provision_start_seconds=(datetime.now(timezone.utc)-start).total_seconds())
        write_json(output / 'completion.json', dict(results=results, budget=budget.snapshot(), provisioning=provisioning,
                    production_started=False, research_fitting=False, E_scoring=False,
                    recommendation='review measured cost and correctness before authorizing complete coherent fresh extraction'))
    except BaseException as exc:
        write_json(output / 'failure.json', dict(error_type=type(exc).__name__, error=str(exc),
                    traceback=traceback.format_exc(), partial_results=results, budget=budget.snapshot(),
                    recommendation='fix the recorded issue; preserve campaign budget and do not reset limits'))
        raise


def supervise(command, output):
    supervisor_started = time.monotonic()
    process = subprocess.Popen(command, start_new_session=True)
    reason = None
    try:
        while process.poll() is None:
            state = json.loads((output / 'budget.json').read_text())
            reason = exceeded(state, time.monotonic())
            if reason:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                break
            time.sleep(.1)
    except BaseException:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        raise
    finally:
        write_json(output / 'supervisor-exit.json', dict(returncode=process.returncode, cap_reason=reason,
                   last_budget=json.loads((output / 'budget.json').read_text()),
                   supervisor_total_billable_wall_seconds=time.monotonic() - supervisor_started,
                   ended_monotonic=time.monotonic(), budget_reset_allowed=False))
    return 124 if reason else process.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['dry-run', 'run', 'check-shard'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--expected-commit')
    parser.add_argument('--expected-receipt-sha256')
    parser.add_argument('--hourly-price', type=float)
    parser.add_argument('--internal-worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.mode == 'check-shard':
        require(args.expected_receipt_sha256 is not None and file_hash(args.output / 'receipt.json') == args.expected_receipt_sha256,
                'independently supplied receipt checksum required')
        values, rows, receipt = load_features(args.output)
        print(json.dumps(dict(shape=list(values.shape), rows=len(rows), receipt=pin(args.output / 'receipt.json'),
                              payload_sha256=receipt['payload_sha256'])))
        return
    cfg, manifest = plan()
    if args.mode == 'dry-run':
        write_json(args.output, dict(status='local_metadata_verified_no_inference', contract_sha256=CONFIG_SHA256,
                   manifest_sha256=cfg['manifest_sha256'], input_count=320, calibration_count=80, verification_count=240,
                   raw_inventory=cfg['raw_inventory'], runtime_lock=cfg['runtime_lock'], numerical_acceptance=cfg['numerical_acceptance']))
        return
    require(args.hourly_price is None or args.hourly_price >= 0, 'hourly price must be nonnegative')
    require(platform_is_linux(), 'model inference must execute on remote Linux GPU host')
    require(args.cache is not None and args.expected_commit, 'fresh cache and exact pushed code commit required')
    output, cache = args.output.resolve(), args.cache.resolve()
    require(not output.is_relative_to(ROOT) and not cache.is_relative_to(ROOT)
            and not output.is_relative_to(cache) and not cache.is_relative_to(output), 'external separate fresh output/cache paths')
    require(not str(output).startswith(('/tmp/', '/private/tmp/')), 'durable output required')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    require(head == args.expected_commit and len(head) == 40, 'exact remote code revision mismatch')
    require(not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=all'], text=True).strip(),
            'remote code checkout must be clean')
    if args.internal_worker:
        launch = json.loads((output / 'launch.json').read_text())
        require(launch['parent_pid'] == os.getppid(), 'worker must be supervised')
        worker(output, cache, args.hourly_price)
        return
    require(output.parent.is_dir() and not output.exists() and not cache.exists(), 'fresh output/cache only')
    # One campaign sentinel per durable destination. Failures cannot silently reset the caps.
    write_json(output.parent / 'r2-fresh-pilot-20261002-campaign.json', dict(output=str(output), cache=str(cache),
               commit=head, caps=cfg['caps'], status='single_attempt_no_automatic_retry', billable_started_utc=time.time()))
    output.mkdir()
    write_json(output / 'launch.json', dict(commit=head, parent_pid=os.getpid(), argv=sys.argv,
               contract_sha256=CONFIG_SHA256, manifest_sha256=cfg['manifest_sha256'], hourly_price=args.hourly_price))
    Budget(output)
    command = [sys.executable, '-B', '-m', 'src.checkpoint_r2_fresh_pilot', *sys.argv[1:], '--internal-worker']
    raise SystemExit(supervise(command, output))


def platform_is_linux():
    return sys.platform.startswith('linux')


if __name__ == '__main__':
    main()
