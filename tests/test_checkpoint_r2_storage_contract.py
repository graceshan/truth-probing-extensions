"""Non-secret permission reporting never substitutes for locking/integrity/quota."""
import copy
import errno
import json
import subprocess
from pathlib import Path
import pytest
import src.checkpoint_r2_fresh_store as store
import src.checkpoint_r2_full_support as support


POLICY=support.NON_SECRET_ARTIFACT_CONTRACT


def emulate_mode_reporting(monkeypatch):
    original=Path.chmod
    def emulated(path,mode,*args,**kwargs):
        return original(path,0o666 if path.name=='lock' else mode,*args,**kwargs)
    monkeypatch.setattr(Path,'chmod',emulated)


def test_native_modes_are_reported_without_claiming_access_enforcement(tmp_path):
    receipt=support.filesystem_prerequisites(tmp_path,artifact_contract=POLICY)
    permissions=receipt['permissions']
    assert permissions['requested_mode']==permissions['observed_mode_after']=='0o600'
    assert permissions['chmod_call']['status']==permissions['exact_mode_reporting']['status']=='passed'
    assert permissions['access_enforcement']['status']==permissions['owner_only_protection']['status']=='untested'
    assert receipt['writer_and_full_capacity']['status']=='untested'
    assert receipt['git_initialization']['status']==receipt['cross_process_flock_exclusion']['status']=='passed'
    assert receipt['mount_evidence']['volume']==str(tmp_path)
    assert not list(tmp_path.iterdir())


def test_mode_emulation_only_reports_unsupported_under_nonsecret_policy(tmp_path,monkeypatch):
    emulate_mode_reporting(monkeypatch)
    receipt=support.filesystem_prerequisites(tmp_path,artifact_contract=POLICY)
    permissions=receipt['permissions']
    assert permissions['requested_mode']=='0o600' and permissions['observed_mode_after']=='0o666'
    assert permissions['chmod_call']['status']=='passed'
    assert permissions['exact_mode_reporting']['status']=='unsupported'
    assert permissions['access_enforcement']['status']==permissions['owner_only_protection']['status']=='untested'
    support.validate_filesystem_prerequisites(receipt)


@pytest.mark.parametrize('contract',[None,'','private','credential'])
def test_private_unknown_and_implicit_contracts_are_rejected(tmp_path,contract):
    with pytest.raises(ValueError,match='non-secret filesystem contract'):
        support.filesystem_prerequisites(tmp_path,artifact_contract=contract)
    assert not list(tmp_path.iterdir())
    with pytest.raises(TypeError):support.filesystem_prerequisites(tmp_path)


@pytest.mark.parametrize('error',[errno.EACCES,errno.EIO,errno.EOPNOTSUPP])
def test_chmod_errors_are_not_treated_as_mode_emulation(tmp_path,monkeypatch,error):
    def fail(*args,**kwargs):raise OSError(error,'chmod failed')
    monkeypatch.setattr(Path,'chmod',fail)
    with pytest.raises(OSError) as caught:support.filesystem_prerequisites(tmp_path,artifact_contract=POLICY)
    assert caught.value.errno==error


@pytest.mark.parametrize('child_returncode',[0,1,42])
def test_failed_exclusion_child_errors_and_failed_release_block(tmp_path,monkeypatch,child_returncode):
    original=support.subprocess.run
    def child(args,**kwargs):
        if '-c' in args:return subprocess.CompletedProcess(args,child_returncode,b'',b'BlockingIOError')
        return original(args,**kwargs)
    monkeypatch.setattr(support.subprocess,'run',child)
    with pytest.raises(ValueError,match='filesystem lock'):
        support.filesystem_prerequisites(tmp_path,artifact_contract=POLICY)


def test_parent_lock_error_is_not_swallowed(tmp_path,monkeypatch):
    def fail(*args):raise OSError(errno.EIO,'lock failed')
    monkeypatch.setattr(support.fcntl,'flock',fail)
    with pytest.raises(OSError,match='lock failed'):support.filesystem_prerequisites(tmp_path,artifact_contract=POLICY)


def test_git_failure_still_blocks(tmp_path,monkeypatch):
    def fail(args,**kwargs):raise subprocess.CalledProcessError(1,args)
    monkeypatch.setattr(support.subprocess,'run',fail)
    with pytest.raises(subprocess.CalledProcessError):support.filesystem_prerequisites(tmp_path,artifact_contract=POLICY)


@pytest.mark.parametrize('change',['policy','locking','git','false_protection','false_mode_success'])
def test_capability_receipts_reject_false_passes(tmp_path,monkeypatch,change):
    emulate_mode_reporting(monkeypatch)
    receipt=support.filesystem_prerequisites(tmp_path,artifact_contract=POLICY)
    if change=='policy':receipt['artifact_contract']='private'
    if change=='locking':receipt['cross_process_flock_exclusion']['status']='unsupported'
    if change=='git':receipt['git_initialization']['status']='untested'
    if change=='false_protection':receipt['permissions']['owner_only_protection']['status']='passed'
    if change=='false_mode_success':receipt['permissions']['exact_mode_reporting']['status']='passed'
    with pytest.raises(ValueError):support.validate_filesystem_prerequisites(receipt)


@pytest.mark.parametrize('failure',['publication','overwrite','fsync','hash_readback','recovery'])
def test_full_storage_entry_still_blocks_integrity_failures_after_mode_emulation(tmp_path,monkeypatch,failure):
    """Tiny synthetic storage plan only, through the real unchanged storage_probe."""
    import src.checkpoint_r2_full_raw as full
    emulate_mode_reporting(monkeypatch)
    monkeypatch.setattr(support.platform,'system',lambda:'Linux')
    monkeypatch.setattr(support.os.path,'ismount',lambda _:True)
    original_check=support.subprocess.check_output
    def mount(args,**kwargs):
        if args[0]=='findmnt':return str(tmp_path)+' source fuse rw'
        return original_check(args,**kwargs)
    monkeypatch.setattr(support.subprocess,'check_output',mount)
    monkeypatch.setattr(full,'valid_primary_bytes',lambda *args:0)
    monkeypatch.setattr(support,'verify_cached_models',lambda *args:{})
    monkeypatch.setattr(support,'token_storage_plan',lambda *args:dict(complete_copy_reserved_bytes=1024,
        second_copy_reserved_bytes=1024,temporary_shard_reserved_bytes=512))
    if failure=='publication':
        def fail(*args):raise OSError(errno.EIO,'publication failed')
        monkeypatch.setattr(store.os,'link',fail)
    elif failure=='overwrite':
        original=store.publish
        def unsafe(path,write):
            if Path(path).exists():return None  # broken exclusive-publication contract
            return original(path,write)
        monkeypatch.setattr(store,'publish',unsafe)
    elif failure=='fsync':
        def fail(*args):raise OSError(errno.EIO,'fsync failed')
        monkeypatch.setattr(store.os,'fsync',fail)
    elif failure=='hash_readback':
        original=store.pin
        def corrupt(path):
            value=original(path)
            return dict(value,sha256='0'*64) if Path(path).name=='reserved' else value
        monkeypatch.setattr(store,'pin',corrupt)
    else:
        def fail(*args):raise OSError(errno.EIO,'recovery failed')
        monkeypatch.setattr(store.os,'replace',fail)
    cache=tmp_path/'cache-root/cache';cache.mkdir(parents=True)
    receipt=tmp_path/'storage.json'
    with pytest.raises((ValueError,OSError)):
        support.storage_check(dict(storage=dict(safety_margin_bytes=512)),{},cache,tmp_path,tmp_path/'output',receipt)
    assert not receipt.exists()  # no passing storage proof and hence no execution


def test_storage_receipt_cannot_skip_mandatory_capability_validation():
    with pytest.raises(ValueError,match='non-secret filesystem contract'):
        support.check_storage_receipt(dict(filesystem_prerequisites=dict(schema='old',artifact_contract='private')),
            {},{},Path('/unused'),Path('/unused'),{})
