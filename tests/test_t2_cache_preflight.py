"""Synthetic row identity and read-only physical-check regressions; no live SSH."""
import copy
import hashlib
import io
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src import t2_cache_remote as remote
from src.t2_cache_preflight import typed_rows, verify_correspondence, remote_preflight


def rows():
    return [dict(dataset='cities',row_index=str(i),statement='The city of A is in B.',entity_id='entity_a',
                 topic='cities',form='affirmative',split='train',label='1') for i in [2,7]]


def originals(typed):
    return {(r['dataset'],r['row_index']):dict(row=r,source_sha256='a'*64,
        statement_sha256=hashlib.sha256(r['statement'].encode()).hexdigest()) for r in typed}


def test_duplicate_text_keeps_distinct_source_rows():
    typed=typed_rows(rows(),'train'); source=originals(typed)
    receipt=verify_correspondence(typed,source,set(source))
    assert receipt['unique_source_row_ids']==2
    assert receipt['duplicate_statement_groups']==1 and receipt['duplicate_statement_rows']==2
    assert receipt['final_selection_map_created'] is False
    assert receipt['admitted_rows_or_person_keys_used'] is False
    assert 'person_key' not in receipt['correspondence_fields']
    reverse=verify_correspondence(list(reversed(typed)),source,set(source))
    assert receipt['preliminary_correspondence_sha256']!=reverse['preliminary_correspondence_sha256']


@pytest.mark.parametrize('column,value',[('statement','changed text'),('label',0),('entity_id','other'),('split','validation')])
def test_exact_source_binding_rejects_changes(column,value):
    typed=typed_rows(rows(),'train');source=originals(copy.deepcopy(typed))
    typed[0][column]=value
    with pytest.raises(ValueError,match='source statement'): verify_correspondence(typed,source,set(source))


def test_source_coverage_and_unique_ids():
    typed=typed_rows(rows(),'train');source=originals(typed)
    with pytest.raises(ValueError,match='coverage'): verify_correspondence(typed[:1],source,set(source))
    bad=rows();bad[1]['row_index']='2'
    with pytest.raises(ValueError,match='duplicate'):typed_rows(bad,'train')
    bad=rows();bad[0]['split']='test'
    with pytest.raises(ValueError,match='partition'):typed_rows(bad)


def test_llama_example_id_must_be_source_identity():
    raw=[dict(r,example_id=r['dataset']+':'+r['row_index']) for r in rows()]
    assert len(typed_rows(raw,'train',True))==2
    raw[0]['example_id']='cities:7'
    with pytest.raises(ValueError,match='Llama example'):typed_rows(raw,'train',True)


def npy(dtype='<f2',fortran=False):
    stream=io.BytesIO();values=np.zeros((2,3,4),dtype=dtype)
    if fortran:values=np.asfortranarray(values)
    np.save(stream,values,allow_pickle=False);return stream.getvalue()


def test_header_only_shape_dtype_and_payload_size():
    payload=npy();stream=io.BytesIO(payload);header=remote.npy_header(stream)
    assert header['shape']==[2,3,4] and header['dtype']=='<f2'
    assert stream.tell()==header['header_bytes']<len(payload)
    assert header['computed_file_bytes']==len(payload)
    for bad in [npy('<f4'),npy(fortran=True),b'badmagic',payload[:20]]:
        with pytest.raises((ValueError,struct.error,SyntaxError)):
            remote.npy_header(io.BytesIO(bad))


def test_fixed_allowlist_and_hash_not_counts(tmp_path,monkeypatch):
    monkeypatch.setattr(remote,'ROOT',tmp_path)
    directories={key:tmp_path/key for key in ['qwen/train','qwen/validation','llama/train','llama/validation']}
    monkeypatch.setattr(remote,'CACHES',directories)
    requests={}
    for key,directory in directories.items():
        directory.mkdir(parents=True)
        (directory/'activations.npy').write_bytes(npy())
        companions=[directory/'metadata.csv']
        companions+=([directory.parent/n for n in ['completion.json','extraction_manifest.json']] if key.startswith('qwen/') else
                     [directory/n for n in ['progress.json','extraction_manifest.json']])
        for path in companions:path.write_text('synthetic sidecar')
        requests[key]=dict(path=str(directory/'activations.npy'),bytes=len(npy()),sha256=hashlib.sha256(npy()).hexdigest(),shape=[2,3,4],
            companions={str(path):dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in companions})
    assert all(r['status']=='verified' for r in remote.verify(requests).values())
    bad=copy.deepcopy(requests);bad['qwen/train']['sha256']='0'*64
    assert remote.verify(bad)['qwen/train']['status']=='conflict'
    bad=copy.deepcopy(requests);bad['qwen/train']['path']=str(tmp_path/'qwen/test/activations.npy')
    assert remote.verify(bad)['qwen/train']['status']=='conflict'
    (directories['llama/train']/'metadata.csv').write_text('changed')
    assert remote.verify(requests)['llama/train']['status']=='conflict'
    with pytest.raises(ValueError,match='four atomic'):remote.verify({'test':{}})


def test_measure_rejects_links_and_truncated_array(tmp_path,monkeypatch):
    monkeypatch.setattr(remote,'ROOT',tmp_path)
    path=tmp_path/'data.npy';path.write_bytes(npy()[:-1])
    with pytest.raises(ValueError,match='payload size'):remote.measure(path,True)
    link=tmp_path/'alias.npy';link.symlink_to(path)
    with pytest.raises(ValueError,match='linked'):remote.measure(link,True)


def test_ssh_failure_is_pending_not_success(tmp_path,monkeypatch):
    program=tmp_path/'remote.py';program.write_text('print("fixture")')
    def fake(command,**kwargs):
        assert command[0]=='ssh' and '-T' in command and 'StrictHostKeyChecking=yes' in command
        assert kwargs['capture_output'] is True
        return SimpleNamespace(returncode=255,stderr='Permission denied (publickey,password).',stdout='')
    monkeypatch.setattr('src.t2_cache_preflight.subprocess.run',fake)
    result=remote_preflight({},program,Path('~/.ssh/id_ed25519'))
    assert result['status']=='pending' and result['cache_results']=={}
