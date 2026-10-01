"""Pilot v2: verify and export both required layers in one sequential source read.

Stdlib only; no model execution, persistent remote writes or test-cache interface.
The original canonical exporter and the pilot's aborted seek-based v1 are preserved.
"""
import ast,hashlib,io,json,math,os,socket,stat,struct,sys,tarfile,csv
from pathlib import Path
ROOT=Path('/workspace/truth-probing-artifacts')
DIRECTORIES={
 'qwen_train':'atomic_repair/qwen25_a09a354_bs1_bf16_v1/train',
 'qwen_validation':'atomic_repair/qwen25_a09a354_bs1_bf16_v1/validation',
 'llama_train':'llama31_replication_v1/atomic/train',
 'llama_validation':'llama31_replication_v1/atomic/validation',
 'qwen_raw':'pinned_compound_qwen2_5/qwen2_5_7b_a09a354_bs1_bf16_v1',
 'llama_transfer':'llama31_replication_v1/transfer'}

def require(ok,message):
    if not ok:raise ValueError(message)
def sha(data):return hashlib.sha256(data).hexdigest()
def token(st):return (st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)
def header(handle):
    require(handle.read(6)==b'\x93NUMPY','not NPY');version=tuple(handle.read(2));require(version in ((1,0),(2,0)),'NPY version')
    width=2 if version==(1,0) else 4;n=struct.unpack('<H' if width==2 else '<I',handle.read(width))[0]
    require(0<n<=65536,'header length');v=ast.literal_eval(handle.read(n).decode('latin1').strip())
    require(set(v)=={'descr','shape','fortran_order'} and v['descr']=='<f2' and v['fortran_order'] is False,'storage policy')
    require(isinstance(v['shape'],tuple) and len(v['shape'])==3 and all(type(i)is int and i>0 for i in v['shape']),'tensor shape')
    return dict(shape=list(v['shape']),dtype='<f2',fortran_order=False,header_bytes=handle.tell(),computed_file_bytes=handle.tell()+2*math.prod(v['shape']),version=list(version))
def open_regular(path):
    path=Path(path);require(path.is_absolute() and path.is_relative_to(ROOT) and path.resolve()==path,'unsafe path')
    st=path.stat();require(stat.S_ISREG(st.st_mode),'regular file required')
    f=os.fdopen(os.open(path,os.O_RDONLY|os.O_NOFOLLOW),'rb');require(token(st)==token(os.fstat(f.fileno())),'changed before open')
    return f,st

def companion(path):
    f,st=open_regular(path)
    with f:
        data=f.read();require(token(st)==token(os.fstat(f.fileno()))==token(Path(path).stat()),'companion changed')
    return data,dict(path=str(path),bytes=len(data),sha256=sha(data),header=None,stable_during_read=True)

def output_header(rows,width):
    data=repr({'descr':'<f2','fortran_order':False,'shape':(rows,width)}).encode('latin1')
    data+=b' '*((64-(10+len(data)+1)%64)%64)+b'\n'
    return b'\x93NUMPY\x01\x00'+struct.pack('<H',len(data))+data

def stream_select(handle,segments,outputs,chunk_size=8*1024*1024):
    """Copy only selected byte spans while hashing every source byte once."""
    digest=hashlib.sha256();offset=0;cursor=0
    for block in iter(lambda:handle.read(chunk_size),b''):
        digest.update(block);end=offset+len(block)
        while cursor<len(segments) and segments[cursor][0]<end:
            start,stop,key=segments[cursor]
            lo=max(start,offset);hi=min(stop,end)
            if hi>lo:outputs[key].write(block[lo-offset:hi-offset])
            if stop<=end:cursor+=1
            else:break
        offset=end
    require(cursor==len(segments),'truncated source segments')
    return offset,digest.hexdigest()

def export_source(key,jobs):
    directory=ROOT/DIRECTORIES[key];first=next(iter(jobs.values()))
    layers=[18,22] if key.startswith('qwen') else [10,20]
    require(sorted(v['layer'] for v in jobs.values())==layers and len(jobs)==2,'exact additional layers required')
    expected_companions={str(directory/'metadata.csv')}
    expected_companions|=({str(directory.parent/n) for n in ('completion.json','extraction_manifest.json')}
                         if key in ('qwen_train','qwen_validation') else {str(directory/n) for n in ('extraction_manifest.json','progress.json')})
    require(set(first['companions'])==expected_companions and first['path']==str(directory/'activations.npy'),'source allowlist')
    checked={};metadata=None
    for path,pin in first['companions'].items():
        data,actual=companion(path);require(all(actual[k]==pin[k] for k in ('bytes','sha256')),'companion identity')
        checked[path]=actual
        if path.endswith('/metadata.csv'):metadata=list(csv.DictReader(io.StringIO(data.decode())))
    f,st=open_regular(first['path'])
    with f:
        meta=header(f);require(meta['shape']==first['shape'] and meta['computed_file_bytes']==st.st_size==first['bytes'],'source header/size')
        outputs={};segments=[]
        for name,spec in jobs.items():
            require(all(spec[k]==first[k] for k in ('path','bytes','sha256','shape','companions','row_indices','row_identity_sha256')),'mixed source jobs')
            rows=spec['row_indices'];require(rows and rows==sorted(set(rows)) and all(type(i)is int and 0<=i<len(metadata)==meta['shape'][0] for i in rows),'source row indices')
            for i in rows:
                require(metadata[i]['split']==('train' if key.endswith('train') else 'validation'),'nondevelopment source row')
                if key in ('qwen_raw','llama_transfer'):
                    require(metadata[i].get('condition_id','raw_reference')=='raw_reference' and metadata[i]['evaluation_phase']=='development','nonbare/nondevelopment row')
            outputs[name]=io.BytesIO(output_header(len(rows),meta['shape'][2]))
            outputs[name].seek(0,2)
            for row in rows:
                begin=meta['header_bytes']+(row*meta['shape'][1]+spec['layer'])*meta['shape'][2]*2
                segments.append((begin,begin+meta['shape'][2]*2,name))
        segments.sort();f.seek(0)
        size,digest=stream_select(f,segments,outputs)
        require(size==first['bytes'] and digest==first['sha256'],'tensor SHA/size mismatch')
        require(token(st)==token(os.fstat(f.fileno()))==token(Path(first['path']).stat()),'source changed during stream')
    require(checked=={path:companion(path)[1] for path in checked},'companion changed through stream')
    measured=dict(path=first['path'],bytes=size,sha256=digest,header=meta,stable_during_read=True)
    result={}
    for name,spec in jobs.items():
        data=outputs[name].getvalue()
        result[name]=(data,dict(status='verified',source=measured,companions=checked,original_row_indices=spec['row_indices'],
            saved_layer_index=spec['layer'],row_identity_sha256=spec['row_identity_sha256'],source_stable_through_export=True,
            export=dict(bytes=len(data),sha256=sha(data),shape=[len(spec['row_indices']),meta['shape'][2]],dtype='<f2',fortran_order=False)))
    return result

def main():
    require(socket.gethostname()=='ef7f7534c328','unexpected remote hostname')
    data=sys.stdin.buffer.read();request=json.loads(data)
    expected={k+'_L'+str(l) for k in DIRECTORIES for l in ([18,22] if k.startswith('qwen') else [10,20])}
    require(set(request['caches'])==expected,'exact twelve exports')
    receipt=dict(remote_hostname=socket.gethostname(),request_sha256=sha(data),caches={},model_extraction=False,final_test_access=False,remote_files_written=False,algorithm='one sequential SHA-256 read per source; selected byte spans only')
    with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as archive:
        for key in sorted(DIRECTORIES):
            jobs={name:spec for name,spec in request['caches'].items() if name.rsplit('_L',1)[0]==key}
            try:
                for name,(payload,record) in export_source(key,jobs).items():
                    info=tarfile.TarInfo(name+'.npy');info.size=len(payload);archive.addfile(info,io.BytesIO(payload));receipt['caches'][name]=record
            except (OSError,ValueError,SyntaxError,struct.error) as exc:
                for name in jobs:receipt['caches'][name]=dict(status='not_computed',dependency_error=str(exc))
        payload=(json.dumps(receipt,sort_keys=True,indent=2)+'\n').encode();info=tarfile.TarInfo('export_receipt.json');info.size=len(payload);archive.addfile(info,io.BytesIO(payload))
if __name__=='__main__':main()
