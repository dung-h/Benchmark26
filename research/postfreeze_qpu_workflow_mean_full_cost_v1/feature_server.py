"""Persistent CPU feature worker in the original Qiskit2.5.2 environment."""
import sys,struct,pickle,ctypes,signal,os
ctypes.CDLL(None).prctl(1,signal.SIGTERM)
os.environ.update({k:'1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')})
import numpy as np
from circuit_routes import Routes
routes=Routes()
def pack(x):
    if isinstance(x,np.ndarray):return {'__array__':True,'dtype':x.dtype.str,'shape':x.shape,'bytes':x.tobytes()}
    if isinstance(x,dict):return {k:pack(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [pack(v) for v in x]
    return x
while True:
    header=sys.stdin.buffer.read(8)
    if not header:break
    size=struct.unpack('!Q',header)[0];raw=sys.stdin.buffer.read(size)
    try:
        req=pickle.loads(raw)
        value=routes.get(*req['args']) if req['op']=='get' else routes.extract(*req['args'])
        response={'ok':True,'value':pack(value)}
    except Exception as e:response={'ok':False,'error':repr(e)}
    data=pickle.dumps(response,protocol=4);sys.stdout.buffer.write(struct.pack('!Q',len(data))+data);sys.stdout.buffer.flush()
