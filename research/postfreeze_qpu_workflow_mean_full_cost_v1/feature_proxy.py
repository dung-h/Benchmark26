"""Explicit two-environment pipeline; IPC is included in Clock C elapsed time."""
import sys,subprocess,pickle,struct
from pathlib import Path
import numpy as np
class Routes:
    def __init__(self):
        self.worker=subprocess.Popen(['/home/server/Documents/.venv-qonductor/bin/python',str(Path(__file__).with_name('feature_server.py'))],stdin=subprocess.PIPE,stdout=subprocess.PIPE)
    def call(self,op,args):
        data=pickle.dumps({'op':op,'args':args},protocol=4)
        self.worker.stdin.write(struct.pack('!Q',len(data))+data);self.worker.stdin.flush()
        header=self.worker.stdout.read(8)
        if len(header)!=8:raise RuntimeError('feature worker exited: '+str(self.worker.poll()))
        size=struct.unpack('!Q',header)[0];data=bytearray()
        while len(data)<size:
            block=self.worker.stdout.read(size-len(data))
            if not block:raise RuntimeError('feature worker response truncated')
            data.extend(block)
        result=pickle.loads(data)
        if not result['ok']:raise RuntimeError(result['error'])
        return self.decode(result['value'])
    def decode(self,x):
        if isinstance(x,dict):
            if x.get('__array__'):return np.frombuffer(x['bytes'],dtype=np.dtype(x['dtype'])).reshape(x['shape']).copy()
            return {k:self.decode(v) for k,v in x.items()}
        if isinstance(x,list):return [self.decode(v) for v in x]
        return x
    def get(self,*args):return self.call('get',args)
    def extract(self,*args):return self.call('extract',args)
    def close(self):
        if self.worker.poll() is None:self.worker.terminate();self.worker.wait(timeout=10)
    def __del__(self):
        try:self.close()
        except Exception:pass
