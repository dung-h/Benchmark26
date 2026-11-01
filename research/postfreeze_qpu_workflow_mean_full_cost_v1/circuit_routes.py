"""Target-free reconstruction of exact existing input stages, no quantum execution."""
import csv,json,sys,zipfile,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'benchmark_v1/scripts'))
import mali_full_features as mf
# Load only the two pure reconstruction functions from the exact producing
# source. Importing its CLI would unnecessarily require an IBM provider.
import ast
from types import SimpleNamespace
from qiskit import QuantumCircuit
_recovery_source=ROOT/'benchmark_v1/scripts/run_recovery_c133_c135_v3.py'
_tree=ast.parse(_recovery_source.read_text())
_scope={'QuantumCircuit':QuantumCircuit}
_nodes=[n for n in _tree.body if isinstance(n,ast.FunctionDef) and n.name in {'qpack_circuit','regular_graph_edges'}]
exec(compile(ast.fix_missing_locations(ast.Module(body=_nodes,type_ignores=[])),str(_recovery_source),'exec'),_scope)
c135=SimpleNamespace(MALI_QASM_ROOT=ROOT.parent/'Quantum-Execution-Time-Prediction/data/quantum_circuits',qpack_circuit=_scope['qpack_circuit'])
from qiskit import qasm2,qasm3

def read(path,columns=None):
    with Path(path).open(newline='') as f:
        r=csv.reader(f);head=next(r);cols=columns or head;indices=[head.index(c) for c in cols]
        return [dict(zip(cols,(row[i] for i in indices))) for row in r]

def digest(raw):return hashlib.sha256(raw).hexdigest()
class Routes:
    def __init__(self):
        cols=['canonical_row_id','source_id','qasm_path_or_member','qasm_bytes_sha256','shots']
        self.canonical={r['canonical_row_id']:r for r in read(mf.CANONICAL,cols)}
        self.rep={r['canonical_row_id']:r for r in read(mf.REPRESENTATION)}
        self.attempts={r['canonical_observation_id']:r for r in read(ROOT/'artifacts/real_qpu/logical_inputs/feature_attempts.csv')}
        p=ROOT/'artifacts/real_qpu/logical_recovery/candidate_recipe_manifest.json'
        self.candidates={r['qasm_sha256']:p.parent/r['path'] for r in json.loads(p.read_text())['records']}
        self.registry=mf.load_snapshot_registry();self.extractor,_=mf.load_pinned_global_extractor()
        self.archive=zipfile.ZipFile(mf.QONDUCTOR_ZIP)
        from qonductor_native_adapter import load_pinned_qonductor_methods
        contract=json.loads((ROOT/'benchmark_v1/protocol/qonductor_native_features.json').read_text())
        self.native_extractor,_,self.native_parser,_=load_pinned_qonductor_methods(ROOT.parent/'Qonductor-SC25',contract['source_pin'])
        from qasm3_hardware_wire_adapter import restore_saved_qasm3_hardware_wires
        self.restore=restore_saved_qasm3_hardware_wires
    def get(self,panel,ledger,method):
        fid=panel['feature_source_observation_id'];rep=self.rep[fid];source=panel['source_id']
        if method=='C0':return {'stage':'no_circuit_required','raw':b'','shots':float(panel['shots'])}
        if method=='R0':
            raw,h,meta=mf.qasm_bytes_for(self.canonical[fid],rep,self.archive)
            return {'stage':'exact_submitted_physical' if source=='qonductor_single_circuit_ibm' else 'saved_nominal_compiled','raw':raw,'shots':float(panel['shots']),'width':int(rep['register_width']),'sha256':h}
        expected=ledger['source_qasm_sha256'] if 'source_qasm_sha256' in ledger else panel['feature_source_qasm_sha256']
        if source=='mali_real_qpu':raw=(c135.MALI_QASM_ROOT/self.canonical[fid]['qasm_path_or_member']).read_bytes();stage='source_logical'
        elif source=='qpack_mcp':
            recipe=json.loads(rep['source_recipe_json']);raw=qasm3.dumps(c135.qpack_circuit(int(recipe['size']),int(recipe['p']))).encode();stage='frozen_workflow_reconstruction'
        else:
            h=self.attempts.get(fid,{}).get('candidate_or_recipe_hash',expected)
            if h in self.candidates:raw=self.candidates[h].read_bytes()
            else:raw=(ROOT/'artifacts/postfreeze_qonductor_generator_version_semantics_v1/generated/1.0.3/graphstate__q3.qasm').read_bytes()
            stage='qualified_logical_reconstruction'
        if digest(raw)!=expected:raise RuntimeError('logical_byte_hash_mismatch:'+fid)
        calibration,_=mf.snapshot_t1_t2(rep,self.registry)
        return {'stage':stage,'raw':raw,'shots':float(panel['shots']),'calibration':calibration,'sha256':expected}
    def extract(self,route,method):
        if method=='C0':return {}
        raw=route['raw']
        if method=='R0':
            if route['stage']=='exact_submitted_physical':c=self.native_parser(raw.decode())
            else:
                text,_=self.restore(raw.decode(),route['width']);c=qasm3.loads(text)
            return {'native5':np.asarray(self.native_extractor._extract_features([c],int(route['shots'])).tolist()[0],dtype=np.float64)}
        if raw.lstrip().startswith(b'OPENQASM 3'):c=qasm3.loads(raw.decode())
        else:c=qasm2.loads(raw.decode(),custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
        out={'globals':np.asarray(mf.extract_global_features(c,self.extractor),dtype=np.float32)}
        if method.startswith('G'):out['graph_record']=mf.compact_graph_record(c,route['calibration'])[0]
        if method.endswith('1'):out['shots']=route['shots']
        return out
