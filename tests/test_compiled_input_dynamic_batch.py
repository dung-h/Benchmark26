import copy
import pytest
torch = pytest.importorskip('torch')
from experiments.compiled_input.dynamic_batch import partition,train_step


def test_partition_preserves_order_and_oversize_singleton():
    assert list(partition([0,1,2,3],{0:(2,2),1:(100,1),2:(2,2),3:(3,2)},5,4)) == [[0],[1],[2,3]]


def test_accumulation_matches_full_and_oom_retry():
    torch.manual_seed(12)
    a=torch.nn.Linear(3,1); b=copy.deepcopy(a); oa=torch.optim.SGD(a.parameters(),lr=.1); ob=torch.optim.SGD(b.parameters(),lr=.1)
    x=torch.randn(5,3); y=torch.randn(5,1); sizes={i:(1,1) for i in range(5)}
    oa.zero_grad(); ((a(x)-y)**2).mean().backward();oa.step()
    hit=[False]
    def loss(ids):
        if len(ids)>1 and not hit[0]:
            hit[0]=True;raise torch.cuda.OutOfMemoryError('simulated')
        return ((b(x[ids])-y[ids])**2).sum()
    report=train_step(b,ob,list(range(5)),sizes,5,5,loss,torch)
    assert report['retries']==1
    for p,q in zip(a.parameters(),b.parameters()):assert torch.allclose(p,q,atol=1e-7)
