#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

// Resource dependencies are exactly the operation-order last-use wire edges
// of the saved s71 DAG. Measurements remain in the full timing path.
extern "C" int paths(int n, int nq, const uint8_t* arity, const int32_t* q0,
                     const int32_t* q1, const uint8_t* measure,
                     const double* durations, double* out) {
    std::vector<int> one(nq,0), two(nq,0), level(nq,0);
    std::vector<double> full(nq,0), unitary(nq,0);
    std::vector<uint8_t> has_two(n+1,0), measured(nq,0);
    int d1=0,d2=0,depth=0,measurements=0;
    double cp_full=0,cp_unitary=0;
    for (int i=0;i<n;++i) {
        int a=arity[i],x=q0[i],y=q1[i];
        if ((a!=1 && a!=2) || x<0 || x>=nq || (a==2 && (y<0 || y>=nq || y==x))) return 1;
        if (!std::isfinite(durations[i]) || durations[i]<0) return 2;
        double f=std::max(full[x],a==2?full[y]:0.0)+durations[i];
        full[x]=f; if(a==2) full[y]=f;
        cp_full=std::max(cp_full,f);
        if(measure[i]) {if(a!=1)return 3;measured[x]=1;++measurements;continue;}
        if(measured[x] || (a==2 && measured[y])) return 4;
        int u=std::max(one[x],a==2?one[y]:0)+(a==1);
        int v=std::max(two[x],a==2?two[y]:0)+(a==2);
        int l=std::max(level[x],a==2?level[y]:0)+1;
        double g=std::max(unitary[x],a==2?unitary[y]:0.0)+durations[i];
        one[x]=u;two[x]=v;level[x]=l;unitary[x]=g;
        if(a==2){one[y]=u;two[y]=v;level[y]=l;unitary[y]=g;has_two[l]=1;}
        d1=std::max(d1,u);d2=std::max(d2,v);depth=std::max(depth,l);
        cp_unitary=std::max(cp_unitary,g);
    }
    int k2=0;for(int l=1;l<=depth;++l)k2+=has_two[l];
    double values[]={double(d1),double(d2),double(depth),double(depth-k2),double(k2),
                     double(d1+10*d2),double(depth+9*k2),cp_full,cp_unitary,double(measurements)};
    std::copy(values,values+10,out);
    return 0;
}
