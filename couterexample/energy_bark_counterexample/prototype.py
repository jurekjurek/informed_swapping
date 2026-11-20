import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigsh, expm_multiply
from scipy.linalg import eigh

def make_h(m,U=None):
    U=20*m*m if U is None else U
    n=1<<m
    a,y,d,c=range(n,n+4)
    H=lil_matrix((n+4,n+4),dtype=float)
    def edge(i,j,v): H[i,j]=v; H[j,i]=v
    for z in range(n):
        H[z,z]=U*z.bit_count()
        for j in range(m):
            other=z^(1<<j)
            if z<other:edge(z,other,-1)
    edge(0,a,-2)
    for j in range(m):edge(1<<j,a,-1)
    edge(0,y,1.8)
    edge(a,y,-1.8)
    edge(y,d,-1)
    edge(d,c,-1)
    H[c,c]=-2.2
    return H.tocsr(), {'s':0,'a':a,'y':y,'d':d,'c':c}

def lambda_min(E,diag,r):
    return 0.5*(E+diag-np.hypot(E-diag,2*r))

def run_bark(H, m, max_steps=None):
    N=H.shape[0]
    v=np.zeros(N);v[0]=1.
    S={0}; selected=[0]; scores={}; E=0.; last=0
    track=[]
    if max_steps is None: max_steps=N-1
    while len(selected)<=max_steps:
        for x in H.indices[H.indptr[last]:H.indptr[last+1]]:
            x=int(x)
            if x in S or x==last:continue
            r=float(H[x,:]@v)
            score=lambda_min(E,float(H[x,x]),r)
            scores[x]=min(scores.get(x,float('inf')),score)
        if not scores: break
        x=min(scores,key=lambda x:(scores[x],x))
        oldscore=scores.pop(x)
        r=float(H[x,:]@v)
        mat=np.array([[E,r],[r,float(H[x,x])]])
        l,vec=eigh(mat)
        vec0=vec[:,0]
        if vec0[0]<0:vec0=-vec0
        v*=vec0[0];v[x]=vec0[1]
        E=float(l[0]); S.add(x);selected.append(x);last=x
        track.append((x,E,oldscore,r,float(v[x])))
        if x>=1<<m: # report first special(s)
            if len(selected)<15: print('special selected',m,len(selected),x,(x-(1<<m)),r)
    return selected,track

if __name__=='__main__':
 for m in range(2,9):
    H,ids=make_h(m)
    selected,track=run_bark(H,m)
    print('m',m,'first a/y/d/c',{k:selected.index(v)+1 if v in selected else None for k,v in ids.items()},'n',H.shape[0], 'num',len(selected))
    if m<=6:
      w,v=eigsh(H,k=2,which='SA',tol=1e-9)
      gs=v[:,np.argmin(w)];w=np.sort(w)
      print('   gap',w[1]-w[0],'overlap',gs[0]**2,'support core',sum(gs[v]**2 for v in ids.values()))
      if m==6:
       for t in [0.5,1,1.5,2.0]:
         psi=expm_multiply(-1j*H*t,np.eye(1,H.shape[0],0,dtype=complex).ravel())
         print('   t',t,{key:round(abs(psi[id])**2,6) for key,id in ids.items()})
