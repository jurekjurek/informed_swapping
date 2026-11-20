"""Sparse exact SKQD-vs-incremental-energy-BARK counterexample.

Requires numpy, scipy, matplotlib. Run:
  python energy_bark_counterexample.py --m 6 --shots 200
No repository files are modified. Active Hamiltonian size is (2**m+4),
which can be embedded into m+1 qubits by padding with decoupled states.
"""
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import expm_multiply, eigsh

COUPLING_ROOT = 2.0
COUPLING_GATEWAY = 1.8
H_TARGET_C = -2.2

def hamiltonian(m, U=None):
    if m < 2:
        raise ValueError('m must be >= 2')
    U = 20.0*m*m if U is None else float(U)
    M=1<<m
    a,y,d,c=range(M,M+4)
    H=lil_matrix((M+4,M+4),dtype=float)
    def connect(i,j,v):
        H[i,j]=v
        H[j,i]=v
    for z in range(M):
        H[z,z]=U*z.bit_count()
        for i in range(m):
            nz=z ^ (1<<i)
            if z<nz:
                connect(z,nz,-1.0)
    connect(0,a,-COUPLING_ROOT)
    for i in range(m):
        connect(a,1<<i,-1.0)
    connect(0,y,+COUPLING_GATEWAY)
    connect(a,y,-COUPLING_GATEWAY)
    connect(y,d,-1.0)
    connect(d,c,-1.0)
    H[c,c]=H_TARGET_C
    return H.tocsr(), dict(s=0,a=a,y=y,d=d,c=c)

def exact_ground_state(m,dim,ids):
    # Target antisymmetric subspace: (|s>-|a>)/sqrt(2), |y>, |d>, |c>.
    A=np.array([[2.,np.sqrt(2.)*COUPLING_GATEWAY,0,0],
                [np.sqrt(2.)*COUPLING_GATEWAY,0,-1,0],
                [0,-1,0,-1],
                [0,0,-1,H_TARGET_C]],dtype=float)
    ev,evec=eigh(A)
    u=evec[:,0]
    gs=np.zeros(dim,dtype=float)
    gs[ids['s']]=u[0]/np.sqrt(2.)
    gs[ids['a']]=-u[0]/np.sqrt(2.)
    gs[ids['y']]=u[1]
    gs[ids['d']]=u[2]
    gs[ids['c']]=u[3]
    return float(ev[0]),gs

def lowest_two_state(E,c,r):
    ev,evec=eigh(np.array([[E,r],[r,c]],dtype=float))
    w=evec[:,0]
    if w[0]<0:
        w=-w
    return float(ev[0]),w

def energy_bark(H):
    N=H.shape[0]
    v=np.zeros(N,dtype=float);v[0]=1.0
    E=0.0
    selected=[0]; seen={0}; last=0
    memory={}
    while len(selected)<N:
        # Score only new neighbors of the last selected vertex; remember best scores.
        for index in range(H.indptr[last],H.indptr[last+1]):
            x=int(H.indices[index])
            if x==last or x in seen:continue
            r=float(np.dot(H.getrow(x).toarray().ravel(),v))
            candidate_E,_=lowest_two_state(E,float(H[x,x]),r)
            memory[x]=min(memory.get(x,float('inf')),candidate_E)
        if not memory: break
        x=min(memory,key=lambda key:(memory[key],key))
        del memory[x]
        r=float(np.dot(H.getrow(x).toarray().ravel(),v))
        E,w=lowest_two_state(E,float(H[x,x]),r)
        v*=w[0]
        v[x]=w[1]
        selected.append(x);seen.add(x);last=x
    return selected

def projected_fidelity(H,gs,indices):
    selected=np.array(indices,dtype=int)
    H_sub=H[selected,:][:,selected].toarray()
    vals,vecs=eigh(H_sub,subset_by_index=[0,0])
    fidelity=float(abs(np.vdot(gs[selected],vecs[:,0]))**2)
    return max(0.,min(1.,fidelity)),float(vals[0])

def skqd(H,m,shots,seed,dt=1.5):
    init=np.zeros(H.shape[0],dtype=complex);init[0]=1
    evolved=expm_multiply((-1j*dt)*H,init)
    prob=np.abs(evolved)**2
    prob=prob/prob.sum()
    rng=np.random.default_rng(seed)
    samples=rng.choice(len(prob),size=shots,p=prob)
    selected=[0];seen={0}
    timepoints=[0]
    for shot,x in enumerate(samples,start=1):
        if int(x) not in seen:
            seen.add(int(x));selected.append(int(x));timepoints.append(shot)
    return selected,timepoints,prob

def savecsv(path,columns,rows):
    with open(path,'w',newline='') as f:
        w=csv.writer(f);w.writerow(columns);w.writerows(rows)

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--m',type=int,default=6)
    p.add_argument('--shots',type=int,default=200)
    p.add_argument('--seed',type=int,default=13)
    p.add_argument('--out',type=Path,default=Path(__file__).parent/'results')
    args=p.parse_args()
    args.out.mkdir(exist_ok=True,parents=True)

    scaling=[]
    for m in range(2,9):
        H,ids=hamiltonian(m)
        order=energy_bark(H)
        y_pos=order.index(ids['y'])+1
        scaling.append((m,2**m,y_pos))
        print(f'm={m}: cube vertices {2**m}, gateway y selected at D={y_pos}')
    savecsv(args.out/'bark_scaling.csv',['m','distracting_vertices','first_gateway_dimension'],scaling)

    m=args.m
    H,ids=hamiltonian(m)
    ground_E,gs=exact_ground_state(m,H.shape[0],ids)
    spectrum=eigsh(H,k=2,which='SA',return_eigenvectors=False,tol=1e-9)
    gap=float(np.diff(np.sort(spectrum))[0])
    if abs(ground_E-np.min(spectrum))>1e-8:
        raise RuntimeError('Target-sector eigenvector is not the global ground state')
    bark_order=energy_bark(H)
    skqd_order,shot_at,prob=skqd(H,m,args.shots,args.seed)
    important=[ids[k] for k in ['s','a','y','d','c']]
    important_found=all(x in skqd_order for x in important)
    print(f'For m={m}: E0={ground_E:.8f}, gap={gap:.5f}, ground-state overlap={gs[0]**2:.5f}')
    print('SKQD t=1.5 sampling probabilities:',{k:round(float(prob[v]),6) for k,v in ids.items()})
    print(f'SKQD {args.shots} shots: {len(skqd_order)} distinct states, all five important? {important_found}')
    print(f'Energy-BARK: first y at D={bark_order.index(ids["y"])+1}, c at D={bark_order.index(ids["c"])+1}')

    bark_rows=[]
    for D in range(1,len(bark_order)+1):
        F,E=projected_fidelity(H,gs,bark_order[:D])
        bark_rows.append((D,F,E-ground_E))
    skqd_rows=[]
    for D in range(1,len(skqd_order)+1):
        F,E=projected_fidelity(H,gs,skqd_order[:D])
        skqd_rows.append((D,shot_at[D-1],F,E-ground_E))
    savecsv(args.out/'bark_fidelity.csv',['D','fidelity','energy_error'],bark_rows)
    savecsv(args.out/'skqd_fidelity.csv',['D','cumulative_shots','fidelity','energy_error'],skqd_rows)

    fig,ax=plt.subplots(figsize=(7.6,4.2))
    ax.plot([r[0] for r in bark_rows],[r[1] for r in bark_rows],label='Incremental energy-BARK')
    ax.plot([r[0] for r in skqd_rows],[r[2] for r in skqd_rows],'-o',ms=4,label=f'SKQD ($t=1.5$, {args.shots} shots)')
    ax.axvline(bark_order.index(ids['y'])+1,ls='--',lw=1,label='BARK first reaches gateway')
    ax.set(xlabel='Number of selected bitstrings $D$',ylabel='Projected ground-state fidelity',ylim=(-0.04,1.05),title=f'Exact sparse-ground-state counterexample ($m={m}$)')
    ax.grid(alpha=0.2);ax.legend(fontsize=9)
    fig.tight_layout();fig.savefig(args.out/'fidelity_vs_subspace.png',dpi=180);plt.close(fig)

    fig,ax=plt.subplots(figsize=(6.9,4.1))
    ax.plot([r[0] for r in scaling],[r[2] for r in scaling],'-o',label='First gateway selection (energy-BARK)')
    ax.plot([r[0] for r in scaling],[r[1] for r in scaling],'--',label='$2^m$')
    ax.set(xlabel='Number of distracting qubits $m$',ylabel='Selected configurations',yscale='log',title='Exponential discovery delay')
    ax.grid(alpha=0.25);ax.legend(fontsize=9)
    fig.tight_layout();fig.savefig(args.out/'exponential_delay.png',dpi=180);plt.close(fig)

if __name__=='__main__':main()
