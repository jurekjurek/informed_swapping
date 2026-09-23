import sys, heapq, warnings, numpy as np, pandas as pd
sys.path.insert(0, ".")
warnings.filterwarnings("ignore")
from EqualNumberOfBitstrings import Cell, cell_seed, budget_grid, with_initial_point, OVERLAP_FLOOR
from RandomSpinModel import make_heisenberg_hamiltonian
from ClusterStudy import solve_ground_state
from BARK import BARK
from SKQD import SKQD, fidelity_at_budget
from Subspace import lowest_eigenpair

def johann_consistent_run(b, gs, init, max_pool):
    """full_bark_run with energy taken from Johann's update, no ED feedback."""
    proj = b.projection; proj.reset(); proj.extend([init])
    pool = {init}; v = np.zeros(b.dimension, complex); v[init]=1
    E = float(b._diagonal[init]); memory={}; q=[]; last=init; sizes=[]; fids=[]
    while proj.size < max_pool:
        cand=[s for s in b.apply_hamiltonian(last) if s not in pool]
        pots,_,_ = b.rank_states(v,E,cand)
        for s,p in zip(cand,pots):
            s=int(s);p=float(p)
            if s not in memory or p<memory[s]:
                memory[s]=p; heapq.heappush(q,(p,s))
        last=None
        while q:
            p,s=heapq.heappop(q)
            if memory.get(s)==p: del memory[s]; last=s; break
        if last is None: break
        ne,a,bb = b.rank_states(v,E,[last])
        v = a[0]*v; v[last]+=bb[0]; E=float(ne[0])
        pool.add(last); proj.extend([last])
        _,g = lowest_eigenpair(proj.block)
        sizes.append(proj.size); fids.append(abs(np.vdot(g, gs[proj.pool]))**2)
    return np.array(sizes), np.array(fids)

def skqd_no_early_stop(s, init, t, shots, gs, max_pool, patience=50):
    proj=s.projection; proj.reset(); proj.extend([init])
    psi=np.zeros(s.dimension,complex); psi[init]=1; sizes=[]; stall=0; it=0
    while proj.size<max_pool and it<20000:
        it+=1
        psi=s.evolve(psi,t); p=np.abs(psi)**2; c=np.cumsum(p)
        idx=np.minimum(np.searchsorted(c,np.random.random_sample(shots)*c[-1],side="right"),s.dimension-1)
        fresh=proj.extend(idx)
        if fresh.size==0:
            stall+=1
            if stall>=patience: break
            continue
        stall=0; sizes.append(proj.size)
    sizes=np.array(sizes); fids=[]
    for k in sizes:
        _,g=lowest_eigenpair(proj.prefix_block(int(k))); fids.append(abs(np.vdot(g,gs[proj.pool[:k]]))**2)
    return sizes, np.array(fids)

def study(ham, N, dim, delta, J, Bz, init):
    c = Cell(ham, N, dim, float(delta), float(J), 0.0, 0.0, float(Bz))
    seed=cell_seed(c); np.random.seed(seed)
    H = make_heisenberg_hamiltonian(num_sites=N, dimension=dim, delta=delta, J=J, h=(0,0,Bz), spin=1)[0].to_matrix(sparse=True)
    gs, ev, evec = solve_ground_state(H, 4096); prob=abs(gs)**2
    D=H.shape[0]; budgets=budget_grid(D,30,0.5); maxp=int(budgets[-1])
    b=BARK(H); s=SKQD(H,eigenvalues=ev,eigenvectors=evec)
    ov=float(prob[init])
    # sector size
    nup=bin(init).count("1"); from math import comb
    print(f"\n=== ham={ham} N={N} dim={dim} delta={delta} J={J} Bz={Bz} init={init}  sector dim={comb(N,nup)}  gap={ev[1]-ev[0]:.3e}")
    offd = np.unique(np.round(np.abs(H[init].toarray().ravel()),12)); print(" |H[init,:]| distinct values:", offd)
    res={}
    ps,fs=b.full_bark_run(gs,init,max_pool_size=maxp); res["BARK (as is)"]=fidelity_at_budget(*with_initial_point(ps,fs,ov),budgets)
    b2=BARK(H, warm_start_threshold=10**9); ps2,fs2=b2.full_bark_run(gs,init,max_pool_size=maxp)
    print(" warm-start vs exact-ED BARK: identical pools?", np.array_equal(b.projection.pool, b2.projection.pool) if len(ps)==len(ps2) else "different lengths", " max |dF| =", np.max(np.abs(fs-fs2)) if len(fs)==len(fs2) else None)
    ps,fs=johann_consistent_run(b,gs,init,maxp); res["BARK (Johann energy)"]=fidelity_at_budget(*with_initial_point(ps,fs,ov),budgets)
    ps,fs=b.simplified_bark_run(gs,init,max_pool_size=maxp); res["Simpl. BARK"]=fidelity_at_budget(*with_initial_point(ps,fs,ov),budgets)
    # SKQD optimizer grid
    tv=(0.1,0.25,0.5,1.0,2.0,4.0); sv=(10,25,50,100)
    grid={}
    for t in tv:
        for n in sv:
            grid[(t,n)]=np.mean([fidelity_at_budget(*s.full_skqd_run(init,t,n,gs,max_pool_size=maxp,budgets=[maxp]),maxp) for _ in range(3)])
    best=max(grid,key=grid.get); top=max(grid.values())
    ties=[k for k,v in grid.items() if v>top-1e-10]
    print(f" SKQD optimize_general: best={best} F={top:.12f}; grid points within 1e-10 of best: {len(ties)}/24")
    t,n=s.optimize_general(init,gs,max_pool_size=maxp,n_repeats=3); print(" optimize_general chose", (t,n))
    runs=[s.full_skqd_run(init,t,n,gs,max_pool_size=maxp,budgets=budgets) for _ in range(3)]
    res["SKQD (as is)"]=np.mean([fidelity_at_budget(*with_initial_point(p,f,ov),budgets) for p,f in runs],axis=0)
    print(" SKQD final pools (as is):", [int(p[-1]) for p,_ in runs])
    # SKQD tuned per budget (oracle envelope) and without early stop
    runs=[skqd_no_early_stop(s,init,t,n,gs,maxp) for _ in range(3)]
    res["SKQD (no early stop)"]=np.mean([fidelity_at_budget(*with_initial_point(p,f,ov),budgets) for p,f in runs],axis=0)
    print(" SKQD final pools (no early stop):", [int(p[-1]) for p,_ in runs])
    # best (t, n) at a small budget
    kmid=int(budgets[len(budgets)//2+3])
    gm={}
    for tt in tv:
        for nn in sv:
            gm[(tt,nn)]=np.mean([fidelity_at_budget(*s.full_skqd_run(init,tt,nn,gs,max_pool_size=kmid,budgets=[kmid]),kmid) for _ in range(3)])
    bm=max(gm,key=gm.get); print(f" at K={kmid}: chosen (t,n) gives F={gm.get((t,n),float('nan')):.4f}; best on grid {bm} gives F={gm[bm]:.4f}")
    res["Ceiling"]=np.cumsum(np.sort(prob)[::-1])[budgets-1]
    df=pd.DataFrame(res,index=budgets); df.index.name="K"
    print(df.iloc[::3].round(4).to_string())
    return df

import sys
for args in [(9,10,1,0.5,0.454189,0.225511,540),(13,10,1,0.0,-0.640511,0.767532,148),(8,10,1,10.0,0.039924,-0.550025,813)]:
    study(*args)
