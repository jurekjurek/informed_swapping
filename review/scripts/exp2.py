import sys, warnings, numpy as np
sys.path.insert(0, "."); warnings.filterwarnings("ignore")
from EqualNumberOfBitstrings import Cell, budget_grid, with_initial_point
from RandomSpinModel import make_heisenberg_hamiltonian
from ClusterStudy import solve_ground_state
from BARK import BARK
from SKQD import fidelity_at_budget
from Subspace import lowest_eigenpair
def rescored(b, gs, init, maxp):
    proj=b.projection; proj.reset(); proj.extend([init]); pool={init}; frontier=set()
    v=np.zeros(b.dimension,complex); v[init]=1; E=float(b._diagonal[init]); last=init; sizes=[]; fids=[]
    while proj.size<maxp:
        frontier |= {int(s) for s in b.apply_hamiltonian(last) if s not in pool}
        if not frontier: break
        cand=np.array(sorted(frontier)); pots,a,bb=b.rank_states(v,E,cand); i=int(np.argmin(pots))
        last=int(cand[i]); v=a[i]*v; v[last]+=bb[i]; E=float(pots[i]); frontier.discard(last)
        pool.add(last); proj.extend([last]); _,g=lowest_eigenpair(proj.block)
        sizes.append(proj.size); fids.append(abs(np.vdot(g,gs[proj.pool]))**2)
    return np.array(sizes),np.array(fids)
for (ham,N,dim,delta,J,Bz,init) in [(9,10,1,0.5,0.454189,0.225511,540),(8,10,1,10.0,0.039924,-0.550025,813)]:
    H=make_heisenberg_hamiltonian(num_sites=N,dimension=dim,delta=delta,J=J,h=(0,0,Bz),spin=1)[0].to_matrix(sparse=True)
    gs,ev,evec=solve_ground_state(H,4096); p=abs(gs)**2; bud=budget_grid(H.shape[0],30,0.5); b=BARK(H)
    a=fidelity_at_budget(*with_initial_point(*b.full_bark_run(gs,init,max_pool_size=int(bud[-1])),p[init]),bud)
    r=fidelity_at_budget(*with_initial_point(*rescored(b,gs,init,int(bud[-1])),p[init]),bud)
    print(f"ham={ham} delta={delta}"); print(" K        ", bud[::3]); print(" as is    ", a[::3].round(3)); print(" rescored ", r[::3].round(3))
