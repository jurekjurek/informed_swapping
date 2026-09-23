import sys, numpy as np
sys.path.insert(0, ".")
import warnings; warnings.filterwarnings("ignore")
from EqualNumberOfBitstrings import enumerate_cells
from RandomSpinModel import make_heisenberg_hamiltonian
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
cells = enumerate_cells(15, [6,8,10,12], [1,2], [0,0.5,1,10,100])
out=[]
for c in cells:
    H = make_heisenberg_hamiltonian(num_sites=c.num_sites, dimension=c.dimensions, delta=c.delta, J=c.J, h=(0,0,c.Bz), spin=1)[0].to_matrix(sparse=True)
    if c.num_sites<=10:
        e = eigh(H.toarray(), eigvals_only=True, subset_by_index=[0,1])
    else:
        e = np.sort(eigsh(H.real, k=2, which="SA", tol=1e-12)[0])
    scale = abs(e[0])
    out.append((c.num_sites,c.dimensions,c.delta,c.J,c.Bz,e[1]-e[0], (e[1]-e[0])/max(scale,1)))
import pandas as pd
df = pd.DataFrame(out, columns=["N","dim","delta","J","Bz","gap","relgap"])
df.to_csv(sys.argv[1], index=False)
print("gap < 1e-8:", (df.gap<1e-8).sum(), " gap < 1e-4:", (df.gap<1e-4).sum(), " gap<1e-2:", (df.gap<1e-2).sum(), "of", len(df))
print(df[df.gap<1e-4].groupby(["delta"]).size())
print(df.sort_values("gap").head(15))
