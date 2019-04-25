import matplotlib.pyplot as plt
import numpy as np
from itertools import product
N = 108                   # number of atoms.
Nbins = 100               # number of bins for RDF
rho = 0.8442              # denstiy
T = 0.728                 # initial temperature
rcut = 2.5                # cut-off radius
Neq = 2500                # number of time steps for equilibration stage
Npr = 5000                # number of time steps for production stage.
dt = 0.001                # time step
L = (N/rho)**(1/3)        # the size of the simulation domain
Ncell = 5                 # number of cells in initialization
a = L / Ncell             # size of lattice in initialization
T = 0.728                 # initial temperature of the system
ind = np.triu_indices(N, k=1)

def InitPos(Ncell, a):
    ppos = [[[x,y,z]] for x, y, z in product(range(Ncell),range(Ncell),range(Ncell))]
    pos = np.array(ppos).reshape((-1,3))[1:N+1,:]*a
    return pos

def InitVel(N, T):
    vel = np.random.randn(N,3)
    vel -= np.average(vel,axis = 0)
    Tc = np.sum(vel**2)/3/N
    factor = np.sqrt(T/Tc)
    vel = factor*vel
    return vel

def CalForce(pos, L = L, rc = 2.5):
    dr = pos[ind[0]] - pos[ind[1]]
    dr -= np.rint(dr / L) * L                     # mic
    rsq = np.sum(dr**2, axis=1)
    F_vec = (48 / rsq ** 7 - 24 / rsq ** 4)[:, None] * dr
    F = np.zeros((N,N,3))
    F[ind[0], ind[1]] = F_vec
    F[ind[1], ind[0]] = -F_vec
    # use truncated potential
    mask = rsq > rc*rc
    ecut = (4 / rc ** 12 - 4 / rc ** 6)
    pmat = (4 / rsq ** 6 - 4 / rsq ** 3) - ecut
    pmat[mask] = 0
    pot = np.sum(pmat)
    return np.sum(F, axis=1), pot

def velverlet(pos, vel, F):
    vel += 0.5*F * dt
    pos = pos + vel * dt
    pos = np.mod(pos, L)                    # folding for re-entering.
    F, pot = CalForce(pos)
    
    vel += 0.5*F * dt
    kin = 0.5*np.sum(vel**2)
    return pos, vel, F, pot, kin

def RDF(N,pos,rcut):
    # No more than rcut
    rvec = pos[ind[0]] - pos[ind[1]]
    rvec -= np.rint(rvec / L) * L
    rabs = np.sqrt(np.sum(rvec**2, axis = 1))
    b = np.histogram(rabs, bins = N, range = (0,rcut))[0]
    return b

# the main run of the simulation
pos = InitPos(Ncell, a)
vel = InitVel(N, T)
F, p = CalForce(pos)
Ueq = []
Keq = []
Upr = 0
Kpr = 0
rdf = np.zeros(Nbins)

# equilibration stage
for ti in range(Neq):
    pos, vel, F, pot, kin = velverlet(pos, vel, F)
    Ueq.append(pot)
    Keq.append(kin)

# production stage: average for kinetic and potential energy as well as RDF.
for ti in range(Npr):
    pos, vel, F, pot, kin = velverlet(pos, vel, F)
    Upr = (Upr*ti + pot) / (ti + 1)
    Kpr = (Kpr*ti + kin) / (ti + 1)
    temp = RDF(Nbins, pos, rcut)
    rdf = rdf + 2*temp

x = np.linspace(1,Neq,Neq)
u = np.array(Ueq) / N
k = np.array(Keq) / N
fig, ax = plt.subplots()
ax.plot(x,u,'r-.',x,k,'g',x,u+k,'m--')
ax.set_xlabel('time steps')
ax.set_ylabel('energy')
plt.legend(['potential energy','kinetic energy','total energy'])

dr = rcut / Nbins
r = np.linspace(0.5*dr,rcut-0.5*dr, Nbins)
for i in range(Nbins):
    rdf[i] /= ((i+1)**3-i**3)
    rdf[i] /= (4/3*np.pi*dr**3*rho)

rdf = rdf/(Npr*N)
plt.plot(r,rdf)
plt.xlabel('r')
plt.ylabel('g(r)')
print('Kinetic Energy:%f, Potential ENergy:%f, Total Energy:%f\n' %(Kpr/N, Upr/N, (Kpr+Upr)/N))