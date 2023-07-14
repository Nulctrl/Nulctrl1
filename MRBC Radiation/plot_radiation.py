import numpy as np
import dedalus.public as d3
import logging
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pathlib
import subprocess
import h5py
import glob
import re


Lx, Lz = 16, 1
Nx, Nz = 1024, 64
Md = 3
Rayleigh = 4e5
Vaisala= 4
Prandtl = 0.7
dealias = 3/2
QD=0
QM=QD/2
kappa = (Lz**3*Md)/(Rayleigh * Prandtl)**(1/2)
nu = (Lz**3*Md)/(Rayleigh / Prandtl)**(1/2)
stop_sim_time = round(1/nu)
timestepper = d3.RK222
max_timestep = 0.125
dtype = np.float64

# Bases
coords = d3.CartesianCoordinates('x', 'z')
dist = d3.Distributor(coords, dtype=dtype)
xbasis = d3.RealFourier(coords['x'], size=Nx, bounds=(0, Lx), dealias=dealias)
zbasis = d3.ChebyshevT(coords['z'], size=Nz, bounds=(0, Lz), dealias=dealias)

# Fields
p = dist.Field(name='p', bases=(xbasis,zbasis))
d = dist.Field(name='d', bases=(xbasis,zbasis))
m = dist.Field(name='m', bases=(xbasis,zbasis))
u = dist.VectorField(coords, name='u', bases=(xbasis,zbasis))
tau_p = dist.Field(name='tau_p')
tau_d1 = dist.Field(name='tau_d1', bases=xbasis)
tau_d2 = dist.Field(name='tau_d2', bases=xbasis)
tau_m1 = dist.Field(name='tau_m1', bases=xbasis)
tau_m2 = dist.Field(name='tau_m2', bases=xbasis)
tau_u1 = dist.VectorField(coords, name='tau_u1', bases=xbasis)
tau_u2 = dist.VectorField(coords, name='tau_u2', bases=xbasis)
tau_c1 = dist.Field(name='tau_c1', bases=xbasis)
tau_c2 = dist.Field(name='tau_c2', bases=xbasis)

# Substitutions
x, z = dist.local_grids(xbasis, zbasis)
ex, ez = coords.unit_vector_fields(dist)
lift_basis = zbasis.derivative_basis(1)
lift = lambda A: d3.Lift(A, lift_basis, -1)
grad_u = d3.grad(u) + ez*lift(tau_u1) # First-order reduction
grad_d = d3.grad(d) + ez*lift(tau_d1) # First-order reduction
grad_m = d3.grad(m) + ez*lift(tau_m1) # First-order reduction
ncc = dist.Field(name='ncc', bases=zbasis)
ncc['g'] = z
ncc.change_scales(3/2)

radiation = dist.Field(name='radiation', bases=zbasis)
radiation['g'] = np.sin(np.pi*z)

fix,ax=plt.subplots()
files = sorted(glob.glob('snapshots0/*.h5'),key=lambda f: int(re.sub('\D', '', f)))

with h5py.File(files[-1], mode='r') as file:
    dflux= file['tasks']['dflux'][:]
    mflux= file['tasks']['mflux'][:]
    bflux= file['tasks']['bflux'][:]
    wflux= file['tasks']['wflux'][:]
    Nusselt= file['tasks']['Nusselt'][:]
    ax.scatter(0,np.mean(dflux),marker="^",color='green')
    ax.scatter(0,np.mean(mflux),marker="x",color='blue')
    ax.scatter(0,np.mean(bflux),marker="s",color='purple')
    ax.scatter(0,np.mean(wflux),marker="+",color='red')
    
files = sorted(glob.glob('snapshots1/*.h5'),key=lambda f: int(re.sub('\D', '', f)))

with h5py.File(files[-1], mode='r') as file:
    dflux= file['tasks']['dflux'][:]
    mflux= file['tasks']['mflux'][:]
    bflux= file['tasks']['bflux'][:]
    wflux= file['tasks']['wflux'][:]
    Nusselt= file['tasks']['Nusselt'][:]
    ax.scatter(0.0028,np.mean(dflux),marker="^",color='green')
    ax.scatter(0.0028,np.mean(mflux),marker="x",color='blue')
    ax.scatter(0.0028,np.mean(bflux),marker="s",color='purple')
    ax.scatter(0.0028,np.mean(wflux),marker="+",color='red')
    
files = sorted(glob.glob('snapshots2/*.h5'),key=lambda f: int(re.sub('\D', '', f)))

with h5py.File(files[-1], mode='r') as file:
    dflux= file['tasks']['dflux'][:]
    mflux= file['tasks']['mflux'][:]
    bflux= file['tasks']['bflux'][:]
    wflux= file['tasks']['wflux'][:]
    Nusselt= file['tasks']['Nusselt'][:]
    ax.scatter(0.014,np.mean(dflux),marker="^",color='green',label="$dflux, u_z\cdot D$")
    ax.scatter(0.014,np.mean(mflux),marker="x",color='blue',label="$mflux, u_z\cdot M$")
    ax.scatter(0.014,np.mean(bflux),marker="s",color='purple',label="$bflux, u_z\cdot B$")
    ax.scatter(0.014,np.mean(wflux),marker="+",color='red',label="$wflux, u_z\cdot(\chi M-D)$")
    
files = sorted(glob.glob('snapshots3/*.h5'),key=lambda f: int(re.sub('\D', '', f)))

with h5py.File(files[-1], mode='r') as file:
    dflux= file['tasks']['dflux'][:]
    mflux= file['tasks']['mflux'][:]
    bflux= file['tasks']['bflux'][:]
    wflux= file['tasks']['wflux'][:]
    Nusselt= file['tasks']['Nusselt'][:]
    ax.scatter(0.07,np.mean(dflux),marker="^",color='green')
    ax.scatter(0.07,np.mean(mflux),marker="x",color='blue')
    ax.scatter(0.07,np.mean(bflux),marker="s",color='purple')
    ax.scatter(0.07,np.mean(wflux),marker="+",color='red')

plt.xlabel('Radiation')
plt.ylabel('Mean fluxes')
ax.legend()
plt.savefig("Flux progression")    
plt.close()

fix,ax=plt.subplots()
files = sorted(glob.glob('snapshots0/*.h5'),key=lambda f: int(re.sub('\D', '', f)))

with h5py.File(files[-1], mode='r') as file:
    dflux= file['tasks']['dflux'][:]
    mflux= file['tasks']['mflux'][:]
    bflux= file['tasks']['bflux'][:]
    wflux= file['tasks']['wflux'][:]
    Nusselt= file['tasks']['Nusselt'][:]

    ax.scatter(0,np.mean(Nusselt[:,0,0]),marker="o",color='blue',label="Nusselt")
    
files = sorted(glob.glob('snapshots1/*.h5'),key=lambda f: int(re.sub('\D', '', f)))

with h5py.File(files[-1], mode='r') as file:
    dflux= file['tasks']['dflux'][:]
    mflux= file['tasks']['mflux'][:]
    bflux= file['tasks']['bflux'][:]
    wflux= file['tasks']['wflux'][:]
    Nusselt= file['tasks']['Nusselt'][:]

    ax.scatter(0.0028,np.mean(Nusselt[:,0,0]),marker="o",color='blue')
    
files = sorted(glob.glob('snapshots2/*.h5'),key=lambda f: int(re.sub('\D', '', f)))

with h5py.File(files[-1], mode='r') as file:
    dflux= file['tasks']['dflux'][:]
    mflux= file['tasks']['mflux'][:]
    bflux= file['tasks']['bflux'][:]
    wflux= file['tasks']['wflux'][:]
    Nusselt= file['tasks']['Nusselt'][:]

    ax.scatter(0.014,np.mean(Nusselt[:,0,0]),marker="o",color='blue')
    
files = sorted(glob.glob('snapshots3/*.h5'),key=lambda f: int(re.sub('\D', '', f)))

with h5py.File(files[-1], mode='r') as file:
    dflux= file['tasks']['dflux'][:]
    mflux= file['tasks']['mflux'][:]
    bflux= file['tasks']['bflux'][:]
    wflux= file['tasks']['wflux'][:]
    Nusselt= file['tasks']['Nusselt'][:]

    ax.scatter(0.07,np.mean(Nusselt[:,0,0]),marker="o",color='blue')

plt.xlabel('Radiation')
plt.ylabel('Nusselt Number')
ax.legend()
plt.savefig("Nusselt progression")    
plt.close()

fig, ax = plt.subplots(3, figsize=(6,12))

files = sorted(glob.glob('snapshots0/*.h5'),key=lambda f: int(re.sub('\D', '', f)))

with h5py.File(files[-1], mode='r') as file:
    KE1 = file['tasks']['KE']
    xgrid=KE1.dims[1][0][:]
    zgrid=KE1.dims[2][0][:]
    u=file['tasks']['velocity'][:]
    uz=u[:,1,:,:]
    ux=u[:,0,:,:]
    ax[0].plot(zgrid, np.mean((uz**2)[-1,:,:],axis=0),label='Q=0')
    ax[1].plot(zgrid, np.mean((ux**2)[-1,:,:],axis=0),label='Q=0')
    ax[2].plot(zgrid, np.mean((uz**2+ux**2)[-1,:,:],axis=0),label='Q=0')
    
files = sorted(glob.glob('snapshots1/*.h5'),key=lambda f: int(re.sub('\D', '', f)))

with h5py.File(files[-1], mode='r') as file:
    u=file['tasks']['velocity'][:]
    uz=u[:,1,:,:]
    ux=u[:,0,:,:]
    ax[0].plot(zgrid, np.mean((uz**2)[-1,:,:],axis=0),label='Q=0.0028',linestyle='dotted')
    ax[1].plot(zgrid, np.mean((ux**2)[-1,:,:],axis=0),label='Q=0.0028',linestyle='dotted')
    ax[2].plot(zgrid, np.mean((uz**2+ux**2)[-1,:,:],axis=0),label='Q=0.0028',linestyle='dotted')
    
files = sorted(glob.glob('snapshots2/*.h5'),key=lambda f: int(re.sub('\D', '', f)))

with h5py.File(files[-1], mode='r') as file:
    u=file['tasks']['velocity'][:]
    uz=u[:,1,:,:]
    ux=u[:,0,:,:]
    ax[0].plot(zgrid, np.mean((uz**2)[-1,:,:],axis=0),label='Q=0.014',linestyle='--')
    ax[1].plot(zgrid, np.mean((ux**2)[-1,:,:],axis=0),label='Q=0.014',linestyle='--')
    ax[2].plot(zgrid, np.mean((uz**2+ux**2)[-1,:,:],axis=0),label='Q=0.014',linestyle='--')
    
files = sorted(glob.glob('snapshots3/*.h5'),key=lambda f: int(re.sub('\D', '', f)))

with h5py.File(files[-1], mode='r') as file:
    u=file['tasks']['velocity'][:]
    uz=u[:,1,:,:]
    ux=u[:,0,:,:]
    ax[0].plot(zgrid, np.mean((uz**2)[-1,:,:],axis=0),label='Q=0.07',linestyle='dashdot')
    ax[1].plot(zgrid, np.mean((ux**2)[-1,:,:],axis=0),label='Q=0.07',linestyle='dashdot')
    ax[2].plot(zgrid, np.mean((uz**2+ux**2)[-1,:,:],axis=0),label='Q=0.07',linestyle='dashdot')
    
ax[0].set_ylabel("$u_z^2$")   
ax[0].set_xlabel("Radiation")
ax[0].set_title("Vertical KE")
ax[0].legend()

ax[1].set_ylabel("$u_x^2+u_y^2$")   
ax[1].set_xlabel("Radiation")
ax[1].set_title("Horizontal KE")
ax[1].legend()

ax[2].set_ylabel("KE")   
ax[2].set_xlabel("Radiation")
ax[2].set_title("Total KE")
ax[2].legend()
plt.tight_layout()
plt.savefig("Vertical KE")    
plt.close()

files = sorted(glob.glob('snapshots0/*.h5'),key=lambda f: int(re.sub('\D', '', f)))
with h5py.File(files[-1], mode='r') as file:
    moist_buoyancy1 = file['tasks']['moist buoyancy']
    xgrid=moist_buoyancy1.dims[1][0][:]
    zgrid=moist_buoyancy1.dims[2][0][:]
    moist_buoyancy = file['tasks']['moist buoyancy'][:,:,:]
    dry_buoyancy = file['tasks']['dry buoyancy'][:, :, :]
    buoyancy=np.maximum(moist_buoyancy,dry_buoyancy-Vaisala*zgrid)
    saturation=moist_buoyancy-dry_buoyancy+Vaisala*zgrid
    extra_buoyancy=buoyancy-(dry_buoyancy-Vaisala*zgrid)

    # Plotting extra buoyancy
    plt.figure()
    plt.contourf(xgrid, zgrid, extra_buoyancy[-1,:,:].T)
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('z')
    plt.title('Final Clouds0')
    plt.savefig('Final Clouds0')
    plt.close()

with h5py.File(files[-1], mode='r') as file:
    vorticity = file['tasks']['vorticity']

    # Plotting extra buoyancy
    plt.figure()
    plt.contourf(xgrid, zgrid, vorticity[-1,:,:].T)
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('z')
    plt.title('Final Vorticity0')
    plt.savefig('Final Vorticity0')
    plt.close()
    
files = sorted(glob.glob('snapshots1/*.h5'),key=lambda f: int(re.sub('\D', '', f)))
with h5py.File(files[-1], mode='r') as file:
    moist_buoyancy1 = file['tasks']['moist buoyancy']
    xgrid=moist_buoyancy1.dims[1][0][:]
    zgrid=moist_buoyancy1.dims[2][0][:]
    moist_buoyancy = file['tasks']['moist buoyancy'][:,:,:]
    dry_buoyancy = file['tasks']['dry buoyancy'][:, :, :]
    buoyancy=np.maximum(moist_buoyancy,dry_buoyancy-Vaisala*zgrid)
    saturation=moist_buoyancy-dry_buoyancy+Vaisala*zgrid
    extra_buoyancy=buoyancy-(dry_buoyancy-Vaisala*zgrid)

    # Plotting extra buoyancy
    plt.figure()
    plt.contourf(xgrid, zgrid, extra_buoyancy[-1,:,:].T)
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('z')
    plt.title('Final Clouds1')
    plt.savefig('Final Clouds1')
    plt.close()

with h5py.File(files[-1], mode='r') as file:
    vorticity = file['tasks']['vorticity']

    # Plotting extra buoyancy
    plt.figure()
    plt.contourf(xgrid, zgrid, vorticity[-1,:,:].T)
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('z')
    plt.title('Final Vorticity1')
    plt.savefig('Final Vorticity1')
    plt.close()
    
files = sorted(glob.glob('snapshots2/*.h5'),key=lambda f: int(re.sub('\D', '', f)))
with h5py.File(files[-1], mode='r') as file:
    moist_buoyancy1 = file['tasks']['moist buoyancy']
    xgrid=moist_buoyancy1.dims[1][0][:]
    zgrid=moist_buoyancy1.dims[2][0][:]
    moist_buoyancy = file['tasks']['moist buoyancy'][:,:,:]
    dry_buoyancy = file['tasks']['dry buoyancy'][:, :, :]
    buoyancy=np.maximum(moist_buoyancy,dry_buoyancy-Vaisala*zgrid)
    saturation=moist_buoyancy-dry_buoyancy+Vaisala*zgrid
    extra_buoyancy=buoyancy-(dry_buoyancy-Vaisala*zgrid)

    # Plotting extra buoyancy
    plt.figure()
    plt.contourf(xgrid, zgrid, extra_buoyancy[-1,:,:].T)
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('z')
    plt.title('Final Clouds2')
    plt.savefig('Final Clouds2')
    plt.close()

with h5py.File(files[-1], mode='r') as file:
    vorticity = file['tasks']['vorticity']

    # Plotting extra buoyancy
    plt.figure()
    plt.contourf(xgrid, zgrid, vorticity[-1,:,:].T)
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('z')
    plt.title('Final Vorticity2')
    plt.savefig('Final Vorticity2')
    plt.close()

files = sorted(glob.glob('snapshots3/*.h5'),key=lambda f: int(re.sub('\D', '', f)))
with h5py.File(files[-1], mode='r') as file:
    moist_buoyancy1 = file['tasks']['moist buoyancy']
    xgrid=moist_buoyancy1.dims[1][0][:]
    zgrid=moist_buoyancy1.dims[2][0][:]
    moist_buoyancy = file['tasks']['moist buoyancy'][:,:,:]
    dry_buoyancy = file['tasks']['dry buoyancy'][:, :, :]
    buoyancy=np.maximum(moist_buoyancy,dry_buoyancy-Vaisala*zgrid)
    saturation=moist_buoyancy-dry_buoyancy+Vaisala*zgrid
    extra_buoyancy=buoyancy-(dry_buoyancy-Vaisala*zgrid)

    # Plotting extra buoyancy
    plt.figure()
    plt.contourf(xgrid, zgrid, extra_buoyancy[-1,:,:].T)
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('z')
    plt.title('Final Clouds3')
    plt.savefig('Final Clouds3')
    plt.close()

with h5py.File(files[-1], mode='r') as file:
    vorticity = file['tasks']['vorticity']

    # Plotting extra buoyancy
    plt.figure()
    plt.contourf(xgrid, zgrid, vorticity[-1,:,:].T)
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('z')
    plt.title('Final Vorticity3')
    plt.savefig('Final Vorticity3')
    plt.close()


files = sorted(glob.glob('snapshots0/*.h5'),key=lambda f: int(re.sub('\D', '', f)))
fig, ax = plt.subplots()

with h5py.File(files[0], mode='r') as file:
    KEs0 = file['tasks']['total KE'][:,:,:]
    
for i in range(1,len(files)):
    with h5py.File(files[i], mode='r') as file:
        KE = file['tasks']['total KE'][:,:,:]
        KEs0=np.append(KEs0,KE,axis=0)

ax.plot(np.arange(len(KEs0[:,0,0]))/4*nu, KEs0[:,0,0],label='Q=0')

files = sorted(glob.glob('snapshots1/*.h5'),key=lambda f: int(re.sub('\D', '', f)))

with h5py.File(files[0], mode='r') as file:
    KEs1 = file['tasks']['total KE'][:,:,:]
    
for i in range(1,len(files)):
    with h5py.File(files[i], mode='r') as file:
        KE = file['tasks']['total KE'][:,:,:]
        KEs1=np.append(KEs1,KE,axis=0)

ax.plot(np.arange(len(KEs1[:,0,0]))/4*nu, KEs1[:,0,0],linestyle='dotted',label='Q=0.0028')

files = sorted(glob.glob('snapshots2/*.h5'),key=lambda f: int(re.sub('\D', '', f)))

with h5py.File(files[0], mode='r') as file:
    KEs2 = file['tasks']['total KE'][:,:,:]
    
for i in range(1,len(files)):
    with h5py.File(files[i], mode='r') as file:
        KE = file['tasks']['total KE'][:,:,:]
        KEs2=np.append(KEs2,KE,axis=0)

ax.plot(np.arange(len(KEs2[:,0,0]))/4*nu, KEs2[:,0,0],linestyle='--',label='Q=0.014')
        
files = sorted(glob.glob('snapshots3/*.h5'),key=lambda f: int(re.sub('\D', '', f)))

with h5py.File(files[0], mode='r') as file:
    KEs3 = file['tasks']['total KE'][:,:,:]
    
for i in range(1,len(files)):
    with h5py.File(files[i], mode='r') as file:
        KE = file['tasks']['total KE'][:,:,:]
        KEs3=np.append(KEs3,KE,axis=0)

ax.plot(np.arange(len(KEs3[:,0,0]))/4*nu, KEs3[:,0,0], linestyle='dashdot', label='Q=0.07')

ax.set_title('Total KE vs Time')
ax.grid(True)
ax.set_xlabel(r"Normalized Time $\nu t/H^2$")
ax.set_ylabel(r"Total KE  $\log_{10}E_k(t)$")
ax.set_yscale('log')
plt.legend()

plt.savefig('Total_KE_vs_Time')


fig, ax = plt.subplots(3, figsize=(6,12))


files = sorted(glob.glob('snapshots0/*.h5'),key=lambda f: int(re.sub('\D', '', f)))

with h5py.File(files[-1], mode='r') as file:
    moist_buoyancy = file['tasks']['moist buoyancy']
    xgrid=moist_buoyancy.dims[1][0][:]
    zgrid=moist_buoyancy.dims[2][0][:]
    dry_buoyancy = file['tasks']['dry buoyancy'][:,:,:]
    moist_buoyancy = file['tasks']['moist buoyancy'][:,:,:]
    buoyancy = file['tasks']['buoyancy'][:,:,:]
    avgmbs=np.mean(moist_buoyancy,axis=1)
    avgdbs=np.mean(dry_buoyancy,axis=1)
    avgbs=np.mean(buoyancy,axis=1)
    
ax[0].plot(zgrid, avgmbs[-1,:],label='Q=0')
ax[1].plot(zgrid, avgdbs[-1,:],label='Q=0')
ax[2].plot(zgrid, avgbs[-1,:],label='Q=0')

#M
files = sorted(glob.glob('snapshots1/*.h5'),key=lambda f: int(re.sub('\D', '', f)))

with h5py.File(files[-1], mode='r') as file:
    moist_buoyancy = file['tasks']['moist buoyancy']
    xgrid=moist_buoyancy.dims[1][0][:]
    zgrid=moist_buoyancy.dims[2][0][:]
    dry_buoyancy = file['tasks']['dry buoyancy'][:,:,:]
    moist_buoyancy = file['tasks']['moist buoyancy'][:,:,:]
    buoyancy = file['tasks']['buoyancy'][:,:,:]
    avgmbs=np.mean(moist_buoyancy,axis=1)
    avgdbs=np.mean(dry_buoyancy,axis=1)
    avgbs=np.mean(buoyancy,axis=1)
    
ax[0].plot(zgrid, avgmbs[-1,:],linestyle='dotted',label='Q=0.0028')
ax[1].plot(zgrid, avgdbs[-1,:],linestyle='dotted',label='Q=0.0028')
ax[2].plot(zgrid, avgbs[-1,:],linestyle='dotted',label='Q=0.0028')

#M
files = sorted(glob.glob('snapshots2/*.h5'),key=lambda f: int(re.sub('\D', '', f)))

with h5py.File(files[-1], mode='r') as file:
    moist_buoyancy = file['tasks']['moist buoyancy']
    xgrid=moist_buoyancy.dims[1][0][:]
    zgrid=moist_buoyancy.dims[2][0][:]
    dry_buoyancy = file['tasks']['dry buoyancy'][:,:,:]
    moist_buoyancy = file['tasks']['moist buoyancy'][:,:,:]
    buoyancy = file['tasks']['buoyancy'][:,:,:]
    avgmbs=np.mean(moist_buoyancy,axis=1)
    avgdbs=np.mean(dry_buoyancy,axis=1)
    avgbs=np.mean(buoyancy,axis=1)
    
ax[0].plot(zgrid, avgmbs[-1,:],linestyle='--',label='Q=0.014')
ax[1].plot(zgrid, avgdbs[-1,:],linestyle='--',label='Q=0.014')
ax[2].plot(zgrid, avgbs[-1,:],linestyle='--',label='Q=0.014')

#M
files = sorted(glob.glob('snapshots3/*.h5'),key=lambda f: int(re.sub('\D', '', f)))

with h5py.File(files[-1], mode='r') as file:
    moist_buoyancy = file['tasks']['moist buoyancy']
    xgrid=moist_buoyancy.dims[1][0][:]
    zgrid=moist_buoyancy.dims[2][0][:]
    dry_buoyancy = file['tasks']['dry buoyancy'][:,:,:]
    moist_buoyancy = file['tasks']['moist buoyancy'][:,:,:]
    buoyancy = file['tasks']['buoyancy'][:,:,:]
    avgmbs=np.mean(moist_buoyancy,axis=1)
    avgdbs=np.mean(dry_buoyancy,axis=1)
    avgbs=np.mean(buoyancy,axis=1)
    
ax[0].plot(zgrid, avgmbs[-1,:],linestyle='dashdot',label='Q=0.07')
ax[1].plot(zgrid, avgdbs[-1,:],linestyle='dashdot',label='Q=0.07')
ax[2].plot(zgrid, avgbs[-1,:],linestyle='dashdot',label='Q=0.07')

ax[0].set_title('Vertical M Profile')
ax[0].grid(True)
ax[0].set_xlabel(r"<z>")
ax[0].set_ylabel(r"<M>")
ax[0].legend()

ax[1].set_title('Vertical D Profile')
ax[1].grid(True)
ax[1].set_xlabel(r"<z>")
ax[1].set_ylabel(r"<D>")
ax[1].legend()

ax[2].set_title('Vertical B Profile')
ax[2].grid(True)
ax[2].set_xlabel(r"<z>")
ax[2].set_ylabel(r"<B>")
ax[2].legend()

plt.tight_layout()
plt.savefig('Vertical Profile')
plt.show()


fig, ax = plt.subplots(3, figsize=(6,12))

#M
files = sorted(glob.glob('snapshots0/*.h5'),key=lambda f: int(re.sub('\D', '', f)))

with h5py.File(files[10], mode='r') as file:
    moist_buoyancy = file['tasks']['moist buoyancy']
    xgrid=moist_buoyancy.dims[1][0][:]
    zgrid=moist_buoyancy.dims[2][0][:]
    dry_buoyancy = file['tasks']['dry buoyancy'][:,:,:]
    moist_buoyancy = file['tasks']['moist buoyancy'][:,:,:]
    buoyancy = file['tasks']['buoyancy'][:,:,:]
    avgmbs=np.mean(moist_buoyancy,axis=1)
    avgdbs=np.mean(dry_buoyancy,axis=1)
    avgbs=np.mean(buoyancy,axis=1)
    avgmbs1=np.mean(avgmbs,axis=0)
    avgdbs1=np.mean(avgdbs,axis=0)
    avgbs1=np.mean(avgbs,axis=0)
for i in range(11,len(files)):
    with h5py.File(files[i], mode='r') as file:
        moist_buoyancy = file['tasks']['moist buoyancy']
        xgrid=moist_buoyancy.dims[1][0][:]
        zgrid=moist_buoyancy.dims[2][0][:]
        dry_buoyancy = file['tasks']['dry buoyancy'][:,:,:]
        moist_buoyancy = file['tasks']['moist buoyancy'][:,:,:]
        buoyancy = file['tasks']['buoyancy'][:,:,:]
        avgmbs=np.mean(moist_buoyancy,axis=1)
        avgdbs=np.mean(dry_buoyancy,axis=1)
        avgbs=np.mean(buoyancy,axis=1)
        avgmbs2=np.mean(avgmbs,axis=0)
        avgdbs2=np.mean(avgdbs,axis=0)
        avgbs2=np.mean(avgbs,axis=0)
        avgmbs1+=avgmbs2
        avgdbs1+=avgdbs2
        avgbs1+=avgbs2

    
ax[0].plot(zgrid, avgmbs1[:],label='Q=0')
ax[1].plot(zgrid, avgdbs1[:],label='Q=0')
ax[2].plot(zgrid, avgbs1[:],label='Q=0')

#M
files = sorted(glob.glob('snapshots1/*.h5'),key=lambda f: int(re.sub('\D', '', f)))

with h5py.File(files[10], mode='r') as file:
    moist_buoyancy = file['tasks']['moist buoyancy']
    xgrid=moist_buoyancy.dims[1][0][:]
    zgrid=moist_buoyancy.dims[2][0][:]
    dry_buoyancy = file['tasks']['dry buoyancy'][:,:,:]
    moist_buoyancy = file['tasks']['moist buoyancy'][:,:,:]
    buoyancy = file['tasks']['buoyancy'][:,:,:]
    avgmbs=np.mean(moist_buoyancy,axis=1)
    avgdbs=np.mean(dry_buoyancy,axis=1)
    avgbs=np.mean(buoyancy,axis=1)
    avgmbs1=np.mean(avgmbs,axis=0)
    avgdbs1=np.mean(avgdbs,axis=0)
    avgbs1=np.mean(avgbs,axis=0)  
for i in range(11,len(files)):
    with h5py.File(files[i], mode='r') as file:
        moist_buoyancy = file['tasks']['moist buoyancy']
        xgrid=moist_buoyancy.dims[1][0][:]
        zgrid=moist_buoyancy.dims[2][0][:]
        dry_buoyancy = file['tasks']['dry buoyancy'][:,:,:]
        moist_buoyancy = file['tasks']['moist buoyancy'][:,:,:]
        buoyancy = file['tasks']['buoyancy'][:,:,:]
        avgmbs=np.mean(moist_buoyancy,axis=1)
        avgdbs=np.mean(dry_buoyancy,axis=1)
        avgbs=np.mean(buoyancy,axis=1)
        avgmbs2=np.mean(avgmbs,axis=0)
        avgdbs2=np.mean(avgdbs,axis=0)
        avgbs2=np.mean(avgbs,axis=0)
        avgmbs1+=avgmbs2
        avgdbs1+=avgdbs2
        avgbs1+=avgbs2
        
ax[0].plot(zgrid, avgmbs1[:],linestyle='dotted',label='Q=0.0028')
ax[1].plot(zgrid, avgdbs1[:],linestyle='dotted',label='Q=0.0028')
ax[2].plot(zgrid, avgbs1[:],linestyle='dotted',label='Q=0.0028')

#M
files = sorted(glob.glob('snapshots2/*.h5'),key=lambda f: int(re.sub('\D', '', f)))

with h5py.File(files[10], mode='r') as file:
    moist_buoyancy = file['tasks']['moist buoyancy']
    xgrid=moist_buoyancy.dims[1][0][:]
    zgrid=moist_buoyancy.dims[2][0][:]
    dry_buoyancy = file['tasks']['dry buoyancy'][:,:,:]
    moist_buoyancy = file['tasks']['moist buoyancy'][:,:,:]
    buoyancy = file['tasks']['buoyancy'][:,:,:]
    avgmbs=np.mean(moist_buoyancy,axis=1)
    avgdbs=np.mean(dry_buoyancy,axis=1)
    avgbs=np.mean(buoyancy,axis=1)
    avgmbs1=np.mean(avgmbs,axis=0)
    avgdbs1=np.mean(avgdbs,axis=0)
    avgbs1=np.mean(avgbs,axis=0)
for i in range(11,len(files)):
    with h5py.File(files[i], mode='r') as file:
        moist_buoyancy = file['tasks']['moist buoyancy']
        xgrid=moist_buoyancy.dims[1][0][:]
        zgrid=moist_buoyancy.dims[2][0][:]
        dry_buoyancy = file['tasks']['dry buoyancy'][:,:,:]
        moist_buoyancy = file['tasks']['moist buoyancy'][:,:,:]
        buoyancy = file['tasks']['buoyancy'][:,:,:]
        avgmbs=np.mean(moist_buoyancy,axis=1)
        avgdbs=np.mean(dry_buoyancy,axis=1)
        avgbs=np.mean(buoyancy,axis=1)
        avgmbs2=np.mean(avgmbs,axis=0)
        avgdbs2=np.mean(avgdbs,axis=0)
        avgbs2=np.mean(avgbs,axis=0)
        avgmbs1+=avgmbs2
        avgdbs1+=avgdbs2
        avgbs1+=avgbs2
        
ax[0].plot(zgrid, avgmbs1[:],linestyle='--',label='Q=0.014')
ax[1].plot(zgrid, avgdbs1[:],linestyle='--',label='Q=0.014')
ax[2].plot(zgrid, avgbs1[:],linestyle='--',label='Q=0.014')

#M
files = sorted(glob.glob('snapshots3/*.h5'),key=lambda f: int(re.sub('\D', '', f)))

with h5py.File(files[10], mode='r') as file:
    moist_buoyancy = file['tasks']['moist buoyancy']
    xgrid=moist_buoyancy.dims[1][0][:]
    zgrid=moist_buoyancy.dims[2][0][:]
    dry_buoyancy = file['tasks']['dry buoyancy'][:,:,:]
    moist_buoyancy = file['tasks']['moist buoyancy'][:,:,:]
    buoyancy = file['tasks']['buoyancy'][:,:,:]
    avgmbs=np.mean(moist_buoyancy,axis=1)
    avgdbs=np.mean(dry_buoyancy,axis=1)
    avgbs=np.mean(buoyancy,axis=1)
    avgmbs1=np.mean(avgmbs,axis=0)
    avgdbs1=np.mean(avgdbs,axis=0)
    avgbs1=np.mean(avgbs,axis=0)
for i in range(11,len(files)):
    with h5py.File(files[i], mode='r') as file:
        moist_buoyancy = file['tasks']['moist buoyancy']
        xgrid=moist_buoyancy.dims[1][0][:]
        zgrid=moist_buoyancy.dims[2][0][:]
        dry_buoyancy = file['tasks']['dry buoyancy'][:,:,:]
        moist_buoyancy = file['tasks']['moist buoyancy'][:,:,:]
        buoyancy = file['tasks']['buoyancy'][:,:,:]
        avgmbs=np.mean(moist_buoyancy,axis=1)
        avgdbs=np.mean(dry_buoyancy,axis=1)
        avgbs=np.mean(buoyancy,axis=1)
        avgmbs2=np.mean(avgmbs,axis=0)
        avgdbs2=np.mean(avgdbs,axis=0)
        avgbs2=np.mean(avgbs,axis=0)
        avgmbs1+=avgmbs2
        avgdbs1+=avgdbs2
        avgbs1+=avgbs2
        
ax[0].plot(zgrid, avgmbs1[:],linestyle='dashdot',label='Q=0.07')
ax[1].plot(zgrid, avgdbs1[:],linestyle='dashdot',label='Q=0.07')
ax[2].plot(zgrid, avgbs1[:],linestyle='dashdot',label='Q=0.07')

ax[0].set_title('Vertical M Profile')
ax[0].grid(True)
ax[0].set_xlabel(r"$z$")
ax[0].set_ylabel(r"$M$")
ax[0].legend()

ax[1].set_title('Vertical D Profile')
ax[1].grid(True)
ax[1].set_xlabel(r"$z$")
ax[1].set_ylabel(r"$D$")
ax[1].legend()

ax[2].set_title('Vertical B Profile')
ax[2].grid(True)
ax[2].set_xlabel(r"$z$")
ax[2].set_ylabel(r"$B$")
ax[2].legend()

plt.tight_layout()
plt.savefig('Vertical Profile Time Average')
plt.show()

print("Start Animating")

files = sorted(glob.glob('snapshots0/*.h5'),key=lambda f: int(re.sub('\D', '', f)))
with h5py.File(files[0], mode='r') as file:
    extra_buoyancy = file['tasks']['additional buoyancy'][:, :, :] 
    clouds = np.where(extra_buoyancy < 0, 0, extra_buoyancy)
for i in range(1,len(files)):
    with h5py.File(files[i], mode='r') as file:
        extra_buoyancy = file['tasks']['additional buoyancy'][:, :, :] 
        clouds=np.append(clouds,extra_buoyancy,axis=0)

clouds = np.where(clouds < 0, 0, clouds)     
global_min=np.min(clouds) 
global_max=np.max(clouds)
conditon = (clouds == global_max)
max_pos = np.where(conditon)
print("finished processing")
fig, ax = plt.subplots()
cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Define colorbar axes position
img = ax.contourf(xgrid, zgrid, clouds[int(max_pos[0]), :, :].T, vmin=global_min, vmax=global_max,cmap='Blues_r' )
cb = fig.colorbar(img, cax=cax)
cb.set_label('Extra Buoyancy')
def animate(frame):
    ax.clear()
    img = ax.contourf(xgrid, zgrid, clouds[frame, :, : ].T,vmin=global_min, vmax=global_max,cmap='Blues_r')
    ax.set_title('Frame: {}'.format(frame))
    ax.set_xlabel('x')  # Add x-axis label
    ax.set_ylabel('z')  # Add y-axis label

# Call animate method
animation = FuncAnimation(fig, animate, frames=len(clouds), interval=100, blit=False)
animation.save('radiation0.gif', writer='imagemagick')
# Display the plot
plt.show()

files = sorted(glob.glob('snapshots1/*.h5'),key=lambda f: int(re.sub('\D', '', f)))
with h5py.File(files[0], mode='r') as file:
    extra_buoyancy = file['tasks']['additional buoyancy'][:, :, :] 
    clouds = np.where(extra_buoyancy < 0, 0, extra_buoyancy)
for i in range(1,len(files)):
    with h5py.File(files[i], mode='r') as file:
        extra_buoyancy = file['tasks']['additional buoyancy'][:, :, :] 
        clouds=np.append(clouds,extra_buoyancy,axis=0)

clouds = np.where(clouds < 0, 0, clouds)     
global_min=np.min(clouds) 
global_max=np.max(clouds)
conditon = (clouds == global_max)
max_pos = np.where(conditon)
print("finished processing")
fig, ax = plt.subplots()
cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Define colorbar axes position
img = ax.contourf(xgrid, zgrid, clouds[int(max_pos[0]), :, :].T, vmin=global_min, vmax=global_max,cmap='Blues_r' )
cb = fig.colorbar(img, cax=cax)
cb.set_label('Extra Buoyancy')
def animate(frame):
    ax.clear()
    img = ax.contourf(xgrid, zgrid, clouds[frame, :, : ].T,vmin=global_min, vmax=global_max,cmap='Blues_r')
    ax.set_title('Frame: {}'.format(frame))
    ax.set_xlabel('x')  # Add x-axis label
    ax.set_ylabel('z')  # Add y-axis label

# Call animate method
animation = FuncAnimation(fig, animate, frames=len(clouds), interval=100, blit=False)
animation.save('radiation1.gif', writer='imagemagick')
# Display the plot
plt.show()

files = sorted(glob.glob('snapshots2/*.h5'),key=lambda f: int(re.sub('\D', '', f)))
with h5py.File(files[0], mode='r') as file:
    extra_buoyancy = file['tasks']['additional buoyancy'][:, :, :] 
    clouds = np.where(extra_buoyancy < 0, 0, extra_buoyancy)
for i in range(1,len(files)):
    with h5py.File(files[i], mode='r') as file:
        extra_buoyancy = file['tasks']['additional buoyancy'][:, :, :] 
        clouds=np.append(clouds,extra_buoyancy,axis=0)

clouds = np.where(clouds < 0, 0, clouds)     
global_min=np.min(clouds) 
global_max=np.max(clouds)
conditon = (clouds == global_max)
max_pos = np.where(conditon)
print("finished processing")
fig, ax = plt.subplots()
cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Define colorbar axes position
img = ax.contourf(xgrid, zgrid, clouds[int(max_pos[0]), :, :].T, vmin=global_min, vmax=global_max,cmap='Blues_r' )
cb = fig.colorbar(img, cax=cax)
cb.set_label('Extra Buoyancy')
def animate(frame):
    ax.clear()
    img = ax.contourf(xgrid, zgrid, clouds[frame, :, : ].T,vmin=global_min, vmax=global_max,cmap='Blues_r')
    ax.set_title('Frame: {}'.format(frame))
    ax.set_xlabel('x')  # Add x-axis label
    ax.set_ylabel('z')  # Add y-axis label

# Call animate method
animation = FuncAnimation(fig, animate, frames=len(clouds), interval=100, blit=False)
animation.save('radiation2.gif', writer='imagemagick')
# Display the plot
plt.show()

files = sorted(glob.glob('snapshots3/*.h5'),key=lambda f: int(re.sub('\D', '', f)))
with h5py.File(files[0], mode='r') as file:
    extra_buoyancy = file['tasks']['additional buoyancy'][:, :, :] 
    clouds = np.where(extra_buoyancy < 0, 0, extra_buoyancy)
for i in range(1,len(files)):
    with h5py.File(files[i], mode='r') as file:
        extra_buoyancy = file['tasks']['additional buoyancy'][:, :, :] 
        clouds=np.append(clouds,extra_buoyancy,axis=0)

clouds = np.where(clouds < 0, 0, clouds)     
global_min=np.min(clouds) 
global_max=np.max(clouds)
conditon = (clouds == global_max)
max_pos = np.where(conditon)
print("finished processing")
fig, ax = plt.subplots()
cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Define colorbar axes position
img = ax.contourf(xgrid, zgrid, clouds[int(max_pos[0]), :, :].T, vmin=global_min, vmax=global_max,cmap='Blues_r' )
cb = fig.colorbar(img, cax=cax)
cb.set_label('Extra Buoyancy')
def animate(frame):
    ax.clear()
    img = ax.contourf(xgrid, zgrid, clouds[frame, :, : ].T,vmin=global_min, vmax=global_max,cmap='Blues_r')
    ax.set_title('Frame: {}'.format(frame))
    ax.set_xlabel('x')  # Add x-axis label
    ax.set_ylabel('z')  # Add y-axis label

# Call animate method
animation = FuncAnimation(fig, animate, frames=len(clouds), interval=100, blit=False)
animation.save('radiation3.gif', writer='imagemagick')
# Display the plot
plt.show()
