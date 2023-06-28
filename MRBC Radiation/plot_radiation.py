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


Lx, Lz = 4, 1
Nx, Nz = 256, 64
Md = 3
Rayleigh = 1e6
Vaisala= 4
Prandtl = 0.7
dealias = 3/2
QD=0
QM=QD/2
stop_sim_time = 200
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
kappa = (Lz**3*Md)/(Rayleigh * Prandtl)**(1/2)
nu = (Lz**3*Md)/(Rayleigh / Prandtl)**(1/2)
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
plt.savefig('Vertical Profile')
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


