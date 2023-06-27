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

files = sorted(glob.glob('snapshots0/*.h5'),key=lambda f: int(re.sub('\D', '', f)))
fig, ax = plt.subplots()

with h5py.File(files[0], mode='r') as file:
    KEs0 = file['tasks']['total KE'][:,:,:]
    
for i in range(1,len(files)):
    with h5py.File(files[i], mode='r') as file:
        KE = file['tasks']['total KE'][:,:,:]
        KEs0=np.append(KEs0,KE,axis=0)

ax.plot(np.arange(len(KEs0[:,0,0]))/4*nu, KEs0[:,0,0])

files = sorted(glob.glob('snapshots1/*.h5'),key=lambda f: int(re.sub('\D', '', f)))

with h5py.File(files[0], mode='r') as file:
    KEs1 = file['tasks']['total KE'][:,:,:]
    
for i in range(1,len(files)):
    with h5py.File(files[i], mode='r') as file:
        KE = file['tasks']['total KE'][:,:,:]
        KEs1=np.append(KEs1,KE,axis=0)

ax.plot(np.arange(len(KEs1[:,0,0]))/4*nu, KEs1[:,0,0])

files = sorted(glob.glob('snapshots2/*.h5'),key=lambda f: int(re.sub('\D', '', f)))

with h5py.File(files[0], mode='r') as file:
    KEs2 = file['tasks']['total KE'][:,:,:]
    
for i in range(1,len(files)):
    with h5py.File(files[i], mode='r') as file:
        KE = file['tasks']['total KE'][:,:,:]
        KEs2=np.append(KEs2,KE,axis=0)

ax.plot(np.arange(len(KEs2[:,0,0]))/4*nu, KEs2[:,0,0])

ax.set_title('Total KE vs Time')
ax.grid(True)
ax.set_xlabel(r"Normalized Time $\nu t/H^2$")
ax.set_ylabel(r"Total KE  $\log_{10}E_k(t)$")
ax.set_yscale('log')


plt.savefig('Total_KE_vs_Time')


files = sorted(glob.glob('snapshots0/*.h5'),key=lambda f: int(re.sub('\D', '', f)))
with h5py.File(files[0], mode='r') as file:
    moist_buoyancy = file['tasks']['moist buoyancy'][:, :, :] 
    dry_buoyancy = file['tasks']['dry buoyancy'][:, :, :]
    buoyancy=np.maximum(moist_buoyancy,dry_buoyancy-Vaisala*z)
    extra_buoyancy=buoyancy-(dry_buoyancy-Vaisala*z)
    clouds = np.where(extra_buoyancy < 0, 0, extra_buoyancy)
for i in range(1,len(files)):
    with h5py.File(files[i], mode='r') as file:
        moist_buoyancy = file['tasks']['moist buoyancy'][:, :, :] 
        dry_buoyancy = file['tasks']['dry buoyancy'][:, :, :]
        buoyancy=np.maximum(moist_buoyancy,dry_buoyancy-Vaisala*z)
        extra_buoyancy=buoyancy-(dry_buoyancy-Vaisala*z)
        cloud = np.where(extra_buoyancy < 0, 0, extra_buoyancy)
        clouds=np.append(clouds,cloud,axis=0)
        
fig, ax = plt.subplots()
cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Define colorbar axes position

def animate(frame):
    ax.clear()
    img = ax.contourf(clouds[frame, :, : ].T,vmin=np.min(clouds[frame, :, :]), vmax=np.max(clouds[frame, :, :]),cmap='Blues_r')
    ax.set_title('Frame: {}'.format(frame))
    ax.set_xlabel('x')  # Add x-axis label
    ax.set_ylabel('z')  # Add y-axis label
    vmin = np.min(clouds[frame, :, :])
    vmax = np.max(clouds[frame, :, :])
    cb = plt.colorbar(img, cax=cax)
    cb.set_label('Extra Buoyancy')

# Call animate method
animation = FuncAnimation(fig, animate, frames=len(clouds), interval=100, blit=False)
animation.save('radiation0.gif', writer='imagemagick')
# Display the plot
plt.show()


files = sorted(glob.glob('snapshots1/*.h5'),key=lambda f: int(re.sub('\D', '', f)))
with h5py.File(files[0], mode='r') as file:
    moist_buoyancy = file['tasks']['moist buoyancy'][:, :, :] 
    dry_buoyancy = file['tasks']['dry buoyancy'][:, :, :]
    buoyancy=np.maximum(moist_buoyancy,dry_buoyancy-Vaisala*z)
    extra_buoyancy=buoyancy-(dry_buoyancy-Vaisala*z)
    clouds = np.where(extra_buoyancy < 0, 0, extra_buoyancy)
for i in range(1,len(files)):
    with h5py.File(files[i], mode='r') as file:
        moist_buoyancy = file['tasks']['moist buoyancy'][:, :, :] 
        dry_buoyancy = file['tasks']['dry buoyancy'][:, :, :]
        buoyancy=np.maximum(moist_buoyancy,dry_buoyancy-Vaisala*z)
        extra_buoyancy=buoyancy-(dry_buoyancy-Vaisala*z)
        cloud = np.where(extra_buoyancy < 0, 0, extra_buoyancy)
        clouds=np.append(clouds,cloud,axis=0)
        
fig, ax = plt.subplots()
cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Define colorbar axes position

def animate(frame):
    ax.clear()
    img = ax.contourf(clouds[frame, :, : ].T,vmin=np.min(clouds[frame, :, :]), vmax=np.max(clouds[frame, :, :]),cmap='Blues_r')
    ax.set_title('Frame: {}'.format(frame))
    ax.set_xlabel('x')  # Add x-axis label
    ax.set_ylabel('z')  # Add y-axis label
    vmin = np.min(clouds[frame, :, :])
    vmax = np.max(clouds[frame, :, :])
    cb = plt.colorbar(img, cax=cax)
    cb.set_label('Extra Buoyancy')

# Call animate method
animation = FuncAnimation(fig, animate, frames=len(clouds), interval=100, blit=False)
animation.save('radiation1.gif', writer='imagemagick')
# Display the plot
plt.show()

files = sorted(glob.glob('snapshots2/*.h5'),key=lambda f: int(re.sub('\D', '', f)))
with h5py.File(files[0], mode='r') as file:
    moist_buoyancy = file['tasks']['moist buoyancy'][:, :, :] 
    dry_buoyancy = file['tasks']['dry buoyancy'][:, :, :]
    buoyancy=np.maximum(moist_buoyancy,dry_buoyancy-Vaisala*z)
    extra_buoyancy=buoyancy-(dry_buoyancy-Vaisala*z)
    clouds = np.where(extra_buoyancy < 0, 0, extra_buoyancy)
for i in range(1,len(files)):
    with h5py.File(files[i], mode='r') as file:
        moist_buoyancy = file['tasks']['moist buoyancy'][:, :, :] 
        dry_buoyancy = file['tasks']['dry buoyancy'][:, :, :]
        buoyancy=np.maximum(moist_buoyancy,dry_buoyancy-Vaisala*z)
        extra_buoyancy=buoyancy-(dry_buoyancy-Vaisala*z)
        cloud = np.where(extra_buoyancy < 0, 0, extra_buoyancy)
        clouds=np.append(clouds,cloud,axis=0)
        
fig, ax = plt.subplots()
cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # Define colorbar axes position

def animate(frame):
    ax.clear()
    img = ax.contourf(clouds[frame, :, : ].T,vmin=np.min(clouds[frame, :, :]), vmax=np.max(clouds[frame, :, :]),cmap='Blues_r')
    ax.set_title('Frame: {}'.format(frame))
    ax.set_xlabel('x')  # Add x-axis label
    ax.set_ylabel('z')  # Add y-axis label
    vmin = np.min(clouds[frame, :, :])
    vmax = np.max(clouds[frame, :, :])
    cb = plt.colorbar(img, cax=cax)
    cb.set_label('Extra Buoyancy')

# Call animate method
animation = FuncAnimation(fig, animate, frames=len(clouds), interval=100, blit=False)
animation.save('radiation2.gif', writer='imagemagick')
# Display the plot
plt.show()

