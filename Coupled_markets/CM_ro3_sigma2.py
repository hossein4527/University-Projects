import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

def tot5(lattice):
    total_dev = np.zeros(lattice.shape)
    for i in range(lattice.shape[0]):
        for j  in range(lattice.shape[1]):
            if lattice[i,j] > 5:
                total_dev[i,j] = 5
            else:
                total_dev[i,j] = lattice[i,j]
    return total_dev
def tot(lattice):
    total_dev = np.zeros(lattice.shape)
    for i in range(lattice.shape[0]):
        for j  in range(lattice.shape[1]):
            if lattice[i,j] > 10:
                total_dev[i,j] = 10
            else:
                total_dev[i,j] = lattice[i,j]
    return total_dev

num_of_fig = '12'
number_of_toys = 100
ro = 1
var = 0.008975847639977273
sx = 2*var
sy = var

pdf = np.zeros((number_of_toys,number_of_toys))
Q_xy = np.zeros(pdf.shape)
Q_yx = np.zeros(pdf.shape)
Q_tot = np.zeros(pdf.shape)
ret_list = np.linspace(-0.05 , 0.05 , number_of_toys)

for index1 , x in enumerate(ret_list) :
    for index2 , y in enumerate(ret_list) :
        pdf[index1,index2] = np.exp(0.5*(-(x/sx)**2-(y/sy)**2 +2*ro*(x/sx)*(y/sy))) 
        Q_xy[index1,index2] = (x/sx + ro*y/sy)**2 -1
        Q_yx[index1,index2] = (y/sy + ro*x/sx)**2 -1
pdf/=np.sum(pdf)
Q_tot = Q_xy + Q_yx


fig = plt.figure(figsize=(7,5.5))
plt.imshow(tot5(pdf),interpolation='sinc',cmap='Greys',origin='lower',extent=[ret_list[0],ret_list[-1],ret_list[0],ret_list[-1]])
plt.xlabel('$y$',fontsize=24)
plt.ylabel(r'$\rho$ = '+str(abs(ro)),fontsize=27)
plt.colorbar()
plt.savefig('Fig:('+num_of_fig+'a)' ,dpi=200)

fig = plt.figure(figsize=(7,5.5))
plt.imshow(tot5(Q_xy),interpolation='sinc',cmap='Greys',extent=[ret_list[0],ret_list[-1],ret_list[0],ret_list[-1]])
plt.xlabel('$y$',fontsize=24)
plt.ylabel('$x$',fontsize=24)
plt.colorbar()
plt.savefig('Fig:('+num_of_fig+'b)' ,dpi=200)

fig = plt.figure(figsize=(7,5.5))
plt.imshow(tot5(Q_yx),interpolation='sinc',cmap='Greys',extent=[ret_list[0],ret_list[-1],ret_list[0],ret_list[-1]])
plt.xlabel('$y$',fontsize=24)
plt.ylabel('$x$',fontsize=24)
plt.colorbar()
plt.savefig('Fig:('+num_of_fig+'c)' ,dpi=200)

fig = plt.figure(figsize=(7,5.5))
plt.imshow(tot(Q_tot),cmap='Greys',interpolation='sinc',extent=[ret_list[0],ret_list[-1],ret_list[0],ret_list[-1]],origin='upper')
plt.ylabel('$x$',fontsize=24)
plt.xlabel('$y$',fontsize=24)
plt.colorbar()
plt.savefig('Fig:('+num_of_fig+'d)' ,dpi=200)
