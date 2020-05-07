import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm

num_of_fig = '3'
number_of_toys = 100
ro = 1
sx = 0.008975847639977273
sy = sx

pdf = np.zeros((number_of_toys,number_of_toys))
Q_xy = np.zeros(pdf.shape)
Q_yx = np.zeros(pdf.shape)
Q_tot = np.zeros(pdf.shape)
return_list = np.linspace(-0.05 , 0.05 , number_of_toys)

for index1 , x in enumerate(return_list) :
    for index2 , y in enumerate(return_list) :
        pdf[index1,index2] = np.exp(0.5*(-(x/sx)**2-(y/sy)**2 +2*ro*(x/sx)*(y/sy))) 
        Q_xy[index1,index2] = (x/sx + ro*y/sy)**2 -1
        Q_yx[index1,index2] = (y/sy + ro*x/sx)**2 -1
pdf/=np.sum(pdf)
Q_tot = Q_xy + Q_yx

X = return_list
Y = return_list
X , Y = np.meshgrid(Y , X)

funcs = [pdf,Q_xy,Q_yx,Q_tot]
sub_figs = ['a','b','c','d']
for i,f in enumerate(funcs):
    fig = plt.figure()
    ax = fig.gca(projection = '3d')
    surf = ax.plot_surface(X , Y , f , rstride=1, cstride=1 , cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)
    ax.set_xlabel('$y$',fontsize=17)
    ax.set_ylabel('$x$',fontsize=17)
    plt.savefig('Fig:('+num_of_fig+sub_figs[i]+')' , dpi = 200)
#     plt.show()
 
