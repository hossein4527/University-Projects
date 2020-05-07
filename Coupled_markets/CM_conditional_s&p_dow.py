import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from my_packs import coupled_markets as cm

data1 = pd.read_csv('data-s&p.csv')
data2 = pd.read_csv('data-dow.csv')

data1_price = data1['price']
data2_price = data2['price']

ret1 = cm.price_return(data1_price)
ret2 = cm.price_return(data2_price)

pdf = cm.joint_pdf(ret1,ret2,0.001,0.001)
kerd = cm.kernel_2d(pdf['pdf'],5,5)
lat_pot = cm.quantum_potential(pdf['pdf'])
linebazi = cm.linebazi(lat_pot['Q_y'])

simple_pdf = cm.simple_pdf(ret2,0.001)
ret2_pdf = simple_pdf['simple_pdf']

cond_pdf_xy = np.zeros(kerd.shape)
for i in range(cond_pdf_xy.shape[0]):
    for j in range(cond_pdf_xy.shape[1]):
        cond_pdf_xy[i][j] = pdf['pdf'][i][j]/ret2_pdf[j]

simple_pdf = cm.simple_pdf(ret1,0.001)
ret1_pdf = simple_pdf['simple_pdf']

cond_pdf_yx = np.zeros(kerd.shape)
for i in range(cond_pdf_yx.shape[0]):
    for j in range(cond_pdf_yx.shape[1]):
        cond_pdf_yx[i][j] = pdf['pdf'][i][j]/ret1_pdf[j]


cond_pot_yx = cm.quantum_potential(cond_pdf_yx)
# cond_line_yxx = cm.linebazi(cond_pot_yx['Q_x'])
# cond_line_yxy = cm.linebazi(cond_pot_yx['Q_y'])

cond_pot_xy = cm.quantum_potential(cond_pdf_xy)
# cond_line_xyx = cm.linebazi(cond_pot_xy['Q_x'])
# cond_line_xyy = cm.linebazi(cond_pot_xy['Q_y'])

fig = plt.figure(figsize=(16,9))
fig.add_subplot(2,2,1)
plt.imshow(abs(cond_pot_xy['Q_x']) ,cmap='Greys' ,origin = 'lower',extent=[pdf['y_values'][0],pdf['y_values'][-1],pdf['x_values'][0],pdf['x_values'][-1]])
plt.colorbar()

fig.add_subplot(2,2,2)
plt.imshow(abs(cond_pot_xy['Q_y']) ,cmap='Greys' ,origin = 'lower',extent=[pdf['y_values'][0],pdf['y_values'][-1],pdf['x_values'][0],pdf['x_values'][-1]])
plt.colorbar()

fig.add_subplot(2,2,3)
plt.imshow(abs(cond_pot_yx['Q_x']) ,cmap='Greys' ,origin = 'lower',extent=[pdf['y_values'][0],pdf['y_values'][-1],pdf['x_values'][0],pdf['x_values'][-1]])
plt.colorbar()


fig.add_subplot(2,2,4)
plt.imshow(abs(cond_pot_yx['Q_y']) ,cmap='Greys' ,origin = 'lower',extent=[pdf['y_values'][0],pdf['y_values'][-1],pdf['x_values'][0],pdf['x_values'][-1]])
plt.colorbar()


plt.show()

print(cond_pot_xy['Q_x'])
