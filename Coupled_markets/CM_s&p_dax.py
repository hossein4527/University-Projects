import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from my_packs import coupled_markets as cm

data1 = pd.read_csv('data-s&p.csv')
data2 = pd.read_csv('data-dax.csv')

data1_price = data1['price']
data2_price = data2['price']

for betta in [1/100,1/10,1,10,100]:
    ret1 = cm.price_return(data1_price)
    ret2 = cm.price_return(data2_price)

    pdf = cm.joint_pdf(ret1,ret2,0.001,0.001)
    kerd = cm.kernel_2d(pdf['pdf'],5,5)
    lat_pot = cm.quantum_potential(kerd,betta)
    # linebazi = cm.linebazi(lat_pot['Q_y'])
    linebazi = lat_pot['Q_tot']
    fig = plt.figure(figsize=(16,9))
    plt.imshow(abs(linebazi), cmap='Greys' ,origin = 'lower',extent=[pdf['y_values'][0],pdf['y_values'][-1],pdf['x_values'][0],pdf['x_values'][-1]])
    plt.colorbar()
    plt.title('betta = '+str(betta))
    plt.xlabel('Dax',fontsize=22)
    plt.ylabel('S&P500',fontsize=22)
    # plt.savefig('Dax_S&P',dpi=300)
    plt.show()
