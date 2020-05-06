import pandas as pd
import numpy as np
from mod_genetic import run_genetic as rg
from mod_packs import quantum_potential as qp
import matplotlib.pyplot as plt
import itertools as it

price_df = pd.read_excel('PRICE INDEX.xlsx').dropna().set_index('Date')
price_val = price_df.T.values
names = [price_df.columns[i][:3] for i in range(8)]

staff = [0,1,2,3,4,5,6,7]
subs = []
for i in range(len(staff)):
    for subset in it.combinations(staff,i):
        subs.append(subset)
subs = subs[9:]        
# kk = 11
for kk in range(240,258):
    data = [qp.scaled_return(price_val[i],1) for i  in subs[kk]]

    p_count = 50
    steps = 100
    retain = 0.3
    random_select = 0.3
    mutate = 0.3
    band_width = 0.0005
    fitness = rg(data , p_count , steps, retain , random_select , mutate, band_width )

    np.save(str(kk)+'fitness_p_count'+str(p_count)+'steps'+str(steps)+'retain'+str(retain)+'rnd'+\
    str(random_select)+'mutate'+str(mutate)+str(np.random.random()) , np.array(fitness), allow_pickle=True)