import pandas as pd
import numpy as np
from mod_genetic import run_genetic_ret as rg
from mod_packs import quantum_potential as qp
import matplotlib.pyplot as plt

price_df = pd.read_excel('PRICE INDEX.xlsx').dropna().set_index('Date')
price_val = price_df.T.values
names = [price_df.columns[i][:3] for i in range(8)]

data = [qp.scaled_return(price_val[i],1) for i  in range(8)]
p_count = 50
steps = 1000
retain = 0.3
random_select = 0.3
mutate = 0.3
band_width = 0.0005
fitness = rg(data , p_count , steps, retain , random_select , mutate, band_width )

np.save('fitness_p_count'+str(p_count)+'steps'+str(steps)+'retain'+str(retain)+'rnd'+\
str(random_select)+'mutate'+str(mutate)+str(np.random.random()) , np.array(fitness), allow_pickle=True)