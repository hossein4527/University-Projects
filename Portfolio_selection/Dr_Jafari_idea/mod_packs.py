#-----------------------------------------------------------------------#
#----Used libraried inside class functions below------------------------#
#-----------------------------------------------------------------------#
import numpy as np
from numpy import sum,zeros,amin,amax,ceil,log,array,exp,sort,diff
from random import random , randint
from functools import reduce
from operator import add
from sklearn.neighbors.kde import KernelDensity
from scipy.signal import argrelextrema
#-----------------------------------------------------------------------#
#----Coupled markets modules as 2-D quantum potential---
#-----------------------------------------------------------------------#
class coupled_markets(object):
    """docstring for price_return."""
    def __init__(self, price_series):
        # super(price_return, self).__init__()
        self.price = price_series

    def price_return(p):
        pr = []
        for i in range(len(p)-1):
            pr.append(np.log(p[i+1])-np.log(p[i]))
        return pr

    def joint_pdf(data1,data2,delta_x,delta_y):
        dlim_data1 = int(min(data1)/delta_x) * delta_x
        dlim_data2 = int(min(data2)/delta_y) * delta_y
        ulim_data1 = int(max(data1)/delta_x) * delta_x
        ulim_data2 = int(max(data2)/delta_y) * delta_y
        x_values = np.zeros(int(((ulim_data1 - dlim_data1) / delta_x) + 1))
        y_values = np.zeros(int(((ulim_data2 - dlim_data2) / delta_y) + 1))
        pdf = np.zeros((int(((ulim_data1 - dlim_data1) / delta_x) + 1),\
        int(((ulim_data2 - dlim_data2) / delta_y) + 1)))
        for i in range(0,len(x_values)):
            x_values[i] = dlim_data1 + (i * delta_x)
        for i in range(0,len(y_values)):
            y_values[i] = dlim_data2 + (i * delta_y)

        for i in range(min(len(data1),len(data2))):
            block_one = int((data1[i]-dlim_data1)/delta_x)
            block_two = int((data2[i]-dlim_data2)/delta_y)
            pdf[block_one][block_two]+=1
        # self.pdf/=(len(self.pdf)**2 * self.delta_x)
        # self.pdf/=(sum(self.pdf) * self.delta_x)
        # pdf/=sum(pdf)
        # self.pdf/=(len(self.data1)*len(self.data2))
        return {'pdf':pdf,'x_values':x_values,'y_values':y_values,'dlim':dlim_data1}

    def quantum_potential(pdf,betta):
        sec_xdev = np.zeros(pdf.shape)
        for i in range(2,pdf.shape[0]-2):
            for j in range(pdf.shape[1]):
                sec_xdev[i][j] = (-pdf[i+2][j]+8*pdf[i+1][j]\
                -8*pdf[i-1][j]+pdf[i-2][j])/(12*1)

        sec_ydev = np.zeros(pdf.shape)
        for i in range(pdf.shape[0]):
            for j in range(2,pdf.shape[1]-2):
                sec_ydev[i][j] = (-pdf[i][j+2]+8*pdf[i][j+1]\
                -8*pdf[i][j-1]+pdf[i][j-2])/(12*1)

        Q_x = np.array(sec_xdev)/np.array(pdf)
        Q_y = np.array(sec_ydev)/np.array(pdf)
        Q_tot = Q_x + betta*Q_y

        return {'Q_x':Q_x , 'Q_y':Q_y , 'Q_tot':Q_tot}

    def kernel_2d(grid,sigma1,sigma2):
        data_kernel=np.zeros(grid.shape)
        for i in range(sigma1,grid.shape[0]-sigma1):
            for k in range(sigma2,grid.shape[1]-sigma2):
                for j in range(-sigma1,sigma1):
                    for l in range(-sigma2,sigma2):
                        data_kernel[i,k]+=grid[i+j][k+l]*np.exp(-0.5*((j/sigma1)**2+(l/sigma2)**2))

        data_kernel/=np.sum(data_kernel)
        return data_kernel

    def linebazi(grid1):
        grid = grid1.copy()
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if np.isnan(grid[i][j]):
                    grid[i][j] = 0
                elif grid[i][j] > 100000:
                    grid[i][j] = 1
                elif grid[i][j] < -100000:
                    grid[i][j] = 1   
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if grid[i,j] == 0 :
                    grid[i,j] = 3

        return grid        


    def simple_pdf(data,delta) :
        down_limit = int(min(data)/delta)*delta
        up_limit = int(max(data)/delta)*delta
        pdf = np.zeros(int((up_limit-down_limit)/delta)+1)
        x_axis = np.zeros(pdf.shape[0])
        for i in range(len(x_axis)):
            x_axis[i] = down_limit + i*delta
        for i in range(len(data)):
            block = int((data[i]-down_limit)/delta)
            pdf[block] +=1

        pdf /= np.sum(pdf)    
        return {'x_axis':x_axis , 'simple_pdf':pdf}

    def kernel(data,sigma):
        data_kernel=zeros(len(data))
        for i in range(sigma,len(data)-sigma):
            for j in range(-sigma,sigma):
                data_kernel[i]+=data[i+j]*exp(-0.5*(j/sigma)**2)
        data_kernel/=sum(data_kernel)
        return data_kernel    


    def quasi_qpot(pdf):
        sec_xdev = np.zeros(pdf.shape)
        for i in range(2,pdf.shape[0]-2):
            for j in range(pdf.shape[1]):
                sec_xdev[i][j] = (-pdf[i+2][j]+8*pdf[i+1][j]\
                -8*pdf[i-1][j]+pdf[i-2][j])/(12*1)

        sec_ydev = np.zeros(pdf.shape)
        for i in range(pdf.shape[0]):
            for j in range(2,pdf.shape[1]-2):
                sec_ydev[i][j] = (-pdf[i][j+2]+8*pdf[i][j+1]\
                -8*pdf[i][j-1]+pdf[i][j-2])/(12*1)

        xdev = np.zeros(pdf.shape)
        for i in range(2,pdf.shape[0]-2):
            for j in range(pdf.shape[1]):
                xdev[i][j] = (pdf[i+1][j]-pdf[i-1][j])/2

        ydev = np.zeros(pdf.shape)
        for i in range(pdf.shape[0]):
            for j in range(2,pdf.shape[1]-2):
                ydev[i][j] = (pdf[i][j+1]-pdf[i][j-1])/2        

        vq1 = ((1/2*np.array(pdf))*np.array(xdev)**2 - sec_xdev)/np.array(pdf)
        vq2 = ((1/2*np.array(pdf))*np.array(ydev)**2 - sec_ydev)/np.array(pdf)
        vq_tot = vq1 + vq2
        return {'Q_x':vq1 , 'Q_y':vq2 , 'Q_tot':vq_tot}     

    def quantum_pot1d1(pdf):
        sec_dev = np.zeros(len(pdf))
        for i in range(2,len(pdf)-2):
            sec_dev[i] = (-pdf[i+2]+8*pdf[i+1] -8*pdf[i-1]+pdf[i-2])/(12*1)

        Q = np.array(sec_dev)/np.array(pdf)

        return Q

    def quantum_pot1d2(pdf):
        sec_dev = np.zeros(len(pdf))
        for i in range(2,len(pdf)-2):
            sec_dev[i] = (pdf[i+2]+pdf[i-2] -2*pdf[i])/4

        Q = np.array(sec_dev)/np.array(pdf)

        return Q
    def numpy_qpot(pdf):
        sec_dev = np.gradient(np.gradient(pdf))
        qpot = sec_dev/np.array(pdf)
        return qpot    

    def quantum_pot1d3(pdf):
        sec_dev = np.zeros(len(pdf))
        for i in range(2,len(pdf)-2):
            sec_dev[i] = (469/90)*pdf[i]-(223/10)*pdf[i-1]+\
            (879/20)*pdf[i-2]-(949/18)*pdf[i-3]+\
            41*pdf[i-4]-(201/10)*pdf[i-5]+\
            (1019/180)*pdf[i-6]-(7/10)*pdf[i-7]

        Q = np.array(sec_dev)/np.array(pdf)

        return Q  
#-----------------------------------------------------------------------#
#----A handwritten non_uniform pdf estimation and numpy fitting proccess#
#-----------------------------------------------------------------------#
class non_uniform_pdf(object):
    """bin such that in every bin there is a particle"""
    def __init__(self, dx1,dx2):
        self.delta_x1 = dx1
        self.delta_x2 = dx2
        self.sort_ret = []
        self.data_count = 0
        self.d_limit1 = 0
        self.u_limit1 = 0
        self.d_limit2 = 0
        self.u_limit2 = 0
        self.d_limit3 = 0
        self.u_limit3 = 0
        self.Final_x = []
        self.Final_pdf = []


    def set_data_list(self, data_list):
        self.sort_ret = sort(data_list)
        self.data_count = len(self.sort_ret)

    def pdf_nu_func(self):
        sort_ret1 = self.sort_ret[(self.sort_ret < -0.01)]
        sort_ret2 = self.sort_ret[(self.sort_ret >= -0.01 )&(self.sort_ret <= 0.01)]
        sort_ret3 = self.sort_ret[(self.sort_ret > 0.01)]
        self.d_limit1 = int(amin(sort_ret1) / self.delta_x2) * self.delta_x2
        self.d_limit2 = int(amin(sort_ret2) / self.delta_x1) * self.delta_x1
        self.d_limit3 = int(amin(sort_ret3) / self.delta_x2) * self.delta_x2
        self.u_limit1 = int(amax(sort_ret1) / self.delta_x2 + 1) * self.delta_x2
        self.u_limit2 = int(amax(sort_ret2) / self.delta_x1 + 1) * self.delta_x1
        self.u_limit3 = int(amax(sort_ret3) / self.delta_x2 + 1) * self.delta_x2
        num_oF_bins1 = int((amax(sort_ret1) - amin(sort_ret1))/self.delta_x2) + 1
        num_oF_bins2 = int((amax(sort_ret2) - amin(sort_ret2))/self.delta_x1) + 1
        num_oF_bins3 = int((amax(sort_ret3) - amin(sort_ret3))/self.delta_x2) + 1
        pdf1 = zeros((num_oF_bins1,2))
        pdf2 = zeros((num_oF_bins2,2))
        pdf3 = zeros((num_oF_bins3,2))
        for i in range(pdf1.shape[0]):
            pdf1[i,0] = self.d_limit1 + i*self.delta_x2
        for i in range(pdf2.shape[0]):
            pdf2[i,0] = self.d_limit2 + i*self.delta_x1
        for i in range(pdf3.shape[0]):
            pdf3[i,0] = self.d_limit3 + i*self.delta_x2

        for i in range(len(sort_ret1)):
            block = int((sort_ret1[i]-self.d_limit1)/self.delta_x2)
            pdf1[block,1] += 1/self.delta_x2
        for i in range(len(sort_ret2)):
            block = int((sort_ret2[i]-self.d_limit2)/self.delta_x1)
            pdf2[block,1] += 1/self.delta_x1
        for i in range(len(sort_ret3)):
            block = int((sort_ret3[i]-self.d_limit3)/self.delta_x2)
            pdf3[block,1] += 1/self.delta_x2

        pdf1[:,1]/=self.data_count
        pdf2[:,1]/=self.data_count
        pdf3[:,1]/=self.data_count

        # pdf1[:,1]/=(sum(pdf1[:,1])*self.delta_x2)
        # pdf2[:,1]/=(sum(pdf2[:,1])*self.delta_x1)
        # pdf3[:,1]/=(sum(pdf3[:,1])*self.delta_x2)


        pdf2[0,1] = 2*(pdf2[0,1]+pdf1[-1,1])/4
        pdf1[-1,1] = pdf2[0,1]/1.2
        pdf2[-1,1] = (pdf2[-2,1]+pdf3[0,1])/2

        left_fit = np.polyfit(pdf1[:,0],pdf1[:,1],4)
        left_fet = np.poly1d(left_fit)

        right_fit = np.polyfit(pdf3[:,0],pdf3[:,1],4)
        right_fet = np.poly1d(right_fit)

        xx1 = np.arange(pdf1[0,0],pdf1[-1,0], self.delta_x1)
        for i in xx1:
            self.Final_x.append(i)
            self.Final_pdf.append(left_fet(i))

        # for i in range(pdf1.shape[0]):
        #     self.Final_x.append(pdf1[i,0])
        #     self.Final_pdf.append(pdf1[i,1])

        for i in range(pdf2.shape[0]):
            self.Final_x.append(pdf2[i,0])
            self.Final_pdf.append(pdf2[i,1])

        # for i in range(pdf3.shape[0]):
        #     self.Final_x.append(pdf3[i,0])
        #     self.Final_pdf.append(pdf3[i,1])

        xx2 = np.arange(pdf3[0,0],pdf3[-1,0], self.delta_x1)
        for i in xx2:
            self.Final_x.append(i)
            self.Final_pdf.append(right_fet(i))
#-----------------------------------------------------------------------#
#---Genetic algorithm for optimizing risk function of a given portfolio-#
#-----------------------------------------------------------------------#
class GA(object):
    """genetic algorithm written for risk."""
    def __init__(self, portfo_prices):
        # super(price_return, self).__init__()
        self.ret = portfo_prices
        self.length = len(portfo_prices)
        
    def individual(self):
        suu = np.array([ random() for x in range(self.length) ])
        return suu/np.sum(suu)

    def population(self,count):
        return [self.individual() for x in range(count)]

    def fitness(self,individual, markets):
        portfo_return = np.zeros(len(markets[0]))
        for j in range(len(markets[0])):
            portfo_return[j] = np.dot(np.array(individual) , np.array(markets).T[j])
        
        pdf = non_uniform_pdf(0.001,0.01)
        pdf.set_data_list(portfo_return)
        pdf.pdf_nu_func()
        kerd = coupled_markets.kernel(pdf.Final_pdf,5)
        pot = coupled_markets.quantum_pot1d2(kerd)
        
        andis1 = []
        andis2 = []
        for i in range(len(pot)-1):
            if abs(pot[i]) == np.inf and abs(pot[i+1]) != np.inf:
                andis1.append(i)
        for i in range(len(pot)):
            if abs(pot[len(pot)-1-i]) == np.inf and abs(pot[len(pot)-2-i]) != np.inf:
                andis2.append(len(pot)-1-i)
        risk = pdf.Final_x[andis2[0]]-pdf.Final_x[andis1[0]]
        
        return risk

    def grade(self,pop , markets):
        summed = reduce(add, (self.fitness(x, markets) for x in pop), 0)
        return summed/(len(pop)*1.0)

    def evolve(self,pop, markets, retain=0.5, random_select=0.05, mutate=0.01):
        graded1 = [ (self.fitness(x, markets),list(x)) for x in pop]
        graded = [ x[1] for x in sorted(list(graded1))]
        

        retain_length = int(len(graded)*retain)
        parents = graded[:retain_length]
        
        for individual in graded[retain_length:]:
            if random_select > random():
                parents.append(individual)
        
        for individual in parents:
            if mutate > random():
                pos_to_mutate = randint(0, len(individual)-1)
                individual[pos_to_mutate] = random()*(-min(individual)+max(individual)) + min(individual)
                # individual[pos_to_mutate] = random()
                individual/=sum(individual)
        
        parents_length = len(parents)
        desired_length = len(pop) - parents_length
        children = []
        while len(children) < desired_length:
            male = randint(0, parents_length-1)
            female = randint(0, parents_length-1)
            if male != female:
                male = parents[male]
                female = parents[female]
                half = int(len(male) / 2)
                child = np.concatenate((male[:half] , female[half:]))
                child/=sum(child)
                children.append(child)
                
        parents.extend(children)
        
        return parents    
class GA_new(object):
    """genetic algorithm written for risk."""
    def __init__(self, portfo_prices):
        # super(price_return, self).__init__()
        self.ret = portfo_prices
        self.length = len(portfo_prices)
        
    def individual(self):
        suu = np.array([ random() for x in range(self.length) ])
        return suu/np.sum(suu)

    def population(self,count):
        return [self.individual() for x in range(count)]

    def fitness(self,individual, markets):
        portfo_return = np.zeros(len(markets[0]))
        for j in range(len(markets[0])):
            portfo_return[j] = np.dot(np.array(individual) , np.array(markets).T[j])

        ret = portfo_return
        X = ret[:, np.newaxis]
        X_plot = np.linspace(min(ret),max(ret), 200)[:, np.newaxis]
        kde = KernelDensity(kernel='gaussian', bandwidth=np.std(X)*0.009).fit(X)
        log_dens = kde.score_samples(X_plot)
        pdf = np.exp(log_dens)

        sec_dev = np.gradient(np.gradient(pdf))
        qpot = [500]
        for i in range(1,len(pdf)-1):
            try:
                jj = sec_dev[i-1]/pdf[i]
            except ZeroDivisionError:
                jj=500
            qpot.append(jj) 
        qpot.append(500)

        # dd = X_plot[argrelextrema(qpot, np.greater)]
        # risk = dd[dd>0][0] - dd[dd<0][-1]

        xx =[]
        x = X_plot.reshape(len(qpot))
        for i in range(len(qpot)):
            if qpot[i] >= 10000:
                xx.append(i)
        x_list = np.array(x)[xx]
        d_lim = x_list[x_list<0][-1]
        u_lim = x_list[x_list>0][0]

        return u_lim-d_lim


    def grade(self,pop , markets):
        summed = reduce(add, (self.fitness(x, markets) for x in pop), 0)
        return summed/(len(pop)*1.0)

    def evolve(self,pop, markets, retain=0.3, random_select=0.2, mutate=0.1):
        graded1 = [ (self.fitness(x, markets),list(x)) for x in pop]
        graded = [ x[1] for x in sorted(list(graded1))]
        

        retain_length = int(len(graded)*retain)
        parents = graded[:retain_length]
        
        for individual in graded[retain_length:]:
            if random_select > random():
                parents.append(individual)
        
        for i in range(len(parents)):
            if mutate > random():
                pos_to_mutate = randint(0, len(individual)-1)
                parents[i][pos_to_mutate] = random()*(-min(individual)+max(individual)) + min(individual)
                parents[i] /= np.sum(parents[i])
        
        parents_length = len(parents)
        desired_length = len(pop) - parents_length
        children = []
        while len(children) < desired_length:
            male = randint(0, parents_length-1)
            female = randint(0, parents_length-1)
            if male != female:
                male = parents[male]
                female = parents[female]
                half = int(len(male) / 2)
                child = np.concatenate((male[:half] , female[half:]))
                child/=np.sum(child)
                children.append(child)
                
        parents.extend(children)
        
        return parents   
class GA_std(object):
    """genetic algorithm written for risk."""
    def __init__(self, portfo_prices):
        # super(price_return, self).__init__()
        self.ret = portfo_prices
        self.length = len(portfo_prices)
        
    def individual(self):
        suu = np.array([ random() for x in range(self.length) ])
        return suu/np.sum(suu)

    def population(self,count):
        return [self.individual() for x in range(count)]

    def fitness(self,individual, markets):
        portfo_return = np.zeros(len(markets[0]))
        for j in range(len(markets[0])):
            portfo_return[j] = np.dot(np.array(individual) , np.array(markets).T[j])
        return np.std(portfo_return)


    def grade(self,pop , markets):
        summed = reduce(add, (self.fitness(x, markets) for x in pop), 0)
        return summed/(len(pop)*1.0)

    def evolve(self,pop, markets, retain=0.3, random_select=0.2, mutate=0.1):
        graded1 = [ (self.fitness(x, markets),list(x)) for x in pop]
        graded = [ x[1] for x in sorted(list(graded1))]
        

        retain_length = int(len(graded)*retain)
        parents = graded[:retain_length]
        
        for individual in graded[retain_length:]:
            if random_select > random():
                parents.append(individual)
        
        for i in range(len(parents)):
            if mutate > random():
                pos_to_mutate = randint(0, len(individual)-1)
                parents[i][pos_to_mutate] = random()*(-min(individual)+max(individual)) + min(individual)
                parents[i] /= np.sum(parents[i])
        
        parents_length = len(parents)
        desired_length = len(pop) - parents_length
        children = []
        while len(children) < desired_length:
            male = randint(0, parents_length-1)
            female = randint(0, parents_length-1)
            if male != female:
                male = parents[male]
                female = parents[female]
                half = int(len(male) / 2)
                child = np.concatenate((male[:half] , female[half:]))
                child/=np.sum(child)
                children.append(child)
                
        parents.extend(children)
        
        return parents        
class GA_less(object):
    """genetic algorithm written for risk."""
    def __init__(self, portfo_prices):
        # super(price_return, self).__init__()
        self.ret = portfo_prices
        self.length = len(portfo_prices)
        
    def individual(self):
        suu = np.array([ random() for x in range(self.length) ])
        return suu/np.sum(suu)

    def population(self,count):
        return [self.individual() for x in range(count)]

    def fitness(self,individual, markets):
        portfo_return = np.zeros(len(markets[0]))
        for j in range(len(markets[0])):
            portfo_return[j] = np.dot(np.array(individual) , np.array(markets).T[j])

        ret = portfo_return
        X = ret[:, np.newaxis]
        X_plot = np.linspace(min(ret),max(ret), 200)[:, np.newaxis]
        kde = KernelDensity(kernel='gaussian', bandwidth=np.std(X)).fit(X)
        log_dens = kde.score_samples(X_plot)
        pdf = np.exp(log_dens)

        sec_dev = np.gradient(np.gradient(pdf))
        qpot = sec_dev/np.array(pdf)

        dd = X_plot[argrelextrema(qpot, np.greater)]
        risk = dd[dd<0][-1]
        return abs(risk)

    def grade(self,pop , markets):
        summed = reduce(add, (self.fitness(x, markets) for x in pop), 0)
        return summed/(len(pop)*1.0)

    def evolve(self,pop, markets, retain=0.3, random_select=0.2, mutate=0.1):
        graded1 = [ (self.fitness(x, markets),list(x)) for x in pop]
        graded = [ x[1] for x in sorted(list(graded1))]
        

        retain_length = int(len(graded)*retain)
        parents = graded[:retain_length]
        
        for individual in graded[retain_length:]:
            if random_select > random():
                parents.append(individual)
        
        for i in range(len(parents)):
            if mutate > random():
                pos_to_mutate = randint(0, len(individual)-1)
                parents[i][pos_to_mutate] = random()*(-min(individual)+max(individual)) + min(individual)
                parents[i] /= np.sum(parents[i])
        
        parents_length = len(parents)
        desired_length = len(pop) - parents_length
        children = []
        while len(children) < desired_length:
            male = randint(0, parents_length-1)
            female = randint(0, parents_length-1)
            if male != female:
                male = parents[male]
                female = parents[female]
                half = int(len(male) / 2)
                child = np.concatenate((male[:half] , female[half:]))
                child/=np.sum(child)
                children.append(child)
                
        parents.extend(children)
        
        return parents  
#-----------------------------------------------------------------------#
#----1-D quantumm potential using sklearn and numpy---------------------#
#-----------------------------------------------------------------------#
class quantum_potential(object):
    """docstring for price_return."""
    # def __init__(self):
    #     # super(price_return, self).__init__()
    #     self.ret

    def pdf_sklearn( ret,num):
        X = ret[:, np.newaxis]
        X_plot = np.linspace(min(ret),max(ret), 200)[:, np.newaxis]
        kde = KernelDensity(kernel='gaussian', bandwidth=np.std(X)*num).fit(X)
        log_dens = kde.score_samples(X_plot)
        pdf = np.exp(log_dens)
        return X_plot.reshape(len(pdf)) , pdf

    def risk( x , pot):
        dd = x[argrelextrema(pot, np.greater)]
        risk = dd[dd>0][0] - dd[dd<0][-1]
        return risk

    def new_risk(x,pot):
        xx =[]
        for i in range(len(pot)):
            if pot[i] >= 499:
                xx.append(i)
        x_list = np.array(x)[xx]
        d_lim = x_list[x_list<0][-1]
        u_lim = x_list[x_list>0][0]
        return d_lim , u_lim

    def loss(  x , pot):
        dd = x[argrelextrema(pot, np.greater)]
        loss = dd[dd<0][-1]
        return abs(loss)   

    def gain( x , pot):
        dd = x[argrelextrema(pot, np.greater)]
        loss = dd[dd>0][0]
        return (loss)        

    def numpy_qpot( data , band_width):

        X = data[:, np.newaxis]
        X_plot = np.linspace(min(data),max(data), 200)[:, np.newaxis]
        kde = KernelDensity(kernel='gaussian', bandwidth=band_width).fit(X)
        log_dens = kde.score_samples(X_plot)
        pdf = np.exp(log_dens)

        sec_dev = np.diff(pdf,2)
        qpot = [500.0]
        for i in range(1,len(pdf)-1):
            if pdf[i] > 0.0001:
                jj = sec_dev[i-1]/pdf[i]
            else:
                jj=500
            qpot.append(jj) 
        qpot.append(500)
        return X_plot.reshape(len(pdf)) , qpot    

    def scaled_return(data , scale):
        ret_list = []
        for i in np.arange(0,len(data)-scale):
            ret_list.append(log(data[i+scale])-log(data[i]))       
        return np.array(ret_list)
        
    def scaled_return_of_return(data, scale):
        sc_data = np.zeros(len(data)-scale)
        for i in range(len(sc_data)-scale):
            sc_data[i] = np.sum(data[i:i+scale])
        return sc_data      

    def ind_generator(markets , ind):
        portfo_return = np.zeros(len(markets[0]))
        for j in range(len(markets[0])):
            portfo_return[j] = np.dot(np.array(ind) , np.array(markets).T[j])
        return portfo_return   

    def risk_data_input(ret, scale):
        X = ret[:, np.newaxis]
        X_plot = np.linspace(min(ret),max(ret), 200)[:, np.newaxis]
        kde = KernelDensity(kernel='gaussian', bandwidth=scale).fit(X)
        log_dens = kde.score_samples(X_plot)
        pdf = np.exp(log_dens)
        x = X_plot.reshape(len(pdf))

        sec_dev = np.diff(pdf,2)
        qpot = [500.0]
        for i in range(1,len(pdf)-1):
            if pdf[i] > 0.0001:
                jj = sec_dev[i-1]/pdf[i]
            else:
                jj=500
            qpot.append(jj) 
        qpot.append(500)

        xx =[]
        for i in range(len(qpot)):
            if qpot[i] >= 499:
                xx.append(i)
        x_list = np.array(x)[xx]
        d_lim = x_list[x_list<0][-1]
        u_lim = x_list[x_list>0][0]
        return d_lim , u_lim

        
#-----------------------------------------------------------------------#