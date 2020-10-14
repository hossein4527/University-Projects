class Gennetic_Algorithm(object):
    """genetic algorithm written for risk."""
    def __init__(self, portfo_prices, band_width):
        # super(price_return, self).__init__()
        self.ret = portfo_prices
        self.length = len(portfo_prices)
        self.band_width = band_width
        
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
        kde = KernelDensity(kernel='gaussian', bandwidth=self.band_width).fit(X)
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
        # dd = X_plot[argrelextrema(qpot, np.greater)]
        # risk = dd[dd>0][0] - dd[dd<0][-1]

        xx =[]
        x = X_plot.reshape(len(qpot))
        for i in range(len(qpot)):
            if qpot[i] >= 499:
                xx.append(i)
        x_list = np.array(x)[xx]
        d_lim = x_list[x_list<0][-1]
        u_lim = x_list[x_list>0][0]

        return u_lim-d_lim


    def grade(self,pop , markets):
        summed = reduce(add, (self.fitness(x, markets) for x in pop), 0)
        return summed/(len(pop)*1.0)

    def evolve(self,pop, markets, retain, random_select, mutate):
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