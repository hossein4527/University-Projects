import numpy as np
from numpy import amin,amax,zeros,ceil

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

    def linebazi(grid):
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


            
