import numpy as np

def sigmoid(x):
    s = 1/(1+np.exp(-1*x))
    return s

def reLU(x):
    if x<=0 :
        return 0
    else:
        return x

if __name__ == '__main__':
    a = np.array([0,1,2,-3])
    print(sigmoid(a))

