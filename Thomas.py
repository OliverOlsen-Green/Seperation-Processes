import numpy as np
#class for the functions provided in MATLAB converted to python
class Thomas:
    def Thomas_general(a, b, c, d):

        # lower diagonal (a) (n-1)
        # main diagonal (n) : b
        # upper diagonal (n+1) : c
        # rhs vector : d (n)

        n = len(b)

        c_star = np.zeros((n - 1))
        d_star = np.zeros(n)

        c_star[0] = c[0] / b[0]
        d_star[0] = d[0] / b[0]

        for i in range(1, (n - 1)):
            m = b[i] - a[i - 1] * c_star[i - 1]

            c_star[i] = c[i] / m

            d_star[i] = (d[i] - a[i - 1] * d_star[i - 1]) / m

        m = b[n - 1] - a[n - 2] * c_star[n - 2]

        d_star[n - 1] = (d[n - 1] - a[n - 2] * d_star[n - 2]) / m

        x = np.zeros(n)
        x[n - 1] = d_star[n - 1]

        for i in range((n - 2), -1, -1):
            x[i] = d_star[i] - c_star[i] * x[i + 1]

        return x

    def Thomas(strip,feed):
        Stages,Components = strip.shape
        adia = strip
        adix = adia + 1

        feed = feed.copy()

        odix = np.zeros((Stages,Components))

        #construct the upper diagonal terms
        for i in range(0,Stages-1):
            odix[i,:] = -strip[i+1,:]

        #forward substitution
        for i in range(0,Stages-1):
            odix[i,:] = odix[i,:] / adix[i,:]
            feed[i,:] = feed[i,:] / adix[i,:]

            adix[i + 1,:] = adix[i + 1,:] + odix[i,:]
            feed[i + 1,:] = feed[i + 1,:] + feed[i,:]

        #final row
        feed[Stages-1,:] = feed[Stages-1,:] / adix[Stages-1,:]

        for i in range(Stages-2,-1,-1):
            feed[i,:] = feed[i,:] - odix[i,:] * feed[i+1,:]


        X = feed

        return X




a = np.array([-1, -1, -1, -1])
b = np.array([ 4,  4,  4, 4, 4])
c = np.array([-1, -1, -1,-1])


d = np.array([1, 2, 3, 4, 5])

x = Thomas.Thomas_general(a,b,c,d)
#print(x)

feed = np.array([
    [0.0,   0.0,   100.0, 300.0],
    [0,   0.0,   0.0,   0.0],
    [0.0,   0.0,   0.0,   0.0],
    [0.0,   0.0,   0.0,   0.0],
    [250, 750, 0,   0],
])

strip = np.array([
    [200, 1000, 0.9, 0.01],
    [200, 1000, 0.9, 0.01],
    [200, 1000, 0.9, 0.01],
    [200, 1000, 0.9, 0.01],
    [200, 1000, 0.9, 0.01],
])

X = Thomas.Thomas(strip,feed)

print(X)
