"""Z Plane poles and zeros plot Matplotlib
    Plot the complex z-plane given a transfer function.
    ---
    TODO LIST
        [ ] 1. Add sos filter version
        [ ] 2. Add plotting several b, a pairs
"""
import numpy as np
import matplotlib.pyplot as plt
from  matplotlib import patches
from matplotlib.figure import Figure
from matplotlib import rcParams
    
def zplane(b, a, filename=None):
    # get a figure/plot
    ax = plt.subplot(111)

    # create the unit circle
    uc = patches.Circle((0,0), radius=1, fill=False,
                        color='black', ls='dashed')
    ax.add_patch(uc)

    #Converting b and a to np.array if they are list format
    a=np.asarray(a)
    b=np.asarray(b)

    # The coefficients are less than 1, normalize the coeficients
    if np.max(b) > 1:
        kn = np.max(b)
        b = b/float(kn)
    else:
        kn = 1

    if np.max(a) > 1:
        kd = np.max(a)
        a = a/float(kd)
    else:
        kd = 1
        
    # Get the poles and zeros
    p = np.roots(a)
    z = np.roots(b)
    k = kn/float(kd)
    
    # Plot the zeros and set marker properties    
    t1 = plt.plot(z.real, z.imag, 'go', ms=10)
    plt.setp( t1, markersize=10.0, markeredgewidth=3.0,
              markeredgecolor='b', markerfacecolor='w')

    # Plot the poles and set marker properties
    t2 = plt.plot(p.real, p.imag, 'rx', ms=10)
    plt.setp( t2, markersize=10.0, markeredgewidth=3.0,
              markeredgecolor='r', markerfacecolor='r')

    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # set the ticks
    plt.axis('scaled');
    # r = 1.5; plt.axis('scaled'); plt.axis([-r, r, -r, r])
    # ticks = [-1, -.5, .5, 1]; plt.xticks(ticks); plt.yticks(ticks)

    plt.title("Poles and Zeros",fontsize=14)

    if filename is None:
        plt.show()
    else:
        plt.savefig(filename)

    return z, p, k

if __name__=='__main__':
    coeff = [-0.00014145100081455295, 0.00014355742002773923, 0.0, 1.0, -1.9993387569654149, 0.9993457156679053]    
    b_coeff, a_coeff = coeff[:len(coeff)//2], coeff[len(coeff)//2:]
    zplane(b_coeff, a_coeff)
