"""
Principle of convolution
1. Invert the sequence of the second array
2. Move one by one and sigma of multiplication

            1   2   3
  0.5   1   0
            0           = 0

        1   2   3
  0.5   1   0
        1   0           = 1
        
    1   2   3
    0.5 1   0
    0.5 2   0           = 2.5
                
    1   2   3
        0.5 1   0
        1   3           = 4         
    
    1   2   3
            0.5   1   0
            1.5         = 1.5
            
            result = [0 1 2.5 4 1.5]

Principle of correlate
1. the sequence of the second array
2. Move one by one and sigma of multiplication

        1   2   3
0   1   0.5
        0.5                 = 0.5

        1   2   3
    0   1   0.5
        1   1               = 2

        1   2   3
        0   1   0.5
            2   1.5         = 3.5

        1   2   3
            0   1   0.5
            0   3           = 3

        1   2   3
                0   1   0.5
                0           = 0

            result = [0.5 2 3.5 3 0]
"""
import numpy as np

in1 = [1, 2, 3]
in2 = [0, 1, 0.5]
func = np.convolve
print("\t Full mode numpy convolution")
print(func(in1, in2, mode="full")) 

print("\t Valid mode numpy convolution") # max(M, N)
print(func(in1, in2, mode="valid")) 

print("\t Same mode numpy convolution") # max(M, N) - min(M, N) + 1
print(func(in1, in2, mode="same")) 

print("-"*50)
import scipy.signal as signal
func = signal.convolve
print("\t Full mode scipy convolution")
print(func(in1, in2, mode="full", method='auto')) 

print("\t Valid mode scipy convolution") # max(M, N), centered
print(func(in1, in2, mode="valid", method='auto')) 

print("\t Same mode scipy convolution") # max(M, N) - min(M, N) + 1, centered
print(func(in1, in2, mode="same", method='auto')) 

print("-"*50)
# method divides fft convolve and convolve in time
"""
scipy fft convolve
    ...

    sp1 = fft(in1, fshape, axes=axes)
    sp2 = fft(in2, fshape, axes=axes)

    ret = ifft(sp1 * sp2, fshape, axes=axes)
    
    ...
"""
func = signal.fftconvolve
print("\t Full mode scipy fft convolution")
print(func(in1, in2, mode="full"))

print("\t Valid mode scipy fft convolution") # max(M, N), centered
print(func(in1, in2, mode="valid"))

print("\t Same mode scipy fft convolution") # max(M, N) - min(M, N) + 1, centered
print(func(in1, in2, mode="same"))

print("-"*50)

func = np.correlate
print("\t Full mode numpy correlate")
print(func(in1, in2, mode="full")) 

print("\t Valid mode numpy correlate") # max(M, N)
print(func(in1, in2, mode="valid")) 

print("\t Same mode numpy correlate") # max(M, N) - min(M, N) + 1
print(func(in1, in2, mode="same")) 
