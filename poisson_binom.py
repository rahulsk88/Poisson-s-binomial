import numpy as np
from math import atan2, sqrt, log, cos, sin, exp, pi
from scipy.fft import fft

def Arg_zj(pj, w, l):
    return np.arctan2(pj * sin(w*l), 1 - pj + pj * cos(w*l))

def pdf_poi_bin(probs):
    n = len(probs)
    w = 2 * pi / (n + 1)
    outputs = np.empty(n+1, dtype=complex) #Complex numbers

    v_atan2 = np.vectorize(atan2) #for faster computation still takes 4 mins 
    v_sqrt = np.vectorize(sqrt)
    v_log = np.vectorize(log)
    v_cos = np.vectorize(cos)
    v_sin = np.vectorize(sin)
    v_exp = np.vectorize(exp)

    for l in range((n+1)//2 + 1):
        if l == 0:
            outputs[l] = complex(1, 0)
        else:
            wl = w * l
            real_part = np.subtract(1,  np.array(probs)) + probs * v_cos(wl)
            imag_part = probs * v_sin(wl)
            mod = v_sqrt(real_part**2 + imag_part**2)
            arg = v_atan2(imag_part, real_part)
            d = v_exp(v_log(mod).sum())
            arg_sum = arg.sum()
            outputs[l] = complex(d * v_cos(arg_sum), d * v_sin(arg_sum))
    for l in range(1, (n+1)//2 + 1): # Do not neeed to do both because apparently we care only about the real value so we can set the rest to be the conjugate. 
        outputs[n + 1 -l] = outputs[l].conjugate()
    pmf = fft(outputs).real / (n + 1)
    return pmf

