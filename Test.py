import math
import numpy as np

# coefficients of xi in terms of lamba xi=x_coeff/lambda
x_coef = np.array([104 / 18, 128 / 16, 135 / 14, 139 / 14, 150 / 12, 153 / 12, 162 / 10, 168 / 4, 195 / 2, 198 / 2])

# coefficients of constraint function h(x)
constraint= np.array([9,8,7,7,6,6,5,2,1,1])

# coefficients of functions to maximize F(x)
fx = np.array([104,128,135,139,150,153,162,168,195,198])

def calc_fx(x_coef,constraint,fx):

    # Calculating lambda
    lam_sq = np.dot(constraint, np.square(x_coef)) / 68644
    lam = math.sqrt(lam_sq)

    # Calculating real coefficients
    x = np.array(x_coef)
    x = x / lam

    # Calculating upper bound ie F(real values of x)
    result = np.matmul(fx, np.transpose(x))

    return result,x

result,x = calc_fx(x_coef,constraint,fx)
print(result,"\n",x)

max_sum= np.dot(fx,np.floor(x))
print(max_sum)
