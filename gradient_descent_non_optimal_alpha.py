import numpy as np
import random
import sys
import matplotlib.pyplot as plt

# Configurations to show the whole output in console
np.set_printoptions(threshold=sys.maxsize, linewidth=200)
np.set_printoptions(suppress=True)

# Function to generate a standard normal column vector
def std_norm_vector(d):
    c = np.random.standard_normal(size=d)
    return c

# Generates matrix with each column as a standard normal vector
def gen_A(d):
    row_matrix=[]
    for i in range(d):
        col = std_norm_vector(d)
        row_matrix.append(col)

    np_row_matrix = np.array(row_matrix)
    a = np.transpose(np_row_matrix)

    return a



d = 10
c = std_norm_vector(d)
a = gen_A(d)
print("A is")
print(a)

q = np.matmul(np.transpose(a),a)
print(" q is")
print(q)

x_now=std_norm_vector(d)
alpha = 0.02

error = 999

x_star= np.matmul(np.linalg.inv(q),c)
angle=0
output=[]
angles=[]
angle=0

# Run untill error is very small
while error > 0.0001:

    error = np.linalg.norm( np.subtract(x_now, x_star))
    print(error)
    output.append(error)
    aplha_grad_fx= np.dot(alpha,np.subtract(np.matmul(q,x_now),c))

    x_next = np.subtract(x_now, aplha_grad_fx)

    angle = (np.matmul(np.transpose(np.subtract( x_next, x_now)),np.subtract(x_star, x_now)) )/(np.linalg.norm(np.subtract(x_next,x_now)) * np.linalg.norm(np.subtract(x_star,x_now)))

    angles.append(angle)
    x_now = x_next

print("-"*100)
print(output)
plt.plot(output)
plt.plot(np.log(output))
plt.show()

plt.plot(angles)
plt.show()


