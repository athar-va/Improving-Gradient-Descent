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

# Run untill error is very small
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

# Generate q
q = np.matmul(np.transpose(a),a)
print(" q is")
print(q)

x_now=std_norm_vector(d)
x_now_copy=x_now.copy()

error = 999

# Find minimizer
x_star= np.matmul(np.linalg.inv(q),c)

angle=0
output1=[]
output2=[]
angles_optimal_alpha=[]
angles=[]
angle=0


# Gradient descent with optimal Alpha

# Run untill error is very small
while error > 0.0001:

    pk = np.subtract(np.matmul(q, x_now), c)
    alpha = (np.matmul(np.transpose(pk),pk)) / (np.matmul( np.matmul(np.transpose(pk),q),pk))

    error = np.linalg.norm( np.subtract(x_now, x_star))
    print(error)
    output1.append(error)
    aplha_grad_fx= np.dot(alpha, np.subtract(np.matmul(q,x_now),c))

    x_next = np.subtract(x_now, aplha_grad_fx)

    angle = (np.matmul(np.transpose(np.subtract( x_next, x_now)),np.subtract(x_star, x_now)) )/(np.linalg.norm(np.subtract(x_next,x_now)) * np.linalg.norm(np.subtract(x_star,x_now)))

    angles_optimal_alpha.append(angle)
    x_now = x_next

alpha = 0.03
x_now = x_now_copy

# Vanilla gradient descent

# Run untill error is very small
while error > 0.0001:
    error = np.linalg.norm( np.subtract(x_now, x_star))
    print(error)
    output2.append(error)
    aplha_grad_fx= np.dot(alpha,np.subtract(np.matmul(q,x_now),c))

    x_next = np.subtract(x_now, aplha_grad_fx)

    angle = (np.matmul(np.transpose(np.subtract( x_next, x_now)),np.subtract(x_star, x_now)) )/(np.linalg.norm(np.subtract(x_next,x_now)) * np.linalg.norm(np.subtract(x_star,x_now)))

    angles.append(angle)
    x_now = x_next

print("-"*100)


plt.plot(output1,label="Optimal Aplha")
plt.plot(output2, label="Constant Alpha")

leg = plt.legend(loc='upper center')
plt.show()

plt.plot(angles_optimal_alpha)
plt.show()


