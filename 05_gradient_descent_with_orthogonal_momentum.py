import math

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

def gen_A(d):
    row_matrix=[]
    for i in range(d):
        col = std_norm_vector(d)
        row_matrix.append(col)

    np_row_matrix = np.array(row_matrix)
    a = np.transpose(np_row_matrix)

    return a

def solve_for_alpha_beta(q, x_now, c):
    pk = np.subtract(np.matmul(q, x_now), c)
    qk = np.subtract(x_now,x_prev)

    P = np.matmul(np.matmul(np.transpose(pk),q),pk)
    Q = np.matmul(np.matmul(np.transpose(qk), q), qk)
    R = np.matmul(np.matmul(np.transpose(pk), q), qk)

    c1 = np.matmul(np.transpose(pk),pk)
    c2 = np.matmul(np.transpose(pk),qk)

    A = [[P,-1*R],
         [R,-1*Q]]

    B = [c1,c2]

    # print(A)
    # print(B)

    result = np.linalg.solve(A,B)
    # print("Result is")
    # print(result)

    return result








# (0.0010848760922248513, 0.25751267354294916)
# (0.09618725952031704, 0.19111177327782614)





d = 10
c = std_norm_vector(d)
a = gen_A(d)
# print("A is")
# print(a)

q = np.matmul(np.transpose(a),a)

# print(find_alpha_beta(q))


# print(" q is")
# print(q)

x_now=std_norm_vector(d)
x_now1 = x_now.copy()
x_prev = np.zeros(d)


error = 999

x_star = np.matmul(np.linalg.inv(q), c)

angle = 0
output = []
output1= []
output2= []
output3=[]
angles = []
angles1 = []
angles2 = []
angle = 0



# Gradient Descent with momentum with optimal alpha beta with orthogonal qk
# while error > 0.001:
for i in range(500):
    error = np.linalg.norm(np.subtract(x_now, x_star))

    result = solve_for_alpha_beta(q, x_now, c)
    alpha = result[0]
    beta = result [1]
    print(error)
    output.append(error)

    pk = np.subtract(np.matmul(q,x_now),c)
    qk = np.subtract(x_now,x_prev)

    # unit vector q
    qk_norm = qk/np.linalg.norm(qk)

    # unit vector p
    pk_norm = pk / np.linalg.norm(pk)

    # finds orthogonal vector to pk by deleting the projection of qk on pk
    ortho_qk = qk_norm - np.dot(qk_norm,pk_norm) * pk_norm

    aplha_grad_fx = np.dot(alpha,np.subtract(np.matmul(q,x_now),c))

    beta_qk = np.dot(beta,ortho_qk)

    x_next = np.add(np.subtract(x_now, aplha_grad_fx), beta_qk)

    angle = (np.matmul(np.transpose(np.subtract( x_next, x_now)),np.subtract(x_star, x_now)) )/(np.linalg.norm(np.subtract(x_next,x_now)) * np.linalg.norm(np.subtract(x_star,x_now)))

    angles.append(angle)
    x_prev = x_now
    x_now = x_next

error = 999
x_now = x_now1.copy()

alpha = 0.03
beta = 0.75

# Gradient Descent with momentum with optimal alpha beta stepsize
# while error > 0.001:
for i in range(500):
    error = np.linalg.norm(np.subtract(x_now, x_star))

    result = solve_for_alpha_beta(q, x_now, c)
    alpha = result[0]
    beta = result [1]
    print(error)
    output2.append(error)

    aplha_grad_fx = np.dot(alpha,np.subtract(np.matmul(q,x_now),c))

    beta_qk = np.dot(beta,np.subtract(x_now,x_prev))

    x_next = np.add(np.subtract(x_now, aplha_grad_fx) ,np.dot(beta,np.subtract(x_now,x_prev)))

    angle = (np.matmul(np.transpose(np.subtract( x_next, x_now)),np.subtract(x_star, x_now)) )/(np.linalg.norm(np.subtract(x_next,x_now)) * np.linalg.norm(np.subtract(x_star,x_now)))

    angles1.append(angle)
    x_prev = x_now
    x_now = x_next



error = 999
alpha = 0.02
x_now = x_now1.copy()

# Vanilla gradient descent


# while error > 0.001:
for i in range(500):
    error = np.linalg.norm( np.subtract(x_now, x_star))
    print(error)
    output1.append(error)
    aplha_grad_fx= np.dot(alpha,np.subtract(np.matmul(q,x_now),c))

    x_next = np.subtract(x_now, aplha_grad_fx)

    angle = (np.matmul(np.transpose(np.subtract( x_next, x_now)),np.subtract(x_star, x_now)) )/(np.linalg.norm(np.subtract(x_next,x_now)) * np.linalg.norm(np.subtract(x_star,x_now)))

    angles1.append(angle)
    x_now = x_next


# Optimized Gradient Descent

error = 999
x_now = x_now1.copy()

# while error > 0.001:
for i in range(500):
    pk = np.subtract(np.matmul(q, x_now), c)
    alpha = (np.matmul(np.transpose(pk),pk)) / (np.matmul( np.matmul(np.transpose(pk),q),pk))

    error = np.linalg.norm( np.subtract(x_now, x_star))
    print(error)
    output3.append(error)
    aplha_grad_fx = np.dot(alpha, np.subtract(np.matmul(q,x_now),c))

    x_next = np.subtract(x_now, aplha_grad_fx)

    angle = (np.matmul(np.transpose(np.subtract( x_next, x_now)),np.subtract(x_star, x_now)) )/(np.linalg.norm(np.subtract(x_next,x_now)) * np.linalg.norm(np.subtract(x_star,x_now)))

    angles2.append(angle)
    x_now = x_next




print("-"*100)
#print(output)

plt.plot(output,label=" Optimal Aplha Beta with Orthogonal qk")
plt.plot(output1, label=" Optimal Aplha Beta w/o Orthogonal qk")
plt.plot(output2, label=" Optimal alpha Gradient Descent")
plt.plot(output3, label=" Vanilla Gradient Descent")


# plt.plot(np.log(output))
leg = plt.legend(loc='upper center')
plt.show()

plt.plot(angles, label="Optimal Aplha Beta with Orthogonal qk")
plt.plot(angles1, label=" Optimal Aplha Beta w/o Orthogonal qk")
# plt.plot(angles2, label=" Optimal Alpha Gradient Descent")
leg = plt.legend(loc='upper center')
plt.show()

