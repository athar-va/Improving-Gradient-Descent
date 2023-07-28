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


def find_alpha_beta(q):

    # Generate eigen values
    ev = np.linalg.eigvals(q)
    print("Eigen Values are:")
    print(ev)

    flag = 1
    loop_flag = 1

    while loop_flag == 1:
        alpha = random.random()
        beta = random.random()

        # Check if -1<lam<1 for all eigen vectors of q
        for mu in ev:

            if 1 - ( (4 * alpha * mu)/math.pow(1-beta+(alpha*mu), 2)) < 0:
                break

            lam1 = 1 - (0.5*(1- alpha + (alpha * mu))) * (1 + math.sqrt(1 - ( (4 * alpha * mu)/math.pow(1-beta+(alpha*mu), 2))))
            lam2 = 1 - (0.5*(1- alpha + (alpha * mu))) * (1 - math.sqrt(1 - ( (4 * alpha * mu)/math.pow(1-beta+(alpha*mu), 2))))

            print("Lam1 and Lam2 are")
            print(lam1,lam2)

            if lam1 > -1 and lam1 < 1 and lam2 > -1 and lam2 < 1 :
                flag = 0
            else:
                flag = 1
                break

        if flag == 0:
            loop_flag = 0

    return alpha,beta

# (0.00118725952031704, 0.8997177327782614)

d = 10
c = std_norm_vector(d)
a = gen_A(d)

q = np.matmul(np.transpose(a),a)

print(" q is")
print(q)

x_now = std_norm_vector(d)
x_now1 = x_now.copy()
x_prev = np.zeros(d)
# alpha = 0.01
# beta = 0.9

alpha = 0.01
beta = 0.899

error = 999

x_star = np.matmul(np.linalg.inv(q), c)

angle = 0
output = []
output1= []
angles = []
angles1 = []
angle = 0



# Gradient Descent with Momentum with best alpha beta
while error > 0.001:

    error = np.linalg.norm(np.subtract(x_now, x_star))

    print(error)
    output.append(error)

    aplha_grad_fx = np.dot(alpha,np.subtract(np.matmul(q,x_now),c))

    beta_qk = np.dot(beta,np.subtract(x_now,x_prev))

    x_next = np.add(np.subtract(x_now, aplha_grad_fx) ,np.dot(beta,np.subtract(x_now,x_prev)))

    angle = (np.matmul(np.transpose(np.subtract( x_next, x_now)),np.subtract(x_star, x_now)) )/(np.linalg.norm(np.subtract(x_next,x_now)) * np.linalg.norm(np.subtract(x_star,x_now)))

    angles.append(angle)
    x_prev = x_now
    x_now = x_next


error = 999
alpha = 0.02
x_now = x_now1.copy()

# Vanilla gradient descent

#
# while error > 0.001:
#
#     error = np.linalg.norm( np.subtract(x_now, x_star))
#     print(error)
#     output1.append(error)
#     aplha_grad_fx= np.dot(alpha,np.subtract(np.matmul(q,x_now),c))
#
#     x_next = np.subtract(x_now, aplha_grad_fx)
#
#     angle = (np.matmul(np.transpose(np.subtract( x_next, x_now)),np.subtract(x_star, x_now)) )/(np.linalg.norm(np.subtract(x_next,x_now)) * np.linalg.norm(np.subtract(x_star,x_now)))
#
#     angles1.append(angle)
#     x_now = x_next
#

# Optimized Gradient Descent

error = 999
x_now = x_now1.copy()

while error > 0.001:

    pk = np.subtract(np.matmul(q, x_now), c)
    alpha = (np.matmul(np.transpose(pk),pk)) / (np.matmul( np.matmul(np.transpose(pk),q),pk))

    error = np.linalg.norm( np.subtract(x_now, x_star))
    print(error)
    output1.append(error)
    aplha_grad_fx = np.dot(alpha, np.subtract(np.matmul(q,x_now),c))

    x_next = np.subtract(x_now, aplha_grad_fx)

    angle = (np.matmul(np.transpose(np.subtract( x_next, x_now)),np.subtract(x_star, x_now)) )/(np.linalg.norm(np.subtract(x_next,x_now)) * np.linalg.norm(np.subtract(x_star,x_now)))

    angles1.append(angle)
    x_now = x_next




print("-"*100)
#print(output)

plt.plot(output,label="Aplha Beta")
# plt.plot(output1, label=" Vanilla Gradient Descent")

plt.plot(output1, label=" Optimal alpha Gradient Descent")

# plt.plot(np.log(output))
leg = plt.legend(loc='upper center')
plt.show()

plt.plot(angles, label="Aplha Beta")
# plt.plot(angles1, label=" Vanilla Gradient Descent")
plt.plot(angles1, label=" Optimal Alpha Gradient Descent")
leg = plt.legend(loc='upper center')
plt.show()

