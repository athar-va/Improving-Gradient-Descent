import math

import numpy as np


def calc_fx(x_coef,constraint,fx,slice_at,fixed):
    fixed_square=[]
    slice_at = slice_at
    # print("IN CALC Fx")
    # print(slice_at)
    for i in fixed:
        fixed_square.append(i*i)

    diff = np.dot(fixed_square, np.transpose(constraint[:slice_at]))

    # print("\nconstraint[:slice_at]")
    # print(constraint[:slice_at])
    # print("constraint[slice_at:]")
    # print(constraint[slice_at:])
    # print("x_coef[slice_at:]")
    # print(x_coef[slice_at:])
    # print("fx[slice_at:]")
    # print(fx[slice_at:])

    lam_sq = np.dot(constraint[slice_at:], np.square(x_coef[slice_at:])) / (68644-diff)
    lam = math.sqrt(lam_sq)
    # print("Lambda",lam)
    x = np.array(x_coef[slice_at:])
    x = x / lam
    result = np.matmul(fx[slice_at:], np.transpose(x))

    return result

def calc_fixed (fixed,fx,slice_at):
    try:
        # print("In calc_fixed fx[:slice_at] : ",fx[:slice_at])
        return np.dot(fixed,np.transpose(fx[:slice_at]))
    except:
        print("Error")
        print(fx)
        print(fx[:slice_at])
        print(fixed)


def bnb(x_coef,constraint,fx,fixed,slice_at, flag):

    global result_val
    global lower_bound
    global upper_bound
    global valid_complete_assignments

    # Calculate partial assignment value of objective function
    fixed_sum = calc_fixed(fixed, fx, slice_at)

    # Calculate real valued solution to the function
    fx_sum = calc_fx(x_coef, constraint, fx, slice_at,fixed)

    result = fx_sum + fixed_sum

    # Base Condtion : if all the integer values are assigned
    if len(fixed) == 10:

        if result<upper_bound:
            valid_complete_assignments += 1
        if result > lower_bound and result < upper_bound:

            lower_bound = result
            result_val = fixed.copy()

            print("Possible Solution:", fixed, result)

    # checking pruning condition (bounds)
    if result > lower_bound and result <= upper_bound :

        fixed1=fixed.copy()
        fixed2=fixed.copy()

        # Branch

        fixed1.append(math.floor(real_coef[slice_at]))

        bnb(x_coef,constraint,fx,fixed1,slice_at+1, flag+1)

        fixed2.append(math.ceil(real_coef[slice_at]))

        bnb(x_coef, constraint, fx, fixed2, slice_at + 1, flag +1 )

    else:
        return False


# Real Valued Solution derived earlier
real_coef=[  9.01222693,  12.47846805,  15.0410106 ,  15.48667017,
        19.49760633,  19.88755846,  25.26889781,  65.51195728,
       152.08132941, 154.42104217]

# coefficients of xi in terms of lamba xi=x_coeff/lambda
x_coef = np.array([104 / 18, 128 / 16, 135 / 14, 139 / 14, 150 / 12, 153 / 12, 162 / 10, 168 / 4, 195 / 2, 198 / 2])

# coefficients of constraint function h(x)
constraint= np.array([9,8,7,7,6,6,5,2,1,1])

# coefficients of the objective function
fx = np.array([104,128,135,139,150,153,162,168,195,198])
fixed=[]


result= calc_fx(x_coef,constraint,fx,0,fixed)

# Upper bound = 88015.93234412931
upper_bound = result
lower_bound = 87441  # found in earlier solution
result_val=[]
valid_complete_assignments=0
max = lower_bound
flag=0

bnb(x_coef, constraint, fx, fixed ,0, flag)

print("final Lower bound",lower_bound)
print("final coefs", result_val)
print("Valid Complete Assignments", valid_complete_assignments)








