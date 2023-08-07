# Improving Gradient Descent Algorithm

**Summary:**  
The project explores how Gradient Descent algorithms can be improved for optimization. It aims to demonstrate the convergence behavior and rate of convergence of the improved versions.

**Problems Solved:**  
- Analyzing convergence behavior of Gradient Descent with optimal step sizes, momentum, and othognal momentum.
- Comparing the performance of the above techniques.

**Major Learnings:** 
- Gradient descent has an exponential convergence.
- As we approach convergence, the algorithm tries to minimize the angle to the minimizer.
- Optimal alpha gradient descent:
  * The rate of convergence for gradient descent with optimal beta is higher than that of the vanilla gradient descent.
  * As the iterate approaches the minimum, its angle changes rapidly.

- Gradient Descent with Momentum:
  * The gradient descent with momentum does not always converges faster than the gradient descent with optimal alpha and vanilla gradient descent. The rate of convergence depends on the alpha, beta values and setup. For alpha ~ 0.01 and beta ~ 0.85 we can observe that momentum gradient descent performs better that vanilla gradient descent
  * The angle of approach of the momentum gradient descent shows an eccentric behaviour and not necessarily approach 0 deg. The graph is also jagged. However, the jaggedness decreases while approaching the minimizer. This is in contrast with the angle of approach for gradient descents which is a smooth graph. 
We also see no wiggling while approaching minimizer.


**Technologies and Libraries Used:**  
- Python
- NumPy
- Matplotlib


**Concepts Explored:**  
- Optimization Algorithms
- Convergence Analysis

#optimization #machinelearning #gradientdescent #momentum #datascience #algorithms
