# Optimizers-from-Scratch

## Optimization in Deep Learning

Optimization in deep learning is the process of adjusting model parameters to minimize a loss function. However, while optimization and deep learning share a common ground, their goals differ fundamentally.

## Goal of Optimization vs. Goal of Deep Learning

### Optimization's Goal:
- Focuses on minimizing an objective function (loss function) using mathematical techniques.
- Primarily concerned with achieving the lowest possible **training error**.

### Deep Learning's Goal:
- Aims to build a model that **generalizes well** to unseen data, reducing **generalization error**.
- Requires balancing **optimization** with **regularization** techniques to avoid **overfitting**.

# Challenges in Optimization for Deep Learning

Optimization in deep learning is not always straightforward. Several challenges arise that can make training deep neural networks difficult. Three major challenges are **Local Minima**, **Saddle Points**, and **Vanishing Gradients**.

## Local Minima
Local minima are points where the loss function has a **lower value** than nearby points, but it is **not the global minimum**. In high-dimensional spaces, the probability of getting stuck in a true local minimum is relatively low, but small variations in parameters may lead to **suboptimal convergence**.

For example, given the function

$$f(x) = x \cdot \textrm{cos}(\pi x) \textrm{ for } -1.0 \leq x \leq 2.0,$$

![Function Plot](Images/LocalMinima.svg)

we can approximate the local minimum and global minimum of this function.
### Impact:
- Model may converge to a **suboptimal** solution.
- Training may get stuck and fail to reach the best possible performance.

### Mitigation Strategies:
- Using **momentum-based optimizers** (e.g., Adam, RMSProp) to escape local minima.
- Applying **learning rate scheduling** to navigate better through the loss surface.


## Saddle Points
Saddle points occur when the gradient is **zero**, but the point is **not a minimum** (it is higher in some directions and lower in others). In high-dimensional spaces, saddle points are more common than local minima.

Consider the function $f(x, y) = x^2 - y^2$. It has its saddle point at $(0, 0)$. This is a maximum with respect to $y$ and a minimum with respect to $x$. Moreover, it *looks* like a saddle, which is where this mathematical property got its name.

![Function Plot](Images/Saddlepoint.svg)

### Impact:
- Slows down training, as gradients become small and updates become inefficient.
- Can cause optimization algorithms to stall.

### Mitigation Strategies:
- Using **adaptive learning rate methods** like **Adam** or **AdaGrad** to speed up training.
- Increasing **batch size** to stabilize gradients.


## Vanishing Gradients
Vanishing gradients occur when gradients **become too small** during backpropagation, especially in deep networks. This makes it hard for earlier layers to learn meaningful representations.

For instance, assume that we want to minimize the function $f(x) = \tanh(x)$ and we happen to get started at $x = 4$. As we can see, the gradient of $f$ is close to nil.
More specifically, $f'(x) = 1 - \tanh^2(x)$ and thus $f'(4) = 0.0013$. Consequently, optimization will get stuck for a long time before we make progress.

![Function Plot](Images/VanishingGradient.svg)

### Impact:
- Slows down or **prevents learning** in deep networks.
- Earlier layers fail to update, affecting overall model performance.

### Mitigation Strategies:
- Using **ReLU** (Rectified Linear Unit) activation functions instead of **sigmoid/tanh** to prevent small gradients.
- Implementing **batch normalization** to stabilize gradients.
- Using **residual connections** (ResNets) to allow direct gradient flow.



Deep learning optimization faces multiple challenges, but **modern techniques and algorithms** help mitigate these issues. By using **advanced optimizers**, **better activation functions**, and **network architectures**, we can improve convergence and achieve **better generalization**.

---

# Convexity in Deep Learning  

## Role of Convexity in Optimization  
Convexity plays a significant role in optimization because **convex functions** have desirable mathematical properties that make optimization easier and more efficient.  

### **Convex Function**  
A function $f(x)$ is convex if its second derivative is always non-negative:  

$$
f''(x) \geq 0 \quad \forall x
$$

This ensures that any local minimum is also a global minimum.  

### **Convex Optimization**  
When the objective function is convex, gradient-based methods like **Gradient Descent** can efficiently find the optimal solution without getting trapped in local minima.  

### **Non-Convexity in Deep Learning**  
Neural networks typically have **non-convex loss landscapes**, meaning they contain multiple local minima, saddle points, and flat regions.  

**Example of a non-convex function:** 

$$
f(x) = x \cdot \cos(\pi x), \quad -1.0 \leq x \leq 2.0
$$

# Newton's Method for Optimization

Newton's method is a second-order optimization algorithm that updates parameters using both the gradient and the Hessian (second derivative) of the function. It provides faster convergence compared to standard gradient descent, especially near the optimal solution.

## Update Rule

The update rule for Newton’s method is given by:

$$
x_{t+1} = x_t - H_f^{-1} \nabla f(x_t)
$$

where:
- **$$∇f(xₜ)$$** is the gradient (first derivative) of **$$f(x)$$**.
- **$$H_f$$** is the Hessian matrix (second derivative) of **$$f(x)$$**.

---

## Gradient Descent vs. Newton's Method

### **Gradient Descent**
Gradient Descent uses only the gradient information for updates:

$$
x_{t+1} = x_t - \eta \nabla f(x_t)
$$

where:
- **η** is the learning rate, a scalar that controls the step size.
- **∇f(xₜ)** is the gradient (first derivative) of **f(x)** at **xₜ**.

### **Newton's Method**
Newton's Method incorporates second-order curvature (Hessian) for more precise updates:

$$
x_{t+1} = x_t - H_f^{-1} \nabla f(x_t)
$$

---
Below is a Python implementation of Newton's method using PyTorch:

```python
import torch

def f(x, c):
    return torch.cosh(c*x)

def f_grad(x, c):
    return c * torch.sinh(c*x)

def f_hess(x, c):
    return c**2 * torch.cosh(c*x)

def newton(eta = 1, c = torch.tensor(0.5), num_epochs = 10):
    x = torch.tensor(10.0, requires_grad = True)
    results = [x.item()]

    for i in range(num_epochs):
        x = x - eta*f_grad(x,c)/f_hess(x,c)
        results.append(x.item())
        print(f'epoch {i+1}, x: ',x)
    
    return results
```

![Function Plot](Images/NewtonMethod.png)

---

## **Advantages of Newton’s Method**
- **Faster convergence**: Newton's method has a quadratic convergence rate, making it significantly faster near the optimum.
- **Effective for convex functions**: Works well when the Hessian is positive definite.

## **Disadvantages of Newton’s Method**
- **Computational cost**: Computing the Hessian and its inverse is expensive, especially in high-dimensional spaces.
- **Hessian issues**: May fail or perform poorly if the Hessian is singular, ill-conditioned, or not positive definite.
- Newton’s method is a lot faster **once** it has started working properly in convex problems.

---

This method is particularly useful for optimization problems where the function is smooth and well-behaved, but it requires careful handling of the Hessian matrix to avoid computational bottlenecks or numerical instability.


🚀 **Follow for more insights on Deep Learning and Optimization!** 🚀


