# Optimizers-from-Scratch

# Optimization in Deep Learning

Optimization in deep learning is the process of adjusting model parameters to minimize a loss function. However, while optimization and deep learning share a common ground, their goals differ fundamentally.

## Goal of Optimization vs. Goal of Deep Learning

### 🔹 Optimization's Goal:
- Focuses on minimizing an objective function (loss function) using mathematical techniques.
- Primarily concerned with achieving the lowest possible **training error**.

### 🔹 Deep Learning's Goal:
- Aims to build a model that **generalizes well** to unseen data, reducing **generalization error**.
- Requires balancing **optimization** with **regularization** techniques to avoid **overfitting**.

### 📌 Key Difference:
- An optimizer might find a solution that **minimizes training loss perfectly**, but that doesn’t mean the model will **perform well on new data**.

# Challenges in Optimization for Deep Learning

Optimization in deep learning is not always straightforward. Several challenges arise that can make training deep neural networks difficult. Three major challenges are **Local Minima**, **Saddle Points**, and **Vanishing Gradients**.

## 1️⃣ Local Minima
Local minima are points where the loss function has a **lower value** than nearby points, but it is **not the global minimum**. In high-dimensional spaces, the probability of getting stuck in a true local minimum is relatively low, but small variations in parameters may lead to **suboptimal convergence**.

For example, given the function

$$f(x) = x \cdot \textrm{cos}(\pi x) \textrm{ for } -1.0 \leq x \leq 2.0,$$

![Function Plot](images/LocalMinima.svg)

we can approximate the local minimum and global minimum of this function.
### 🔹 Impact:
- Model may converge to a **suboptimal** solution.
- Training may get stuck and fail to reach the best possible performance.

### 🔹 Mitigation Strategies:
- Using **momentum-based optimizers** (e.g., Adam, RMSProp) to escape local minima.
- Applying **learning rate scheduling** to navigate better through the loss surface.

---

## 2️⃣ Saddle Points
Saddle points occur when the gradient is **zero**, but the point is **not a minimum** (it is higher in some directions and lower in others). In high-dimensional spaces, saddle points are more common than local minima.

### 🔹 Impact:
- Slows down training, as gradients become small and updates become inefficient.
- Can cause optimization algorithms to stall.

### 🔹 Mitigation Strategies:
- Using **adaptive learning rate methods** like **Adam** or **AdaGrad** to speed up training.
- Increasing **batch size** to stabilize gradients.

---

## 3️⃣ Vanishing Gradients
Vanishing gradients occur when gradients **become too small** during backpropagation, especially in deep networks. This makes it hard for earlier layers to learn meaningful representations.

### 🔹 Impact:
- Slows down or **prevents learning** in deep networks.
- Earlier layers fail to update, affecting overall model performance.

### 🔹 Mitigation Strategies:
- Using **ReLU** (Rectified Linear Unit) activation functions instead of **sigmoid/tanh** to prevent small gradients.
- Implementing **batch normalization** to stabilize gradients.
- Using **residual connections** (ResNets) to allow direct gradient flow.

---

Deep learning optimization faces multiple challenges, but **modern techniques and algorithms** help mitigate these issues. By using **advanced optimizers**, **better activation functions**, and **network architectures**, we can improve convergence and achieve **better generalization**.

---

🚀 **Follow for more insights on Deep Learning and Optimization!** 🚀

