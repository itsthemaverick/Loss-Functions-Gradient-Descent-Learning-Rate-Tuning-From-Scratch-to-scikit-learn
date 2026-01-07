# Loss Functions, Gradient Descent & Learning Rate Tuning — From Scratch to scikit-learn

This project is a **ground-up implementation and visualization** of core Machine Learning optimization concepts.  
Instead of treating ML libraries as black boxes, I rebuild the mathematics using **NumPy**, validate it visually, and then **map it directly to scikit-learn’s implementation**.

The goal of this project is not accuracy alone — it is **understanding**.

---

## 📌 What This Project Covers

### 1. Loss Functions
Implemented from scratch:
- **Mean Squared Error (MSE)** – regression
- **Mean Absolute Error (MAE)** – robust regression
- **Log Loss (Cross Entropy)** – classification

Each loss function is used exactly where it is mathematically appropriate.

---

### 2. Gradient Descent (From Scratch)
- Batch Gradient Descent implemented using NumPy
- Explicit gradient calculations for:
  - Weight (`w`)
  - Bias (`b`)
- Loss tracked across epochs for convergence analysis

This demonstrates how model parameters are **iteratively optimized**, not magically “fit”.

---

### 3. Learning Rate Tuning Strategies
Implemented and compared visually:
- Constant learning rate
- Time-based decay
- Adaptive learning rate (loss-based adjustment)

The effect of learning rate choice is shown using **loss vs epoch plots**, highlighting:
- Slow convergence
- Stable convergence
- Divergence / instability

---

### 4. Visualization-Driven Understanding
Every major concept is visualized using **matplotlib**:
- Loss vs Epochs
- Regression fit over data
- Learning rate comparison
- **NumPy model vs scikit-learn model comparison**

Visualization is treated as a **debugging and validation tool**, not decoration.

---

### 5. scikit-learn Mapping
The same problem is solved using:
- `SGDRegressor`
- Feature scaling via `StandardScaler`
- Pipeline-based design

The NumPy implementation is then **directly compared** against scikit-learn’s learned model to validate correctness.

---

## 🧠 Why This Project Matters

Most ML projects show *results*.  
This project shows **reasoning**.

It demonstrates:
- Understanding of optimization mathematics
- Awareness of numerical stability
- Ability to debug learning behavior
- Knowledge of how libraries work internally
- Engineering discipline (modular code, visualization, comparison)

---

## 📂 Project Structure
loss-functions-and-gradient-descent/
│
├── README.md
├── data_generation.py
├── loss_functions.py
├── gradient_descent_numpy.py
├── learning_rate_tuning.py
├── sklearn_implementation.py
├── visualizations.py
└── requirements.txt

