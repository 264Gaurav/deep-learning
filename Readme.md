# Deep Learning: Underfitting, Overfitting & Regularization

## 📌 Core Libraries

- **NumPy** → Arrays & matrix operations
- **Matplotlib** → Visualization
- **Pandas** → Data handling (CSV, DataFrames)
- **TensorFlow & Keras** → Neural networks
  - **Sequential** → Linear stack of layers
  - **Dense** → Fully connected layer
  - **Dropout** → Reduces overfitting

---

## ⚖️ Underfitting vs Overfitting

### 🔹 Underfitting

Model too simple → poor performance on train & test.
**Fix:** Add layers/neurons, train longer, lower LR, reduce regularization.

### 🔹 Overfitting

Model too complex → memorizes training, fails on test.
**Fix:** Dropout, L2 regularization, early stopping, data augmentation, reduce complexity.

---

## 📊 Comparison

| Feature           | Underfitting 🟡 | Good Fit 🟢 | Overfitting 🔴 |
| ----------------- | --------------- | ----------- | -------------- |
| Training Accuracy | Low             | High        | Very High      |
| Test Accuracy     | Low             | High        | Low            |
| Model Complexity  | Too simple      | Balanced    | Too complex    |
| Generalization    | Poor            | Good        | Poor           |

---

## 🎯 Dropout

- Randomly drops neurons (prob _p_) during training.
- Prevents reliance on specific neurons → robust features.
- **Training:** scale surviving neurons `1/(1-p)`
- **Testing:** dropout off, scaling handled automatically.
- Best for large models, avoid overuse on small datasets.

---

## 🛠️ Ways to Reduce Overfitting

- Dropout
- L1/L2 Regularization
- Early Stopping
- Data Augmentation
- Batch Normalization
- Reduce Model Complexity
- Cross-Validation

---

## ✅ Rule of Thumb

- **Underfitting:** Increase complexity, train longer.
- **Overfitting:** Add regularization, dropout, early stopping, augmentation.

📌 **Goal:** Achieve balance → model should **generalize well** to unseen data.

# Keras Tuner - HyperParameter Tuning of models:

### For understanding of KerasTuner, refer to the [KerasTuner Readme](./kerasTuner/Readme.md).
