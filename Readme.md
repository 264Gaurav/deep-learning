## Core Libraries:

- **NumPy**: For numerical arrays and matrix operations.
- **Matplotlib**: For data visualization and plotting.
- **Pandas**: For data manipulation and handling (e.g., reading CSV files, working with DataFrames).
- **TensorFlow & Keras**: For building and training neural networks.
  - **Sequential**: A linear stack of layers.
  - **Dense**: A fully connected layer in an ANN.
  - **Dropout**: Reduces overfitting by randomly deactivating neurons during training.

## Overfitting vs Underfitting

### What is Underfitting?

Underfitting happens when a model is too simple to capture the underlying patterns in the data.
It performs poorly on both training and test sets.

#### Causes:

Model has too few layers/neurons (low capacity).

Training stopped too early.

Learning rate too high (model can’t converge).

Features are not informative enough.

#### Symptoms:

Low training accuracy.

Low validation/test accuracy.

Both losses remain high.

#### Solutions:

Increase model complexity (more layers, neurons).

Train longer / reduce learning rate.

Add better features or preprocessing.

Reduce regularization (dropout, weight decay).

### What is Overfitting?

Overfitting happens when a model is too complex and memorizes the training data, instead of learning general patterns.
It performs very well on training data, but poorly on test/validation data.

#### Causes:

Model has too many layers/neurons (high capacity).

Not enough training data.

Training too long without regularization.

#### Symptoms:

High training accuracy.

Low validation/test accuracy.

Training loss keeps decreasing, validation loss increases.

#### Solutions:

Use Dropout layers (e.g., 0.2–0.5).

Add L2 regularization (weight decay).

Use Early Stopping (stop when validation loss increases).

Apply Data Augmentation (especially for images).

Reduce model complexity.

## Underfitting vs Overfitting (Comparison)

| Feature           | Underfitting 🟡 | Good Fit 🟢 | Overfitting 🔴 |
| ----------------- | --------------- | ----------- | -------------- |
| Training Accuracy | Low             | High        | Very High      |
| Test Accuracy     | Low             | High        | Low            |
| Model Complexity  | Too simple      | Balanced    | Too complex    |
| Generalization    | Poor            | Good        | Poor           |

## What Dropout Does

Dropout randomly drops neurons during training with probability p.

Prevents the network from becoming too dependent on specific neurons.

Forces the model to learn redundant, robust representations.

The effect = like training many different smaller networks and averaging them.

👉 Key use: reduce overfitting (too much memorization of training data).

## The Weight Scaling Trick

When using dropout:

### Training phase:

Neurons are dropped with prob p.

Surviving neurons are scaled (1/(1-p) in modern frameworks).

Keeps expected output the same.

### Testing phase:

Dropout is turned off.

No scaling needed in modern frameworks (Keras, TF, PyTorch).

👉 Without scaling, predictions would be too small or too large, breaking consistency.

### When to Use Dropout

Your model is large (many parameters).

You see overfitting (train accuracy >> test accuracy).

### Avoid/dropout less when:

Dataset is very small (too much dropout → underfitting).

Using batch normalization (sometimes BN + Dropout hurts).

## Rule of Thumb

### If underfitting:

Increase model capacity (more layers/neurons).

Reduce dropout (or remove it).

Train longer / with better learning rate schedule.

### If overfitting:

Add dropout (start small, e.g., 0.2–0.5).

Add regularization (L2).

Use early stopping.

Use data augmentation.

### Key Takeaway

Underfitting = model too weak → increase complexity, train longer.

Overfitting = model too strong → add regularization, reduce complexity.

Goal: Balance between the two → a model that generalizes well to unseen data.
