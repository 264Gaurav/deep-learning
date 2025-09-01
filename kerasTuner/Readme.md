# 🔧 Hyperparameter Tuning with Keras Tuner

This project demonstrates **Hyperparameter Tuning** using **Keras Tuner** for both **classification** and **regression** tasks.

---

## 📌 What is Hyperparameter Tuning?

- **Hyperparameters** are settings you choose **before training** a model (e.g., learning rate, number of layers, dropout rate).
- They strongly affect model performance.
- **Hyperparameter tuning** is the process of finding the **best set of hyperparameters** for a given problem.

Keras Tuner provides strategies like:

- **RandomSearch** – tries random combinations.
- **Hyperband** – early stopping, efficient for deep learning.
- **Bayesian Optimization** – guided search using probability models.

---

## 📊 Datasets Used

### 1. **Regression Problem (Air Quality Prediction)**

- **Input features**: Meteorological variables (temperature, humidity, pressure, etc.).
- **Target**: `PM 2.5` (continuous air quality value).
- **Goal**: Predict PM 2.5 levels using regression.

### 2. **Classification Problem (Customer Churn Prediction)**

- **Input features**: Customer details (`CreditScore`, `Geography`, `Gender`, `Age`, `Balance`, etc.).
- **Target**: `Exited` (0 = customer stayed, 1 = customer left).
- **Goal**: Predict whether a customer will churn (binary classification).

---

## ⚙️ Project Workflow

1. **Data Preprocessing**

   - Drop irrelevant columns (IDs, names).
   - Encode categorical variables (`LabelEncoder`).
   - Normalize features using `StandardScaler`.
   - Train-test split for validation.

2. **Model Building with Keras Tuner**

   - Define a `build_model(hp)` function:
     - Tunable parameters:
       - Number of hidden layers.
       - Units (neurons per layer).
       - Dropout (on/off, dropout rate).
       - Learning rate.
     - Output layer:
       - Regression → 1 neuron, `linear` activation.
       - Classification → 1 neuron (`sigmoid`) for binary, `softmax` for multi-class.
   - Compile model with appropriate:
     - Loss function:
       - Regression → `mean_absolute_error`
       - Classification → `binary_crossentropy` / `sparse_categorical_crossentropy`
     - Metrics:
       - Regression → `MAE`
       - Classification → `accuracy`

3. **Hyperparameter Tuning**

   - Use `RandomSearch` tuner:
     ```python
     tuner = RandomSearch(
         build_model,
         objective='val_accuracy'  # or 'val_mean_absolute_error'
         max_trials=10,
         executions_per_trial=2,
         directory='project',
         project_name='Tuning_Example'
     )
     ```
   - Run tuning:
     ```python
     tuner.search(X_train, y_train,
                  epochs=20,
                  validation_data=(X_val, y_val))
     ```

4. **Best Model Selection**

   - View results:
     ```python
     tuner.results_summary()
     ```
   - Retrieve best model:
     ```python
     best_model = tuner.get_best_models(num_models=1)[0]
     ```

5. **Evaluation**
   - Regression → Evaluate using MAE on validation data.
   - Classification → Evaluate using accuracy on validation data.

---
