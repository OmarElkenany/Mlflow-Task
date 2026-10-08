# Mlflow-Task
# MLflow Experiment Tracking – California Housing

## 1. Task description
Train the same model **XGBoost** on the
California Housing dataset three times with different hyperparameters, track
everything with MLflow (parameters, metrics, model artifacts), compare the runs,
and select the best model using **RMSE** as the primary metric.

- Split: 80% train / 20% validation (`random_state=42`)
- Tracked params: `max_depth`, `learning_rate`
- Tracked metrics: `val_rmse`, `val_mae`, `val_r2`
- Model for each run saved as an MLflow artifact (`xgboost-model`)

## 2. How to run
bash
pip install -r requirements.txt
mlflow ui                # terminal 1 -> http://127.0.0.1:5000
python train.py          # terminal 2

In the UI open the California_Housing_Optimization experiment, select the three runs and click Compare


## 3. Screenshot of MLflow results
<img width="1512" height="807" alt="image" src="https://github.com/user-attachments/assets/87e7ebc4-c302-4885-b70e-eb602278b26e" />




## 4. Comparison table

| Run   | Max Depth | Learning Rate | RMSE  | MAE   | R²    |
|-------|-----------|---------------|-------|-------|-------|
| Run 1 | 3         | 0.1           | 0.539 | 0.368 | 0.779 |
| Run 2 | 5         | 0.05          | **0.522** | **0.355** | **0.792** |
| Run 3 | 7         | 0.01          | 0.700 | 0.533 | 0.626 |

## 5. Selected best model
**Run 2** (`max_depth=5`, `learning_rate=0.05`)

## 6. Why it was selected
Used RMSE for primary selection which its metric lower is better. Run 2 has the lowest
validation RMSE which is (0.522), and it is also best on MAE (0.355) and R² (0.792), so
the choice is consistent across all three metrics
