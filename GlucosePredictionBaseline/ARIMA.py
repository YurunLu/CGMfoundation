import pmdarima as pm
from pmdarima import model_selection
from sklearn.metrics import mean_squared_error, mean_absolute_error
from GluPreBaselineutils import load_data
import pandas as pd
import numpy as np
HORIZON = 12
STRIDE = 6
def main(data):
    # 3. 训练模型 (为速度, 使用固定参数。真实场景请用 auto_arima)
    arima_model = pm.auto_arima(data, m=30, stepwise=True, suppress_warnings=True)
    arima_model.fit(data)
    print("ARIMA 训练完毕。")
    predictions_rolling = []
    ground_truth_rolling = []

    test_indices = range(0, len(data) - HORIZON + 1, STRIDE)

    if not list(test_indices):
        print("ARIMA: 测试集太短，无法执行滚动预测。")
        return

    for i in test_indices:
        # 1. 预测
        preds_12_step = arima_model.predict(n_periods=HORIZON)
        predictions_rolling.append(preds_12_step)

        # 2. 收集真实值
        true_values = data[i: i + HORIZON]
        ground_truth_rolling.append(true_values)

        # 3. 更新模型 (使用步长 STRIDE)
        next_true_values = data[i: i + STRIDE]
        arima_model.append(next_true_values, suppress_warnings=True)

    y_pred_arima = np.array(predictions_rolling)
    y_val_arima = np.array(ground_truth_rolling)
    # 打印即时MA_E
    mae = mean_absolute_error(y_val_arima, y_pred_arima)
    print(f"ARIMA (即时) MAE: {mae:.4f}")
if __name__=='__main__':
    data = load_data('../data/2022-06_1.csv')
    main(data)
