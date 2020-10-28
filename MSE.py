import numpy as np
import matplotlib.pyplot as plt

w = 3
b = 0.5

x_lin = np.linspace(0, 100, 101)

y = (x_lin + np.random.randn(101) * 5) * w + b

plt.plot(x_lin, y, 'b.', label = 'data points')
plt.title("Assume we have data points")
plt.legend(loc = 2)
plt.show()

y_hat = x_lin * w + b
plt.plot(x_lin, y, 'b.', label = 'data')

plt.plot(x_lin, y_hat, 'r-', label = 'prediction')
plt.title("Assume we have data points (And the prediction)")
plt.legend(loc = 2)
plt.show()


def mean_square_error(y, yp):
    """
    計算 MAE
    Args:
        - y: 實際值
        - yp: 預測值
    Return:
        - mse: MSE
    """
    # MAE : 將兩個陣列相減後, 取平方(**2), 再將整個陣列加總成一個數字(sum), 最後除以y的長度(len), 因此稱為"均方誤差"
    mse = MSE = sum((y - yp)**2) / len(y)
    return mse

# 呼叫上述函式, 傳回 y(藍點高度)與 y_hat(紅線高度) 的 MAE
MSE = mean_square_error(y, y_hat)
print("The Mean square error is %.3f" % (MSE))