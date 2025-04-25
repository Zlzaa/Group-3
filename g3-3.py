import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred)
plt.xlabel("Actual math score")
plt.ylabel("Predicted math score")
plt.title("Actual vs Predicted")
plt.grid(True)
plt.show()
