import numpy as np

# เลือกแค่ 20 ตัวอย่าง
n = 20
indices = np.arange(n)

plt.figure(figsize=(12, 6))
plt.bar(indices - 0.2, y_test[:n], width=0.4, label='Actual')
plt.bar(indices + 0.2, y_pred[:n], width=0.4, label='Predicted')
plt.xlabel("Student Index")
plt.ylabel("Math Score")
plt.title("Actual vs Predicted (First 20 Samples)")
plt.legend()
plt.grid(True)
plt.show()
