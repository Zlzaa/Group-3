import matplotlib.pyplot as plt

errors = y_test - y_pred

plt.hist(errors, bins=20, edgecolor='black')
plt.title("Error Distribution")
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()
