import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
import matplotlib.pyplot as plt

# 1. โหลดข้อมูลและเลือกเฉพาะคอลัมน์ตัวเลข
df = pd.read_csv('StudentsPerformance.csv')
df_numeric = df.select_dtypes(include=['number'])

# 2. แยก X (features) และ y (target) — ใช้ math score เป็นตัวอย่าง
X = df_numeric.drop('math score', axis=1)
y = df_numeric['math score']

# 3. แบ่งข้อมูลเป็น Train/Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. สร้างและฝึกโมเดล Lasso
model = Lasso(alpha=0.1)
model.fit(X_train, y_train)

# 5. ทำนาย
y_pred = model.predict(X_test)

# 6. แสดงผล Predicted vs Actual ในรูป “Digit Classification”
n = 12  # จำนวนตัวอย่างที่จะแสดง
plt.figure(figsize=(12, 6))

for i in range(n):
    plt.subplot(3, 4, i + 1)
    # แสดงค่าทำนายเป็นตัวเลขใหญ่ตรงกลาง
    plt.text(
        0.5, 0.5,
        str(int(round(y_pred[i]))),
        fontsize=36,
        ha='center', va='center'
    )
    # แสดงค่าจริงไว้ใน title
    plt.title(f"Actual: {int(y_test.values[i])}", fontsize=10)
    plt.axis('off')

plt.suptitle("Predicted Math Scores (Lasso Regression)", fontsize=16)
plt.tight_layout()
plt.show()
