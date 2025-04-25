import pandas as pd

# โหลดไฟล์
df = pd.read_csv('StudentsPerformance.csv')

# เลือกเฉพาะคอลัมน์ที่เป็นตัวเลข
df_numeric = df.select_dtypes(include=['number'])

# แสดงชื่อคอลัมน์ดูว่าเราจะใช้คอลัมน์ไหนเป็น target
print(df_numeric.columns)
