import pandas as pd
df = pd.read_csv('parkinsons.csv')
df.head()

features = ['PPE', 'DFA']
target = 'status'
x = df[features]
y = df[target]

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
x_scaled = scaler.fit_transform(x)

from sklearn.model_selection import train_test_split
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

from sklearn.svm import SVC
model = SVC()
model.fit(x_train, y_train)

from sklearn.metrics import accuracy_score
y_pred = model.predict(x_val)
accuracy = accuracy_score(y_val, y_pred)
print(f'Accuracy: {accuracy}')
