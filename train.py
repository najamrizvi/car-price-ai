import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

from config import DATA_PATH, MODEL_PATH, TEST_SIZE, RANDOM_STATE
from utils import preprocess_data

df = pd.read_csv(DATA_PATH)
df = preprocess_data(df)

X = df.drop('price', axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

model = LinearRegression()
model.fit(X_train, y_train)

joblib.dump(model, MODEL_PATH)

print("✅ Model trained and saved!")