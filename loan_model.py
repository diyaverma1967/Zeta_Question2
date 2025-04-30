import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib

os.makedirs('output/checkpoints', exist_ok=True)

df = pd.read_csv('Loan_Data.csv')
df['Dependents']      = df['Dependents'].replace('3+', 3).astype(float)
df['Loan_Status']     = df['Loan_Status'].map({'Y':1, 'N':0})
df['TotalIncome']     = df['ApplicantIncome'] + df['CoapplicantIncome']
df['LoanAmount']      = df['LoanAmount'].fillna(df['LoanAmount'].median())

X = df.drop(['Loan_ID','Loan_Status'], axis=1)
y = df['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

numeric_features = ['ApplicantIncome','CoapplicantIncome','LoanAmount',
                    'Loan_Amount_Term','TotalIncome']
numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler())
])

categorical_features = ['Gender','Married','Dependents','Education',
                        'Self_Employed','Credit_History','Property_Area']
categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot',  OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features),
])

X_train_pp = preprocessor.fit_transform(X_train)
X_test_pp  = preprocessor.transform(X_test)

joblib.dump(preprocessor, 'output/preprocessor.joblib')


def create_model(input_dim):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_dim,)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

model = create_model(input_dim=X_train_pp.shape[1])


checkpoint_path = "output/checkpoints/model-{epoch:02d}-{val_loss:.2f}.h5"
best_model_path = "output/best_model.h5"

callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint(
        filepath=checkpoint_path,
        save_best_only=False,
        save_weights_only=False
    ),
    ModelCheckpoint(
        filepath=best_model_path,
        save_best_only=True,
        monitor='val_loss',
        save_weights_only=False
    )
]


history = model.fit(
    X_train_pp, y_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=callbacks
)


plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title('Accuracy')
plt.xlabel('Epoch'); plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title('Loss')
plt.xlabel('Epoch'); plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig('output/training_history.png')
plt.show()


best_model = load_model(best_model_path)
loss, acc = best_model.evaluate(X_test_pp, y_test, verbose=0)
print(f"\nBest model test accuracy: {acc:.4f}")


def generate_recommendation(score: float) -> str:
    """Translate numeric score (0–100) into user guidance."""
    if score >= 75:
        return "Excellent credit likelihood. You should qualify for premium loans at great rates."
    elif score >= 50:
        return "Good credit likelihood. Standard loan offers are likely available to you."
    elif score >= 25:
        return "Fair credit likelihood. Consider improving your on-time payments and reducing debts."
    else:
        return "Low credit likelihood. Focus on reducing missed payments and boosting stable income."

probas = best_model.predict(X_test_pp)[:,0]
scores = np.round(probas * 100, 2)

report_df = pd.DataFrame({
    'Actual':    y_test.values[:5],
    'Score(%)':  scores[:5],
    'Recommendation': [generate_recommendation(s) for s in scores[:5]]
})
print("\nSample recommendations on test set:")
print(report_df.to_string(index=False))


best_model.save('output/final_model.h5')
print("\nSaved final model to output/final_model.h5")
print("Saved preprocessor to output/preprocessor.joblib")
