import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

df = pd.read_csv("Training.csv/Training.csv")

# Drop duplicate columns (fluid_overload appears twice in header)
df = df.loc[:, ~df.columns.duplicated()]

# Strip whitespace from column names and target values
df.columns = df.columns.str.strip()
df["prognosis"] = df["prognosis"].str.strip()

X = df.drop("prognosis", axis=1)
y = df["prognosis"]

symptoms = list(X.columns)

model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X, y)

# Cross-val gives a fair accuracy estimate
scores = cross_val_score(model, X, y, cv=5)
print(f"CV accuracy: {scores.mean()*100:.1f}% ± {scores.std()*100:.1f}%")
print(f"Training accuracy: {model.score(X, y)*100:.1f}%")
print(f"Symptoms: {len(symptoms)}, Diseases: {len(y.unique())}")

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("symptoms.pkl", "wb") as f:
    pickle.dump(symptoms, f)

print("Saved model.pkl and symptoms.pkl")
