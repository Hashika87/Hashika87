{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "e2651d48-f1c2-4df2-ad51-9f9ef9525ac6",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "✅ Model Accuracy: 0.7832167832167832\n",
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.79      0.84      0.81        80\n",
      "           1       0.78      0.71      0.74        63\n",
      "\n",
      "    accuracy                           0.78       143\n",
      "   macro avg       0.78      0.78      0.78       143\n",
      "weighted avg       0.78      0.78      0.78       143\n",
      "\n",
      "✅ Model and scaler saved.\n"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import LabelEncoder, StandardScaler\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.metrics import accuracy_score, classification_report\n",
    "import joblib\n",
    "\n",
    "# Load dataset\n",
    "data = pd.read_csv(\"titanic.csv\")\n",
    "\n",
    "# Drop unnecessary or high-missing columns\n",
    "data.drop([\"deck\", \"embark_town\", \"alive\", \"who\", \"adult_male\", \"class\"], axis=1, inplace=True, errors='ignore')\n",
    "\n",
    "# Drop rows with essential missing values\n",
    "data.dropna(subset=[\"age\", \"embarked\", \"sex\", \"fare\"], inplace=True)\n",
    "\n",
    "# Encode categorical features\n",
    "le = LabelEncoder()\n",
    "data[\"sex\"] = le.fit_transform(data[\"sex\"])\n",
    "data[\"embarked\"] = le.fit_transform(data[\"embarked\"])\n",
    "\n",
    "# Drop non-numeric/non-relevant columns\n",
    "data.drop([\"alone\"], axis=1, inplace=True, errors='ignore')\n",
    "\n",
    "# Features & target\n",
    "X = data.drop(\"survived\", axis=1)\n",
    "y = data[\"survived\"]\n",
    "\n",
    "# Split\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "\n",
    "# Scale\n",
    "scaler = StandardScaler()\n",
    "X_train_scaled = scaler.fit_transform(X_train)\n",
    "X_test_scaled = scaler.transform(X_test)\n",
    "\n",
    "# Train model\n",
    "model = RandomForestClassifier(n_estimators=100, random_state=42)\n",
    "model.fit(X_train_scaled, y_train)\n",
    "\n",
    "# Evaluate\n",
    "y_pred = model.predict(X_test_scaled)\n",
    "print(\"✅ Model Accuracy:\", accuracy_score(y_test, y_pred))\n",
    "print(classification_report(y_test, y_pred))\n",
    "\n",
    "# Save model and scaler\n",
    "joblib.dump(model, \"titanic_model.pkl\")\n",
    "joblib.dump(scaler, \"scaler.pkl\")\n",
    "print(\"✅ Model and scaler saved.\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bc31ef7a-4801-4698-9ff2-5d58cbeb5353",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
