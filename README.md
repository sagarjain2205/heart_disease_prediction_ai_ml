# 💓 Heart Disease Prediction AI System

This project is an **AI-based system that predicts the risk of heart disease using common clinical inputs** that doctors typically collect during routine checkups.

The application takes patient health parameters such as **age, blood pressure, cholesterol, heart rate, etc.** and uses trained machine learning models to determine whether a patient is at **high or low risk of heart disease**.

The trained models are integrated into an **interactive Streamlit dashboard** where users can enter patient data and receive predictions in real time.

---

# 🧠 Models Used

The system uses three different AI models:

* **Random Forest**
* **XGBoost**
* **Neural Network (PyTorch)**

### Approximate Model Accuracy

| Model          | Accuracy |
| -------------- | -------- |
| Random Forest  | ~90%     |
| XGBoost        | ~88–90%  |
| Neural Network | ~85–89%  |

Random Forest showed the most stable performance on the dataset.

---

# 📊 Streamlit Application

The trained models are integrated into a **Streamlit web application** where users can:

* Enter patient vitals
* Run predictions using different AI models
* Compare predictions from multiple models
* View patient input summary
* Download a diagnostic report

---

# 📥 Input Features

The system uses **13 clinical indicators** including:

* Age
* Sex
* Chest pain type
* Resting blood pressure
* Cholesterol
* Fasting blood sugar
* Resting ECG
* Maximum heart rate
* Exercise induced angina
* ST depression
* Slope of ST segment
* Number of major vessels
* Thalassemia indicator

---

# 🧪 Model Training Notebook

The repository includes **`heart_ml.ipynb`**, which contains the full machine learning experimentation workflow.

In this notebook:

* The heart disease dataset is loaded and explored.

* Data preprocessing and **feature scaling using StandardScaler** is performed.

* Multiple machine learning models are trained and compared, including:

  * Logistic Regression
  * KNN
  * Decision Tree
  * Random Forest
  * Support Vector Machine
  * XGBoost

* Models are evaluated using metrics such as **accuracy, precision, recall, and confusion matrix**.

After evaluating all models, the **best performing models were selected** and exported for deployment.

The final system integrates the **top 3 models**:

* Random Forest
* XGBoost
* Neural Network (PyTorch)

These models achieved the best performance and were saved inside the **models/** directory for use in the Streamlit application.

---

# ⚙️ Run the Application

```bash
git clone https://github.com/sagarjain2205/heart_disease_prediction_ai_ml.git
cd heart_disease_prediction_ai_ml
pip install -r requirements.txt
streamlit run app.py
```

---

# 🛠 Tech Stack

* Python
* Streamlit
* Scikit-Learn
* XGBoost
* PyTorch
* Pandas / NumPy

---

# 📂 Project Structure

```
heart_disease_prediction_ai_ml

app.py
model_training.py
heart_ml.ipynb

models/
   random_forest_model.pkl
   xgboost_model.json
   pytorch_model.pth
   scaler.pkl

heart.csv
requirements.txt
```

---

# 👨‍💻 Author

**Sagar Jain**
AI / Machine Learning Projects

GitHub:
https://github.com/sagarjain2205
