
# 🧠 Disease Prediction from Symptoms Using Machine Learning

This project predicts diseases based on symptoms entered by users. It uses multiple machine learning classifiers—**Support Vector Machine (SVC)**, **Random Forest (RF)**, **Gaussian Naive Bayes (GNB)**, and an **ensemble Voting Classifier**—to enhance prediction reliability. The application is built with **Python** and presented through an intuitive **Streamlit** web interface.

## 📂 Project Structure

```plaintext
.
├── combine.py               # Voting Classifier (SVC + RF + GNB)
├── svc.PY                   # Support Vector Classifier (SVC)
├── rain.py                  # Random Forest Classifier
├── gaus.py                  # Gaussian Naive Bayes
├── Training data.csv        # Training dataset with symptoms and prognosis
├── Testing data.csv         # Testing dataset for evaluation
├── outputt.png              # Output screenshot (optional for visualization)
├── implementation paper     # paper based on the implementation of models
├── presentation ppt         # Ppt based on the project for better understanding 
|__ README.md                # Project documentation
```

## 🚀 How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/disease-prediction.git
cd disease-prediction
```

### 2. Install Requirements

Make sure you have Python 3.7+ installed. Then install the required libraries:

```bash
pip install pandas numpy scikit-learn streamlit
```

### 3. Run the Application

You can run any of the models independently or use the ensemble model:

```bash
streamlit run combine.py     # Ensemble Voting Classifier
streamlit run svc.PY         # SVC model
streamlit run rain.py        # Random Forest model
streamlit run gaus.py        # GaussianNB model
```

## 🧠 Model Details

| File        | Algorithm             | Notes                          |
|-------------|------------------------|--------------------------------|
| `combine.py`| Voting (SVC + RF + GNB)| Best accuracy via soft voting |
| `svc.PY`    | Support Vector Classifier | Works well on linear data    |
| `rain.py`   | Random Forest Classifier | Robust against overfitting   |
| `gaus.py`   | Gaussian Naive Bayes     | Simple and fast               |

Each model:
- Loads the `Training data.csv`
- Lets users select symptoms from a multi-select dropdown
- Outputs top 3 predicted diseases with confidence scores (where applicable)

## 📊 Dataset

- **`Training data.csv`**: Binary matrix of symptoms (columns) and diseases (last column: `prognosis`)
- **`Testing data.csv`**: Used for separate evaluation (optional in the current version)

## 🧪 Testing & Accuracy

- Voting Classifier uses an 80-20 train-test split.
- Model performance is measured using `accuracy_score`.
- Accuracy may vary based on model tuning and data balancing.

## 📷 Sample Output

A screenshot of the app interface showing predicted disease and probabilities is included for demo purposes.
![outputt](https://github.com/user-attachments/assets/c765175b-ea24-4a51-9915-a897ac144f79)


## 📌 Future Improvements

- Add NLP-based symptom input
- Integrate real-time medical APIs
- Improve dataset size and balance
- Add confusion matrix and F1-score metrics

## 👨‍💻 Authors

- Ritesh Kumar  
- Udit Maurya  
- Vishal Yadav  
- Department of IT, Galgotias College of Engineering and Technology
