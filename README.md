
# 🫀 Heart Disease Risk Prediction & Health Scoring System

## 📄 Project Description
This project is a machine learning-based diagnostic tool designed to predict heart disease risk using patient health data. It includes comprehensive analysis such as correlation and feature interaction, risk factor evaluation, demographic trend analysis, patient clustering into risk groups, and a custom patient health scoring system. It is also designed to support a multi-disease prediction platform.

---

## 🧾 Instructions
1. Clone the repository to your local machine.
2. Open the Jupyter notebooks to explore data analysis, model training, and scoring system.
3. Run the Streamlit app for a real-time interactive interface (optional).
4. Input patient health metrics to predict heart disease risk and receive a personalized health score.

---

## 📦 Dependencies
- Python 3.8+
- pandas
- numpy
- scikit-learn
- seaborn
- matplotlib
- plotly
- joblib
- streamlit (for app interface)

---

## ⚙️ Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/heart-risk-prediction.git
   cd heart-risk-prediction
   ```
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) Run the app:
   ```bash
   streamlit run streamlit_app/app.py
   ```

---

## 🚀 Usage Examples

### Example 1: Predict risk using model
```python
import joblib
model = joblib.load("models/top_features_heart_model.pkl")
sample_input = [[1, 0, 1, 55, 180, 27.5, 14.0, 6, 210, 1]]
prediction = model.predict(sample_input)
```

### Example 2: Run scoring system in Jupyter Notebook
Open `notebooks/health_scoring.ipynb` to evaluate individual patients based on normalized health metrics.

---

## 👥 Credits
- **Project Lead**: Sammyst Analytics Solutions  
- **Data Source**: Publicly available health datasets (cleaned and anonymized)
- **ML Models**: Trained using Scikit-learn
- Special thanks to the open-source community for tools and libraries.

---

## 📜 License
This project is licensed under the MIT License.  
You are free to use, modify, and distribute this code with attribution.

---

## 🤝 Contributing Guidelines
We welcome contributions!  
To contribute:
1. Fork the repo.
2. Create a new branch (`git checkout -b feature-name`).
3. Make your changes.
4. Push to your fork and open a pull request.
5. Ensure code is clean, commented, and tested.

---

## 📬 Contact Information
- **Name**: Sammyst The Analyst  
- **Email**: sammysttheanalyst@gmail.com  
- **YouTube**: [@sammysttheanalyst](https://www.youtube.com/@sammysttheanalyst)  
- **Website**: [sammysttheanalyst.github.io](https://sammysttheanalyst.github.io)

---
