# Student Academic Performance Prediction Using Deep Neural Networks and Random Forest

## Overview

This project uses Machine Learning and Deep Learning techniques to predict students' final academic performance based on demographic, academic, social, and behavioural factors.

It implements two regression models:
- **Deep Neural Network (DNN):** Learns complex patterns through multiple neural network layers.
- **Random Forest Regressor:** Uses an ensemble of decision trees to predict final grades.

The project compares both models using standard regression evaluation metrics and provides an interactive **Streamlit web application** for making predictions.

## Objectives

- Predict students' final grades using academic and personal information.
- Implement a Deep Neural Network using TensorFlow/Keras.
- Compare DNN performance with Random Forest.
- Perform Exploratory Data Analysis (EDA).
- Apply preprocessing and feature encoding techniques.
- Optimize DNN hyperparameters to improve performance.
- Build an interactive interface for real-time predictions.

## Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas & NumPy | Data manipulation |
| Matplotlib & Seaborn | Data visualization |
| Scikit-learn | Preprocessing and Random Forest |
| TensorFlow / Keras | Deep Neural Network |
| Streamlit | Interactive web application |
| Joblib | Saving and loading ML components |
| Jupyter Notebook | Model development and experimentation |

## Dataset

**Dataset:** UCI Student Performance Dataset (Portuguese Language)

- Total Records: 649 students
- Original Features: 32 input features
- Target Variable: `G3`
- Target Description: Final student grade (0–20)

The dataset contains information about:
- Demographic factors
- Family background
- Academic performance
- Study habits
- Social activities
- Behavioural factors

## Project Workflow

1. **Data Collection:** Load the UCI Student Performance dataset.
2. **Exploratory Data Analysis:** Analyze distributions, correlations, and patterns.
3. **Data Preprocessing:** Handle categorical and numerical features.
4. **Feature Encoding:** Apply One-Hot Encoding to categorical variables.
5. **Feature Scaling:** Apply StandardScaler to numerical variables.
6. **Model Development:** Build and train the DNN.
7. **Hyperparameter Tuning:** Experiment with learning rates, batch sizes, and dropout rates.
8. **Model Comparison:** Train Random Forest and compare its performance with DNN.
9. **Evaluation:** Evaluate models using MAE, MSE, RMSE, and R² Score.
10. **Deployment:** Integrate the trained DNN into a Streamlit application.

## Deep Neural Network Architecture

The DNN uses the following architecture:

| Layer | Configuration |
|---|---|
| Input Layer | Processed input features |
| Hidden Layer 1 | 64 neurons, ReLU |
| Dropout | 0.2 |
| Hidden Layer 2 | 32 neurons, ReLU |
| Dropout | 0.2 |
| Hidden Layer 3 | 16 neurons, ReLU |
| Output Layer | 1 neuron, Linear |

**Training Configuration:**
- Optimizer: Adam
- Learning Rate: 0.001 (default)
- Loss Function: Mean Squared Error
- Evaluation Metric: Mean Absolute Error
- Maximum Epochs: 200
- Batch Size: 16
- Early Stopping: Patience of 15 epochs

## Random Forest Model

Random Forest Regressor is implemented as the traditional Machine Learning comparison model.

**Configuration:**
- Number of Estimators: 100
- Maximum Depth: None
- Random State: 42
- Parallel Processing: Enabled (`n_jobs=-1`)

## Model Evaluation

Both models are evaluated using:

| Metric | Description |
|---|---|
| MAE | Average absolute prediction error |
| MSE | Average squared prediction error |
| RMSE | Square root of mean squared error |
| R² Score | Measures how well the model explains variation in the target |

The models are evaluated on the same test dataset for a fair comparison.

## Streamlit Application

The application provides:

- Student information input form.
- Predicted final academic grade.
- Academic performance category.
- Model analytics and evaluation visualizations.
- Project information and model details.

The DNN predicts a numerical grade, which is mapped into performance categories:

- Below 10: Needs Improvement
- 10 to below 14: Satisfactory
- 14 to below 17: Good
- 17 and above: Excellent

These categories are application-defined, not official UCI classifications.

## Project Structure

```text
Student-Performance-DNN/
│
├── app/
│   └── app.py
│
├── datasets/
│   └── student-por.csv
│
├── models/
│   ├── student_performance_dnn.keras
│   ├── random_forest_model.pkl
│   └── preprocessor.pkl
│
├── notebooks/
│   └── student_performance.ipynb
│
├── requirements.txt
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/Student-Performance-DNN.git
```

### 2. Navigate to the project directory

```bash
cd Student-Performance-DNN
```

### 3. Create a virtual environment (optional)

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit application

```bash
streamlit run app/app.py
```

The application will open in your browser.

## Future Improvements

- Experiment with additional ML and Deep Learning architectures.
- Improve generalization using larger and more diverse datasets.
- Add explainability techniques for understanding prediction factors.
- Improve hyperparameter optimization.
- Explore additional academic performance indicators.

## Contributors

- **Atuliya Bansal** — Machine Learning & Deep Learning Development
- **[Friend's Name]** — [Her Contribution]

## Acknowledgements

Dataset: [UCI Student Performance Dataset](https://archive.ics.uci.edu/dataset/320/student+performance)

Developed as an academic Machine Learning and Deep Learning project.

---

**Disclaimer:** This project is intended for educational and research purposes. Predictions are estimates and should not be treated as definitive assessments of a student's academic potential.
