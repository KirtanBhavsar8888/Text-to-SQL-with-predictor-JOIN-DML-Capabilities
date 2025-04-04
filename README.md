---

# Text-to-SQL with Predictor JOIN DML Capabilities

A powerful Python-based application that combines **Text-to-SQL query generation** with **data prediction capabilities**. This project leverages advanced machine learning models, including **LSTM for time series forecasting**, to provide seamless query execution and predictive analytics on uploaded datasets.

---

## 🚀 Features

- **Text-to-SQL Query Generation**: Convert natural language questions into SQL queries for efficient database interaction.
- **Multi-Table JOIN Support**: Execute complex SQL queries involving multiple tables.
- **Predictive Analytics**:
  - Regression for continuous value prediction.
  - Classification for categorical prediction.
  - Time Series Forecasting using LSTM and other models.
- **Manual Problem Type Override**: Flexibility to switch between regression, classification, and time series detection.
- **Interactive Web Interface**: Built with Streamlit for an intuitive user experience.
- **Synthetic Data Generation**: Generate large datasets for testing predictive models.

---

## 📂 Project Structure

| File/Folder          | Description                                                                 |
|----------------------|-----------------------------------------------------------------------------|
| `app.py`             | Main Streamlit application integrating Text-to-SQL and prediction modules. |
| `predictor.py`       | Handles predictive analytics, model training, and visualization.           |
| `requirements.txt`   | List of dependencies required to run the application.                      |
| `data.db`            | Sample SQLite database used for testing Text-to-SQL queries.               |
| `synthetic_dataset.csv` | Synthetic dataset for regression, classification, and time series testing. |
| `Time_Series_data.csv` | Sample dataset for time series forecasting.                               |

---

## 🛠️ Installation

Follow these steps to set up the project in your local environment:

1. Clone the repository:
   ```bash
   git clone https://github.com/KirtanBhavsar8888/Text-to-SQL-with-predictor-JOIN-DML-Capabilities.git
   cd Text-to-SQL-with-predictor-JOIN-DML-Capabilities
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:
   - **Windows**:
     ```bash
     venv\Scripts\activate
     ```
   - **macOS/Linux**:
     ```bash
     source venv/bin/activate
     ```

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

5. Run the application:
   ```bash
   streamlit run app.py
   ```

---

## 📊 Usage

### Text-to-SQL Query Generation
1. Upload your database file (e.g., `data.db`).
2. Enter a natural language query into the text box (e.g., "Show all employees earning more than $5000").
3. View the generated SQL query and its execution results.

### Predictive Analytics
1. Upload your dataset (e.g., `synthetic_dataset.csv`).
2. Select the target column and input features.
3. Choose the problem type (regression, classification, or time series).
4. Train models and visualize results, including feature importance and predictions.

### Time Series Forecasting with LSTM
1. Upload a time series dataset (e.g., `Time_Series_data.csv`).
2. Select the target column and date column.
3. Train the LSTM model and view forecast results.

---

## 📋 Requirements

- Python 3.8 or higher
- Streamlit 1.x
- TensorFlow 2.x (for LSTM models)
- Scikit-learn
- Plotly

Refer to `requirements.txt` for a complete list of dependencies.

---

### Synthetic Data Generation
Run the following script to generate synthetic datasets:
```bash
python data_generator.py
```

---

## 🤝 Contributing

Contributions are welcome! To contribute:
1. Fork this repository.
2. Create a new branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m "Added new feature"`).
4. Push to your branch (`git push origin feature-name`).
5. Create a pull request.


## 📧 Contact

For any queries or suggestions, feel free to reach out:

- **Author**: Kirtan Bhavsar  
- **Email**: [katha.kirtan2@gmail.com](mailto:katha.kirtan2@gmail.com)  
- **GitHub**: [KirtanBhavsar8888](https://github.com/KirtanBhavsar8888)

---

This README provides a professional overview of your project while being easy to navigate for users and contributors alike! Let me know if you'd like further tweaks or additions! 😊


---

