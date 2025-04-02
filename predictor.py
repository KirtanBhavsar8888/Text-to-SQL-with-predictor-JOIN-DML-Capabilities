import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Any, Optional, Union
import traceback
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error,
                            accuracy_score, f1_score, precision_score, recall_score,
                            confusion_matrix, classification_report)
from sklearn.preprocessing import LabelEncoder

# Regression models
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Classification models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Time series models (core ones that are part of standard libraries)
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Optional imports with fallbacks
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False


def detect_problem_type(df: pd.DataFrame, target_col: str) -> str:
    """
    Automatically detect the type of problem based on the target column and dataset structure.

    Args:
        df: DataFrame containing the data.
        target_col: Name of the target column.

    Returns:
        String indicating problem type: "regression", "classification", or "time_series".
    """
    # Check if there's a datetime column in the dataset
    date_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]

    # Check if it's likely a time series problem
    if date_cols:
        # If there is at least one datetime column and the target is numeric, it's likely time series
        if pd.api.types.is_numeric_dtype(df[target_col]):
            return "time_series"

    # Check if classification or regression
    if pd.api.types.is_numeric_dtype(df[target_col]):
        unique_ratio = df[target_col].nunique() / len(df)
        # If few unique values compared to total, likely classification
        if df[target_col].nunique() < 10 and unique_ratio < 0.05:
            return "classification"
        else:
            return "regression"
    else:
        # Categorical target
        return "classification"



def get_model_options(problem_type: str) -> List[str]:
    """
    Return appropriate models for the problem type.
    
    Args:
        problem_type: String indicating problem type
        
    Returns:
        List of model names appropriate for the problem type
    """
    if problem_type == "regression":
        models = ["Linear Regression", "Decision Tree", "Random Forest", "Gradient Boosting"]
        if XGBOOST_AVAILABLE:
            models.append("XGBoost")
        return models
    
    elif problem_type == "classification":
        models = ["Logistic Regression", "Decision Tree", "Random Forest", "Gradient Boosting"]
        if XGBOOST_AVAILABLE:
            models.append("XGBoost")
        return models
    
    else:  # time_series
        models = ["ARIMA", "Exponential Smoothing"]
        if PROPHET_AVAILABLE:
            models.append("Prophet")
        return models


def preprocess_data(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    """
    Create a preprocessor for the data.
    
    Args:
        X: Feature DataFrame
        
    Returns:
        Tuple containing:
        - ColumnTransformer preprocessor
        - List of numerical feature names
        - List of categorical feature names
    """
    # Identify categorical and numerical features
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_features = X.select_dtypes(include=['number']).columns.tolist()
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'
    )
    
    return preprocessor, numerical_features, categorical_features


def get_feature_importance(model, preprocessor, numerical_features, categorical_features) -> Optional[pd.DataFrame]:
    """
    Extract feature importance if available.
    
    Args:
        model: Trained model
        preprocessor: ColumnTransformer for preprocessing
        numerical_features: List of numerical feature names
        categorical_features: List of categorical feature names
        
    Returns:
        DataFrame with feature importance or None if not available
    """
    if hasattr(model, 'feature_importances_'):
        try:
            # For models with feature_importances_ attribute
            importances = model.feature_importances_
            
            # Get feature names after preprocessing
            feature_names = []
            
            # Add numerical feature names directly
            feature_names.extend(numerical_features)
            
            # For categorical features, get the one-hot encoded column names
            if categorical_features:
                # Get the one-hot encoder
                ohe = preprocessor.named_transformers_['cat']
                if hasattr(ohe, 'get_feature_names_out'):
                    cat_feature_names = ohe.get_feature_names_out(categorical_features)
                    feature_names.extend(cat_feature_names)
                else:
                    # Fallback for older scikit-learn versions
                    for cat_feature in categorical_features:
                        cats = ohe.categories_[categorical_features.index(cat_feature)]
                        feature_names.extend([f"{cat_feature}_{cat}" for cat in cats])
            
            # Ensure lengths match (sometimes they don't due to feature selection)
            if len(importances) == len(feature_names):
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                return importance_df
            else:
                # If lengths don't match, create a simpler version with indices
                importance_df = pd.DataFrame({
                    'Feature': [f"Feature {i}" for i in range(len(importances))],
                    'Importance': importances
                }).sort_values('Importance', ascending=False)
                return importance_df
                
        except Exception as e:
            print(f"Error getting feature importance: {e}")
            return None
    
    # For models with coef_ attribute (like Linear/Logistic Regression)
    elif hasattr(model, 'coef_'):
        try:
            coefficients = model.coef_
            
            # For multi-class, take the mean absolute coefficient
            if len(coefficients.shape) > 1:
                importances = np.mean(np.abs(coefficients), axis=0)
            else:
                importances = np.abs(coefficients)
            
            # Get feature names (simplified approach)
            feature_names = [f"Feature {i}" for i in range(len(importances))]
            
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            return importance_df
            
        except Exception as e:
            print(f"Error getting coefficients: {e}")
            return None
    
    return None


def train_regression_model(X_train, X_test, y_train, y_test, 
                         preprocessor, model_name, numerical_features, 
                         categorical_features) -> Dict[str, Any]:
    """
    Train and evaluate a regression model.
    
    Args:
        X_train: Training features
        X_test: Testing features
        y_train: Training target
        y_test: Testing target
        preprocessor: ColumnTransformer for preprocessing
        model_name: Name of the model to use
        numerical_features: List of numerical feature names
        categorical_features: List of categorical feature names
        
    Returns:
        Dictionary with model, metrics, and feature importance
    """
    # Initialize the model
    if model_name == "Linear Regression":
        model = LinearRegression()
    elif model_name == "Decision Tree":
        model = DecisionTreeRegressor(random_state=42)
    elif model_name == "Random Forest":
        model = RandomForestRegressor(random_state=42, n_estimators=100)
    elif model_name == "Gradient Boosting":
        model = GradientBoostingRegressor(random_state=42)
    elif model_name == "XGBoost" and XGBOOST_AVAILABLE:
        model = xgb.XGBRegressor(random_state=42)
    else:
        # Default to Random Forest if model not recognized
        model = RandomForestRegressor(random_state=42)
        model_name = "Random Forest"
    
    # Create pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Train model
    pipeline.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = pipeline.predict(X_test)
    metrics = {
        'R² Score': r2_score(y_test, y_pred),
        'Mean Squared Error': mean_squared_error(y_test, y_pred),
        'Mean Absolute Error': mean_absolute_error(y_test, y_pred),
        'Root Mean Squared Error': np.sqrt(mean_squared_error(y_test, y_pred))
    }
    
    # Get feature importance
    feature_importance = get_feature_importance(model, preprocessor, numerical_features, categorical_features)
    
    return {
        'model': pipeline,
        'model_name': model_name,
        'metrics': metrics,
        'feature_importance': feature_importance
    }


from sklearn.preprocessing import LabelEncoder

def train_classification_model(X_train, X_test, y_train, y_test,
                               preprocessor, model_name, numerical_features,
                               categorical_features) -> Dict[str, Any]:
    """
    Train and evaluate a classification model.

    Args:
        X_train: Training features
        X_test: Testing features
        y_train: Training target
        y_test: Testing target
        preprocessor: ColumnTransformer for preprocessing
        model_name: Name of the model to use
        numerical_features: List of numerical feature names
        categorical_features: List of categorical feature names

    Returns:
        Dictionary with model, metrics, and feature importance
    """
    # Encode target labels if they are not numeric
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_test_encoded = label_encoder.transform(y_test)

    # Initialize the model
    if model_name == "Logistic Regression":
        model = LogisticRegression(max_iter=1000, random_state=42)
    elif model_name == "Decision Tree":
        model = DecisionTreeClassifier(random_state=42)
    elif model_name == "Random Forest":
        model = RandomForestClassifier(random_state=42, n_estimators=100)
    elif model_name == "Gradient Boosting":
        model = GradientBoostingClassifier(random_state=42)
    elif model_name == "XGBoost" and XGBOOST_AVAILABLE:
        model = xgb.XGBClassifier(random_state=42)
    else:
        # Default to Random Forest if model not recognized
        model = RandomForestClassifier(random_state=42)
        model_name = "Random Forest"

    # Create pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    # Train the model
    pipeline.fit(X_train, y_train_encoded)

    # Evaluate the model
    y_pred = pipeline.predict(X_test)

    # Handle binary vs multi-class metrics
    classes = label_encoder.classes_
    if len(classes) == 2:
        # Binary classification metrics
        metrics = {
            'Accuracy': accuracy_score(y_test_encoded, y_pred),
            'Precision': precision_score(y_test_encoded, y_pred, pos_label=1),
            'Recall': recall_score(y_test_encoded, y_pred, pos_label=1),
            'F1 Score': f1_score(y_test_encoded, y_pred, pos_label=1)
        }
    else:
        # Multi-class classification metrics
        metrics = {
            'Accuracy': accuracy_score(y_test_encoded, y_pred),
            'Macro F1': f1_score(y_test_encoded, y_pred, average='macro'),
            'Weighted F1': f1_score(y_test_encoded, y_pred, average='weighted')
        }

    # Get feature importance
    feature_importance = get_feature_importance(model, preprocessor,
                                                 numerical_features,
                                                 categorical_features)

    # Get confusion matrix (as list of lists for JSON serialization)
    cm = confusion_matrix(y_test_encoded, y_pred).tolist()

    return {
        'model': pipeline,
        'model_name': model_name,
        'metrics': metrics,
        'feature_importance': feature_importance,
        'confusion_matrix': cm,
        'classes': list(classes)  # Save original class labels for display
    }



def train_time_series_model(df, target_col, model_name) -> Dict[str, Any]:
    """
    Train and evaluate a time series model.
    
    Args:
        df: DataFrame containing the data
        target_col: Name of the target column
        model_name: Name of the model to use
        
    Returns:
        Dictionary with model, metrics, and forecast
    """
    # Ensure data is sorted by time
    if pd.api.types.is_datetime64_any_dtype(df[target_col]):
        df = df.sort_values(by=target_col)
        date_col = target_col
    else:
        # If target is not datetime, look for a date column
        date_cols = [col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])]
        if date_cols:
            date_col = date_cols[0]
            df = df.sort_values(by=date_col)
        else:
            # Create a date index if no date column exists
            df['date_index'] = pd.date_range(start='2022-01-01', periods=len(df))
            date_col = 'date_index'
            df = df.sort_values(by=date_col)
    
    # Find a numeric column to forecast if target is datetime
    if pd.api.types.is_datetime64_any_dtype(df[target_col]):
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        if numeric_cols:
            value_col = numeric_cols[0]
        else:
            raise ValueError("No numeric column found to forecast")
    else:
        value_col = target_col
    
    # Prepare training data (use last 20% for testing)
    train_size = int(len(df) * 0.8)
    train = df.iloc[:train_size]
    test = df.iloc[train_size:]
    
    # Forecast periods
    forecast_periods = len(test)
    
    # Train the model based on type
    if model_name == "ARIMA":
        try:
            # Fit ARIMA model
            model = ARIMA(train[value_col].values, order=(5, 1, 0))
            model_fit = model.fit()
            
            # Make forecast
            forecast = model_fit.forecast(steps=forecast_periods)
            
            # Evaluate
            if len(test) > 0:
                mse = mean_squared_error(test[value_col].values, forecast)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(test[value_col].values, forecast)
            else:
                mse = mae = rmse = np.nan
            
            metrics = {
                'Mean Squared Error': mse,
                'Root Mean Squared Error': rmse,
                'Mean Absolute Error': mae
            }
            
            # Create forecast DataFrame
            forecast_dates = test[date_col] if len(test) > 0 else pd.date_range(
                start=train[date_col].iloc[-1], 
                periods=forecast_periods+1, 
                closed='right'
            )
            
            forecast_df = pd.DataFrame({
                'Date': forecast_dates[:len(forecast)],
                'Forecast': forecast,
                'Actual': test[value_col].values[:len(forecast)] if len(test) > 0 else None
            })
            
            return {
                'model': model_fit,
                'model_name': model_name,
                'metrics': metrics,
                'forecast': forecast_df,
                'date_col': date_col,
                'value_col': value_col
            }
            
        except Exception as e:
            print(f"ARIMA error: {e}")
            # Fallback to simpler method
            model_name = "Exponential Smoothing"
    
    if model_name == "Exponential Smoothing":
        try:
            # Fit Exponential Smoothing model
            model = ExponentialSmoothing(
                train[value_col].values,
                trend='add',
                seasonal=None,
                initialization_method="estimated"
            )
            model_fit = model.fit()
            
            # Make forecast
            forecast = model_fit.forecast(forecast_periods)
            
            # Evaluate
            if len(test) > 0:
                mse = mean_squared_error(test[value_col].values[:len(forecast)], forecast)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(test[value_col].values[:len(forecast)], forecast)
            else:
                mse = mae = rmse = np.nan
            
            metrics = {
                'Mean Squared Error': mse,
                'Root Mean Squared Error': rmse,
                'Mean Absolute Error': mae
            }
            
            # Create forecast DataFrame
            forecast_dates = test[date_col].values[:len(forecast)] if len(test) > 0 else pd.date_range(
                start=train[date_col].iloc[-1], 
                periods=forecast_periods+1, 
                closed='right'
            )
            
            forecast_df = pd.DataFrame({
                'Date': forecast_dates[:len(forecast)],
                'Forecast': forecast,
                'Actual': test[value_col].values[:len(forecast)] if len(test) > 0 else None
            })
            
            return {
                'model': model_fit,
                'model_name': model_name,
                'metrics': metrics,
                'forecast': forecast_df,
                'date_col': date_col,
                'value_col': value_col
            }
            
        except Exception as e:
            print(f"Exponential Smoothing error: {e}")
            # Use simple moving average as last resort
            model_name = "Moving Average"
    
    if model_name == "Prophet" and PROPHET_AVAILABLE:
        try:
            # Prepare data for Prophet
            prophet_df = pd.DataFrame({
                'ds': df[date_col],
                'y': df[value_col]
            })
            
            # Fit Prophet model
            model = Prophet()
            model.fit(prophet_df)
            
            # Make forecast
            future = model.make_future_dataframe(periods=forecast_periods)
            forecast = model.predict(future)
            
            # Evaluate
            if len(test) > 0:
                forecast_subset = forecast.iloc[train_size:train_size+len(test)]
                mse = mean_squared_error(test[value_col].values, forecast_subset['yhat'].values)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(test[value_col].values, forecast_subset['yhat'].values)
            else:
                mse = mae = rmse = np.nan
            
            metrics = {
                'Mean Squared Error': mse,
                'Root Mean Squared Error': rmse,
                'Mean Absolute Error': mae
            }
            
            # Create forecast DataFrame
            forecast_df = pd.DataFrame({
                'Date': forecast['ds'].iloc[-forecast_periods:],
                'Forecast': forecast['yhat'].iloc[-forecast_periods:],
                'Actual': test[value_col].values if len(test) > 0 else None
            })
            
            return {
                'model': model,
                'model_name': model_name,
                'metrics': metrics,
                'forecast': forecast_df,
                'date_col': date_col,
                'value_col': value_col
            }
            
        except Exception as e:
            print(f"Prophet error: {e}")
            # Fallback to Moving Average
            model_name = "Moving Average"
    
    # Simple Moving Average (fallback method)
    if model_name == "Moving Average":
        # Calculate moving average
        window = min(len(train) // 4, 7)  # Use at most 7 day window
        if window < 1:
            window = 1
            
        ma = train[value_col].rolling(window=window).mean()
        
        # Last valid MA value
        last_ma = ma.dropna().iloc[-1]
        
        # Create forecast (flat line at last MA value)
        forecast = np.array([last_ma] * forecast_periods)
        
        # Evaluate
        if len(test) > 0:
            mse = mean_squared_error(test[value_col].values[:len(forecast)], forecast)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(test[value_col].values[:len(forecast)], forecast)
        else:
            mse = mae = rmse = np.nan
        
        metrics = {
            'Mean Squared Error': mse,
            'Root Mean Squared Error': rmse,
            'Mean Absolute Error': mae
        }
        
        # Create forecast DataFrame
        forecast_dates = test[date_col].values[:len(forecast)] if len(test) > 0 else pd.date_range(
            start=train[date_col].iloc[-1], 
            periods=forecast_periods+1, 
            closed='right'
        )
        
        forecast_df = pd.DataFrame({
            'Date': forecast_dates[:len(forecast)],
            'Forecast': forecast,
            'Actual': test[value_col].values[:len(forecast)] if len(test) > 0 else None
        })
        
        # Create a simple "model" as a dictionary with the MA parameters
        model_dict = {
            'window': window,
            'last_value': last_ma
        }
        
        return {
            'model': model_dict,
            'model_name': "Moving Average",
            'metrics': metrics,
            'forecast': forecast_df,
            'date_col': date_col,
            'value_col': value_col
        }


def train_model(df: pd.DataFrame, target_col: str, selected_features: List[str], 
              problem_type: str, selected_model: str = "Auto-select best model") -> Dict[str, Any]:
    """
    Train a model based on problem type and selected options.
    
    Args:
        df: DataFrame containing the data
        target_col: Name of the target column
        selected_features: List of feature column names
        problem_type: Type of problem (regression, classification, time_series)
        selected_model: Name of the model to use or "Auto-select best model"
        
    Returns:
        Dictionary with trained model and performance metrics
    """
    try:
        # For time series problems, we use a different approach
        if problem_type == "time_series":
            if selected_model == "Auto-select best model":
                # Try different models and select the best one
                models_to_try = get_model_options(problem_type)
                best_result = None
                best_metric = float('inf')  # Lower is better for MSE
                
                for model_name in models_to_try:
                    try:
                        result = train_time_series_model(df, target_col, model_name)
                        if 'metrics' in result and 'Mean Squared Error' in result['metrics']:
                            mse = result['metrics']['Mean Squared Error']
                            if mse < best_metric:
                                best_metric = mse
                                best_result = result
                    except Exception as e:
                        print(f"Error training {model_name}: {e}")
                        continue
                
                if best_result:
                    return best_result
                else:
                    # If all models fail, fall back to Moving Average
                    return train_time_series_model(df, target_col, "Moving Average")
            else:
                # Train the selected model
                return train_time_series_model(df, target_col, selected_model)
        
        # For regression and classification, we prepare data similarly
        X = df[selected_features]
        y = df[target_col]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create preprocessor
        preprocessor, numerical_features, categorical_features = preprocess_data(X)
        
        if selected_model == "Auto-select best model":
            # Get available models for the problem type
            models_to_try = get_model_options(problem_type)
            
            best_result = None
            best_metric = -float('inf')  # Higher is better for R² and accuracy
            
            for model_name in models_to_try:
                try:
                    if problem_type == "regression":
                        result = train_regression_model(
                            X_train, X_test, y_train, y_test,
                            preprocessor, model_name, numerical_features, categorical_features
                        )
                        # Use R² for model selection
                        if 'metrics' in result and 'R² Score' in result['metrics']:
                            metric = result['metrics']['R² Score']
                            if metric > best_metric:
                                best_metric = metric
                                best_result = result
                    else:  # classification
                        result = train_classification_model(
                            X_train, X_test, y_train, y_test,
                            preprocessor, model_name, numerical_features, categorical_features
                        )
                        # Use accuracy for model selection
                        if 'metrics' in result and 'Accuracy' in result['metrics']:
                            metric = result['metrics']['Accuracy']
                            if metric > best_metric:
                                best_metric = metric
                                best_result = result
                except Exception as e:
                    print(f"Error training {model_name}: {e}")
                    continue
            
            if best_result:
                return best_result
            else:
                # If all models fail, use a default
                if problem_type == "regression":
                    return train_regression_model(
                        X_train, X_test, y_train, y_test,
                        preprocessor, "Linear Regression", numerical_features, categorical_features
                    )
                else:
                    return train_classification_model(
                        X_train, X_test, y_train, y_test,
                        preprocessor, "Logistic Regression", numerical_features, categorical_features
                    )
        else:
            # Train the selected model
            if problem_type == "regression":
                return train_regression_model(
                    X_train, X_test, y_train, y_test,
                    preprocessor, selected_model, numerical_features, categorical_features
                )
            else:  # classification
                return train_classification_model(
                    X_train, X_test, y_train, y_test,
                    preprocessor, selected_model, numerical_features, categorical_features
                )
    
    except Exception as e:
        # Return error information
        print(f"Training error: {str(e)}")
        traceback.print_exc()
        return {
            'error': str(e),
            'traceback': traceback.format_exc()
        }


def make_prediction(model, model_info: Dict[str, Any], input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Make a prediction using the trained model.
    
    Args:
        model: Trained model
        model_info: Dictionary with model metadata
        input_data: Dictionary with input feature values
        
    Returns:
        Dictionary with prediction results
    """
    try:
        problem_type = model_info['problem_type']
        
        if problem_type in ["regression", "classification"]:
            # Convert input data to DataFrame
            input_df = pd.DataFrame([input_data])
            
            # Make prediction
            if problem_type == "regression":
                prediction = model.predict(input_df)[0]
                
                # Try to get prediction interval (if model supports it)
                prediction_range = None
                try:
                    if hasattr(model, 'predict_interval'):
                        lower, upper = model.predict_interval(input_df, alpha=0.05)
                        prediction_range = (float(lower[0]), float(upper[0]))
                except:
                    # Fall back to estimated range based on MSE
                    if 'metrics' in model_info and 'Root Mean Squared Error' in model_info['metrics']:
                        rmse = model_info['metrics']['Root Mean Squared Error']
                        prediction_range = (prediction - 1.96 * rmse, prediction + 1.96 * rmse)
                
                result = {
                    'prediction': float(prediction),
                    'range': prediction_range
                }
                
            else:  # classification
                prediction = model.predict(input_df)[0]
                
                # Get probabilities if available
                probabilities = {}
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(input_df)[0]
                    classes = model.classes_
                    probabilities = {str(c): float(p) for c, p in zip(classes, proba)}
                
                result = {
                    'prediction': str(prediction),
                    'probabilities': probabilities
                }
            
            return result
            
        elif problem_type == "time_series":
            # For time series, we return the forecast directly
            if 'forecast' in model_info:
                return {
                    'forecast': model_info['forecast']
                }
            else:
                return {
                    'error': "No forecast available"
                }
        
        else:
            return {
                'error': f"Unsupported problem type: {problem_type}"
            }
    
    except Exception as e:
        return {
            'error': str(e)
        }


def display_regression_results(results):
    """Display regression model results."""
    st.subheader("Model Performance")
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("R² Score", f"{results['metrics']['R² Score']:.4f}")
    with col2:
        st.metric("RMSE", f"{results['metrics'].get('Root Mean Squared Error', np.sqrt(results['metrics']['Mean Squared Error'])):.4f}")
    with col3:
        st.metric("MAE", f"{results['metrics']['Mean Absolute Error']:.4f}")
    
    # Feature importance
    if 'feature_importance' in results and results['feature_importance'] is not None:
        st.subheader("Feature Importance")
        
        # Get the top 10 features
        feature_importance = results['feature_importance']
        if len(feature_importance) > 10:
            feature_importance = feature_importance.head(10)
        
        fig = px.bar(
            feature_importance,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Top Features',
            height=400
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)


def display_classification_results(results):
    """Display classification model results."""
    st.subheader("Model Performance")
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", f"{results['metrics']['Accuracy']:.4f}")
    with col2:
        if 'F1 Score' in results['metrics']:
            st.metric("F1 Score", f"{results['metrics']['F1 Score']:.4f}")
        elif 'Weighted F1' in results['metrics']:
            st.metric("Weighted F1", f"{results['metrics']['Weighted F1']:.4f}")
    with col3:
        if 'Precision' in results['metrics']:
            st.metric("Precision", f"{results['metrics']['Precision']:.4f}")
        elif 'Macro F1' in results['metrics']:
            st.metric("Macro F1", f"{results['metrics']['Macro F1']:.4f}")
    
    # Confusion Matrix
    if 'confusion_matrix' in results and 'classes' in results:
        st.subheader("Confusion Matrix")
        cm = np.array(results['confusion_matrix'])
        classes = results['classes']
        
        # Create heatmap
        fig = px.imshow(
            cm,
            x=classes,
            y=classes,
            color_continuous_scale='Blues',
            labels=dict(x="Predicted", y="Actual", color="Count"),
            title="Confusion Matrix"
        )
        
        # Add text annotations
        annotations = []
        for i in range(len(classes)):
            for j in range(len(classes)):
                annotations.append({
                    'x': classes[j],
                    'y': classes[i],
                    'text': str(cm[i, j]),
                    'showarrow': False,
                    'font': {'color': 'white' if cm[i, j] > cm.max() / 2 else 'black'}
                })
                
        fig.update_layout(annotations=annotations)
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance
    if 'feature_importance' in results and results['feature_importance'] is not None:
        st.subheader("Feature Importance")
        
        # Get the top 10 features
        feature_importance = results['feature_importance']
        if len(feature_importance) > 10:
            feature_importance = feature_importance.head(10)
        
        fig = px.bar(
            feature_importance,
            x='Importance',
            y='Feature',
            orientation='h',
            title='Top Features',
            height=400
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)


def display_time_series_results(results):
    """Display time series model results."""
    st.subheader("Forecast Results")
    
    # Metrics
    if 'metrics' in results:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("RMSE", f"{results['metrics'].get('Root Mean Squared Error', np.sqrt(results['metrics'].get('Mean Squared Error', 0))):.4f}")
        with col2:
            st.metric("MAE", f"{results['metrics']['Mean Absolute Error']:.4f}")
    
    # Forecast plot
    if 'forecast' in results:
        forecast_df = results['forecast']
        
        # Create the plot
        fig = go.Figure()
        
        # Add actual values if available
        if 'Actual' in forecast_df.columns and not forecast_df['Actual'].isna().all():
            fig.add_trace(go.Scatter(
                x=forecast_df['Date'],
                y=forecast_df['Actual'],
                mode='lines',
                name='Actual',
                line=dict(color='blue')
            ))
        
        # Add forecast values
        fig.add_trace(go.Scatter(
            x=forecast_df['Date'],
            y=forecast_df['Forecast'],
            mode='lines',
            name='Forecast',
            line=dict(color='red', dash='dash')
        ))
        
        # Update layout
        fig.update_layout(
            title='Time Series Forecast',
            xaxis_title='Date',
            yaxis_title='Value',
            legend=dict(x=0, y=1, traceorder='normal'),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show forecast table
        with st.expander("View Forecast Data"):
            st.dataframe(forecast_df)


def prediction_page():
    """Main prediction page UI function."""
    # st.header("🔮 Smart Prediction Engine")
    
    if not st.session_state.get('uploaded_files', []):
        st.warning("⚠️ Please upload data files first in the upload section.")
        return
    
    # 1. Select dataset from uploaded files
    file_options = [f"{file['name']}" for file in st.session_state.uploaded_files]
    selected_file = st.selectbox("📊 Select dataset for prediction", file_options)
    df = next((file['df'] for file in st.session_state.uploaded_files if file['name'] == selected_file), None)
    
    if df is None:
        st.warning("Please select a valid dataset")
        return
    
    # Display dataset preview
    with st.expander("Dataset Preview"):
        st.dataframe(df.head())
        st.write(f"Shape: {df.shape}")
    
    # 2. User selects target column
    col1, col2 = st.columns([3, 1])
    with col1:
        target_col = st.selectbox("📌 Select column to predict (target)", df.columns.tolist())
    
    # 3. Auto-detect problem type
    problem_type = detect_problem_type(df, target_col)
    
    with col2:
        st.info(f"Detected: {problem_type.title()}")

    # 3.5 Feature selection
    remaining_cols = [col for col in df.columns if col != target_col]

    # Add checkbox to select all features except the target
    select_all_features = st.checkbox(
        "Select all input features (exclude target column)",
        value=True,
        help="Check this box to automatically select all columns except the target column."
    )

    if select_all_features:
        selected_features = remaining_cols
    else:
        selected_features = st.multiselect(
            "🔍 Select input features (columns to use for prediction)",
            remaining_cols,
            default=remaining_cols[:min(len(remaining_cols), 5)],
            help="Manually select columns to use as input features."
        )

    if not selected_features:
        st.warning("⚠️ Please select at least one feature for prediction")
        return


    # Model selection - optional for user
    st.subheader("Model Configuration")
    
    model_options = get_model_options(problem_type)
    custom_model = st.checkbox("I want to select a specific model", value=False)
    
    if custom_model:
        selected_model = st.selectbox("Select model", model_options)
    else:
        selected_model = "Auto-select best model"
    
    # 4. Train model button
    if st.button("🚀 Train Model"):
        with st.spinner("Training model... This may take a moment"):
            try:
                # Train the model based on problem type
                results = train_model(df, target_col, selected_features, problem_type, selected_model)
                
                if 'error' in results:
                    st.error(f"Error training model: {results['error']}")
                    if 'traceback' in results:
                        with st.expander("View error details"):
                            st.code(results['traceback'])
                    return
                
                # Store model in session state
                st.session_state['current_model'] = results['model']
                st.session_state['model_info'] = {
                    'features': selected_features,
                    'target': target_col,
                    'problem_type': problem_type,
                    'model_name': results['model_name'],
                    'results': results
                }
                
                # Display model performance
                st.success(f"✅ Successfully trained {results['model_name']}!")
                
                # Display appropriate results based on problem type
                if problem_type == "regression":
                    display_regression_results(results)
                elif problem_type == "classification":
                    display_classification_results(results)
                else:  # time_series
                    display_time_series_results(results)
                    
            except Exception as e:
                st.error(f"Error during model training: {str(e)}")
                st.error(traceback.format_exc())
    
    # 5. Make predictions with trained model
    if 'current_model' in st.session_state and 'model_info' in st.session_state:
        st.header("Make Predictions")
        st.write(f"Using model: {st.session_state['model_info']['model_name']}")
        
        # For time series, we display the forecast already generated
        if st.session_state['model_info']['problem_type'] == 'time_series':
            if 'results' in st.session_state['model_info']:
                display_time_series_results(st.session_state['model_info']['results'])
            return
        
        # Create input fields for each feature
        st.subheader("Enter values for prediction")
        input_data = {}

        for feature in st.session_state['model_info']['features']:
            # Determine appropriate input widget based on data type
            if pd.api.types.is_numeric_dtype(df[feature]):
                # Numeric input with slider
                feature_min = float(df[feature].min())
                feature_max = float(df[feature].max())
                feature_mean = float(df[feature].mean())

                # Adjust step size based on the range
                range_size = feature_max - feature_min
                step = max(range_size / 100, 0.01)  # Ensure step is never zero

                input_data[feature] = st.slider(
                    f"{feature}",
                    min_value=feature_min,
                    max_value=feature_max,
                    value=feature_mean,
                    step=step,
                    format="%.2f" if range_size < 100 else "%.0f"
                )
            elif df[feature].nunique() < 10:
                # For categorical with few values
                options = df[feature].dropna().unique().tolist()
                input_data[feature] = st.selectbox(f"{feature}", options)
            else:
                # Text input for other cases
                input_data[feature] = st.text_input(
                    f"{feature}",
                    value=str(df[feature].mode().iloc[0])
                )

        # Predict button
        if st.button("🔮 Predict"):
            with st.spinner("Generating prediction..."):
                try:
                    prediction_result = make_prediction(
                        st.session_state['current_model'],
                        st.session_state['model_info'],
                        input_data
                    )
                    
                    if 'error' in prediction_result:
                        st.error(f"Error making prediction: {prediction_result['error']}")
                        return
                    
                    # Display prediction result
                    st.subheader("Prediction Result")
                    
                    if st.session_state['model_info']['problem_type'] == 'regression':
                        # Create a gauge chart for regression results
                        target = st.session_state['model_info']['target']
                        predicted_value = prediction_result['prediction']
                        
                        # Find min and max of target in dataset for gauge scale
                        min_val = float(df[target].min())
                        max_val = float(df[target].max())
                        
                        # Format the prediction value
                        if abs(predicted_value) < 0.01:
                            formatted_value = f"{predicted_value:.6f}"
                        elif abs(predicted_value) < 1:
                            formatted_value = f"{predicted_value:.4f}"
                        elif abs(predicted_value) < 10:
                            formatted_value = f"{predicted_value:.3f}"
                        else:
                            formatted_value = f"{predicted_value:.2f}"
                        
                        # Display the prediction
                        st.markdown(f"### Predicted {target}: **{formatted_value}**")
                        
                        # Create a gauge or indicator
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=predicted_value,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': f"Predicted {target}"},
                            gauge={
                                'axis': {'range': [min_val, max_val]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [min_val, (min_val + max_val) / 2], 'color': "lightgray"},
                                    {'range': [(min_val + max_val) / 2, max_val], 'color': "gray"}
                                ],
                            }
                        ))
                        fig.update_layout(height=300)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Show prediction range if available
                        if 'range' in prediction_result and prediction_result['range'] is not None:
                            low, high = prediction_result['range']
                            st.write(f"Prediction range: {low:.4f} to {high:.4f} (95% confidence)")
                    
                    elif st.session_state['model_info']['problem_type'] == 'classification':
                        target = st.session_state['model_info']['target']
                        predicted_class = prediction_result['prediction']
                        
                        # Display the prediction with an icon
                        st.markdown(f"### Predicted {target}: **{predicted_class}**")
                        
                        # Show probabilities if available
                        if 'probabilities' in prediction_result and prediction_result['probabilities']:
                            probs = prediction_result['probabilities']
                            
                            # Convert to DataFrame for chart
                            probs_df = pd.DataFrame({
                                'Class': list(probs.keys()),
                                'Probability': list(probs.values())
                            }).sort_values('Probability', ascending=False)
                            
                            # Create bar chart
                            fig = px.bar(
                                probs_df,
                                x='Class',
                                y='Probability',
                                title='Prediction Probabilities',
                                color='Probability',
                                color_continuous_scale='Blues'
                            )
                            fig.update_traces(texttemplate='%{y:.2f}', textposition='outside')
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Show in table format too
                            st.write("Probabilities:")
                            for cls, prob in probs.items():
                                st.write(f"- {cls}: {prob:.4f}")
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
                    st.error(traceback.format_exc())


def add_to_app(app_file_path="app.py"):
    """
    Utility function to import this predictor into the main app.py
    
    Args:
        app_file_path: Path to the app.py file
    """
    import_statement = "from predictor import prediction_page"
    
    try:
        with open(app_file_path, 'r') as file:
            content = file.read()
        
        if import_statement not in content:
            # Add import at the top of the file
            with open(app_file_path, 'w') as file:
                lines = content.split('\n')
                # Find the right place for the import
                import_idx = 0
                for i, line in enumerate(lines):
                    if line.startswith('import') or line.startswith('from'):
                        import_idx = i + 1
                
                lines.insert(import_idx, import_statement)
                file.write('\n'.join(lines))
            
            print(f"Added import to {app_file_path}")
        else:
            print(f"Import already exists in {app_file_path}")
            
    except Exception as e:
        print(f"Error adding import to app.py: {str(e)}")


if __name__ == "__main__":
    # This allows testing the predictor module independently
    import streamlit as st
    
    st.set_page_config(page_title="Prediction Engine", page_icon="🔮", layout="wide")
    st.title("🔮 Prediction Engine")
    
    # Create a sample dataframe for testing
    if 'test_df' not in st.session_state:
        # Generate sample data
        import numpy as np
        
        np.random.seed(42)
        n_samples = 1000
        
        X1 = np.random.normal(0, 1, n_samples)
        X2 = np.random.normal(0, 1, n_samples)
        X3 = np.random.choice(['A', 'B', 'C'], n_samples)
        X4 = np.random.choice(['Red', 'Blue', 'Green', 'Yellow'], n_samples)
        
        # Regression target
        y_reg = 2*X1 + 3*X2 + np.random.normal(0, 0.5, n_samples)
        
        # Classification target
        y_cls = (y_reg > 0).astype(str)
        y_cls[y_reg > 2] = 'High'
        y_cls[y_reg < -2] = 'Low'
        y_cls[(y_reg >= -2) & (y_reg <= 2)] = 'Medium'
        
        # Create DataFrame
        df = pd.DataFrame({
            'feature1': X1,
            'feature2': X2,
            'category1': X3,
            'category2': X4,
            'regression_target': y_reg,
            'classification_target': y_cls
        })
        
        # Add date column for time series
        dates = pd.date_range(start='2020-01-01', periods=n_samples)
        df['date'] = dates
        
        # Create time series target (with trend and noise)
        ts_target = np.arange(n_samples) * 0.1 + np.sin(np.arange(n_samples)/50) * 10 + np.random.normal(0, 1, n_samples)
        df['time_series_target'] = ts_target
        
        st.session_state.test_df = df
        st.session_state.uploaded_files = [{
            'name': 'sample_data_2.csv',
            'df': df,
            'table_name': 'sample_data_2'
        }]
    
    # Run the prediction page
    prediction_page()
