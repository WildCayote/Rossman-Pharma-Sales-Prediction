import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import joblib
from datetime import datetime

class SalesModel:
    """
    Preprocess data and train a RandomForestRegressor.

    Attributes
    ----------
        pipeline : sklearn Pipeline
            Pipeline for preprocessing and training.
        X_train : pd.DataFrame
            Training features.
        X_test : pd.DataFrame
            Testing features.
        y_train : pd.Series
            Training targets.
        y_test : pd.Series
            Testing targets.
    """

    def __init__(self):
        """Initialize with a RandomForestRegressor pipeline."""
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', RandomForestRegressor(
                n_estimators=100,
                max_depth=64,
                min_samples_split=10,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=42
            ))
        ])
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def preprocess_data(self, data, target, test_size=0.2, random_state=42):
        """
            Split and scale data.

            Parameters
            ----------
            data : pd.DataFrame
                Dataset with features and target.
            target : str
                Target column name.
            test_size : float
                Test set proportion (default 0.2).
            random_state : int
                Seed for reproducibility (default 42).
        """
        X = data.drop(columns=[target])
        y = data[target]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
    def train_model(self):
        """Train the model."""
        self.pipeline.fit(self.X_train, self.y_train)

    def evaluate_model(self):
        """Evaluate the model and print RMSE and RMSLE."""
        y_pred = self.pipeline.predict(self.X_test)
        rmse = np.sqrt(mean_squared_error(self.y_test, y_pred))
        rmsle = self.rmsle(self.y_test, y_pred)
        print(f"Model RMSE: {rmse:.2f}")
        print(f"Model RMSLE: {rmsle:.4f}")
    
    def rmsle(self, y_true, y_pred):
        """Calculate RMSLE."""
        return np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_true))**2))

    def tune_model(self, params):
        """
            Tune model with GridSearchCV.

            Parameters
            ----------
            params : dict
                GridSearchCV parameters.

            Returns
            -------
            dict
                Best parameters.
        """
        grid_search = GridSearchCV(self.pipeline, params, cv=5, scoring='neg_root_mean_squared_error', n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)
        self.pipeline = grid_search.best_estimator_
        return grid_search.best_params_

    def save_model(self):
        """Save the model to a file."""
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        filename = f"../src/model/sales_model_{timestamp}.pkl"
        joblib.dump(self.pipeline, filename)
        print(f"Model saved as {filename}")

    def load_model(self, filepath):
        """
            Load a model from a file.

            Parameters
            ----------
            filepath : str
                Model file path.
        """
        self.pipeline = joblib.load(filepath)

    def feature_importance(self):
        """Get feature importances."""
        importances = self.pipeline.named_steps['model'].feature_importances_
        features = self.X_train.columns
        return pd.Series(importances, index=features).sort_values(ascending=False)

    def plot_actual_vs_predicted(self):
        """Plot actual vs predicted values."""
        y_pred = self.pipeline.predict(self.X_test)
        plt.figure(figsize=(12, 6))
        sns.set_palette("Set2")
        sns.scatterplot(x=self.y_test, y=y_pred, alpha=0.6, s=100, edgecolor='w', linewidth=0.5)
        plt.plot([min(self.y_test), max(self.y_test)], 
                 [min(self.y_test), max(self.y_test)], 
                 color='darkorange', linestyle='--', linewidth=2, label='Ideal Prediction')
        plt.title('Actual vs Predicted Sales', fontsize=18)
        plt.xlabel('Actual Sales', fontsize=14)
        plt.ylabel('Predicted Sales', fontsize=14)
        plt.grid(visible=True, linestyle='--', alpha=0.7)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=12)
        plt.gca().set_facecolor('lightgrey')
        plt.tight_layout()
        plt.show()

    def make_predictions(self, test_data):
        """
            Predict sales for test data.

            Parameters
            ----------
            test_data : pd.DataFrame
                Test dataset.

            Returns
            -------
            np.array
                Predicted sales.
        """
        return self.pipeline.predict(test_data)

    def create_submission_file(self, test_data, filepath):
        """
            Create a Kaggle submission file.

            Parameters
            ----------
            test_data : pd.DataFrame
                Test dataset.
            filepath : str
                Submission file path.
        """
        predictions = self.make_predictions(test_data)
        submission_df = pd.DataFrame({
            'Id': test_data.reset_index().Id,
            'Sales': predictions
        })
        submission_df.to_csv(filepath, index=False)
        print(f"Submission file saved as {filepath}")
