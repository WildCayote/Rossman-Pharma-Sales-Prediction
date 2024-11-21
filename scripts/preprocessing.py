import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

class DataPreprocessor:
    def __init__(self, train_filepath: str, test_filepath: str, test_id_filepath: str):
        """
        Initialize the DataPreprocessor with the file paths to the training and testing datasets.
        
        Parameters
        ----------
        train_filepath : str
            File path to the training dataset.
        test_filepath : str
            File path to the testing dataset.
        test_id_filepath : str
            File path to the CSV file containing test IDs.
        """
        dtype_dict = {
            'Store': int,
            'Sales': float,
            'Open': float,
            'StateHoliday': str,
            'SchoolHoliday': float,
            'Promo': float
        }
        
        self.train_df = pd.read_csv(train_filepath, dtype=dtype_dict, low_memory=False)
        self.test_df = pd.read_csv(test_filepath, dtype=dtype_dict, low_memory=False)
        self.test_id_df = pd.read_csv(test_id_filepath, dtype={'Id': int}, low_memory=False)
        self.test_df['Id'] = self.test_id_df['Id']
        
        self.train_df = self.train_df[self.train_df['Open'] == 1]
        self.train_copy = self.train_df.copy()
        self.test_copy = self.test_df.copy()
        self.scaler = StandardScaler()

    def clean_data(self):
        """Clean the data by resetting indexes and removing unnecessary columns."""
        self.train_copy.reset_index(drop=True, inplace=True)
        self.test_copy.reset_index(drop=True, inplace=True)
        self.train_copy.drop(columns=['Customers'], errors='ignore', inplace=True)
        self.test_copy.drop(columns=['Id'], errors='ignore', inplace=True)
        self.handle_missing_values()

    def handle_missing_values(self):
        """Manage missing values consistently across training and testing datasets."""
        combined_df = pd.concat([self.train_copy, self.test_copy], axis=0, keys=['train', 'test'])
        combined_df.fillna({'Open': combined_df['Open'].mode()[0]}, inplace=True)
        self.train_copy = combined_df.xs('train')
        self.test_copy = combined_df.xs('test')

    def extract_datetime_features(self):
        """Extract features from datetime data, including weekdays, months, and holiday-related variables."""
        for df in [self.train_copy, self.test_copy]:
            df['Date'] = pd.to_datetime(df['Date'])
            df['DayOfWeek'] = df['Date'].dt.dayofweek  # 0=Monday, 6=Sunday
            df['IsWeekday'] = df['DayOfWeek'].apply(lambda x: 1 if x <= 5 else 0)
            df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)
            df['Month'] = df['Date'].dt.month
            df['IsBeginningOfMonth'] = (df['Date'].dt.day <= 7).astype(int)
            df['IsMidMonth'] = ((df['Date'].dt.day > 7) & (df['Date'].dt.day <= 21)).astype(int)
            df['IsEndOfMonth'] = (df['Date'].dt.day > 21).astype(int)
            # df.drop(columns=['Date', 'Dataset', 'CompetitionOpenSinceMonth', 'CompetitionOpenSinceYear'], inplace=True)

    def feature_engineering(self):
        """Create new features based on existing data, such as holiday flags and promotional duration."""
        for df in [self.train_copy, self.test_copy]:
            df['IsHoliday'] = df.apply(lambda x: 1 if (x['StateHoliday'] != '0' or x['SchoolHoliday'] == 1) else 0, axis=1)
            df['PromoDuration'] = df.groupby('Store')['Promo'].cumsum()

    def encode_categorical_data(self):
        """Convert categorical variables into numeric form using label encoding."""
        label_columns = ['StateHoliday']
        label_encoder = LabelEncoder()

        for column in label_columns:
            # Combine train and test data for consistent encoding
            combined_data = pd.concat([self.train_copy[column], self.test_copy[column]], axis=0).astype(str)
            label_encoder.fit(combined_data)

            # Apply the encoding
            self.train_copy[column] = label_encoder.transform(self.train_copy[column].astype(str))
            self.test_copy[column] = label_encoder.transform(self.test_copy[column].astype(str))

    def preprocess(self):
        """Run the entire data preprocessing pipeline."""
        print("Cleaning data...")
        self.clean_data()
        print("Extracting datetime features...")
        self.extract_datetime_features()
        print("Performing feature engineering...")
        self.feature_engineering()
        print("Encoding categorical data...")
        self.encode_categorical_data()

        self.test_copy.drop(columns=['Sales'], errors='ignore', inplace=True)
        self.test_copy.reset_index(drop=True, inplace=True)
        self.test_copy.set_index(self.test_df['Id'], inplace=True)
        self.train_copy.reset_index(drop=True, inplace=True)
        self.train_copy.set_index(self.train_df['Date'], inplace=True)
        
        print("Preprocessing complete.")
        return self.train_copy, self.test_copy

    def save_data(self, train_output_filepath='../data/train_processed.csv', test_output_filepath='../data/test_processed.csv'):
        """Save the processed training and testing datasets to CSV files."""
        self.train_copy.to_csv(train_output_filepath, index=True)
        self.test_copy.to_csv(test_output_filepath, index=True)
        print(f"Processed data saved to {train_output_filepath} and {test_output_filepath}.")
