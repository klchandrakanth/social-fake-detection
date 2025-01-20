import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess_data():
    # Load dataset
    dataset_path = "data/social_media_fake_accounts.csv"
    df = pd.read_csv(dataset_path)

    # Encode categorical features
    categorical_columns = ['account_type', 'profile_picture', 'status_count']
    for col in categorical_columns:
        df[col] = LabelEncoder().fit_transform(df[col])

    # Feature-target split
    X = df.drop(columns=['is_fake'])
    y = df['is_fake']

    # Data normalization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

