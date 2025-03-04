from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import torch

class Preprocessor:
    
    def __init__(self):
        pass
    
    # Preprocessing: Scaling numerical values & one-hot encoding categorical values
    def preprocessing(self, df, numerical_features: list, categorical_features: list):
        
        preprocessor = ColumnTransformer([
            ("num", StandardScaler(), numerical_features),  
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)  
        ])
        
        print(df)

        # Apply transformations
        processed_features = preprocessor.fit_transform(df)
        
        tabular_tensor = torch.tensor(processed_features, dtype=torch.float32)
        return tabular_tensor