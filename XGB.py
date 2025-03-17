#!/usr/bin/env python3

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import DMatrix, train, callback
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Define path (adjust if needed for server)
path = "C:/Users/yasmin_s1/sanjida_projects/projects/mater_thesis/ML/Datasets/Dataset_7months/"

def main():
    # Step 1: Load Processed Dataset
    df = pd.read_csv(path + "processed_dataset.csv")
    print(f"Loaded rows: {len(df)}, columns: {len(df.columns)}")

    # Step 2: Filter Classes with <2 Instances (Initial Clean)
    class_counts = df['maintenance_comments_encoded'].value_counts()
    rare_classes = class_counts[class_counts < 2].index
    df = df[~df['maintenance_comments_encoded'].isin(rare_classes)]
    print(f"Rows after initial <2 filter: {len(df)}, unique labels: {df['maintenance_comments_encoded'].nunique()}")

    # Step 3: Define Features and Target
    features = (['TopologyId_encoded', 'response_time_minutes', 'msg_hour', 'msg_dayofweek', 
                 'msg_weekend', 'event_hour', 'event_dayofweek', 'event_weekend', 
                 'msg_shift_encoded', 'event_shift_encoded'] + 
                [f'msg_emb_{i}' for i in range(384)] + 
                [f'op_emb_{i}' for i in range(384)])
    X = df[features]
    y = df['maintenance_comments_encoded']
    print(f"Features shape: {X.shape}, unique labels: {y.nunique()}")

    # Step 4: Train-Test Split with Dynamic Filter
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    print(f"After first split - Train: {X_train.shape}, Temp: {X_temp.shape}")

    temp_df = pd.DataFrame({'y_temp': y_temp}, index=X_temp.index)
    temp_counts = temp_df['y_temp'].value_counts()
    temp_rare = temp_counts[temp_counts < 2].index
    keep_indices = ~temp_df['y_temp'].isin(temp_rare)
    X_temp = X_temp[keep_indices]
    y_temp = y_temp[keep_indices]
    print(f"After filtering y_temp - Temp rows: {len(X_temp)}, unique labels: {y_temp.nunique()}")

    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    print(f"Final split - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    # Step 5: Re-encode Labels After Filtering
    final_labels = np.concatenate([y_train, y_val, y_test])
    le_final = LabelEncoder()
    final_labels_encoded = le_final.fit_transform(final_labels)
    train_size = len(y_train)
    val_size = len(y_val)
    y_train = final_labels_encoded[:train_size]
    y_val = final_labels_encoded[train_size:train_size + val_size]
    y_test = final_labels_encoded[train_size + val_size:]
    print(f"Re-encoded y_train min/max: {y_train.min()}/{y_train.max()}")
    print(f"Re-encoded y_val min/max: {y_val.min()}/{y_val.max()}")
    print(f"Re-encoded y_test min/max: {y_test.min()}/{y_test.max()}")

    # Step 6: Recreate LabelEncoder for Reporting
    le_target = LabelEncoder()
    le_target.fit(df['maintenance_comments'])

    # Step 7: Convert to DMatrix and Train
    dtrain = DMatrix(X_train, label=y_train)
    dval = DMatrix(X_val, label=y_val)
    dtest = DMatrix(X_test, label=y_test)

    params = {
        'max_depth': 6,
        'learning_rate': 0.1,
        'objective': 'multi:softmax',
        'num_class': len(np.unique(final_labels)),
        'random_state': 42,
        'tree_method': 'hist',
        'eval_metric': 'mlogloss'
    }

    early_stopping = callback.EarlyStopping(
        rounds=10,
        metric_name='mlogloss',
        data_name='val'
    )

    evals = [(dtrain, 'train'), (dval, 'val')]
    model = train(
        params,
        dtrain,
        num_boost_round=100,
        evals=evals,
        callbacks=[early_stopping],
        verbose_eval=True
    )
    print(f"Best iteration: {model.best_iteration}, Score: {model.best_score}")

    # Step 8: Evaluate Model and Feature Importance
    y_pred_val = model.predict(dval).astype(int)
    y_pred_test = model.predict(dtest).astype(int)

    y_val_orig = le_final.inverse_transform(y_val)
    y_test_orig = le_final.inverse_transform(y_test)
    y_pred_val_orig = le_final.inverse_transform(y_pred_val)
    y_pred_test_orig = le_final.inverse_transform(y_pred_test)

    print("\nValidation Results:")
    print(f"Accuracy: {accuracy_score(y_val_orig, y_pred_val_orig):.4f}")
    print(classification_report(y_val_orig, y_pred_val_orig, target_names=le_target.classes_[~np.isin(np.arange(len(le_target.classes_)), rare_classes)], output_dict=False))

    print("\nTest Results:")
    print(f"Accuracy: {accuracy_score(y_test_orig, y_pred_test_orig):.4f}")
    print(classification_report(y_test_orig, y_pred_test_orig, target_names=le_target.classes_[~np.isin(np.arange(len(le_target.classes_)), rare_classes)], output_dict=False))

    importances = model.get_score(importance_type='weight')
    feature_importance_df = pd.DataFrame({
        'Feature': [features[int(k[1:])] for k in importances.keys()],
        'Importance': list(importances.values())
    })
    print("\nFeature Importance (Top 10):")
    print(feature_importance_df.sort_values(by='Importance', ascending=False).head(10))

    # Step 9: Save Model and LabelEncoders with Confirmation
    model_file = path + "xgboost_model.joblib"
    joblib.dump(model, model_file)
    joblib.dump(le_target, path + "label_encoder_target.joblib")
    joblib.dump(le_final, path + "label_encoder_final.joblib")

    # Confirm files exist
    for file in [model_file, path + "label_encoder_target.joblib", path + "label_encoder_final.joblib"]:
        if os.path.exists(file):
            print(f"Saved: {file} (Size: {os.path.getsize(file) / 1024:.2f} KB)")
        else:
            print(f"Error: {file} not saved!")

if __name__ == "__main__":
    main()