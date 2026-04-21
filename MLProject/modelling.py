import os
import shutil
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def main():

    # ==========================================
    # 1. LOAD DATA
    # ==========================================
    print("Memuat dataset dari folder MLProject...")
    df = pd.read_csv("telco_churn_preprocessing/telco_churn_clean.csv")
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ==========================================
    # 2. TRAINING DENGAN AUTOLOG & SIMPAN MODEL
    # ==========================================
    print("Melatih ulang model untuk Production dengan Autolog...")
    
    # Autolog akan otomatis menempel pada "Run" yang dibuat oleh GitHub Actions
    mlflow.sklearn.autolog()

    # KITA TIDAK LAGI MENGGUNAKAN block `with mlflow.start_run():`
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf.fit(X_train, y_train)

    # Direktori tempat model akan disimpan untuk Docker
    model_path = "model_dir"
    
    # Hapus folder jika sebelumnya sudah ada agar tidak error
    if os.path.exists(model_path):
        shutil.rmtree(model_path)
        
    # Simpan model untuk dibungkus menjadi Docker
    mlflow.sklearn.save_model(rf, model_path)
    print(f"Model berhasil disimpan di folder: {model_path}")

if __name__ == "__main__":
    main()