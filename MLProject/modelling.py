import os
import shutil
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def main():
    # ==========================================
    # 1. SETUP DAGSHUB 
    # ==========================================
    REPO_OWNER = "rfahur11"
    REPO_NAME = "Telco-Churn-MLOps"
    dagshub.init(repo_owner=REPO_OWNER, repo_name=REPO_NAME, mlflow=True)
    mlflow.set_experiment("Telco_Churn_Experiment")

    # ==========================================
    # 2. LOAD DATA
    # ==========================================
    print("Memuat dataset dari folder MLProject...")
    df = pd.read_csv("telco_churn_preprocessing/telco_churn_clean.csv")
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ==========================================
    # 3. TRAINING DENGAN AUTOLOG & SIMPAN MODEL
    # ==========================================
    print("Melatih ulang model untuk Production dengan Autolog...")
    
    # Aktifkan Autolog sesuai permintaan reviewer
    mlflow.sklearn.autolog()

    # Gunakan block start_run agar rapi di DagsHub
    with mlflow.start_run(run_name="CI_CD_Retraining"):
        # Tambahkan max_depth agar parameternya sama dengan baseline Kriteria 2
        rf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
        rf.fit(X_train, y_train)

        # Direktori tempat model akan disimpan untuk Docker
        model_path = "model_dir"
        
        # Hapus folder jika sebelumnya sudah ada agar tidak error
        if os.path.exists(model_path):
            shutil.rmtree(model_path)
            
        #  UNTUK CI/CD 
        mlflow.sklearn.save_model(rf, model_path)
        print(f"Model berhasil disimpan di folder: {model_path}")

if __name__ == "__main__":
    main()