from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
import pickle
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
warnings.filterwarnings("ignore")
import mlflow
import mlflow.sklearn

from featuring import create_preprocessing_pipeline


def train_model(df, target_column):

    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    preprocessor = create_preprocessing_pipeline(df, target_column)

    models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": XGBClassifier(eval_metric='logloss', use_label_encoder=False, random_state=42),
    "LightGBM": LGBMClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "CatBoost": CatBoostClassifier(verbose=0, random_state=42)
    }

    best_model = None
    best_score = 0
    best_model_name = None

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            pipeline = ImbPipeline([
                ("preprocess", preprocessor),
                ("smote", SMOTE(random_state=42)),  
                ("model", model)
            ])
    
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
    
            acc = accuracy_score(y_test, y_pred)

            mlflow.log_param("model_name", name)
            mlflow.log_metric("accuracy", acc)
            mlflow.sklearn.log_model(pipeline, artifact_path="model")
    
            if acc > best_score:
                best_score = acc
                best_model = pipeline
                best_model_name = name

    print(f"\nThe best model: {best_model_name} (Accuracy: {best_score:.4f})")

    # Save the best model
    with open("models/best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)
    
    with mlflow.start_run(run_name="Best_Model"):
        mlflow.log_param("best_modelm", best_model_name)
        mlflow.log_metric("best_accuracy", best_score)
        mlflow.sklearn.log_model(best_model, artifact_path="best_model")

    return best_model