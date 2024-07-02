from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import random
import numpy as np
import os
import joblib

# Default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 7, 2),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Initialize the DAG
dag = DAG(
    'adaptive_ml_firewall',
    default_args=default_args,
    description='Adaptive ML Firewall with Dynamic Rule Generation',
    schedule_interval=None,  # Set your desired schedule interval
)

# Function definitions

def generate_random_traffic(port_weights, ip_weights, traffic_type_weights, action_weights, **kwargs):
    """Generate random traffic data with weighted probabilities."""
    ports = [random.randint(1000, 9999) for _ in range(50)]
    ips = [f"192.168.1.{i}" for i in range(1, 9)]
    traffic_types = ["tcp", "udp", "http", "https", "ftp", "ssh"]
    actions = ["allow", "deny"]

    data = [{
        "port": np.random.choice(ports, p=port_weights),
        "ip": np.random.choice(ips, p=ip_weights),
        "traffic_type": np.random.choice(traffic_types, p=traffic_type_weights),
        "action": np.random.choice(actions, p=action_weights)
    } for _ in range(50)]

    kwargs['ti'].xcom_push(key='traffic_data', value=data)

def load_data(**kwargs):
    """Load data from a list of dictionaries."""
    ti = kwargs['ti']
    input_data = ti.xcom_pull(key='traffic_data', task_ids='traffic')
    df = pd.DataFrame(input_data)
    ti.xcom_push(key='loaded_data', value=df.to_dict(orient='records'))

def load_user_feedback(filename="user_feedback.csv", **kwargs):
    """Load user feedback from a CSV file."""
    if os.path.isfile(filename):
        feedback_data = pd.read_csv(filename).to_dict(orient='records')
    else:
        print(f"Feedback file {filename} does not exist.")
        feedback_data = []
    kwargs['ti'].xcom_push(key='feedback_data', value=feedback_data)

def combine_data_with_feedback(**kwargs):
    """Combine the main data with user feedback, prioritizing feedback."""
    ti = kwargs['ti']
    data = pd.DataFrame(ti.xcom_pull(key='loaded_data', task_ids='firewall'))
    feedback_data = pd.DataFrame(ti.xcom_pull(key='feedback_data', task_ids='user_feedback'))
    
    if not feedback_data.empty:
        combined = pd.concat([data, feedback_data]).drop_duplicates(subset=['port', 'ip', 'traffic_type'], keep='last')
    else:
        combined = data

    ti.xcom_push(key='combined_data', value=combined.to_dict(orient='records'))

def train_model_and_save(**kwargs):
    """Train a random forest model with hyperparameter tuning on the given data and save it."""
    ti = kwargs['ti']
    data = pd.DataFrame(ti.xcom_pull(key='combined_data', task_ids='firewall_logs_and_user_feedback'))
    
    if not data.empty:
        X = pd.get_dummies(data[['port', 'ip', 'traffic_type']])
        y = data['action']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

        rf = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
        grid_search.fit(X_train, y_train)

        best_rf = grid_search.best_estimator_
        best_rf.fit(X_train, y_train)

        y_pred = best_rf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model accuracy: {accuracy:.2%}")

        model_filename = '/Users/tamilselvans/airflow/dags/trained_firewall_model.pkl'

        if accuracy >= 0.60:
            print("Model meets the accuracy requirement. Saving the model.")
            joblib.dump(best_rf, model_filename)
            ti.xcom_push(key='model_filename', value=model_filename)
        else:
            print("Model does not meet the accuracy requirement. Retrying...")
            ti.xcom_push(key='model_filename', value=None)
    else:
        print("No data available for training.")
        ti.xcom_push(key='model_filename', value=None)

def generate_rules(**kwargs):
    """Generate firewall rules based on model predictions and data features."""
    ti = kwargs['ti']
    model_filename = ti.xcom_pull(key='model_filename', task_ids='adaptive_ML_model')
    data = pd.DataFrame(ti.xcom_pull(key='combined_data', task_ids='firewall_logs_and_user_feedback'))

    if model_filename and os.path.exists(model_filename) and not data.empty:
        model = joblib.load(model_filename)
        feature_df = pd.DataFrame(columns=pd.get_dummies(data[['port', 'ip', 'traffic_type']]).columns)
        rules = []
        for index, row in data.iterrows():
            single_row_df = pd.DataFrame([row]).pipe(pd.get_dummies)
            prepared_features = single_row_df.reindex(columns=feature_df.columns, fill_value=0)
            prediction = model.predict(prepared_features)[0]
            rules.append({
                "action": prediction,
                "port": row['port'],
                "ip": row['ip'],
                "traffic_type": row['traffic_type']
            })
        ti.xcom_push(key='rules', value=rules)
    else:
        print("No model or data available to generate rules.")
        ti.xcom_push(key='rules', value=[])

def save_rules_to_csv(**kwargs):
    """Save firewall rules to a CSV file."""
    ti = kwargs['ti']
    rules = ti.xcom_pull(key='rules', task_ids='ML_Predicted_rule')
    df = pd.DataFrame(rules)
    filename = "firewall_rules.csv"
    df.to_csv(filename, mode='w', header=True, index=False)

def update_firewall_rules(**kwargs):
    """Update firewall rules in a CSV file."""
    ti = kwargs['ti']
    rules = ti.xcom_pull(key='rules', task_ids='ML_Predicted_rule')
    df = pd.DataFrame(rules)
    filename = "firewall_rules.csv"
    if os.path.isfile(filename):
        df.to_csv(filename, mode='a', header=False, index=False)
    else:
        df.to_csv(filename, index=False)

# Initial weights for random generation
port_weights = [1/50] * 50
ip_weights = [1/8] * 8
traffic_type_weights = [1/6] * 6
action_weights = [0.5, 0.5]

# Define Airflow tasks

generate_traffic_task = PythonOperator(
    task_id='traffic',
    python_callable=generate_random_traffic,
    op_args=[port_weights, ip_weights, traffic_type_weights, action_weights],
    provide_context=True,
    dag=dag,
)

load_data_task = PythonOperator(
    task_id='firewall',
    python_callable=load_data,
    provide_context=True,
    dag=dag,
)

load_feedback_task = PythonOperator(
    task_id='user_feedback',
    python_callable=load_user_feedback,
    provide_context=True,
    dag=dag,
)

combine_data_task = PythonOperator(
    task_id='firewall_logs_and_user_feedback',
    python_callable=combine_data_with_feedback,
    provide_context=True,
    dag=dag,
)

train_model_task = PythonOperator(
    task_id='adaptive_ML_model',
    python_callable=train_model_and_save,
    provide_context=True,
    dag=dag,
)

generate_rules_task = PythonOperator(
    task_id='ML_Predicted_rule',
    python_callable=generate_rules,
    provide_context=True,
    dag=dag,
)

save_rules_task = PythonOperator(
    task_id='updated_rule',
    python_callable=save_rules_to_csv,
    provide_context=True,
    dag=dag,
)

update_rules_task = PythonOperator(
    task_id='update_firewall_rules',
    python_callable=update_firewall_rules,
    provide_context=True,
    dag=dag,
)

# Define task dependencies
generate_traffic_task >> load_data_task >> combine_data_task >> train_model_task
load_feedback_task >> combine_data_task
train_model_task >> generate_rules_task >> [save_rules_task, update_rules_task]

if __name__ == "__main__":
    dag.cli()
