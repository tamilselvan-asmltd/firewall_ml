from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import time
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

def generate_random_traffic(port_weights, ip_weights, traffic_type_weights, action_weights):
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

    return data

def load_data(input_data):
    """Load data from a list of dictionaries."""
    return pd.DataFrame(input_data)

def load_user_feedback(filename="user_feedback.csv"):
    """Load user feedback from a CSV file."""
    if os.path.isfile(filename):
        return pd.read_csv(filename).to_dict(orient='records')
    else:
        print(f"Feedback file {filename} does not exist.")
        return []

def combine_data_with_feedback(data, feedback_data):
    """Combine the main data with user feedback, prioritizing feedback."""
    if feedback_data:
        feedback_df = pd.DataFrame(feedback_data)
        combined = pd.concat([data, feedback_df]).drop_duplicates(subset=['port', 'ip', 'traffic_type'], keep='last')
        return combined
    else:
        return data

def train_model_and_save(data, model_filename="trained_firewall_model.pkl"):
    """Train a random forest model with hyperparameter tuning on the given data and save it."""
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

        if accuracy >= 0.60:
            print("Model meets the accuracy requirement. Saving the model.")
            joblib.dump(best_rf, model_filename)
            return best_rf, True
        else:
            print("Model does not meet the accuracy requirement. Retrying...")
            return best_rf, False
    else:
        print("No data available for training.")
        return None, False

def generate_rules(model, data):
    """Generate firewall rules based on model predictions and data features."""
    if model and not data.empty:
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
        return rules
    else:
        print("No model or data available to generate rules.")
        return []

def save_rules_to_csv(rules, filename="firewall_rules.csv"):
    """Save firewall rules to a CSV file."""
    df = pd.DataFrame(rules)
    if os.path.isfile(filename):
        df.to_csv(filename, mode='a', header=False, index=False)
    else:
        df.to_csv(filename, index=False)

def update_firewall_rules(rules, filename="firewall_rules.csv"):
    """Update firewall rules in a CSV file."""
    df = pd.DataFrame(rules)
    df.to_csv(filename, index=False)

def apply_firewall_rules(rules, log_filename="firewall_log.csv"):
    """Simulate applying firewall rules and log the actions."""
    print("Applying firewall rules:")
    log_entries = []
    for rule in rules:
        print(f"{rule['action']} traffic on port {rule['port']} from IP {rule['ip']} with type {rule['traffic_type']}")
        log_entries.append(rule)

    log_df = pd.DataFrame(log_entries)
    if os.path.isfile(log_filename):
        log_df.to_csv(log_filename, mode='a', header=False, index=False)
    else:
        log_df.to_csv(log_filename, index=False)

def log_user_feedback(feedback_data, log_filename="user_feedback_log.csv"):
    """Log user feedback to a CSV file."""
    feedback_df = pd.DataFrame(feedback_data)
    if os.path.isfile(log_filename):
        feedback_df.to_csv(log_filename, mode='a', header=False, index=False)
    else:
        feedback_df.to_csv(log_filename, index=False)

# Initial weights for random generation
port_weights = [1/50] * 50
ip_weights = [1/8] * 8
traffic_type_weights = [1/6] * 6
action_weights = [0.5, 0.5]

# Define Airflow tasks

generate_traffic_task = PythonOperator(
    task_id='generate_traffic',
    python_callable=generate_random_traffic,
    op_args=[port_weights, ip_weights, traffic_type_weights, action_weights],
    dag=dag,
)

load_data_task = PythonOperator(
    task_id='load_data',
    python_callable=load_data,
    op_args=[generate_traffic_task.output],
    dag=dag,
)

load_feedback_task = PythonOperator(
    task_id='load_user_feedback',
    python_callable=load_user_feedback,
    dag=dag,
)

combine_data_task = PythonOperator(
    task_id='combine_data_with_feedback',
    python_callable=combine_data_with_feedback,
    op_args=[load_data_task.output, load_feedback_task.output],
    dag=dag,
)

train_model_task = PythonOperator(
    task_id='train_model_and_save',
    python_callable=train_model_and_save,
    op_args=[combine_data_task.output, '/Users/tamilselvans/airflow/dags/trained_firewall_model.pkl'],
    dag=dag,
)

generate_rules_task = PythonOperator(
    task_id='generate_rules',
    python_callable=generate_rules,
    op_args=[train_model_task.output],
    dag=dag,
)

save_rules_task = PythonOperator(
    task_id='save_rules_to_csv',
    python_callable=save_rules_to_csv,
    op_args=[generate_rules_task.output],
    dag=dag,
)

update_rules_task = PythonOperator(
    task_id='update_firewall_rules',
    python_callable=update_firewall_rules,
    op_args=[generate_rules_task.output],
    dag=dag,
)

apply_rules_task = PythonOperator(
    task_id='apply_firewall_rules',
    python_callable=apply_firewall_rules,
    op_args=[generate_rules_task.output],
    dag=dag,
)

log_feedback_task = PythonOperator(
    task_id='log_user_feedback',
    python_callable=log_user_feedback,
    op_args=[load_feedback_task.output],
    dag=dag,
)

# Define task dependencies
generate_traffic_task >> load_data_task
load_data_task >> load_feedback_task
[load_data_task, load_feedback_task] >> combine_data_task
combine_data_task >> train_model_task
train_model_task >> generate_rules_task
generate_rules_task >> [save_rules_task, update_rules_task, apply_rules_task]
load_feedback_task >> log_feedback_task

if __name__ == "__main__":
    dag.cli()
