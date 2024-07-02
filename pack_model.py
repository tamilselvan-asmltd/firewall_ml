import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import time
import random
import numpy as np
import os
import joblib

print("\n\n")
print("[    *** Adaptive ML Firewall with Dynamic Rule Generation ***   ]")
print("\n\n")

def generate_random_traffic(port_weights, ip_weights, traffic_type_weights, action_weights):
    """Generate random traffic data with weighted probabilities."""
    ports = [random.randint(1000, 9999) for _ in range(50)]
    ips = [f"192.168.1.{i}" for i in range(1, 9)]
    traffic_types = ["tcp", "udp", "http", "https", "ftp", "ssh"]
    actions = ["allow", "deny"]

    data = {
        "port": np.random.choice(ports, p=port_weights),
        "ip": np.random.choice(ips, p=ip_weights),
        "traffic_type": np.random.choice(traffic_types, p=traffic_type_weights),
        "action": np.random.choice(actions, p=action_weights)
    }
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

def train_model(data):
    """Train a random forest model with hyperparameter tuning on the given data."""
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
            print("Model meets the accuracy requirement.")
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

# Flag to track if rules have been uploaded
rules_uploaded = False

while not rules_uploaded:
    # Generate random input data
    input_data = [generate_random_traffic(port_weights, ip_weights, traffic_type_weights, action_weights) for _ in range(50)]  # Generate 50 random entries
    traffic_data = load_data(input_data)

    # Load user feedback
    user_feedback_data = load_user_feedback()

    # Combine the traffic data with user feedback
    data = combine_data_with_feedback(traffic_data, user_feedback_data)

    # Train model and generate rules, retry until accuracy requirement is met
    model, accuracy_requirement_met = train_model(data)

    while not accuracy_requirement_met:
        time.sleep(5)  # Wait for a while before retrying

        # Generate new input data and combine with feedback for retraining
        input_data = [generate_random_traffic(port_weights, ip_weights, traffic_type_weights, action_weights) for _ in range(50)]
        traffic_data = load_data(input_data)
        data = combine_data_with_feedback(traffic_data, user_feedback_data)

        # Retry training
        model, accuracy_requirement_met = train_model(data)

    # Generate rules
    rules = generate_rules(model, data)

    # Update or save rules based on accuracy requirement
    if accuracy_requirement_met:
        update_firewall_rules(rules)
    else:
        save_rules_to_csv(rules)

    # Apply the generated rules
    apply_firewall_rules(rules)

    # Log user feedback
    log_user_feedback(user_feedback_data)

    # Mark rules as uploaded
    rules_uploaded = True

# Serialize the trained model for production use
joblib.dump(model, 'trained_firewall_model.pkl')
print("Trained model saved as 'trained_firewall_model.pkl'")
