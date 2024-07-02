from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime

# Define the function to print date and time
def print_date_time():
    print(f"Current date and time: {datetime.now()}")

# Define default arguments for the DAG
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
}

# Define the DAG
dag = DAG(
    'print_date_time',
    default_args=default_args,
    description='A simple DAG to print date and time',
    schedule_interval='@daily',  # Change as needed
)

# Define the task using PythonOperator
print_date_time_task = PythonOperator(
    task_id='print_date_time',
    python_callable=print_date_time,
    dag=dag,
)

# Set task dependencies (if any)
