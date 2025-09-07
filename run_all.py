import subprocess
import time

def run_command(command, description):
    print(f"\n>> Running: {description}")
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"❌ Error while running: {description}")
        exit(result.returncode)
    print(f"✅ Finished: {description}")
    time.sleep(1)  # Add 1-second delay after each step

# Step 1: Run model_training.py
run_command("python model_Training.py", "model_Training.py")

# Step 2: Run app.py
run_command("python app.py", "app.py")

# Step 3: Run Streamlit
run_command("streamlit run app.py", "Streamlit app")
