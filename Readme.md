# Q-Learning Agent for Taxi-v3 Environment

This project implements a Q-Learning algorithm to solve the Taxi-v3 environment from the Gymnasium library. The agent learns to pick up a passenger from one location and drop them off at a destination in a grid world.

## Prerequisites (Linux)

* **Python 3:** Ensure you have Python 3 installed (preferably Python 3.7 or newer). You can check your Python version by running:
    ```bash
    python3 --version
    ```
* **pip:** The Python package installer. It usually comes with Python 3. You can check by running:
    ```bash
    pip3 --version
    ```
* **venv:** (Recommended for creating virtual environments). This module should be part of your Python 3 installation.

## Setup and Installation

1.  **Clone the Repository (or Download Files):**
    If your code is in a Git repository, clone it:
    ```bash
    git clone <your-repository-url>
    cd <repository-name>
    ```
    If you just have the `q_learning_taxi.py` and `requirements.txt` files, create a directory, place them inside, and navigate into that directory:
    ```bash
    mkdir q_learning_taxi_project
    cd q_learning_taxi_project
    # (Now copy your q_learning_taxi.py and requirements.txt into this directory)
    ```

2.  **Create and Activate a Virtual Environment (Recommended):**
    It's good practice to create a virtual environment to manage project dependencies in isolation.
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
    Your terminal prompt should now indicate that the virtual environment is active (e.g., `(.venv) user@host:...$`).

3.  **Install Dependencies:**
    Install the required Python libraries using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```
    If you don't have a `requirements.txt` file yet, create one with the following content:
    ```txt
    gymnasium[toy-text]
    numpy
    matplotlib
    ```
    Then run `pip install -r requirements.txt`. The `gymnasium[toy-text]` package includes `pygame`, which is needed for visual rendering.

## Running the Q-Learning Agent

Once the setup is complete, you can run the Q-Learning script:

```bash
python q_learning_taxi.py
