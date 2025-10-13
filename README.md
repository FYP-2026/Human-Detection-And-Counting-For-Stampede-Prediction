## Setup and Installation
Follow these steps to set up the project on your local machine.

1. Clone the Repository
Bash

git clone https://github.com/YourUsername/Human-Detection-And-Counting-For-Stampede-Prediction.git
cd Human-Detection-And-Counting-For-Stampede-Prediction
2. Create the Conda Environment
This will create a new environment named stampede_env and install all the required libraries from the environment.yml file.

Bash

conda env create -f environment.yml
3. Activate the Environment
You must activate the environment every time you want to run the project.

Bash

conda activate stampede_env
4. Download Required Data
Download the fine-tuned model (e.g., best.pt) and place it inside the trained_models/ folder.

Download the test videos and place them inside the data/test_videos/ folder.

## How to Run the Application ▶️
Once the setup is complete, you can run the main script to start the application.

Bash

python main.py
Make sure to configure the correct video file name inside the main.py script before running.