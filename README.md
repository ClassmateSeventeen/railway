# Railway Track Fault Detection Project

<b>GOAL</b>

To predict railway track faults.

In this project, I used Python and TensorFlow to classify images.

<b>DATASET && MODEL</B>

Dataset && Model used: https://pan.baidu.com/s/1jafF_532aymXfJRulVk-sA 
Enter Code：op83


## Steps to Use the Project

1. **Clone the Repository**: Clone this repository to your local machine using the following command:
   
   ```
   git clone https://github.com/ClassmateSeventeen/railway.git
   ```

2. **Navigate to the Project Directory**: Move into the project directory:
   
   ```
   cd railway-track-fault-detection
   ```

3. **Set Up Environment**: Install the required Python packages by running the following command:

   ```
   conda create --name railway python=3.8
   conda activate railway
   pip install -r requirements.txt
   ```

4. **Slove Error**: Install the required by running the following command:

   ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts. onnxsim 0.4.24 requires `rich`, which is not installed.

   ```
   pip install rich
   pip install -r requirements.txt
   ```  
   
5. **Download and Preprocess the Net**: Download the railway track fault detection Resnet50 from the url. 

   ```
   url: https://pan.baidu.com/s/1-_6JyI32T1S-Kk-K7YKFIQ Enter Code: sa1y
   ```

6. **Explore the Jupyter Notebook**: Open the Jupyter notebook named `demo.ipynb` to understand the project implementation and execution steps. This notebook provides a detailed guide on loading the dataset, building and training the deep learning model, evaluating performance, and testing the model on sample images.

7. **Inference**: Use the model to inference the result on photos by running the following command:
   ```
   python inference.py
   ```  

8. **Train**: Use the model to train your datasets by running the following command:
   ```
   python train.py
   ```  
