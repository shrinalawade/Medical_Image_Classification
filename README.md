# Medical_Image_Classification
Classification of Patients X-ray images into Normal and those affected by COVID-19.

Python version-  Python 3.8.1

requirements.txt - It contains all the required libraries list.

build_model.py- 
1. It contains the data import part, preprocessing of the data .i.e. Data Augmentation
2. Model building using Convolutional Neural Networks and saving the model in the model folder with .hd5 extension

main.py
1. A simple UI is developed using the streamlit library in python to upload the X-ray file of patients.
2. Once file is uploaded a file name, file image is displayed on the UI.
3. The prediction results are also displayed below the image giving the percentage of lungs infected by COVID-19.

Steps to run the code:
1. Install all dependencies command: pip install -r requirements.txt
2. Build the model command:  python build_model.py
3. To start the streamlit app RUN COMMAND: streamlit run main.py


Streamlit UI:
<img width="959" alt="image" src="https://github.com/shrinalawade/Medical_Image_Classification/assets/26817905/5ed16ec4-6030-49d7-85d7-c60cc5acfd7c">

