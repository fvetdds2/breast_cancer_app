******Breast cancer risk education and prediction application******

**Overview**

This repository contains code books for a comprehensive application designed to educate individuals about breast cancer risk factors, provide interactive health activity tracking, and facilitate access to support groups. The platform leverages a machine learning model built with XGBoost to predict individual risk of developing breast cancer using data from the Breast Cancer Surveillance Consortium (BCSC), which provides de-identified patient data including demographics, clinical risk factors and screening outcomes. The data can be downloaded from https://www.bcsc-research.org/index.php/datasets/rf/ with subjecting to data use agreement.

**Getting Started**
*Prerequisites*
-Python3 or higher

** Data**
- File: bcsc_concatenated_no_9.csv
- Description: Patient records with clinical, demographic features
- *Target Column*: breast_cancer_history (binary:0 = no cancer, 1 = cancer history)

**Model Training**
An XGBoost classifier was augmented with SMOTE oversampling the minority class to predict probability of breast cancer based on demographic and clinical features.

**Threshold Selection**
A Precision-Recall curve was computed form a test dataset to find a threshold to achieve Precision > 0.40 and Recall > 0.9 to minimize false negative number.

**Model Evaluation**
The chosen model was evaluated by plotting multiple performance metrics including AUC, accuracy, precision, recall, confusion matrix at chosen threshold, and feature importance plots

**Web Application**

The web application serves three main components:

-*1. Risk Calculator*: Users can input personal risk factors to receive an individualized risk probability.

-*2. Activity Tracker*: Interactive dashboard for logging health activities. and educational part providing videos organized by topic.

-*3. List of local and National support groups*: A curated list of organizations and communities to navigate through cancer treatment and recovery journey.

To use the application, click the link https://breastcancerapp-dollada.streamlit.app/


****License****
This project is released under the MIT License.

Developed by Dollada Srisai
Last updated: July 1st 2025