# ju-cmater-project
An end-to-end machine learning pipeline for satellite collision risk prediction using an L1-regularized LSTM, VIF feature screening, hierarchical dendrogram clustering, and SHAP explainability.  

**Overview:**  
This repository contains a complete feature selection and deep learning pipeline designed to predict satellite orbital conjunction and collision risk. Built with PyData tools and Keras/TensorFlow, the project cleans and preprocesses high-dimensional satellite data, filters out redundant features using statistical correlation ($p$-values, VIF, and hierarchical dendrograms), and trains an L1-regularized LSTM network. It incorporates SHAP values for model explainability and incremental feature selection to maximize predictive performance.  

**Key Features / Tech Stack (Optional summary bullets):**  
  
- **Data Preprocessing & Cleaning:** Automated median imputation, standard scaling, and one-hot encoding.
- **Multicollinearity Screening:** Feature reduction via Pearson correlation (p<0.05), VIF thresholding, and Ward's hierarchical dendrogram clustering.
- **Deep Learning Model:** LSTM neural network with L1 regularization for embedded feature importance.
- **Explainable AI (XAI):** Global feature ranking and interpretability using SHAP (DeepExplainer).
- **Iterative Optimization:** Incremental feature selection driven by cross-validated metric gains.
