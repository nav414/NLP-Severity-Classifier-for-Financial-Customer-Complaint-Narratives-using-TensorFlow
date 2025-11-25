This project applies Natural Language Processing with TensorFlow to classify the severity of financial customer complaint narratives. It uses a neural network–based text classification approach to help financial organizations prioritize incoming complaints by automatically predicting whether a complaint is Low, Medium, or High severity based on the customer’s written narrative.

The repository includes the labeled dataset, a Jupyter notebook for model development, and a Streamlit app for real time severity prediction. 



Project Overview :
1. Complaint narratives were labeled into Low, Medium, and High severity using embedding and classification with the help of narratives and metadata associated with each complain.
2. Text data was preprocessed and vectorized using TensorFlow TextVectorization.
3. Multiple deep learning models were tested, including LSTM, Bidirectional LSTM, CNN, and multi-channel CNN.
4. The final model uses a Bidirectional LSTM with a custom Attention mechanism.
5. The trained model was integrated into a Streamlit app for interactive predictions.



Repository Structure :
1. complaints_filtered.csv - The dataset of financial customer complaint narratives along with metadata.
2. Banking_Complaint.ipynb - Jupyter notebook containing preprocessing steps, model training, experimentation, and evaluation.
3. app.py - Streamlit application for real time severity prediction from text input.



Technologies Used :
1. Python - Pandas, NumPy, Matplotlib
2. TensorFlow and Keras
3. NLP preprocessing and TextVectorization
4. BiLSTM and Attention Networks
5. Streamlit for deployment



Model Architecture :
1. Data Preprocessing
2. Manual Severity Labeling of Complaint Narratives
3. NLP Embedding Generation
4. TensorFlow Neural Network Architecture (Bi-LSTM + Attention)
5. Model Training and Validation
6. Severity Classification Output (Low / Medium / High)
7. Model Evaluation
8. Deployment (Streamlit)



Results :
The BiLSTM with Attention model achieved the strongest results among all tested architectures and was selected for deployment in the Streamlit app. The model consistently captures important contextual patterns in complaint narratives and provides reliable severity predictions.
