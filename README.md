
# 📧 Spam Detection using SVM  

🚀 **Overview**  
This project implements a **spam detection system** using a **Support Vector Machine (SVM)** classifier. It analyzes **SMS text messages** and classifies them as **spam or ham (not spam)** using **NLP techniques** and **machine learning**.  

📌 **Features**  
✅ **Classifies SMS messages** as spam or ham  
✅ Uses **TF-IDF vectorization** for text processing  
✅ **SVM model** for high accuracy and robustness  
✅ Model evaluation using **accuracy, precision, recall, F1-score**  
✅ Supports custom datasets  

🔧 **Tech Stack**  
- **Python**, NumPy, Pandas, Scikit-learn  
- **NLTK, spaCy** (for text preprocessing)  
- **Matplotlib, Seaborn** (for visualization)  

📂 **Usage**  

1️⃣ **Install dependencies:**  
   ```bash
   pip install -r requirements.txt
   ```  

2️⃣ **Load and preprocess the dataset:**  
   ```python
   import pandas as pd  
   from sklearn.feature_extraction.text import TfidfVectorizer  

   # Load dataset  
   df = pd.read_csv("spam_dataset.csv")  

   # Convert labels to binary (spam = 1, ham = 0)  
   df['label'] = df['label'].map({'spam': 1, 'ham': 0})  

   # Text vectorization using TF-IDF  
   vectorizer = TfidfVectorizer(stop_words='english')  
   X = vectorizer.fit_transform(df['message'])  
   y = df['label']  
   ```  

3️⃣ **Train an SVM classifier:**  
   ```python
   from sklearn.model_selection import train_test_split  
   from sklearn.svm import SVC  

   # Split dataset  
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)  

   # Train SVM model  
   model = SVC(kernel='linear')  
   model.fit(X_train, y_train)  
   ```  

4️⃣ **Make predictions:**  
   ```python
   prediction = model.predict(X_test)  
   print("Predictions:", prediction)  
   ```  

📌 **Contributions & Issues**  
Feel free to contribute, report bugs, or suggest improvements! 🚀  

