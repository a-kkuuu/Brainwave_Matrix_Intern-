📰 Fake News Detection Web App

A professional-grade machine learning web application to detect fake news using NLP techniques and Streamlit. Built using the ISOT dataset.

🚀 Features

- Detects whether news is **Real** or **Fake**
- Built using multiple ML models (Logistic, Random Forest, Naive Bayes)
- User-friendly UI with examples
- Clean layout and responsive design
- Preprocessing includes lemmatization, stopwords removal, vectorization
- Multilingual support and language detection (optional upgrade in progress)

💻 Tech Stack

- Python
- Scikit-learn
- Pandas, Numpy
- Streamlit
- Pickle
- ISOT Dataset

📸 Screenshots

🏠 Home Page
![Home Page](assets/Screenshot 2025-07-20 002647)

🔍 Prediction Output
![Prediction Output](assets/Screenshot 2025-07-20 002731)

📂 Project Structure

├── app.py # Streamlit web app
├── models/ # Pretrained ML models + vectorizer
│ ├── logistic_model.pkl
│ ├── naive_bayes_model.pkl
│ ├── random_forest_model.pkl
│ ├── fake_news_detector.pkl
│ └── vectorizer.pkl
├── assets/ # Screenshots for documentation
│ ├── screenshot_home.png
│ └── screenshot_prediction.png
├── data/ # Raw datasets (True.csv, Fake.csv)
├── requirements.txt # Required Python libraries
└── README.md # Project overview

yaml
Copy code



## 🛠️ Tech Stack

- Python 3.10
- Streamlit (for UI)
- scikit-learn (ML models)
- NLTK / re (text preprocessing)
- joblib (model serialization)

▶️ How to Run Locally

bash
# 1. Clone the repository
git clone https://github.com/a-kkuuu/Brainwave_Matrix_Intern-.git
cd Brainwave_Matrix_Intern-

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run app.py

🧪 Example
Click the “Generate Example” button in the app to auto-fill a real or fake news article sample and test prediction.

🏷 GitHub Topics
streamlit · fake-news · machine-learning · logistic-regression · text-classification · scikit-learn

📬 Contact
Developer: Akriti Kumari
Email: [akriti91137@gmail.com]
GitHub: https://github.com/a-kkuuu

🌟 Star This Repo!
If you found this helpful or interesting, feel free to ⭐ star the repo to support the project!



✅ Next Steps

1. Save this as README.md in your root folder.
2. Run:

   bash
   git add README.md
   git commit -m "Final polished README with screenshots and details"
   git push
Visit your GitHub repo and enjoy your professional-looking homepage! 😄
