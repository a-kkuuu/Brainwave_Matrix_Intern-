#!/usr/bin/env python3
"""
🚀 FAKE NEWS DETECTION - DAY 1 COMPLETE VERSION
All functionality working with multiple models and predictions
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def create_sample_data():
    """Create sample fake news data for immediate testing"""
    print("📊 Loading dataset...")
    
    # Sample real news headlines/text
    real_news = [
        "Scientists discover new planet in nearby solar system using advanced telescopes",
        "Local mayor announces new infrastructure project for downtown area",
        "Stock market shows steady growth this quarter according to financial reports",
        "New medical breakthrough helps treat rare disease in clinical trials",
        "University researchers publish climate change study in scientific journal",
        "Technology company releases quarterly earnings report showing profits",
        "International trade agreement signed between two countries this week",
        "Archaeological team uncovers ancient artifacts in historical excavation site",
        "Sports team wins championship in overtime victory against rivals",
        "Government announces new education funding initiative for public schools"
    ]
    
    # Sample fake news headlines/text
    fake_news = [
        "Aliens land in major city downtown area, government covers up story completely",
        "Miracle cure discovered that doctors don't want you to know about",
        "Celebrity secretly controls world government from hidden underground base",
        "Dangerous vaccines contain mind control microchips according to insider",
        "Local politician caught in massive conspiracy scandal involving millions",
        "Scientists hide evidence that earth is actually flat, whistleblower reveals",
        "Secret society plans to control all social media platforms next month",
        "Ancient prophecy predicts end of world next month according to expert",
        "Government uses weather machines to control natural disasters and storms",
        "Billionaire entrepreneur is actually time traveling alien from future"
    ]
    
    # Create DataFrame
    texts = real_news + fake_news
    labels = [0] * len(real_news) + [1] * len(fake_news)  # 0 = Real, 1 = Fake
    
    df = pd.DataFrame({
        'text': texts,
        'label': labels
    })
    
    print(f"Dataset loaded: {len(df)} articles")
    print(f"Real news: {len(real_news)}")
    print(f"Fake news: {len(fake_news)}")
    return df

def preprocess_text(text):
    """Basic text preprocessing"""
    if pd.isna(text):
        return ""
    # Convert to lowercase and basic cleaning
    text = str(text).lower()
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def train_models(X_train, X_test, y_train, y_test):
    """Train multiple models and compare results"""
    print("Training models...")
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Naive Bayes': MultinomialNB(),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='linear', probability=True, random_state=42)
    }
    
    results = {}
    
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'predictions': y_pred
        }
        
        print(f"{name} Accuracy: {accuracy:.2f}")
    
    return results

def predict_article(model, vectorizer, article_text):
    """Predict if an article is fake or real"""
    # Preprocess the text
    processed_text = preprocess_text(article_text)
    
    # Vectorize the text
    text_vector = vectorizer.transform([processed_text])
    
    # Make prediction
    prediction = model.predict(text_vector)[0]
    probability = model.predict_proba(text_vector)[0]
    
    # Get confidence
    confidence = max(probability) * 100
    
    # Return result
    label = "FAKE" if prediction == 1 else "REAL"
    return label, confidence

def create_visualizations(results, y_test):
    """Create basic visualizations"""
    print("\n📊 Creating visualizations...")
    
    # Model comparison
    plt.figure(figsize=(12, 5))
    
    # Subplot 1: Model Accuracies
    plt.subplot(1, 2, 1)
    names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in names]
    
    bars = plt.bar(names, accuracies, color=['skyblue', 'lightgreen', 'lightcoral', 'lightsalmon'])
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    
    # Add accuracy values on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.2f}', ha='center', va='bottom')
    
    # Subplot 2: Confusion Matrix for best model
    plt.subplot(1, 2, 2)
    best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
    best_predictions = results[best_model_name]['predictions']
    
    cm = confusion_matrix(y_test, best_predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.show()

def main():
    """Main execution function"""
    print("🔍 Fake News Detection System")
    print("=" * 40)
    
    # Step 1: Create or load data
    df_fake = pd.read_csv("data/Fake.csv")
    df_true = pd.read_csv("data/True.csv")

    df_fake['label'] = 1  # Fake = 1
    df_true['label'] = 0  # Real = 0

    df = pd.concat([df_fake, df_true], ignore_index=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Dataset loaded: {len(df)} articles")
    print(f"Real news: {sum(df['label'] == 0)}")
    print(f"Fake news: {sum(df['label'] == 1)}")

    
    # Step 2: Preprocess text
    df['processed_text'] = (df['title'].astype(str) + " " + df['text'].astype(str)).apply(preprocess_text)

    
    # Step 3: Create features using TF-IDF
    vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1, 2)  # use unigrams + bigrams
    )
    X = vectorizer.fit_transform(df['processed_text'])
    y = df['label']
    
    # Step 4: Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Step 5: Train models
    results = train_models(X_train, X_test, y_train, y_test)
    
    # Step 6: Find best model
    # Use Logistic Regression for better generalization
    best_model_name = 'Logistic Regression'
    best_model = results[best_model_name]['model']
    best_accuracy = results[best_model_name]['accuracy']

    
    # Step 7: Test with custom articles
    print("=" * 50)
    print("Testing with custom articles:")
    print("=" * 50)
    
    test_articles = [
    "The economy grew by 3% last quarter, according to government data.",
    "Doctors warn about fake COVID-19 cures spreading on social media.",
    "BREAKING: NASA confirms aliens are already living on Mars!",
    "Miracle plant can cure cancer in 7 days — experts shocked!",
    "Government launches new scheme to provide free education to all."
]

    
    for i, article in enumerate(test_articles, 1):
        label, confidence = predict_article(best_model, vectorizer, article)
        print(f"Test {i}: {article}")
        print(f"Prediction: {label} ({confidence:.1f}% confidence)")
        print()
    
    # Step 8: Create visualizations
    create_visualizations(results, y_test)
    
    # Step 9: Summary
    print("🎉 DAY 1 COMPLETE!")
    print("✅ Sample dataset created")
    print("✅ Multiple models trained")
    print("✅ Predictions working")
    print("✅ Visualizations created")
    print(f"✅ Best accuracy: {best_accuracy:.2f}")
    print("\n🚀 Ready for Day 2: Real dataset integration!")

if __name__ == "__main__":
    main()
    import joblib
    joblib.dump(best_model, f"models/{best_model_name.replace(' ', '_')}.pkl")
    joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
    print(f"💾 Saved model and vectorizer to 'models/'")
