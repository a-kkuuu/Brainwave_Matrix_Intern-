
# 🎯 FINAL SOLUTION: Unbiased Fake News Detection
# Addresses ISOT dataset bias and creates production-ready model

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import re
import pickle

# 1. LOAD AND COMBINE ISOT DATASET
print("🔍 LOADING ISOT DATASET")
print("="*50)
fake_df = pd.read_csv('data/Fake.csv')
true_df = pd.read_csv('data/True.csv')
fake_df['label'] = 0
true_df['label'] = 1
df = pd.concat([fake_df, true_df], ignore_index=True)
print(f"✅ Total articles: {len(df)}")

# 2. ADVANCED TEXT PREPROCESSING TO REMOVE BIAS
def advanced_preprocessing(df):
    print(f"\n🔧 ADVANCED PREPROCESSING")
    print("="*30)
    df = df.dropna(subset=['text', 'title'])
    df['combined_text'] = df['title'] + ' ' + df['text']

    def clean_text(text):
        text = text.lower()
        bias_patterns = [
            r'\(reuters\)', r'\(ap\)', r'reuters -', r'- reuters', r'- ap',
            r'http[s]?://\S+', r'pic\.twitter\.com/\S+', r'@\w+', r'#\w+',
            r'www\.\w+\.\w+'
        ]
        for pattern in bias_patterns:
            text = re.sub(pattern, '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    df['cleaned_text'] = df['combined_text'].apply(clean_text)
    df = df[(df['cleaned_text'].str.len() > 100) & (df['cleaned_text'].str.len() < 5000)]
    df = df.drop_duplicates(subset=['cleaned_text'])
    print(f"After preprocessing: {len(df)} articles")
    return df

# 3. BALANCED SAMPLING TO REDUCE BIAS
def create_balanced_dataset(df, sample_size_per_class=15000):
    fake_articles = df[df['label'] == 0].sample(n=min(sample_size_per_class, len(df[df['label'] == 0])), random_state=42)
    real_articles = df[df['label'] == 1].sample(n=min(sample_size_per_class, len(df[df['label'] == 1])), random_state=42)
    balanced_df = pd.concat([fake_articles, real_articles], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"\n📊 BALANCED DATASET")
    print("="*30)
    print(f"Fake articles: {len(fake_articles)}")
    print(f"Real articles: {len(real_articles)}")
    print(f"Total balanced: {len(balanced_df)}")
    return balanced_df

# 4. CONTENT-FOCUSED FEATURE EXTRACTION
def create_content_features(X_train, X_test):
    vectorizer = TfidfVectorizer(
        max_features=8000, stop_words='english', ngram_range=(1, 3),
        min_df=3, max_df=0.8, sublinear_tf=True, analyzer='word',
        token_pattern=r'\b[a-zA-Z]{3,}\b'
    )
    print(f"\n🔤 CONTENT-FOCUSED FEATURES")
    print("="*30)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    print(f"Feature matrix: {X_train_tfidf.shape}")
    return X_train_tfidf, X_test_tfidf, vectorizer

# 5. ENSEMBLE MODEL WITH VOTING
def train_ensemble_models(X_train, X_test, y_train, y_test):
    lr = LogisticRegression(C=0.01, random_state=42, max_iter=1000)
    nb = MultinomialNB(alpha=1.0)
    rf_raw = RandomForestClassifier(n_estimators=50, max_depth=15, random_state=42, min_samples_split=20)
    rf = CalibratedClassifierCV(rf_raw, method='sigmoid', cv=3)
    ensemble = VotingClassifier(estimators=[('lr', lr), ('nb', nb), ('rf', rf)], voting='soft')
    models = {'Logistic Regression': lr, 'Naive Bayes': nb, 'Random Forest': rf, 'Ensemble': ensemble}
    results = {}
    print(f"\n🚀 TRAINING ENSEMBLE MODELS")
    print("="*40)
    for name, model in models.items():
        print(f"Training {name}...")
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        results[name] = {'model': model, 'cv_mean': cv_scores.mean(), 'cv_std': cv_scores.std(),
                         'test_accuracy': test_accuracy, 'predictions': y_pred}
        print(f"{name}:")
        print(f"  CV Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})")
        print(f"  Test Accuracy: {test_accuracy:.4f}\n")
    return results

# 6. COMPREHENSIVE REALISTIC TESTING
def comprehensive_testing(results, vectorizer):
    rf_model = results['Random Forest']['model']
    lr_model = results['Logistic Regression']['model']
    test_cases = [
        {"text": "The Federal Reserve announced Wednesday ...", "expected": "REAL", "category": "Economic News"},
        {"text": "Researchers at Johns Hopkins University ...", "expected": "REAL", "category": "Medical Research"},
        {"text": "SHOCKING: Scientists cure cancer overnight ...", "expected": "FAKE", "category": "Health Misinformation"},
        {"text": "BREAKING: Aliens invade Earth ...", "expected": "FAKE", "category": "Conspiracy Theory"},
    ]
    print(f"\n🧪 SMART COMPREHENSIVE TESTING")
    print("=" * 60)
    correct = 0
    for i, case in enumerate(test_cases, 1):
        cleaned_text = case['text'].lower()
        tfidf = vectorizer.transform([cleaned_text])
        rf_pred = rf_model.predict(tfidf)[0]
        rf_conf = rf_model.predict_proba(tfidf)[0].max()
        lr_pred = lr_model.predict(tfidf)[0]
        lr_conf = lr_model.predict_proba(tfidf)[0].max()
        if rf_conf < 0.70 or rf_pred != lr_pred:
            final_pred = lr_pred
            final_conf = lr_conf
            model_used = "Logistic Regression (fallback)"
        else:
            final_pred = rf_pred
            final_conf = rf_conf
            model_used = "Random Forest"
        predicted_label = "REAL" if final_pred == 1 else "FAKE"
        is_correct = predicted_label == case['expected']
        status = "✅" if is_correct else "❌"
        if is_correct:
            correct += 1
        print(f"\n{status} Test {i} ({case['category']}):")
        print(f"Expected: {case['expected']} | Predicted: {predicted_label} ({final_conf:.1%})")
        print(f"Used: {model_used}")
    accuracy = correct / len(test_cases)
    print(f"\n🎯 SMART TEST ACCURACY: {accuracy:.1%} ({correct}/{len(test_cases)})")
    return accuracy

# 7. SAVE MODEL
def save_production_model(model, vectorizer, accuracy):
    print(f"\n💾 SAVING PRODUCTION MODEL")
    print("="*30)
    with open('models/fake_news_detector.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('models/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    model_info = {
        'model_type': type(model).__name__,
        'test_accuracy': accuracy,
        'features': vectorizer.get_feature_names_out()[:100].tolist(),
        'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    with open('models/model_info.pkl', 'wb') as f:
        pickle.dump(model_info, f)
    print(f"✅ Model saved successfully!")

# 8. SAVE INDIVIDUAL MODELS FOR WEB APP
def save_additional_models_for_webapp(results, vectorizer):
    print(f"\n💾 SAVING ADDITIONAL MODELS FOR WEB APP")
    print("="*40)
    with open('models/logistic_model.pkl', 'wb') as f:
        pickle.dump(results['Logistic Regression']['model'], f)
    with open('models/naive_bayes_model.pkl', 'wb') as f:
        pickle.dump(results['Naive Bayes']['model'], f)
    with open('models/random_forest_model.pkl', 'wb') as f:
        pickle.dump(results['Random Forest']['model'], f)
    print(f"✅ Individual models saved for web app!")

# 9. MAIN EXECUTION
if __name__ == "__main__":
    import os
    os.makedirs('models', exist_ok=True)
    df = advanced_preprocessing(df)
    balanced_df = create_balanced_dataset(df)
    X_train, X_test, y_train, y_test = train_test_split(balanced_df['cleaned_text'], balanced_df['label'], test_size=0.2, random_state=42, stratify=balanced_df['label'])
    X_train_tfidf, X_test_tfidf, vectorizer = create_content_features(X_train, X_test)
    results = train_ensemble_models(X_train_tfidf, X_test_tfidf, y_train, y_test)
    best_model_name = max(results.keys(), key=lambda k: results[k]['cv_mean'])
    best_model = results[best_model_name]['model']
    best_accuracy = results[best_model_name]['test_accuracy']
    print(f"\n🏆 BEST MODEL: {best_model_name}")
    print(f"🎯 CV ACCURACY: {results[best_model_name]['cv_mean']:.4f}")
    print(f"🎯 TEST ACCURACY: {best_accuracy:.4f}")
    realistic_accuracy = comprehensive_testing(results, vectorizer)
    save_production_model(best_model, vectorizer, best_accuracy)
    save_additional_models_for_webapp(results, vectorizer)
    print(f"\n🎉 FINAL SOLUTION COMPLETE!")
    print(f"✅ Model saved and ready for production")
    print(f"📊 Realistic accuracy: {realistic_accuracy:.1%}")
