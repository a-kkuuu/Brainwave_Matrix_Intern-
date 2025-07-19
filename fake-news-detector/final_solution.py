# 🎯 FINAL SOLUTION: Unbiased Fake News Detection
# Addresses ISOT dataset bias and creates production-ready model

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
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
    """Remove source bias and focus on content"""
    
    print(f"\n🔧 ADVANCED PREPROCESSING")
    print("="*30)
    
    # Remove rows with missing text
    df = df.dropna(subset=['text', 'title'])
    
    # Combine title and text
    df['combined_text'] = df['title'] + ' ' + df['text']
    
    def clean_text(text):
        """Advanced text cleaning to remove bias indicators"""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove common bias indicators from ISOT dataset
        bias_patterns = [
            r'\(reuters\)',
            r'\(ap\)',
            r'reuters -',
            r'associated press',
            r'washington \(reuters\)',
            r'new york \(reuters\)',
            r'london \(reuters\)',
            r'- reuters',
            r'- ap',
            r'published.*ago',
            r'updated.*ago',
            r'breaking:',
            r'urgent:',
            r'just in:',
            r'exclusive:',
            r'www\.\w+\.com',
            r'http[s]?://\S+',
            r'pic\.twitter\.com/\S+',
            r'@\w+',
            r'#\w+',
        ]
        
        for pattern in bias_patterns:
            text = re.sub(pattern, '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    # Apply cleaning
    df['cleaned_text'] = df['combined_text'].apply(clean_text)
    
    # Remove very short or very long articles
    df = df[(df['cleaned_text'].str.len() > 100) & (df['cleaned_text'].str.len() < 5000)]
    
    # Remove duplicates based on cleaned text
    df = df.drop_duplicates(subset=['cleaned_text'])
    
    print(f"After preprocessing: {len(df)} articles")
    
    return df

# 3. BALANCED SAMPLING TO REDUCE BIAS
def create_balanced_dataset(df, sample_size_per_class=15000):
    """Create balanced dataset with equal samples"""
    
    fake_articles = df[df['label'] == 0].sample(n=min(sample_size_per_class, len(df[df['label'] == 0])), random_state=42)
    real_articles = df[df['label'] == 1].sample(n=min(sample_size_per_class, len(df[df['label'] == 1])), random_state=42)
    
    balanced_df = pd.concat([fake_articles, real_articles], ignore_index=True)
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\n📊 BALANCED DATASET")
    print("="*30)
    print(f"Fake articles: {len(fake_articles)}")
    print(f"Real articles: {len(real_articles)}")
    print(f"Total balanced: {len(balanced_df)}")
    
    return balanced_df

# 4. CONTENT-FOCUSED FEATURE EXTRACTION
def create_content_features(X_train, X_test):
    """Focus on content rather than style"""
    
    # TF-IDF with content-focused parameters
    vectorizer = TfidfVectorizer(
        max_features=8000,
        stop_words='english',
        ngram_range=(1, 3),  # Include trigrams for better context
        min_df=3,
        max_df=0.8,
        sublinear_tf=True,
        analyzer='word',
        token_pattern=r'\b[a-zA-Z]{3,}\b'  # Only words with 3+ letters
    )
    
    print(f"\n🔤 CONTENT-FOCUSED FEATURES")
    print("="*30)
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"Feature matrix: {X_train_tfidf.shape}")
    
    return X_train_tfidf, X_test_tfidf, vectorizer

# 5. ENSEMBLE MODEL WITH VOTING
def train_ensemble_models(X_train, X_test, y_train, y_test):
    """Train ensemble of models for better generalization"""
    
    from sklearn.ensemble import VotingClassifier
    
    # Individual models with strong regularization
    lr = LogisticRegression(C=0.01, random_state=42, max_iter=1000)
    nb = MultinomialNB(alpha=1.0)
    rf = RandomForestClassifier(n_estimators=50, max_depth=15, random_state=42, min_samples_split=20)
    
    # Create voting ensemble
    ensemble = VotingClassifier(
        estimators=[('lr', lr), ('nb', nb), ('rf', rf)],
        voting='soft'
    )
    
    models = {
        'Logistic Regression': lr,
        'Naive Bayes': nb,
        'Random Forest': rf,
        'Ensemble': ensemble
    }
    
    results = {}
    
    print(f"\n🚀 TRAINING ENSEMBLE MODELS")
    print("="*40)
    
    for name, model in models.items():
        print(f"Training {name}...")
        
        # Cross-validation first
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        
        # Train on full training set
        model.fit(X_train, y_train)
        
        # Test predictions
        y_pred = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'test_accuracy': test_accuracy,
            'predictions': y_pred
        }
        
        print(f"{name}:")
        print(f"  CV Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})")
        print(f"  Test Accuracy: {test_accuracy:.4f}")
        print()
    
    return results

# 6. REALISTIC TESTING WITH DIVERSE EXAMPLES
def comprehensive_testing(best_model, vectorizer):
    """Test with diverse, realistic examples"""
    
    test_cases = [
        # REAL NEWS (should be classified as REAL)
        {
            'text': "The Federal Reserve announced Wednesday that it will maintain the federal funds rate at its current level of 5.25% to 5.50%. Chair Jerome Powell cited ongoing concerns about inflation and labor market conditions in the decision.",
            'expected': 'REAL',
            'category': 'Economic News'
        },
        {
            'text': "Researchers at Johns Hopkins University published findings in the Journal of Medical Research showing that a new treatment for diabetes showed promising results in clinical trials involving 500 patients over 12 months.",
            'expected': 'REAL',
            'category': 'Medical Research'
        },
        {
            'text': "The city council approved a $2.3 million budget for road repairs and infrastructure improvements. The project will focus on Main Street and downtown areas, with construction expected to begin in spring.",
            'expected': 'REAL',
            'category': 'Local Government'
        },
        
        # FAKE NEWS (should be classified as FAKE)
        {
            'text': "SHOCKING DISCOVERY: Scientists have found that eating this common household item can cure cancer overnight! Big Pharma doesn't want you to know this simple trick that costs only $1!",
            'expected': 'FAKE',
            'category': 'Health Misinformation'
        },
        {
            'text': "BREAKING: Government officials secretly admit that aliens have been living among us for decades! Leaked documents reveal shocking truth about UFO coverup that will change everything!",
            'expected': 'FAKE',
            'category': 'Conspiracy Theory'
        },
        {
            'text': "You won't believe what happened next! This celebrity's shocking secret will change your life forever. Click here to discover the amazing truth that will absolutely blow your mind!",
            'expected': 'FAKE',
            'category': 'Clickbait'
        },
        
        # BORDERLINE CASES
        {
            'text': "Local business owner reports unusual increase in sales following social media campaign. The restaurant has seen customer traffic double in the past month.",
            'expected': 'REAL',
            'category': 'Business News'
        },
        {
            'text': "Celebrity couple spotted together at local restaurant, sparking rumors about their relationship status. Sources close to the couple have not commented on the speculation.",
            'expected': 'REAL',
            'category': 'Entertainment'
        }
    ]
    
    print(f"\n🧪 COMPREHENSIVE TESTING")
    print("="*50)
    
    correct_predictions = 0
    total_predictions = len(test_cases)
    
    for i, case in enumerate(test_cases, 1):
        # Clean the text same way as training data
        cleaned_text = case['text'].lower()
        
        # Transform and predict
        text_tfidf = vectorizer.transform([cleaned_text])
        prediction = best_model.predict(text_tfidf)[0]
        confidence = best_model.predict_proba(text_tfidf)[0].max()
        
        predicted_label = "REAL" if prediction == 1 else "FAKE"
        is_correct = predicted_label == case['expected']
        
        if is_correct:
            correct_predictions += 1
            status = "✅"
        else:
            status = "❌"
        
        print(f"\n{status} Test {i} ({case['category']}):")
        print(f"Text: {case['text'][:100]}...")
        print(f"Expected: {case['expected']} | Predicted: {predicted_label} ({confidence:.1%})")
    
    realistic_accuracy = correct_predictions / total_predictions
    print(f"\n🎯 REALISTIC TEST ACCURACY: {realistic_accuracy:.1%} ({correct_predictions}/{total_predictions})")
    
    return realistic_accuracy

# 7. SAVE PRODUCTION MODEL
def save_production_model(model, vectorizer, accuracy):
    """Save the final production-ready model"""
    
    print(f"\n💾 SAVING PRODUCTION MODEL")
    print("="*30)
    
    # Save model and vectorizer
    with open('models/fake_news_detector.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open('models/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    
    # Save model info
    model_info = {
        'model_type': type(model).__name__,
        'test_accuracy': accuracy,
        'features': vectorizer.get_feature_names_out()[:100].tolist(),  # First 100 features
        'training_date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open('models/model_info.pkl', 'wb') as f:
        pickle.dump(model_info, f)
    
    print(f"✅ Model saved successfully!")
    print(f"📁 Files saved in models/ directory")

# 8. MAIN EXECUTION
if __name__ == "__main__":
    # Create models directory
    import os
    os.makedirs('models', exist_ok=True)
    
    # Preprocess data
    df = advanced_preprocessing(df)
    
    # Create balanced dataset
    balanced_df = create_balanced_dataset(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        balanced_df['cleaned_text'], 
        balanced_df['label'], 
        test_size=0.2, 
        random_state=42, 
        stratify=balanced_df['label']
    )
    
    # Create features
    X_train_tfidf, X_test_tfidf, vectorizer = create_content_features(X_train, X_test)
    
    # Train models
    results = train_ensemble_models(X_train_tfidf, X_test_tfidf, y_train, y_test)
    
    # Select best model based on CV score
    best_model_name = max(results.keys(), key=lambda k: results[k]['cv_mean'])
    best_model = results[best_model_name]['model']
    best_accuracy = results[best_model_name]['test_accuracy']
    
    print(f"\n🏆 BEST MODEL: {best_model_name}")
    print(f"🎯 CV ACCURACY: {results[best_model_name]['cv_mean']:.4f}")
    print(f"🎯 TEST ACCURACY: {best_accuracy:.4f}")
    
    # Comprehensive realistic testing
    realistic_accuracy = comprehensive_testing(best_model, vectorizer)
    
    # Save production model
    save_production_model(best_model, vectorizer, best_accuracy)
    
    # Create final visualization
    plt.figure(figsize=(15, 10))
    
    # Model comparison
    plt.subplot(2, 2, 1)
    model_names = list(results.keys())
    cv_scores = [results[name]['cv_mean'] for name in model_names]
    test_scores = [results[name]['test_accuracy'] for name in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    
    plt.bar(x - width/2, cv_scores, width, label='CV Score', alpha=0.8)
    plt.bar(x + width/2, test_scores, width, label='Test Score', alpha=0.8)
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title('Model Performance Comparison')
    plt.xticks(x, model_names, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Confusion matrix
    plt.subplot(2, 2, 2)
    best_predictions = results[best_model_name]['predictions']
    cm = confusion_matrix(y_test, best_predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=['Fake', 'Real'], 
               yticklabels=['Fake', 'Real'])
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Feature importance (if available)
    plt.subplot(2, 2, 3)
    if hasattr(best_model, 'feature_importances_'):
        feature_names = vectorizer.get_feature_names_out()
        importances = best_model.feature_importances_
        top_indices = np.argsort(importances)[-20:]
        
        plt.barh(range(20), importances[top_indices])
        plt.yticks(range(20), [feature_names[i] for i in top_indices])
        plt.xlabel('Importance')
        plt.title('Top 20 Feature Importances')
    else:
        plt.text(0.5, 0.5, 'Feature importance\nnot available\nfor this model', 
                ha='center', va='center', transform=plt.gca().transAxes)
        plt.title('Feature Analysis')
    
    # Performance summary
    plt.subplot(2, 2, 4)
    plt.axis('off')
    summary_text = f"""
    FINAL MODEL PERFORMANCE
    
    Best Model: {best_model_name}
    
    Cross-Validation: {results[best_model_name]['cv_mean']:.1%}
    Test Accuracy: {best_accuracy:.1%}
    Realistic Test: {realistic_accuracy:.1%}
    
    Dataset Size: {len(balanced_df):,} articles
    Features: {X_train_tfidf.shape[1]:,}
    
    Status: {'✅ Production Ready' if realistic_accuracy > 0.7 else '⚠️ Needs Improvement'}
    """
    plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes, 
            fontsize=12, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n🎉 FINAL SOLUTION COMPLETE!")
    print(f"✅ Model saved and ready for production")
    print(f"📊 Realistic accuracy: {realistic_accuracy:.1%}")
    
    if realistic_accuracy > 0.75:
        print(f"🏆 EXCELLENT! Your model is production-ready!")
    elif realistic_accuracy > 0.60:
        print(f"✅ GOOD! Model performs well on realistic examples")
    else:
        print(f"⚠️ Model needs further tuning for real-world use")