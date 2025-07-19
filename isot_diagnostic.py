# 🔍 ISOT FAKE NEWS DATASET - DIAGNOSTIC & FIX
# Specifically designed for Fake.csv and True.csv files

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. LOAD ISOT DATASET (Fake.csv + True.csv)
print("🔍 LOADING ISOT DATASET")
print("="*50)

try:
    # Load separate files
    fake_df = pd.read_csv('data/Fake.csv')
    true_df = pd.read_csv('data/True.csv')
    
    # Add labels
    fake_df['label'] = 0  # Fake news = 0
    true_df['label'] = 1  # Real news = 1
    
    # Combine datasets
    df = pd.concat([fake_df, true_df], ignore_index=True)
    
    print(f"✅ Dataset loaded successfully!")
    print(f"Fake articles: {len(fake_df)}")
    print(f"True articles: {len(true_df)}")
    print(f"Total articles: {len(df)}")
    print(f"Columns: {df.columns.tolist()}")
    
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    print("Make sure you have 'Fake.csv' and 'True.csv' in your data/ folder")
    exit()

# 2. INSPECT THE DATA
print(f"\n🔍 DATA INSPECTION")
print("="*30)

print(f"Dataset shape: {df.shape}")
print(f"Label distribution:")
print(df['label'].value_counts())

# Check for missing values
print(f"\nMissing values:")
print(df.isnull().sum())

# Sample articles
print(f"\n📰 SAMPLE FAKE NEWS:")
fake_sample = df[df['label'] == 0]['text'].iloc[0]
print(f"Title: {df[df['label'] == 0]['title'].iloc[0]}")
print(f"Text: {fake_sample[:200]}...")

print(f"\n📰 SAMPLE REAL NEWS:")
real_sample = df[df['label'] == 1]['text'].iloc[0]
print(f"Title: {df[df['label'] == 1]['title'].iloc[0]}")
print(f"Text: {real_sample[:200]}...")

# 3. IMPROVED PREPROCESSING
def improved_preprocessing(df):
    """Better preprocessing for ISOT dataset"""
    
    print(f"\n🔧 PREPROCESSING DATA")
    print("="*30)
    
    # Remove rows with missing text
    initial_size = len(df)
    df = df.dropna(subset=['text', 'title'])
    print(f"Removed missing values: {initial_size - len(df)} articles")
    
    # Combine title and text for better features
    df['combined_text'] = df['title'] + ' ' + df['text']
    
    # Basic text cleaning
    df['combined_text'] = df['combined_text'].str.lower()
    
    # Remove very short articles
    df = df[df['combined_text'].str.len() > 100]
    print(f"Removed short articles: {len(df)} articles remaining")
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['combined_text'])
    print(f"Removed duplicates: {len(df)} articles remaining")
    
    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return df

# 4. PROPER TRAIN-TEST SPLIT
def create_balanced_split(df):
    """Create properly balanced train-test split"""
    
    X = df['combined_text']
    y = df['label']
    
    # Stratified split to maintain class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    
    print(f"\n📊 TRAIN-TEST SPLIT")
    print("="*30)
    print(f"Train set: {len(X_train)} articles")
    print(f"Test set: {len(X_test)} articles")
    print(f"Train - Fake: {sum(y_train == 0)}, Real: {sum(y_train == 1)}")
    print(f"Test - Fake: {sum(y_test == 0)}, Real: {sum(y_test == 1)}")
    
    return X_train, X_test, y_train, y_test

# 5. FEATURE EXTRACTION WITH BETTER PARAMETERS
def create_features(X_train, X_test):
    """Create TF-IDF features with parameters to prevent overfitting"""
    
    # Conservative TF-IDF parameters to prevent overfitting
    vectorizer = TfidfVectorizer(
        max_features=10000,  # Increased for better coverage
        stop_words='english',
        ngram_range=(1, 2),  # Unigrams and bigrams
        min_df=5,  # Word must appear at least 5 times
        max_df=0.7,  # Ignore very common words
        sublinear_tf=True  # Apply log scaling
    )
    
    print(f"\n🔤 CREATING FEATURES")
    print("="*30)
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"Feature matrix shape: {X_train_tfidf.shape}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    
    return X_train_tfidf, X_test_tfidf, vectorizer

# 6. TRAIN MODELS WITH REGULARIZATION
def train_improved_models(X_train, X_test, y_train, y_test):
    """Train models with proper regularization"""
    
    models = {
        'Logistic Regression': LogisticRegression(
            random_state=42, 
            max_iter=1000,
            C=0.1,  # Strong regularization
            solver='liblinear'
        ),
        'Naive Bayes': MultinomialNB(alpha=0.1),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=20,  # Limit depth
            random_state=42,
            min_samples_split=10,
            min_samples_leaf=5
        ),
        'SVM': SVC(
            kernel='linear', 
            random_state=42,
            C=0.1,  # Strong regularization
            probability=True
        )
    }
    
    results = {}
    
    print(f"\n🚀 TRAINING MODELS")
    print("="*40)
    
    for name, model in models.items():
        print(f"Training {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        
        # Predict on test set
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'predictions': y_pred
        }
        
        print(f"{name}: {accuracy:.4f} accuracy")
        
        # Show classification report
        print(f"\n{name} - Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Fake', 'Real']))
        print("-" * 60)
    
    return results

# 7. TEST WITH REALISTIC EXAMPLES
def test_realistic_examples(best_model, vectorizer):
    """Test with realistic news examples"""
    
    test_articles = [
        # Should be REAL
        "The Federal Reserve announced today that it will maintain interest rates at current levels. Chairman Jerome Powell cited ongoing economic uncertainty as a key factor in the decision.",
        
        "Scientists at Stanford University published research in Nature journal showing promising results for a new cancer treatment. The peer-reviewed study followed 200 patients over 18 months.",
        
        "The mayor announced plans for a new public library following approval from the city council. Construction is expected to begin next spring with completion planned for 2025.",
        
        # Should be FAKE
        "SHOCKING: This one weird trick will make you lose 50 pounds overnight! Doctors hate this secret method that big pharma doesn't want you to know about!",
        
        "BREAKING: Government admits to hiding alien technology for 50 years! Leaked documents reveal multiple UFO crash sites across America, officials can no longer deny the truth!",
        
        "You won't believe what happens next! This celebrity secret will change your life forever. Click here to discover the amazing truth that will shock you!",
    ]
    
    print(f"\n🧪 TESTING WITH REALISTIC EXAMPLES")
    print("="*50)
    
    for i, article in enumerate(test_articles, 1):
        # Transform article
        article_tfidf = vectorizer.transform([article])
        
        # Predict
        prediction = best_model.predict(article_tfidf)[0]
        confidence = best_model.predict_proba(article_tfidf)[0].max()
        
        label = "REAL" if prediction == 1 else "FAKE"
        
        print(f"\nTest {i}: {article[:80]}...")
        print(f"Prediction: {label} ({confidence:.1%} confidence)")

# 8. VISUALIZE RESULTS
def create_visualizations(results, y_test):
    """Create accuracy comparison and confusion matrix"""
    
    # Model accuracy comparison
    plt.figure(figsize=(15, 5))
    
    # Subplot 1: Accuracy comparison
    plt.subplot(1, 2, 1)
    model_names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in model_names]
    
    bars = plt.bar(model_names, accuracies, color=['skyblue', 'lightgreen', 'salmon', 'orange'])
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.ylim(0.8, 1.0)  # Focus on the range where differences matter
    
    # Add accuracy values on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f'{acc:.3f}', ha='center', va='bottom')
    
    plt.xticks(rotation=45)
    
    # Subplot 2: Confusion matrix for best model
    plt.subplot(1, 2, 2)
    best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
    y_pred_best = results[best_model_name]['predictions']
    
    cm = confusion_matrix(y_test, y_pred_best)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=['Fake', 'Real'], 
               yticklabels=['Fake', 'Real'])
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.show()

# 9. MAIN EXECUTION
if __name__ == "__main__":
    # Preprocess data
    df = improved_preprocessing(df)
    
    # Create train-test split
    X_train, X_test, y_train, y_test = create_balanced_split(df)
    
    # Create features
    X_train_tfidf, X_test_tfidf, vectorizer = create_features(X_train, X_test)
    
    # Train models
    results = train_improved_models(X_train_tfidf, X_test_tfidf, y_train, y_test)
    
    # Find best model
    best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
    best_model = results[best_model_name]['model']
    best_accuracy = results[best_model_name]['accuracy']
    
    print(f"\n🏆 BEST MODEL: {best_model_name}")
    print(f"🎯 ACCURACY: {best_accuracy:.4f}")
    
    # Test with realistic examples
    test_realistic_examples(best_model, vectorizer)
    
    # Create visualizations
    create_visualizations(results, y_test)
    
    print(f"\n✅ ANALYSIS COMPLETE!")
    print(f"Expected accuracy range: 85-95%")
    print(f"Your result: {best_accuracy:.1%}")
    
    if best_accuracy > 0.98:
        print("⚠️  Still very high accuracy - might indicate overfitting")
        print("This is common with ISOT dataset due to writing style differences")
    else:
        print("✅ Realistic accuracy achieved!")