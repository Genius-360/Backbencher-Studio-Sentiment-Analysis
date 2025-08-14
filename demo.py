import pickle
import re
import nltk
from nltk.corpus import stopwords
import warnings

# Suppress unnecessary warnings for a clean user experience
warnings.filterwarnings('ignore')

# --- Global variable for stopwords, will be initialized later ---
STOP_WORDS = set()

def clean_text(text):
    """
    Cleans the input text by removing HTML, non-alphabetic characters,
    converting to lowercase, and removing stopwords.
    """
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove punctuation and numbers
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # Convert to lowercase
    text = text.lower()
    # Tokenize and remove stopwords
    words = text.split()
    words = [w for w in words if not w in STOP_WORDS]
    return ' '.join(words)

def predict_sentiment(review, model, vectorizer):
    """
    Takes a raw review string and returns the predicted sentiment.
    """
    if not review.strip():
        return "Cannot predict sentiment for an empty review."
    
    # Clean the review using the same steps as in the notebook
    cleaned_review = clean_text(review)
    
    # Transform the text using the loaded TF-IDF vectorizer
    review_vector = vectorizer.transform([cleaned_review])
    
    # Make a prediction with the loaded model
    prediction = model.predict(review_vector)
    
    return "Positive" if prediction[0] == 1 else "Negative"

# This is the main part of the script that runs when you execute `python demo.py`
if __name__ == "__main__":
    # --- Step 1: Ensure NLTK data is available BEFORE anything else ---
    try:
        STOP_WORDS = set(stopwords.words('english'))
    except LookupError:
        print("Stopwords not found. Downloading...")
        nltk.download('stopwords')
        print("Download complete.")
        STOP_WORDS = set(stopwords.words('english'))
    
    # --- Step 2: Load the trained model and vectorizer ---
    try:
        with open('best_logistic_regression_model.pkl', 'rb') as model_file:
            loaded_model = pickle.load(model_file)
        
        with open('tfidf_vectorizer.pkl', 'rb') as vec_file:
            loaded_vectorizer = pickle.load(vec_file)
            
        print("\nModel and vectorizer loaded successfully.")
        print("------------------------------------------")
        
        # --- Step 3: Start the interactive prediction loop ---
        while True:
            user_input = input("Enter a movie review to analyze (or type 'quit' to exit): ")
            if user_input.lower() == 'quit':
                break
            
            # Get the sentiment prediction
            sentiment = predict_sentiment(user_input, loaded_model, loaded_vectorizer)
            print(f"--> The predicted sentiment is: {sentiment}\n")

    except FileNotFoundError:
        print("\nError: Could not find 'best_logistic_regression_model.pkl' or 'tfidf_vectorizer.pkl'.")
        print("Please ensure these files are in the same directory as this script.")
        print("You may need to run the Jupyter Notebook first to create them.")