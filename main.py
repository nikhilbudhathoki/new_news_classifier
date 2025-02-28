import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt

# Define the categories
CATEGORIES = [
    'Arts', 'Automobile', 'Bank', 'Blog', 'Business', 'Crime', 'Economy', 'Education',
    'Entertainment', 'Health', 'Politics', 'Society', 'Sports', 'Technology', 'Tourism', 'World'
]

# Load model and tokenizer from Hugging Face
@st.cache_resource
def load_model():
    model_name = "NikhilBudhathoki/News_classifier"  # Your Hugging Face model path
    try:
        st.info(f"Loading model from Hugging Face: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        device = 'cpu'
        model.to(device)
        model.eval()
        st.success("Model loaded successfully!")
        return model, tokenizer, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

def predict_category(text, model, tokenizer, device):
    # Tokenize the input text
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
        padding="max_length",
    )
    
    # Move inputs to the same device as model
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Get probabilities using softmax
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    
    # Get predicted class and probability
    predicted_class_idx = torch.argmax(probabilities, dim=1).item()
    predicted_class = CATEGORIES[predicted_class_idx]
    
    # Get all class probabilities as dictionary
    all_probs = {CATEGORIES[i]: prob for i, prob in enumerate(probabilities[0].cpu().numpy())}
    
    return predicted_class, all_probs

def create_probability_chart(probabilities):
    # Sort probabilities
    sorted_probs = dict(sorted(probabilities.items(), key=lambda x: x[1], reverse=True))
    categories = list(sorted_probs.keys())
    probs = list(sorted_probs.values())
    
    # Keep only top 5 for cleaner visualization
    if len(categories) > 5:
        top_categories = categories[:5]
        top_probs = probs[:5]
    else:
        top_categories = categories
        top_probs = probs
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(top_categories, top_probs, color='skyblue')
    
    # Add text labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{height:.2%}', ha='center', va='bottom')
    
    plt.title('Top Category Probabilities')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1.0)
    plt.tight_layout()
    
    return fig

def main():
    st.title("News Category Classifier")
    st.write("Enter a news article to classify it into one of the 16 categories.")
    
    # Load model from Hugging Face
    with st.spinner("Loading model... This may take a moment."):
        model, tokenizer, device = load_model()
    
    if model is None:
        st.error("Failed to load model.")
        return
    
    # Create text input area
    news_text = st.text_area("Enter news text:", height=200)
    example_button = st.button("Load Example Text")
    
    # Example text
    example_text = """
    The European Central Bank cut interest rates on Thursday for the third time since June, 
    as the euro zone's economy continues to struggle and inflation edges closer to target. 
    The ECB lowered its benchmark deposit rate by 25 basis points to 3.25%, 
    in line with market expectations.
    """
    
    if example_button:
        news_text = example_text
        st.session_state.news_text = news_text
        st.experimental_rerun()
    
    # Add a classify button
    classify_button = st.button("Classify News")
    
    # Process input and display prediction
    if news_text and classify_button:
        if len(news_text.strip()) < 10:
            st.warning("Please enter a longer news text for better classification.")
        else:
            with st.spinner("Classifying..."):
                predicted_category, probabilities = predict_category(news_text, model, tokenizer, device)
            
            # Display results
            st.success(f"Predicted Category: **{predicted_category}**")
            
            # Show top 3 categories as text
            st.write("### Top 3 Categories:")
            top3 = dict(sorted(probabilities.items(), key=lambda x: x[1], reverse=True)[:3])
            for category, prob in top3.items():
                st.write(f"- {category}: {prob:.2%}")
            
            # Display visualization
            st.write("### Probability Distribution:")
            chart = create_probability_chart(probabilities)
            st.pyplot(chart)
            
            # Display confidence information
            top_prob = max(probabilities.values())
            if top_prob > 0.80:
                st.info("High confidence prediction ✅")
            elif top_prob > 0.50:
                st.info("Moderate confidence prediction ⚠️")
            else:
                st.warning("Low confidence prediction ⚠️ - this article may contain mixed themes")
    
    # Add information section
    with st.expander("About this app"):
        st.write("""
        This app uses a fine-tuned model from Hugging Face to classify news articles into 16 categories.
        
        **Available Categories:**
        - Arts
        - Automobile
        - Bank
        - Blog
        - Business
        - Crime
        - Economy
        - Education
        - Entertainment
        - Health
        - Politics
        - Society
        - Sports
        - Technology
        - Tourism
        - World
        
        For best results, provide complete news articles with at least a full paragraph of text.
        """)

if __name__ == "__main__":
    main()
