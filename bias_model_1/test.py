from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
from transformers import pipeline

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("d4data/bias-detection-model")
model = TFAutoModelForSequenceClassification.from_pretrained("d4data/bias-detection-model")

# Create a text classification pipeline
classifier = pipeline('text-classification', model=model, tokenizer=tokenizer)  # cuda=0,1 based on GPU availability

# Analyze a sample news text
text = "summary: This is a published version of the Forbes Daily newsletter, you can sign-up to get Forbes Daily in your inbox here. Earlier this year, my husband and I traveled to Japan, and we enjoyed how far the U.S. dollar took us-especially when dining on sushi and ramen. We're not the only ones."
result = classifier(text)[0]

# Determine if the news is biased or not
label = result['label']
score = result['score']

if label == 'biased':
    print(f"The news is biased with a confidence score of {score:.2f}.")
else:
    print(f"The news is not biased with a confidence score of {score:.2f}.")
