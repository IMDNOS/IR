import nltk

resources = [
    "punkt",
    "punkt_tab",
    "stopwords",
    "wordnet",
    "omw-1.4",
]

for resource in resources:
    nltk.download(resource)

print("NLTK resources downloaded successfully.")