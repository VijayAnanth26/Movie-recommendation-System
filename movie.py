import pandas as pd
import ast
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity

# Read data
credits_data = pd.read_csv('credits.csv')
movies_data = pd.read_csv('movies.csv')

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Merge data
movies_data = movies_data.merge(credits_data, on='title')

# Handle missing values
movies_data.dropna(inplace=True)

# Select relevant columns
movies_data = movies_data[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

# Convert string representations of lists to actual lists
def convert(obj):
    return [elem['name'] for elem in ast.literal_eval(obj)]

movies_data['genres'] = movies_data['genres'].apply(convert)
movies_data['keywords'] = movies_data['keywords'].apply(convert)

# Extract only the first three names from the cast
def convert3(obj):
    return [elem['name'] for elem in ast.literal_eval(obj)][:3]

movies_data['cast'] = movies_data['cast'].apply(convert3)

# Extract names of directors from the crew
def fetch_director(obj):
    return [elem['name'] for elem in ast.literal_eval(obj) if elem['job'] == 'Director']

movies_data['crew'] = movies_data['crew'].apply(fetch_director)

# Tokenize overview column and preprocess text data
stemmer = PorterStemmer()
movies_data['overview'] = movies_data['overview'].apply(lambda x: x.split())
for col in ['genres', 'keywords', 'cast', 'crew']:
    movies_data[col] = movies_data[col].apply(lambda x: [stemmer.stem(word.replace(" ", "").lower()) for word in x])

# Combine tags
movies_data['tags'] = movies_data['overview'] + movies_data['genres'] + movies_data['keywords'] + movies_data['cast'] + movies_data['crew']

# Select relevant columns
new_data = movies_data[['movie_id', 'title', 'tags']]

# Convert tags to lowercase and join words using .loc to avoid SettingWithCopyWarning
new_data.loc[:, 'tags'] = new_data['tags'].apply(lambda x: " ".join(x).lower())

# Vectorize tags
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_data['tags']).toarray()

# Apply stemming
def stem(text):
    return " ".join([stemmer.stem(word) for word in text.split()])

new_data.loc[:, 'tags'] = new_data['tags'].apply(stem)

# Calculate cosine similarity
similarity = cosine_similarity(vectors)

# Recommend movies
def recommend(movie_title):
    movie_title = movie_title.lower()
    matches = new_data[new_data['title'].str.contains(movie_title, case=False)]

    if matches.empty:
        print(f"Movie '{movie_title}' not found in the dataset.")
        return

    movie_index = matches.index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    for i in movie_list:
        print(new_data.iloc[i[0]].title)

# Example recommendations
recommend('Avatar')
print("\n")
recommend('batman')
print("\n")
recommend('Batman v Superman: Dawn of Justice')
print("\n")
recommend('Batman Begins')
print("\n")
recommend('pi')
print("\n")
recommend("god")
