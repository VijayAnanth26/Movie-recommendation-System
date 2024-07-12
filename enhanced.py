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

# Display merged data
print(movies_data.head())
print(movies_data.shape)
print(movies_data.info())

# Select relevant columns
movies_data = movies_data[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

# Handle missing values
print(movies_data.isnull().sum())
movies_data.dropna(inplace=True)
print(movies_data.duplicated().sum())

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

# Tokenize overview column
movies_data['overview'] = movies_data['overview'].apply(lambda x: x.split())

# Remove spaces and convert to lowercase
for col in ['genres', 'keywords', 'cast', 'crew']:
    movies_data.loc[:, col] = movies_data[col].apply(lambda x: [elem.replace(" ","").lower() for elem in x])

# Combine tags
movies_data['tags'] = movies_data['overview'] + movies_data['genres'] + movies_data['keywords'] + movies_data['cast'] + movies_data['crew']

# Select relevant columns
new_data = movies_data[['movie_id', 'title', 'tags']]

# Convert tags to lowercase and join words
new_data.loc[:, 'tags'] = new_data['tags'].apply(lambda x: " ".join(x).lower())

# Vectorize tags
cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_data['tags']).toarray()

# Stem words
ps = PorterStemmer()
new_data.loc[:, 'tags'] = new_data['tags'].apply(lambda x: " ".join([ps.stem(word) for word in x.split()]))

# Calculate cosine similarity
similarity = cosine_similarity(vectors)

# Recommend movies
def recommend(movie_title):
    try:
        movie_index = new_data[new_data['title'].str.lower() == movie_title.lower()].index[0]
        distances = similarity[movie_index]
        movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

        for i in movie_list:
            print(new_data.iloc[i[0]]['title'])
    except IndexError:
        print(f"Movie '{movie_title}' not found in the dataset.")

# Example recommendation
recommend('Avatar')
recommend('batman')

recommend('Avatar')
recommend('batman')
# Example recommendations
recommend('Avatar')
print("\n")
recommend('batman')
print("\n")
recommend('Avatar')
print("\n")
recommend('batman')

print("\n")
recommend("God")
print("\n")
recommend('batman')
print("\n")
recommend('pi')