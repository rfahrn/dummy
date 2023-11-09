# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# %%
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# %%
data = load_dataset('argilla/medical-domain')['train']
df = data.to_pandas()

# %%
df['label'] = df.prediction.apply(lambda x: x[0]['label'])
df['text_length'] = df.metrics.apply(lambda x: x['text_length'])

# %%
label_frequencies = df.label.value_counts()

fig, ax = plt.subplots(1, 1, figsize = (10, 8))
sns.barplot(
	x=label_frequencies.values,
	y=label_frequencies.index,
	ax=ax
)
ax.set_title('Distribution of Labels')
ax.set_xlabel('frequency')
ax.set_ylabel('label')
plt.show()

# %%
fig, ax = plt.subplots(1, 1, figsize = (10, 5))
sns.histplot(
	df.text_length,
	ax=ax
)
ax.set_title('Distribution of Text Lengths')
ax.set_xlabel('number of characters')
ax.set_ylabel('frequency')
plt.show()

# %%
tokens = df.text.apply(func=word_tokenize).explode()

# %%
def plot_token_frequencies(tokens):
	token_frequencies = tokens.value_counts().head(50)
	fig, ax = plt.subplots(1, 1, figsize = (10, 10))
	sns.barplot(
		x=token_frequencies.values,
		y=token_frequencies.index,
		ax=ax
	)
	ax.set_title('Distribution of Tokens')
	ax.set_xlabel('frequency')
	ax.set_ylabel('token')
	plt.show()

# %%
plot_token_frequencies(tokens)

# %%
filtered_tokens = tokens.str.lower()
plot_token_frequencies(filtered_tokens)

# %%
filtered_tokens = filtered_tokens[~filtered_tokens.isin(stopwords.words('english'))]
filtered_tokens = filtered_tokens[filtered_tokens.str.match('[a-z]')]
plot_token_frequencies(filtered_tokens)

# %%
lemmatizer = WordNetLemmatizer()
filtered_tokens = filtered_tokens.apply(lambda x: lemmatizer.lemmatize(x))
plot_token_frequencies(filtered_tokens)

# %%
