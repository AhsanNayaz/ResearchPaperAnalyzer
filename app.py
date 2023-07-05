from flask import Flask, request, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from datetime import datetime
import openai
from typing import List, Iterator
import pandas as pd
import numpy as np
import os
import nltk
import PyPDF2
import csv 
import re
from ast import literal_eval
import warnings
from getpass import getpass
from openai.embeddings_utils import get_embedding
import pinecone

#nltk.download('punkt')
openai.api_key = os.environ["OPENAI_API_KEY"]
pinecone_api = os.environ["PINECONE_API_KEY"]
environment = os.environ["PINECONE_ENV"]
pinecone.init(api_key=pinecone_api, environment=environment)
EMBEDDING_MODEL = "text-embedding-ada-002"
warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
index = pinecone.Index(index_name='embedded')

app = Flask(__name__) 


def count_tokens(prompt):
    prompt_bytes = bytes(prompt, 'utf-8')

    token_count = 0
    for token in tokenize.tokenize(BytesIO(prompt_bytes).readline):
        if token.type != tokenize.ENDMARKER:
            token_count += 1

    return token_count

def split_prompt(text, max_tokens=1000):
    sentences = nltk.sent_tokenize(text)

    chunks = []
    current_chunk = []
    current_length = 0
    for sentence in sentences:
        sentence_length = len(sentence.split())
        if current_length + sentence_length <= max_tokens:
            current_chunk.append(sentence)
            current_length += sentence_length
        else:
            chunks.append(current_chunk)
            current_chunk = [sentence]
            current_length = sentence_length
    if current_chunk:
        chunks.append(current_chunk)

    prompts = []
    for chunk in chunks:
        prompt = ' '.join(chunk)
        prompts.append(prompt)

    return prompts


def extract_paragraphs(text):
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    return re.split('\s{6,}',text)
    #return ' '.join(paragraphs)
def query_article(query, top_k=5):
    '''Queries an article using its title in the specified
     namespace and prints results.'''

    # Create vector embeddings based on the title column
    embedded_query = openai.Embedding.create(
                                            input=query,
                                            model=EMBEDDING_MODEL,
                                            )["data"][0]['embedding']

    # Query namespace passed as parameter using title vector
    query_result = index.query(embedded_query,
                                      top_k=top_k)
    print(f'\nMost similar results to {query}:')
    if not query_result.matches:
        print('no query result')
    # print(query_result.matches)
    matches = query_result.matches
    ids = [int(res.id) for res in matches]
    scores = [res.score for res in matches]
    res = []
    for i in ids:
      res.append(content_mapped[int(i)])

#asynchronus transmission
#multi threading
    return res


def apply_prompt_template(question: str) -> str:
    """
        A helper function that applies additional template on user's question.
        Prompt engineering could be done here to improve the result. Here I will just use a minimal example.
    """
    prompt = f"""
        By considering above input from me, answer the question: {question}
    """
    return prompt

@app.route('/', methods=['GET', 'POST'])
def main(): 
    # If a form is submitted
    if request.method == "POST":
        # Unpickle classifier
        # Get values through input bars
        prompt = request.form.get("name")
        file = request.files['file']
        
        pdf_reader = PyPDF2.PdfReader(file)
        num_pages = len(pdf_reader.pages)
        page_text = ""
        for page in pdf_reader.pages:
            page_text += ' '.join(page.extract_text().split())
        #page_text = ' '.join(pdf_reader.pages[current_page - 1].extract_text().split())
        paragraphs = extract_paragraphs(page_text)
        #prompts = split_prompt(paragraphs)
        n = 400
        out = [(paragraphs[0][i:i+n]) for i in range(0, len(paragraphs[0]), n)]
        temp = {'text':out}
        new_df = pd.DataFrame(temp)
        new_df.to_csv('books.csv')
        df = pd.read_csv('books.csv')
        df['embedding'] = df['text'].apply(lambda x: get_embedding(x, engine=EMBEDDING_MODEL))
        df.to_csv('embeddings.csv')
        df.info(show_counts=True)
        ind = [i for i in range(0, len(df))]
        df['embedding_id'] = ind
        df.head()
        # Creates new index
        embeddings = df['embedding']
        embedding_ids = df['embedding_id'].astype("string")
        texts = df['text']
        index.delete(deleteAll='true')
        # Create a list of dictionaries with the data
        index.upsert(list(zip(embedding_ids, embeddings)))
        global content_mapped
        content_mapped = dict(zip(embedding_ids.astype(int),texts))
        query_output = query_article(prompt)

        messages = list(
        map(lambda chunk: {
            "role": "user",
            "content": chunk
        }, query_output))
        question = apply_prompt_template(prompt)
        messages.append({"role": "user", "content": question})
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=2048,
            temperature=0.7,  # High temperature leads to a more creative response.
        )
        #prediction = 'sample'
        prediction = response['choices'][0]['message']['content']
    else:
        prediction = ""
    return render_template ("index.html", output = prediction)

if __name__ == '__main__':
    app.run(debug=True)