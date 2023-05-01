# gpt_api.py
import openai
import os
import requests
from bs4 import BeautifulSoup

openai.api_key = "YOUR_API_KEY"


class GPTAPI:
    def get_embeddings(self, texts, model="text-davinci-002"):
        prompt = "Generate embeddings for the following sentences:"
        for text in texts:
            prompt += f"\n- {text}"

        response = openai.Completion.create(
            engine=model,
            prompt=prompt,
            max_tokens=3 * len(texts),
            n=1,
            stop=None,
            temperature=0.7,
        )

        embeddings = response.choices[0].text.strip().split("\n")
        embeddings = [list(map(float, embedding.split(','))) for embedding in embeddings]

        return embeddings

    def get_sentiment_score(self, text, model="text-davinci-002"):
        prompt = f"Sentiment score for the following sentence: '{text}'"

        response = openai.Completion.create(
            engine=model,
            prompt=prompt,
            max_tokens=3,
            n=1,
            stop=None,
            temperature=0.7,
        )

        sentiment_score = float(response.choices[0].text.strip())
        return sentiment_score