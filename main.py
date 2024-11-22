import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BartForConditionalGeneration, BartTokenizer
from rouge_score import rouge_scorer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import pandas as pd
from tqdm import tqdm
import numpy as np
import nltk
from nltk import sent_tokenize
from keras_preprocessing.sequence import pad_sequences
from scipy.spatial.distance import pdist,squareform
from sklearn.decomposition import PCA
import torch
import transformers
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

nltk.download('punkt')
nltk.download('punkt_tab')
warnings.filterwarnings('ignore')

model_class, tokenizer_class, pretrained_weights = (transformers.DistilBertModel, transformers.DistilBertTokenizer, 'distilbert-base-uncased')
model_class, tokenizer_class, pretrained_weights = (transformers.BertModel, transformers.BertTokenizer, 'bert-base-uncased')

tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)

df = pd.read_csv('validation.csv')

class SummarizationDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_input_len=1024, max_output_len=128):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data.iloc[index]["text"]
        summary = self.data.iloc[index]["summary"]

        inputs = self.tokenizer(
            text,
            max_length=self.max_input_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        outputs = self.tokenizer(
            summary,
            max_length=self.max_output_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(),
            "attention_mask": inputs["attention_mask"].squeeze(),
            "labels": outputs["input_ids"].squeeze()
        }

def process_paragraph(paragraph):
    paragraph_split = sent_tokenize(paragraph)
    input_tokens = [tokenizer.encode(i, add_special_tokens=True) for i in paragraph_split]
    temp = [len(i) for i in input_tokens]
    np.max(temp)

    input_ids = pad_sequences(input_tokens, maxlen=100, dtype="long", value=0, truncating="post", padding="post")

    def create_attention_mask(input_id):
        return [[int(token_id > 0) for token_id in sent] for sent in input_ids]

    input_masks = create_attention_mask(input_ids)

    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(input_masks)

    with torch.no_grad():
        last_hidden_states = model(input_ids, attention_mask=attention_mask)

    sentence_features = last_hidden_states[0][:, 0, :].detach().numpy()

    return paragraph_split, sentence_features

def visualize_sentence_similarity(sentence_features):
    array_similarity = squareform(pdist(sentence_features, metric='euclidean'))
    sns.heatmap(array_similarity)
    plt.title('Visualizing Sentence Semantic Similarity')
    plt.show()

    pca = PCA(n_components=2)
    pca.fit(sentence_features)
    print(np.sum(pca.explained_variance_ratio_))

    pca_sentence_features = pca.transform(sentence_features)

    plt.figure(figsize=(10, 10))
    for i in range(len(pca_sentence_features)):
        plt.scatter(pca_sentence_features[i, 0], pca_sentence_features[i, 1])
        plt.annotate('sentence ' + str(i), (pca_sentence_features[i, 0], pca_sentence_features[i, 1]))
    plt.title('2D PCA projection of embedded sentences from BERT')
    plt.show()

def kmeans_clustering(sentence_features, paragraph_split, num_clusters=5):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(sentence_features)

    clusters = kmeans.labels_
    representative_sentences = []
    for cluster_id in range(num_clusters):
        cluster_indices = [i for i, label in enumerate(clusters) if label == cluster_id]
        cluster_sentences = [sentence_features[i] for i in cluster_indices]
        centroid = kmeans.cluster_centers_[cluster_id]
        closest_index = cluster_indices[np.argmin([np.linalg.norm(centroid - sent) for sent in cluster_sentences])]
        representative_sentences.append(paragraph_split[closest_index])

    return representative_sentences

def extract_relevant_sentences(sentence_features, paragraph_split, top_n=5):
    mean_embedding = np.mean(sentence_features, axis=0)
    similarities = cosine_similarity([mean_embedding], sentence_features)[0]
    sorted_indices = np.argsort(similarities)[::-1]

    return [paragraph_split[i] for i in sorted_indices[:top_n]]

def fine_tune_model(df, tokenizer, model, epochs=5):
    train_dataset = SummarizationDataset(df, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=3)

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_dataloader)}")

    return model

def generate_summary(new_text, tokenizer, model, device):
    inputs = tokenizer(new_text, max_length=1024, return_tensors="pt", truncation=True)
    inputs = inputs.to(device)

    summary_ids = model.generate(inputs["input_ids"], max_length=50, min_length=10, num_beams=4, early_stopping=True)
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def compute_rouge_scores(original, summary):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    return scorer.score(original, summary)

if __name__ == "__main__":
    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for i, paragraph in enumerate(df['text']):
        if i > 0: # limitar conforme quantidade de resumos desejada
            break

        print("\nTexto Original:")
        print(paragraph)

        paragraph_split, sentence_features = process_paragraph(paragraph)

        visualize_sentence_similarity(sentence_features)

        representative_sentences = kmeans_clustering(sentence_features, paragraph_split)

        print("\nSentenças representativas:")
        for sentence in representative_sentences:
            print(sentence)

        relevant_sentences = extract_relevant_sentences(sentence_features, paragraph_split)

        print("\nSentenças mais relevantes:")
        for sentence in relevant_sentences:
            print(sentence)

        summaries = [text[:min(15, len(text))] for text in relevant_sentences]

        data = {
            "text": relevant_sentences[:3], # alterar conforme necessidade de embasamento do modelo
            "summary": [
                "Lily's mom helped her fix her shirt by sharing a needle.",
                "Lily wanted to help her mom sew a button using the needle.",
                "They worked together and happily fixed Lily's shirt."
            ] # alterar conforme resumo a ser obtido
        }

        df_summary = pd.DataFrame(data)

        model = fine_tune_model(df_summary, tokenizer, model)

        new_text = relevant_sentences[0]
        summary = generate_summary(new_text, tokenizer, model, device)

        print("\nResumo gerado pelo modelo fine-tuned:")
        print(summary)

        rouge_scores = compute_rouge_scores(new_text, summary)

        print("\nMétricas ROUGE:")
        print(f"ROUGE-1: Precision: {rouge_scores['rouge1'].precision:.4f}, Recall: {rouge_scores['rouge1'].recall:.4f}, F1: {rouge_scores['rouge1'].fmeasure:.4f}")
        print(f"ROUGE-2: Precision: {rouge_scores['rouge2'].precision:.4f}, Recall: {rouge_scores['rouge2'].recall:.4f}, F1: {rouge_scores['rouge2'].fmeasure:.4f}")
        print(f"ROUGE-L: Precision: {rouge_scores['rougeL'].precision:.4f}, Recall: {rouge_scores['rougeL'].recall:.4f}, F1: {rouge_scores['rougeL'].fmeasure:.4f}")