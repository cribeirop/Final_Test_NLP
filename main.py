import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BartForConditionalGeneration, BartTokenizer
from rouge_score import rouge_scorer
import pandas as pd
from tqdm import tqdm
import numpy as np
import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk import sent_tokenize

from keras_preprocessing.sequence import pad_sequences
from scipy.spatial.distance import pdist,squareform
from sklearn.decomposition import PCA

import torch
import transformers as ppb
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

model_class, tokenizer_class, pretrained_weights = (ppb.DistilBertModel, ppb.DistilBertTokenizer, 'distilbert-base-uncased')

model_class, tokenizer_class, pretrained_weights = (ppb.BertModel, ppb.BertTokenizer, 'bert-base-uncased')

tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights)
df = pd.read_csv('tinystories.csv')

paragraphs = [paragraph for paragraph in df['text']]

for paragraph in paragraphs:
    paragraph_split = sent_tokenize(paragraph)

    len(paragraph_split)

    input_tokens = []
    for i in paragraph_split:
        input_tokens.append(tokenizer.encode(i, add_special_tokens=True))

    temp = []
    for i in input_tokens:

        temp.append(len(i))
    np.max(temp)  

    input_ids = pad_sequences(input_tokens, maxlen=100, dtype="long", value=0, truncating="post", padding="post")

    def create_attention_mask(input_id):
        attention_masks = []
        for sent in input_ids:
            att_mask = [int(token_id > 0) for token_id in sent]
            attention_masks.append(att_mask)  
        return attention_masks

    input_masks = create_attention_mask(input_ids)

    input_ids = torch.tensor(input_ids)  
    attention_mask = torch.tensor(input_masks)

    params = list(model.named_parameters())

    print('The BERT model has {:} different named parameters.\n'.format(len(params)))

    print('==== Embedding Layer ====\n')

    for p in params[0:5]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    print('\n==== First Transformer ====\n')

    for p in params[5:21]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    print('\n==== Output Layer ====\n')

    for p in params[-4:]:
        print("{:<55} {:>12}".format(p[0], str(tuple(p[1].size()))))

    with torch.no_grad():
        last_hidden_states = model(input_ids, attention_mask=attention_mask)

    sentence_features = last_hidden_states[0][:,0,:].detach().numpy()

    print(sentence_features.shape)

    array_similarity = squareform(pdist(sentence_features, metric='euclidean'))

    sns.heatmap(array_similarity)
    plt.title('visualizing sentence semantic similarity')

    pca = PCA(n_components=2)
    pca.fit(sentence_features)
    print(np.sum(pca.explained_variance_ratio_))

    pca_sentence_features = pca.transform(sentence_features)

    plt.figure(figsize=(10,10))
    for i in range(len(pca_sentence_features)):
        plt.scatter(pca_sentence_features[i,0],pca_sentence_features[i,1])
        plt.annotate('sentence '+ str(i),(pca_sentence_features[i,0],pca_sentence_features[i,1]))
    plt.title('2D PCA projection of embedded sentences from BERT')

    from sklearn.cluster import KMeans

    num_clusters = 5
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

    print("Sentenças representativas:")
    for sentence in representative_sentences:
        print(sentence)

    from sklearn.metrics.pairwise import cosine_similarity

    mean_embedding = np.mean(sentence_features, axis=0)

    similarities = cosine_similarity([mean_embedding], sentence_features)[0]

    sorted_indices = np.argsort(similarities)[::-1]

    top_n = 5
    relevant_sentences = [paragraph_split[i] for i in sorted_indices[:top_n]]

    print("Sentenças mais relevantes:")
    for sentence in relevant_sentences:
        print(sentence)

    from sklearn.neighbors import NearestNeighbors
    import textwrap

    wrapper = textwrap.TextWrapper(width=70)

    sentence_embeddings = sentence_features
    paragraph_split = paragraph_split      

    number_extract = 5

    nearest_neighbors = NearestNeighbors(n_neighbors=number_extract, metric='cosine')
    nearest_neighbors.fit(sentence_embeddings)

    mean_embedding = np.mean(sentence_embeddings, axis=0).reshape(1, -1)
    distances, indices = nearest_neighbors.kneighbors(mean_embedding)

    relevant_sentences_nn = [paragraph_split[i] for i in indices.flatten()]

    print("\nSentenças mais relevantes usando Nearest Neighbors:")
    for idx, sentence in enumerate(relevant_sentences_nn, 1):
        print(f"{idx}: {sentence}")

    texts = relevant_sentences

    summaries = [text[:min(15, len(text))] for text in texts]

    data = {
        "text": texts,
        "summary": summaries
    }

    df = pd.DataFrame(data)

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

    model_name = "facebook/bart-large-cnn"
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)

    train_dataset = SummarizationDataset(df, tokenizer)
    train_dataloader = DataLoader(train_dataset, batch_size=2)

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    epochs = 5
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
        
            optimizer.zero_grad()
    
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
        
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_dataloader)}")

    model.save_pretrained("./bart-fine-tuned")

    new_text = relevant_sentences[0]

    inputs = tokenizer(new_text, max_length=1024, return_tensors="pt", truncation=True)
    inputs = inputs.to(device)

    summary_ids = model.generate(inputs["input_ids"], max_length=50, min_length=10, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    scores = scorer.score(new_text, summary)

    print('\nTexto Original:')
    print(paragraph)
    print("\nResumo gerado pelo modelo fine-tuned:")
    print(summary)
    print("\nMétricas ROUGE:")
    print(f"ROUGE-1: Precision: {scores['rouge1'].precision:.4f}, Recall: {scores['rouge1'].recall:.4f}, F1: {scores['rouge1'].fmeasure:.4f}")
    print(f"ROUGE-2: Precision: {scores['rouge2'].precision:.4f}, Recall: {scores['rouge2'].recall:.4f}, F1: {scores['rouge2'].fmeasure:.4f}")
    print(f"ROUGE-L: Precision: {scores['rougeL'].precision:.4f}, Recall: {scores['rougeL'].recall:.4f}, F1: {scores['rougeL'].fmeasure:.4f}")
