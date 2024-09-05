# Import libraries

import faiss
from transformers import BertTokenizer, TFBertModel
import pdfplumber
import numpy as np

########
with pdfplumber.open('Data/Biohacking.pdf') as pdf:
    full_text = ""
    passages = []
    passage_with_page = []

    # Parcourir chaque page et extraire le texte
    for page_num in range(len(pdf.pages)):
        page = pdf.pages[page_num]
        text = page.extract_text()

        # Vérification si le texte est bien extrait
        if text:
            print(f"Texte extrait de la page {page_num + 1}")

            # Diviser le texte en paragraphes
            paragraphs = text.split("\n\n")
            for paragraph in paragraphs:
                if paragraph.strip():  # Ignorer les paragraphes vides
                    passages.append(paragraph)
                    # Associer avec le numéro de page
                    passage_with_page.append((paragraph, page_num + 1))

            # Ajouter le texte extrait au texte complet
            full_text += text + "\n\n"
        else:
            print(f"Impossible d'extraire le texte de la page {page_num + 1}")


# Sauvegarde du texte extrait dans un fichier
with open("output_text.txt", "w", encoding="utf-8") as text_file:
    text_file.write(full_text)


# Indexation
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = TFBertModel.from_pretrained('bert-base-uncased')


# Vectoriser le texte
def vectorize_text(text):
    """Vectorise chaque passage du texte

    Args:
        text (str): pdf transform via pdfplumber

    Returns:
        numpy: renvoie le vecteur associé au passage du texte. 
    """
    # Tokenisation du texte
    inputs = tokenizer(text, return_tensors="tf",
                       max_length=512, truncation=True, padding=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Génération du vecteur
    outputs = bert_model(input_ids, attention_mask=attention_mask)

    # Utiliser la sortie du [CLS] token comme représentation du texte
    # La première position correspond au token [CLS]
    cls_output = outputs.last_hidden_state[:, 0, :]
    return cls_output.numpy()


vectors = []
for passage, page_num in passage_with_page:
    vector = vectorize_text(passage)
    vectors.append((vector, page_num))


# Conversion des vacetrus en tableau numpy pour l'indexation future
vectors_array = np.array([v[0] for v in vectors])
pages_array = np.array([v[1] for v in vectors])

print(
    f"Vectorisation terminée. Nombre de vecteurs générés : {vectors_array.shape[0]}")

# Sauvegarde des vecteurs et numéros de pages pour l'indexation future
np.save("vectors.npy", vectors_array)
np.save("pages.npy", pages_array)

print("Vecteurs et numéros de pages sauvegardés.")


# Charger les vecteurs et les numéros de pages
vectors = np.load('vectors.npy')
pages = np.load('pages.npy')
vectors = np.squeeze(vectors)

dimension = vectors.shape[1]

# Création de l'index

index = faiss.IndexFlatL2(dimension)
index.add(vectors)
print(f"Nombre de vecteurs indexés : {index.ntotal}")

# Sauvegarde de l'index

faiss.write_index(index, "faiss_index.bin")
print("Index FAISS sauvegardé avec succès")
