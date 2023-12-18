import PyPDF2
import gdown
import os
import shutil
import chromadb
import os
# import
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.vector_stores import ChromaVectorStore
from llama_index.storage.storage_context import StorageContext
from llama_index.embeddings import HuggingFaceEmbedding
from IPython.display import Markdown, display
import chromadb

# set up OpenAI
import os
import getpass
import openai
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from decouple import config
from llama_index import ServiceContext, VectorStoreIndex, download_loader, SimpleDirectoryReader

from jinja2 import Template
import requests
from decouple import config
import nltk
import pandas as pd

import ssl
import concurrent.futures
from pathlib import Path


def leer_texto(texto):
  with open(texto, 'rb') as archivo:
      lector = PyPDF2.PdfReader(archivo)
      text = ''
      for i in range(len(lector.pages)):
          pagina = lector.pages[i]
          temp = pagina.extract_text()
          temp = temp.replace('\n', '')
          text += temp
          #text += pagina.extract_text()
  return text


# Link con archivos sobre Seguros
url = 'https://drive.google.com/drive/folders/1lDEjs3wxVKs_3TWA2wUN7V-jTwENUq-V?usp=sharing'
# Descarga carpeta 'Seguros'
gdown.download_folder(url, quiet=True, output='Seguros')
# Crear la carpeta 'data' si no existe
carpeta_destino = 'data'
if not os.path.exists(carpeta_destino):
  os.makedirs(carpeta_destino)
  # Mover todos los archivos de 'Seguros' a 'llamaind
carpeta_origen = 'data'
for filename in os.listdir(carpeta_origen):
  ruta_origen = os.path.join(carpeta_origen, filename)
  ruta_destino = os.path.join(carpeta_destino, filename)
  shutil.move(ruta_origen, ruta_destino)
# Eliminar la carpeta 'Seguros'
shutil.rmtree(carpeta_origen)
print("Archivos movidos con éxito.")

chroma_client = chromadb.Client()
if len(chroma_client.list_collections()) != 0:
  chroma_client.delete_collection(name='Seguros')
collection_seguros = chroma_client.create_collection(name='Seguros')



# Ruta de la carpeta "Seguros"
ruta_seguros = 'Seguros'

# Lista de archivos en la carpeta "Seguros"
archivos_seguros = os.listdir(ruta_seguros)

# Imprimir la lista de archivos
print("Archivos en la carpeta 'Seguros':")

documents = []
metadatas = []
ids = []

id=1

for archivo in archivos_seguros:
    texto= leer_texto('Seguros/'+archivo)
    documents.append(texto)
    metadatas.append({"source": archivo +"info"})
    ids.append(str(id))
    id +=1

collection_seguros.add(
    documents=documents,
    metadatas=metadatas,
    ids=ids
)

results = collection_seguros.query(
    query_texts=["que tipos de seguros hay?"],
    n_results=2
)


try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt')

def zephyr_instruct_template(messages, add_generation_prompt=True):
    # Definir la plantilla Jinja
    template_str = "{% for message in messages %}"
    template_str += "{% if message['role'] == 'user' %}"
    template_str += "{{ message['content'] }}</s>\n"
    template_str += "{% elif message['role'] == 'assistant' %}"
    template_str += "{{ message['content'] }}</s>\n"
    template_str += "{% elif message['role'] == 'system' %}"
    template_str += "{{ message['content'] }}</s>\n"
    template_str += "{% else %}"
    template_str += "{{ message['content'] }}</s>\n"
    template_str += "{% endif %}"
    template_str += "{% endfor %}"
    template_str += "{% if add_generation_prompt %}"
    template_str += "\n"
    template_str += "{% endif %}"
    # Crear un objeto de plantilla con la cadena de plantilla
    template = Template(template_str)
    # Renderizar la plantilla con los mensajes proporcionados
    return template.render(messages=messages, add_generation_prompt=add_generation_prompt)


def generate_answer(prompt: str, max_new_tokens: int = 768) -> str:
    try:
        # Tu clave API de Hugging Face
        api_key = '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'

        # URL de la API de Hugging Face para la generación de texto
        api_url = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
        # Cabeceras para la solicitud
        headers = {"Authorization": f"Bearer {api_key}"}
        # Datos para enviar en la solicitud POST
        # Sobre los parámetros: https://huggingface.co/docs/transformers/main_classes/text_generation
        data = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_new_tokens,
                "temperature": 0.7,
                "top_k": 50,
                "top_p": 0.95
            }
        }
        # Realizamos la solicitud POST
        response = requests.post(api_url, headers=headers, json=data)
        # Extraer respuesta
        respuesta = response.json()[0]["generated_text"][len(prompt):]
        return respuesta
    except Exception as e:
        print(f"An error occurred: {e}")
        return ""


def prepare_prompt(query_str: str, nodes: list):
    TEXT_QA_PROMPT_TMPL = (
        "La información de contexto es la siguiente:\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Dada la información de contexto anterior, y sin utilizar conocimiento previo, responde la siguiente pregunta.\n"
        "Pregunta: {query_str}\n"
        "Respuesta: "
    )

    # Construimos el contexto de la pregunta
    context_str = ''
    for node in nodes:
        # Usamos get para obtener la clave, y si no está presente, proporcionamos un valor predeterminado
        page_label = node.metadata.get("page_label", "No Page Label")
        file_path = node.metadata.get("file_path", "No File Path")
        context_str += f"\npage_label: {page_label}\n"
        context_str += f"file_path: {file_path}\n\n"
        context_str += f"{node.text}\n"

    messages = [
        {
            "role": "system",
            "content": "Eres un asistente útil que siempre responde con respuestas veraces, útiles y basadas en hechos.",
        },
        {"role": "user", "content": TEXT_QA_PROMPT_TMPL.format(context_str=context_str, query_str=query_str)},
    ]
    final_prompt = zephyr_instruct_template(messages)
    return final_prompt


# Cargamos nuestro modelo de embeddings
print('Cargando modelo de embeddings...')
embed_model = HuggingFaceEmbeddings(model_name='sentence-transformers/paraphrase-multilingual-mpnet-base-v2')


# Construimos un índice de documentos a partir del archivo CSV con delimitador "|"
print('Indexando documentos...')

#documents = SimpleDirectoryReader("Seguros").load_data()
index = VectorStoreIndex.from_documents(documents, show_progress=True, service_context=ServiceContext.from_defaults(embed_model=embed_model, llm=None))




def embed_and_index(document_path):
    # Embed the document
    embedding = embed_model.embed(document_path)

    # Get the filename without extension as the document ID
    document_id = Path(document_path).stem

    # Index the document
    index.index(document_id, embedding)

# Use ThreadPoolExecutor for parallel processing
with concurrent.futures.ThreadPoolExecutor() as executor:
    # Iterate over documents and parallelize the embedding and indexing process
    futures = [executor.submit(embed_and_index, document) for document in documents]

    # Wait for all threads to complete
    concurrent.futures.wait(futures)

# Now the index is populated with embeddings of all documents


retriever = index.as_retriever(similarity_top_k=2)

# Realizando llamada a HuggingFace para generar respuestas...
queries = ['¿Que es el seguro de cristales?']
for query_str in queries:
    # Traemos los documentos más relevantes para la consulta
    nodes = retriever.retrieve(query_str)
    final_prompt = prepare_prompt(query_str, nodes)
    print('Pregunta:', query_str)
    print('Respuesta:')
    print(generate_answer(final_prompt))

