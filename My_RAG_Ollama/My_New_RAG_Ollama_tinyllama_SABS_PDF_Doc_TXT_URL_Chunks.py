import ollama
import chromadb
##from pypdf import PdfReader
import PyPDF2
import fitz
import cv2
##import docx
from spire.doc import *
from spire.doc.common import *
##from exceptions import PendingDeprecationWarning
import os
import re

import requests
import lxml.html

########################################################################################################################

My_Joined_Text = []

########################################################################################################################
#     FOR WORD .doc FILES

### Create a Document object
##document = Document()
### Load a Word document
##document.LoadFromFile("C:\\Python_Env\\AYEN\\SABS_SANS10142_ED1.docx")
##document.LoadFromFile("C:\\Python_Env\\AYEN\\Regulation_Very_Short.doc")
##
### Extract the text of the document
##document_text = document.GetText()
##print (f"document_text: {document_text}")
##
##My_Joined_Text = document_text
##My_Joined_Text = My_Joined_Text
##print(f"My_Joined_Text : {My_Joined_Text}")
##
##
##documents = [My_Joined_Text]
##print(f"documents : {documents}")


########################################################################################################################
#     FOR PDF 1 x FILES


My_Document_Pdf = open('C:/Python_Env/AYEN/SABS_SANS10142_ED1.pdf', 'rb') 
##My_Document_Pdf = open('C:/Python_Env/AYEN/Electrical_Installation_Regulations.pdf', 'rb')
pdfReader = PyPDF2.PdfReader(My_Document_Pdf)
Num_Pages = len(pdfReader.pages)

print(f"Num_Pages : {Num_Pages}")

for i in range(Num_Pages):
  Page = pdfReader.pages[i]
  My_Extracted_Text = Page.extract_text()
  print(f"Page Number : {i}")
##  print(f"My_Extracted_Text : {My_Extracted_Text}")

  My_Joined_Text.append(My_Extracted_Text)
##  print(f"My_Joined_Text : {My_Joined_Text}")

documents = My_Joined_Text
print(f"documents : {documents}")



########################################################################################################################
#     FOR MORE 1 PDF FILES

##My_Document_Pdf = open('C:/Python_Env/AYEN/Testing/SABS_SANS10142_ED1.pdf', 'rb')
My_Document_Pdf = open('C:/Python_Env/AYEN/Testing/Electrical_Installation_Regulations_Shorter.pdf', 'rb')

pdfReader = PyPDF2.PdfReader(My_Document_Pdf)
Num_Pages = len(pdfReader.pages)

print(f"Num_Pages : {Num_Pages}")

for i in range(Num_Pages):
    Page = pdfReader.pages[i]
    My_Extracted_Text = Page.extract_text()
    print(f"Page Number : {i}")
    print(f"My_Extracted_Text : {My_Extracted_Text}")

    My_Joined_Text.append(My_Extracted_Text)
    print(f"My_Joined_Text : {My_Joined_Text}")


    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text=str(My_Joined_Text))



    documents = chunks
    print(f"documents : {documents}")







########################################################################################################################
#     FOR TEXT .txt FILES

### Create a Document object
##document = Document()
##
### Load a TEXT document
##document.LoadFromFile("C:\\Python_Env\\AYEN\\SABS_SANS10142_ED1.txt")
####document.LoadFromFile("C:\\Python_Env\\AYEN\\Test_Regulations_Test.txt")
##
### Extract the text of the document
##document_text = document.GetText()
##print (f"document_text: {document_text}")
##
##My_Joined_Text = document_text
##My_Joined_Text = My_Joined_Text
##print(f"My_Joined_Text : {My_Joined_Text}")
##
##
##documents = [My_Joined_Text]
##print(f"documents : {documents}")
##

########################################################################################################################
#     FOR WEB PAGE SCRAPING (URL's)

##My_URL = ''
##
### Create a Document object
##document = Document()
##
##dom = lxml.html.fromstring(requests.get('http/www.?.com').content)
##page_list = [x for x in dom.xpath('//td/text()')]
##print(f"page_list: {page_list}")


#######################################################################################################################
#         GETTING EMBEDDINGS FOR LLM's

client = chromadb.Client()
##client = chromadb.HttpClient(host='localhost', port=8000)

collection = client.create_collection(name="docs")
##collection = client.get_or_create_collection(name="docs") # Get a collection object from an existing collection, by name. If it doesn't exist, create it.

# store each document in a vector embedding database
for i, d in enumerate(documents):
  response = ollama.embeddings(model="mxbai-embed-large", prompt=d)
  embedding = response["embedding"]
  collection.add(
    ids=[str(i)],
    embeddings=[embedding],
    documents=[d]
  )


########################################################################################################################

def main():

  Input_From_User = input("ASK AI ANYTHING : ")
##  cv2.waitkey(0)

  prompt = Input_From_User

  # generate an embedding for the prompt and retrieve the most relevant doc
  response = ollama.embeddings(
    prompt=prompt,
    model="mxbai-embed-large"
  )
  results = collection.query(
    query_embeddings=[response["embedding"]],
    n_results=1
  )
  data = results['documents'][0][0]

  # generate a response combining the prompt and data we retrieved in step 2
  output = ollama.generate(
    model="tinyllama",
    prompt=f"Using this data: {data}. Respond to this prompt: {prompt}",
    stream=False
  )

  print(output['response'])

  main()


if __name__ == "__main__":
  main()
