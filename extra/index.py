import os
from docx import Document
from langchain.schema import Document as d
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from langchain.vectorstores import pinecone

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def extract_keywords(text):
    # Tokenize the text
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    # Remove single-character tokens
    keywords = [word for word in lemmatized_tokens if len(word) > 1]
    # Merge adjacent digits and words
    merged_keywords = merge_digits_words(keywords)
    return merged_keywords

def merge_digits_words(tokens):
    merged_tokens = []
    i = 0
    while i < len(tokens):
        if i < len(tokens) - 1 and tokens[i].isdigit() and tokens[i + 1].isalpha():
            merged_tokens.append(tokens[i] + ' ' + tokens[i + 1])
            i += 1
        else:
            merged_tokens.append(tokens[i])
        i += 1
    return merged_tokens

def extract_text_with_metadata(docx_file):
    doc = Document(docx_file)
    text_with_metadata = []
    current_heading = []  # Initialize heading levels dynamically
    current_paragraphs = []
    paragraph_metadata = []  # Store metadata for each paragraph

    for paragraph in doc.paragraphs:
        if paragraph.text.strip() and not paragraph.text.startswith('Table'):  # Check if paragraph is not empty
            style_name = paragraph.style.name
            if style_name.startswith('Heading'):
                # Extract the heading level from the style name (e.g., "Heading 1" -> 1)
                heading_level = int(style_name.split()[1])
                
                # Adjust the current_heading list to match the heading level
                current_heading = current_heading[:heading_level - 1]
                current_heading.append(paragraph.text)

                # If there are previous paragraphs, append them with metadata to text_with_metadata
                if current_paragraphs:
                    # Extract keywords from concatenated headings
                    heading_keywords = extract_keywords(' '.join([word for sublist in paragraph_metadata for word in sublist]))
                    text_with_metadata.append(d(
                        page_content='\n'.join(current_paragraphs),
                        metadata={"titles": ', '.join([word for sublist in paragraph_metadata for word in sublist]),
                                  "keywords": ', '.join(heading_keywords)}
                    ))
                    current_paragraphs = []  # Reset current_paragraphs for new heading
                    paragraph_metadata = [current_heading]  # Update metadata for upcoming paragraphs
                else:
                    paragraph_metadata.append(current_heading)  # Store heading for upcoming paragraphs
            else:
                # Append the paragraph text to current_paragraphs
                current_paragraphs.append(paragraph.text)

    # Append any remaining paragraphs after the loop ends
    if current_paragraphs:
        # Extract keywords from concatenated headings
        heading_keywords = extract_keywords(' '.join([word for sublist in paragraph_metadata for word in sublist]))
        text_with_metadata.append(d(
            page_content='\n'.join(current_paragraphs),
            metadata={"titles": ', '.join([word for sublist in paragraph_metadata for word in sublist]),
                      "keywords": ', '.join(heading_keywords)}
        ))

    return text_with_metadata

if __name__ == "__main__":
    docx_file = r"C:\Users\Bohmid\Desktop\prot\advanced RAG\rep.docx"  # Replace with your file path
    text_with_metadata = extract_text_with_metadata(docx_file)
    for i in text_with_metadata:
        print(i)
    from langchain.vectorstores import Chroma
    import google.generativeai as genai

    os.environ["GOOGLE_API_KEY"]="AIzaSyA0IQJyL9MAqHv1KxmffxLCtNV_tOvp1Xs"


    model = ChatGoogleGenerativeAI(
                                    model="gemini-pro", 
                                    temperature=0.5, 
                                    convert_system_message_to_human=True
                                )
        
    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    pinecone.init(
        api_key="1de088f9-831d-4a95-9a8a-ea7c0bd2bd2e",
        environment="gcp-starter",
    )
    vectorstore = pinecone.Pinecone.from_documents(text_with_metadata,GoogleGenerativeAIEmbeddings(model="models/embedding-001"),index_name="jade-chat")
