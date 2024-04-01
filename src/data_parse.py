from unstructured.partition.pdf import partition_pdf
from src.utils import return_splitter, return_retriever, generate_img_summaries
import uuid
from langchain_core.documents import Document
from src.chatbot import Chatbot


def extract_pdf_elements(path, fname):
    """
    Extract images, tables, and chunk text from a PDF file.
    path: File path, which is used to dump images (.jpg)
    fname: File name
    Categorize extracted elements from a PDF into tables and texts.
    raw_pdf_elements: List of unstructured.documents.elements
    """
    raw_pdf_elements = partition_pdf(
        filename=path + fname,
        extract_images_in_pdf=False,
        infer_table_structure=True,
        chunking_strategy="by_title",
        max_characters=4000,
        new_after_n_chars=3800,
        combine_text_under_n_chars=2000,
        image_output_dir_path=path,
    )
    tables = []
    texts = []
    for element in raw_pdf_elements:
        if "unstructured.documents.elements.Table" in str(type(element)):
            tables.append(str(element))
        elif "unstructured.documents.elements.CompositeElement" in str(type(element)):
            texts.append(str(element))
    return texts, tables


def transform_docs(texts):
    text_splitter = return_splitter()
    joined_texts = " ".join(texts)
    texts_4k_token = text_splitter.split_text(joined_texts)
    return texts_4k_token


def add_documents(retriever, doc_summaries, doc_contents):
    doc_ids = [str(uuid.uuid4()) for _ in doc_contents]
    summary_docs = [
        Document(page_content=s, metadata={"doc_id": doc_ids[i]})
        for i, s in enumerate(doc_summaries)
    ]
    retriever.vectorstore.add_documents(summary_docs)
    retriever.docstore.mset(list(zip(doc_ids, doc_contents)))


def ingestion(
        text_summaries, texts, table_summaries, tables, image_summaries, images
):
    """
    Create retriever that indexes summaries, but returns raw images or texts
    """

    # Initialize the storage layer
    retriever = return_retriever()

    if text_summaries:
        add_documents(retriever, text_summaries, texts)
    if table_summaries:
        add_documents(retriever, table_summaries, tables)
    if image_summaries:
        add_documents(retriever, image_summaries, images)


def workable(path: str = "./cj/", fname: str = "cj.pdf"):
    texts, tables = extract_pdf_elements(path, fname)
    transformed_docs = transform_docs(texts=texts)
    chat = Chatbot()
    text_summaries, table_summaries = chat.generate_text_summaries(
        transformed_docs, tables, summarize_texts=True
    )
    img_base64_list, image_summaries = generate_img_summaries(path)

    ingestion(text_summaries=text_summaries, texts=texts, table_summaries=table_summaries, tables=tables,
              image_summaries=image_summaries, images=img_base64_list)

