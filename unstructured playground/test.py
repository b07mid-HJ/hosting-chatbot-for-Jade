from docx import Document
from xml.etree.ElementTree import Element, SubElement, tostring

def table_to_xml(table):
    root = Element('table')
    for row in table:
        row_element = SubElement(root, 'row')
        for cell in row:
            cell_element = SubElement(row_element, 'cell')
            cell_element.text = cell
    return root

def extract_tables_from_docx(docx_file):
    doc = Document(docx_file)
    langchain_documents = []

    for table in doc.tables:
        # Find the index of the current table within all tables in the document
        table_index = 0
        for i, t in enumerate(doc.tables):
            if t is table:
                table_index = i
                break

        # Find the caption (the paragraph just before the table)
        if table_index > 0:
            caption_paragraph = doc.paragraphs[table_index - 1]
            caption = caption_paragraph.text.strip()

            # Extracting content of the table
            content = []
            for row in table.rows:
                row_content = [cell.text.strip() for cell in row.cells]
                content.append(row_content)

            # Convert table content to XML format
            xml_content = table_to_xml(content)

            # Creating langchain document with metadata as title and content as table
            langchain_document = {
                "metadata": {"title": caption},
                "content": tostring(xml_content, encoding='unicode')
            }
            langchain_documents.append(langchain_document)

    return langchain_documents


# Example usage:
docx_file_path = "./multi+parent/rep.docx"  # Path to your .docx file
langchain_docs = extract_tables_from_docx(docx_file_path)
for doc in langchain_docs:
    print("Title:", doc["metadata"]["title"])
    print("Content (XML format):")
    print(doc["content"])
    print()
