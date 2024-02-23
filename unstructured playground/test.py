from docx import Document
from docx.text.paragraph import Paragraph
from docx.document import Document
from docx.table import _Cell, Table
from docx.oxml.text.paragraph import CT_P
from docx.oxml.table import CT_Tbl
import docx

def iter_block_items(parent):
    if isinstance(parent, Document):
        parent_elm = parent.element.body
    elif isinstance(parent, _Cell):
        parent_elm = parent._tc
    else:
        raise ValueError("something's not right")

    for child in parent_elm.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, parent)
        elif isinstance(child, CT_Tbl):
            table = Table(child, parent)
            for row in table.rows:
                for cell in row.cells:
                    yield from iter_block_items(cell)

def find_tables_with_titles(docx_file):
    document = docx.Document(docx_file)
    tables_with_titles = []
    document_elements = list(iter_block_items(document))

    for i, block_item in enumerate(document_elements):
        if isinstance(block_item, Paragraph) and block_item.style.name == 'Caption':
            print(block_item.text)
            for next_block_item in document_elements[i+1:]:
                if isinstance(next_block_item, Table):
                    print(next_block_item)
                    tables_with_titles.append((block_item.text, next_block_item))
                    break

    return tables_with_titles

docx_file_path = "./multi+parent/rep.docx"  # Replace with your .docx file path
tables_with_titles = find_tables_with_titles(docx_file_path)
for title, table in tables_with_titles:
    print("Title:", title)
    for row in table.rows:
        for cell in row.cells:
            for paragraph in cell.paragraphs:
                print(paragraph.text)
        print()
    print("=" * 20)
