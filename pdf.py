import os
from llama_index.legacy import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.legacy.readers import PDFReader


def get_index(data, index_name):  # get index or create it if it does not exist
    index = None
    if not os.path.exists(index_name):
        print("building index", index_name)
        index = VectorStoreIndex.from_documents(data, show_progress=True)  # create new index
    else:  # index exists
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=index_name)
        )

    return index


pdf_path = os.path.join("data", "Canada.pdf")
canada_pdf = PDFReader().load_data(file=pdf_path)
canada_index = get_index(canada_pdf,"canada")
canada_engine = canada_index.as_query_engine()

