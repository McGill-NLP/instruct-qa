from instruct_qa.collections.dpr_wiki_collection import DPRWikiCollection
from instruct_qa.collections.hotpot_wiki_collection import HotpotWikiCollection
from instruct_qa.collections.topiocqa_wiki_collection import TopiocqaWikiCollection
from instruct_qa.collections.faithdial_collection import FaithDialCollection


def load_collection(document_collection_name, **kwargs):
    """
    Loads a document collection.

    Args:
        document_collection_name (str): The name of the document collection to load.
        kwargs: Additional parameters for the document collection e.g., cachedir, file_name.

    Returns:
        PassageCollection: The loaded document collection.
    """
    document_collection_mapping = {
        "dpr_wiki_collection": DPRWikiCollection,
        "topiocqa_wiki_collection": TopiocqaWikiCollection,
        "hotpot_wiki_collection": HotpotWikiCollection,
        "faithdial_collection": FaithDialCollection,
    }

    if document_collection_name not in document_collection_mapping:
        raise NotImplementedError(
            f"Document collection {document_collection_name} not supported."
        )

    return document_collection_mapping[document_collection_name](name=document_collection_name, **kwargs)
