from rag_eval.prompt import QAPromptTemplate, QAUnaswerablePromptTemplate, ConvQAPromptTemplate, ConvQAUnaswerablePromptTemplate

def load_template(template_name):
    """
    Loads template by name.

    Args:
        template_name (str): Name of template to load.

    Returns:
        PromptTemplate: Template object.
    """
    template_mapping = {
        "qa": QAPromptTemplate,
        "qa_unanswerable": QAUnaswerablePromptTemplate,
        "conv_qa": ConvQAPromptTemplate,
        "conv_qa_unanswerable": ConvQAUnaswerablePromptTemplate,
    }

    if template_name not in template_mapping:
        raise ValueError(f"{template_name} is not a valid template.")

    return template_mapping[template_name]()