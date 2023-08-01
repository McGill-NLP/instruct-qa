from rag_eval.generation import OPT, Llama, Flan, GPTx, PipelineGenerator, StarChatGenerator, Vicuna, FalconPipelineGenerator


def load_model(model_name, **kwargs):
    """
    Loads model by name.

    Args:
        model_name (str): Name of model to load.
        kwargs: Additional parameters for the generator (e.g., temperature).

    Returns:
        BaseGenerator: Generator object.
    """
    if "opt" in model_name:
        model_cls = OPT
    elif "dolly" in model_name or "h2ogpt" in model_name:
        model_cls = PipelineGenerator
    elif any(model_type in model_name for model_type in ["alpaca"]):
        model_cls = Llama
    elif "vicuna" in model_name:
        model_cls = Vicuna
    elif "davinci" in model_name or "gpt" in model_name:
        model_cls = GPTx
    elif "flan" in model_name:
        model_cls = Flan
    elif "falcon" in model_name:
        model_cls = FalconPipelineGenerator
    elif "starchat" in model_name:
        model_cls = StarChatGenerator
    else:
        raise NotImplementedError(f"Model {model_name} not supported.")

    return model_cls(model_name, **kwargs)
