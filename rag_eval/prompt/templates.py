class PromptTemplate:
    """
    Args:
        `variables` (list[str]): a list of the variable names used in the template,
        `template` (str): The template string with {variable} names.
    """

    def __init__(self, variables=None, template=None):
        self.variables = variables
        self.template = template

    def format(self, input_variables):
        """
        Returns the prompt using the `input_variables` in the form of {"query": "text", ...} to a string
        """
        return self.template.format(**input_variables)

    def get_template(self):
        return self.template


class PassageTemplate:
    """
    Args:
        `variables` (list[str]): a list of the variable names used in the template,
        `template` (str): The template string with {variable} for passage
    """

    def __init__(
        self, variables=["title", "text"], template="- Title: {title}\n{text}\n\n"
    ):
        self.variables = variables
        self.template = template

    def serialize_passages(self, passages):
        """
        Serializes the `passages` in the form of [{"context": "text"}, ...] to a string
        """
        return "".join(
            [self.template.format(**passage) for passage in passages]
        ).strip()


class HistoryTemplate:
    """
    Args:
        `templates` (dict{str: str}): The templates dictionary of 'speaker': 'speaker template'.
    """

    def __init__(self, templates={"Human": "User: {}\n", "Assistant": "Agent: {}\n"}):
        self.templates = templates

    def format_utterance(self, statement, speaker):
        assert speaker in self.templates, "{} is not a valid speaker.".format(speaker)
        return self.templates[speaker].format(statement)

    def serialize_history(self, history, max_history=10):
        """
        Serializes the `history` in the form of [{"speaker": "agent", "utterance": "text"}, ...] to a string
        """
        # remove from middle
        while len(history) > max_history:
            mid_point = len(history) // 2

            if mid_point % 2 == 0:
                history = history[: mid_point - 2] + history[mid_point:]
            else:
                history = history[: mid_point - 1] + history[mid_point + 1 :]

        return "".join(
            [
                self.format_utterance(context["utterance"], context["speaker"])
                for context in history
            ]
        ).strip()


class LLMEvalTemplate:
    """
    Args:
        `templates` (dict{str: str}): The templates dictionary of 'speaker': 'speaker template'.
    """

    def __init__(self, templates={"Human": "User: {}\n", "Assistant": "Agent: {}\n"}):
        self.templates = templates


class QAPromptTemplate(PromptTemplate):
    def __init__(self):
        self.variables = ["query", "retrieved_passages"]
        self.template = "Please answer the following question given the following passages:\n{retrieved_passages}\nQuestion: {query}\nAnswer: "

        self.passage_template = PassageTemplate()

    def __call__(self, sample, passages):
        serialized_passages = self.passage_template.serialize_passages(passages)
        prompt = self.format(
            {"query": sample.question, "retrieved_passages": serialized_passages}
        )
        return prompt


class QAUnaswerablePromptTemplate(QAPromptTemplate):
    def __init__(self):
        self.variables = ["query", "retrieved_passages"]
        self.template = 'Please answer the following question given the following passage. If the answer is not in the passage or cannot be inferred from the passage, respond as "I don\'t know".\n{retrieved_passages}\nQuestion: {query}\nAnswer: '

        self.passage_template = PassageTemplate()


class ConvQAPromptTemplate(PromptTemplate):
    def __init__(self):
        self.variables = ["query", "retrieved_passages", "history"]
        self.template = "Please answer the following question given the following passages and the conversation history:\n\n{retrieved_passages}\n\n{history}\nUser: {query}\nAgent: "

        self.history_template = HistoryTemplate()
        self.passage_template = PassageTemplate()

    def __call__(self, sample, passages):
        serialized_passages = self.passage_template.serialize_passages(passages)
        serialized_history = self.history_template.serialize_history(sample.context)
        prompt = self.format(
            {
                "query": sample.question,
                "retrieved_passages": serialized_passages,
                "history": serialized_history,
            }
        )
        return prompt


class ConvQAUnaswerablePromptTemplate(ConvQAPromptTemplate):
    def __init__(self):
        self.variables = ["query", "retrieved_passages", "history"]
        self.template = 'Please answer the following question given the following passage and the conversation history. If the answer is not in the passage or cannot be infered from the passage, respond as "I don\'t know".\n\n{retrieved_passages}\n\n{history}\nUser: {query}\nAgent: '

        self.history_template = HistoryTemplate()
        self.passage_template = PassageTemplate()
