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

    def __init__(self, variables=None, template="- {context}\n"):
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


class StarChatTemplate:
    def __init__(self) -> None:
        self.template = (
            "<|system|>\n{system_prompt}\n<|end|>\n{conversation_history}\n<|assistant|>"
        )
        self.conversation_turn_template_map = {
            "Human": "<|user|>\n{}<|end|>\n",
            "Assistant": "<|assistant|>\n{}<|end|>\n",
        }

    def format_utterance(self, statement, speaker):
        assert speaker in self.conversation_turn_template_map, "{} is not a valid speaker.".format(speaker)
        return self.conversation_turn_template_map[speaker].format(statement)

    def serialize_history(self, system_prompt, history, max_history=10):
        # remove from middle
        while len(history) > max_history:
            mid_point = len(history) // 2

            if mid_point % 2 == 0:
                history = history[: mid_point - 2] + history[mid_point:]
            else:
                history = history[: mid_point - 1] + history[mid_point + 1 :]

        conv_history = "".join(
            [
                self.format_utterance(context["utterance"], context["speaker"])
                for context in history
            ]
        ).strip()

        return self.template.format(
            system_prompt=system_prompt, conversation_history=conv_history
        )


class LLMEvalTemplate:
    """
    Args:
        `templates` (dict{str: str}): The templates dictionary of 'speaker': 'speaker template'.
    """

    def __init__(self, templates={"Human": "User: {}\n", "Assistant": "Agent: {}\n"}):
        self.templates = templates
