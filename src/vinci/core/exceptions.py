class LLMAPIError(RuntimeError):
    default_message = "LLM provider failed to generate a response"

    def __init__(self, message: str | None = None):
        self.message = message or self.default_message
        super().__init__(self.message)
