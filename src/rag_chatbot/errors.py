"""Domain exceptions."""


class RagError(Exception):
    """Base class for application errors."""


class PdfExtractionError(RagError):
    """The PDF could not be read or contained no extractable text."""


class OllamaError(RagError):
    """The Ollama endpoint was unreachable or returned an error."""


class DocumentNotFoundError(RagError):
    pass


class DocumentBusyError(RagError):
    """The document is still being ingested."""
