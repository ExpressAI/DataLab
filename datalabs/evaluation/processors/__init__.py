# flake8: noqa
# when a new processor is implemented, remember to import it here so it gets registered
from . import text_classification
from .processor_registry import get_processor
