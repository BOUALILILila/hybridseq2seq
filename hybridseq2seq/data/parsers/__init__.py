from typing import Optional

from .syntactic_parser import SyntacticParser
from .gold_syntax_parser import GoldSyntaxParser

from ...utils import get_logger

logger = get_logger(__name__)

_SYNTAX_PARSERS = {
    "gold": GoldSyntaxParser,
    "predicted": SyntacticParser,
}


def get_syntax_parser(parser_type: Optional[str] = None):
    parser_type = "predicted" if parser_type is None else parser_type
    if parser_type in _SYNTAX_PARSERS:
        cls = _SYNTAX_PARSERS[parser_type]
        logger.info(f"Using syntax parser {parser_type}")
        return cls(), parser_type
    else:
        raise KeyError(
            f"Unknown {parser_type} syntax parser class. Use one of {_SYNTAX_PARSERS.keys()}"
        )
