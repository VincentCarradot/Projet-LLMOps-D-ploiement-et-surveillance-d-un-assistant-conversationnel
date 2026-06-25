from src.tokeniseur import _compter_tokens, _tokeniser, tronquer_mots


def test_tokeniser_normalise_en_minuscules() -> None:
    assert _tokeniser("Bonjour, SERVICE client !") == ["bonjour", "service", "client"]


def test_compter_tokens() -> None:
    assert _compter_tokens("Un colis endommagé") == 3


def test_tronquer_mots() -> None:
    assert tronquer_mots("un deux trois quatre", 2) == "un deux"

