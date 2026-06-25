from src.modele import generer_texte


def test_generation_fallback_extrait_reponse_du_contexte() -> None:
    prompt = "Contexte:\nQuestion: X\nReponse: Le retour dure 30 jours.\nQuestion: Y"
    assert generer_texte(prompt, modele_llm=None) == "Le retour dure 30 jours."


def test_generation_callable() -> None:
    def modele(prompt: str, max_new_tokens: int) -> str:
        del prompt, max_new_tokens
        return "ok"

    assert generer_texte("prompt", modele_llm=modele) == "ok"

