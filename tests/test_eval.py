from evaluation.evaluer_rag import construire_rapport, rouge_l_f1


def test_rouge_l_f1_borne() -> None:
    score = rouge_l_f1("retour sous 30 jours", "vous pouvez faire un retour sous 30 jours")
    assert 0 <= score <= 1


def test_construire_rapport_contient_metriques() -> None:
    rapport = construire_rapport()
    assert "recall_at_3" in rapport
    assert "rouge_l_moyen" in rapport
    assert rapport["nb_questions"] > 0

