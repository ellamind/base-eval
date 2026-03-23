"""Filter functions for German Grammar Fluency benchmark subtasks."""


def filter_adjective_ending(dataset):
    """Filter for ADJECTIVE_ENDING_ERROR."""
    return dataset.filter(lambda doc: doc.get("error_type") == "ADJECTIVE_ENDING_ERROR")


def filter_als_wie(dataset):
    """Filter for ALS_WIE_ERROR."""
    return dataset.filter(lambda doc: doc.get("error_type") == "ALS_WIE_ERROR")


def filter_article_case(dataset):
    """Filter for ARTICLE_CASE_ERROR."""
    return dataset.filter(lambda doc: doc.get("error_type") == "ARTICLE_CASE_ERROR")


def filter_article_gender(dataset):
    """Filter for ARTICLE_GENDER_ERROR."""
    return dataset.filter(lambda doc: doc.get("error_type") == "ARTICLE_GENDER_ERROR")


def filter_capitalization(dataset):
    """Filter for CAPITALIZATION_ERROR."""
    return dataset.filter(lambda doc: doc.get("error_type") == "CAPITALIZATION_ERROR")


def filter_collocation(dataset):
    """Filter for COLLOCATION_FLUENCY."""
    return dataset.filter(lambda doc: doc.get("error_type") == "COLLOCATION_FLUENCY")


def filter_connector(dataset):
    """Filter for CONNECTOR_FLUENCY."""
    return dataset.filter(lambda doc: doc.get("error_type") == "CONNECTOR_FLUENCY")


def filter_dass_das(dataset):
    """Filter for DASS_DAS_ERROR."""
    return dataset.filter(lambda doc: doc.get("error_type") == "DASS_DAS_ERROR")


def filter_discourse(dataset):
    """Filter for DISCOURSE_FLUENCY."""
    return dataset.filter(lambda doc: doc.get("error_type") == "DISCOURSE_FLUENCY")


def filter_punctuation(dataset):
    """Filter for GERMAN_PUNCTUATION_ERROR."""
    return dataset.filter(lambda doc: doc.get("error_type") == "GERMAN_PUNCTUATION_ERROR")


def filter_konjunktiv(dataset):
    """Filter for KONJUNKTIV_ERROR."""
    return dataset.filter(lambda doc: doc.get("error_type") == "KONJUNKTIV_ERROR")


def filter_naturalness(dataset):
    """Filter for NATURALNESS_FLUENCY."""
    return dataset.filter(lambda doc: doc.get("error_type") == "NATURALNESS_FLUENCY")


def filter_n_deklination(dataset):
    """Filter for N_DEKLINATION_ERROR."""
    return dataset.filter(lambda doc: doc.get("error_type") == "N_DEKLINATION_ERROR")


def filter_perfect_aux(dataset):
    """Filter for PERFECT_AUX_PARTICIPLE_ERROR."""
    return dataset.filter(lambda doc: doc.get("error_type") == "PERFECT_AUX_PARTICIPLE_ERROR")


def filter_pronoun_case(dataset):
    """Filter for PRONOUN_CASE_ERROR."""
    return dataset.filter(lambda doc: doc.get("error_type") == "PRONOUN_CASE_ERROR")


def filter_prose_quality(dataset):
    """Filter for PROSE_QUALITY_FLUENCY."""
    return dataset.filter(lambda doc: doc.get("error_type") == "PROSE_QUALITY_FLUENCY")


def filter_register(dataset):
    """Filter for REGISTER_ERROR."""
    return dataset.filter(lambda doc: doc.get("error_type") == "REGISTER_ERROR")


def filter_seit_seid(dataset):
    """Filter for SEIT_SEID_ERROR."""
    return dataset.filter(lambda doc: doc.get("error_type") == "SEIT_SEID_ERROR")


def filter_ss_eszett(dataset):
    """Filter for SS_ESZETT_ERROR."""
    return dataset.filter(lambda doc: doc.get("error_type") == "SS_ESZETT_ERROR")


def filter_verb_conjugation(dataset):
    """Filter for VERB_CONJUGATION_ERROR."""
    return dataset.filter(lambda doc: doc.get("error_type") == "VERB_CONJUGATION_ERROR")


def filter_word_order_nebensatz(dataset):
    """Filter for WORD_ORDER_NEBENSATZ_ERROR."""
    return dataset.filter(lambda doc: doc.get("error_type") == "WORD_ORDER_NEBENSATZ_ERROR")


def filter_word_order_v2(dataset):
    """Filter for WORD_ORDER_V2_ERROR."""
    return dataset.filter(lambda doc: doc.get("error_type") == "WORD_ORDER_V2_ERROR")
