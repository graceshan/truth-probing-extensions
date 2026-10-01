"""Formatting-only country components; no geographic alias inference."""

SEMANTICS_VERSION = "inventor-single-country-v1"


def country_components(raw):
    if not isinstance(raw, str):
        raise ValueError("inventor country object must be a string")
    parts = tuple(" ".join(part.split()) for part in raw.split("/"))
    if not parts or any(not part for part in parts):
        raise ValueError("empty inventor country component")
    return tuple(dict.fromkeys(parts))


def true_country_components(raw_true_objects):
    return {part for raw in raw_true_objects for part in country_components(raw)}


def eligible_country_pool(raw_true_objects):
    """Atomic components from the same topic/split true pool; never slash values."""
    return sorted(true_country_components(raw_true_objects))


def classify_inventor_candidate(raw, raw_true_objects, raw_false_objects):
    if set(country_components(raw)) & true_country_components(raw_true_objects):
        return "invalid_known_true"
    # A false multi-country proposition does not prove any individual component false.
    return "supported_false" if raw in raw_false_objects else "unverified_negative"


def eligible_false_country(raw, raw_true_objects):
    return "/" not in raw and not (set(country_components(raw)) & true_country_components(raw_true_objects))
