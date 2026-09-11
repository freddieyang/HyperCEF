"""Helpers for role-aware and unordered reaction queries."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def _sorted_unique_ints(values) -> list[int]:
    out = []
    seen = set()
    for value in values or []:
        try:
            idx = int(value)
        except (TypeError, ValueError):
            continue
        if idx < 0 or idx in seen:
            continue
        seen.add(idx)
        out.append(idx)
    return sorted(out)


def make_role_aware_query(substrates, products, reaction_id=None, coefficients=None) -> dict:
    """Create a reaction query that preserves substrate/product roles.

    Existing models can still consume this object through
    :func:`reaction_metabolites`, which exposes the unordered metabolite union.
    """
    substrates = _sorted_unique_ints(substrates)
    products = _sorted_unique_ints(products)
    query = {
        "metabolites": _sorted_unique_ints([*substrates, *products]),
        "substrates": substrates,
        "products": products,
    }
    if reaction_id is not None:
        query["reaction_id"] = str(reaction_id)
    if coefficients is not None:
        query["coefficients"] = coefficients
    return query


def reaction_metabolites(reaction) -> list[int]:
    """Return the sorted unordered metabolite ids for list or role-aware queries."""
    if isinstance(reaction, Mapping):
        if "metabolites" in reaction:
            return _sorted_unique_ints(reaction["metabolites"])
        if "mets" in reaction:
            return _sorted_unique_ints(reaction["mets"])
        return _sorted_unique_ints([*reaction.get("substrates", []), *reaction.get("products", [])])
    if isinstance(reaction, Sequence) and not isinstance(reaction, (str, bytes)):
        return _sorted_unique_ints(reaction)
    return []


def reaction_sides(reaction) -> tuple[list[int], list[int]]:
    """Return substrate and product ids.

    Plain list reactions are treated as unordered participant sets, so both
    returned sides contain the metabolite union.
    """
    if isinstance(reaction, Mapping):
        substrates = _sorted_unique_ints(reaction.get("substrates", reaction.get("reactants", [])))
        products = _sorted_unique_ints(reaction.get("products", []))
        if substrates or products:
            if not substrates:
                substrates = reaction_metabolites(reaction)
            if not products:
                products = reaction_metabolites(reaction)
            return substrates, products
    mets = reaction_metabolites(reaction)
    return mets, mets
