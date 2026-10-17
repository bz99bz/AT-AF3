"""Verified chain / residue / atom correspondence to the ground truth.

The correspondence is built from chemistry and from the *declared* task specification - never
from file order and never from an interaction score.

**Chain level.**  Every chain is reduced to the sequence of its polymer residues' **parent**
chemistry (``DAL`` -> ``ALA``, ``MSE`` -> ``MET``), i.e. a chain is compared through the CCD
components it is made of rather than through the names it happens to be written with.  Model and
GT chains are then paired by one *global, one-to-one* assignment - the maximum of the total
parent-chemistry agreement over every way of pairing the chains - so a GT chain supports at most
one model chain and no model chain can reuse a GT chain to improve its own score.  The single
exception is the **declared target** chain, which claims the model chain it reproduces best before
that assignment runs (see the bullets below); the remaining chains are assigned exactly as above.

* A candidate pair is **compatible** only when the global (Needleman-Wunsch) alignment of the two
  parent sequences has at least :data:`MIN_PAIR_RESIDUES` identical positions *and* those
  positions account for at least :data:`MIN_MODEL_COVERAGE` of the **model** chain
  (``same / n_model``).  The model chain is the denominator on purpose, because that is the
  question the gate answers: does this model chain *as a whole* belong to that reference chain?
  A long model chain that shares only a short window with a short reference chain is then not
  evidence of a correspondence - the model chain takes its better partner and the reference chain
  is reported unmatched instead of being claimed.  A few extra residues (a tag, the unresolved
  ends the models carry) cost coverage but never the pair.
* The primary objective is the total number of identical aligned positions over the assigned pairs -
  the assignment that explains as many of the modelled residues as possible.  Each extra
  identical position outweighs every declared tie-break of the whole matrix, so a nicer label can
  never buy a worse sequence agreement.
* The declared **target** chain - the chain the case configuration names, and therefore the
  reference copy the model has to be scored against - is not a tie-break: it **claims** the model
  chain it reproduces best, before the global assignment decides the remaining chains.  The entries
  that deposit both enantiomers regularly resolve the copy that carries the real D target less
  completely than the L copy (8BFD_2: 30 against 31 residues), and a copy's resolved length is a
  completeness artefact rather than a statement about the model - so no reference copy is ever
  weighed against another reference copy here, only the model chains are, through their parent
  chemistry.  The claim clears the same gate as every other pair, so a target chain the model
  contradicts is reported unmatched instead of being forced onto the model.
* The pairs the global assignment leaves to decide - all pairs of a case without a declared target
  included - are then only ever decided by the **declared** specification at *equal* agreement,
  strongest first: the GT chain is a declared **scope** chain
  (``configs/target_chirality/<case>.json``, cf. ``build_target_scope.py``), then the model and GT
  chain ids are equal.  Those tie-breaks can therefore never force a pair that the chemistry
  contradicts, and where there is no declaration they fall back to the chain ids and finally to the
  sorted chain order, so the result is always reproducible.
* **What is reported.**  Both mapping deliverables name the two sides explicitly -
  ``reference_chain`` (the chain of the reference structure the model is scored against) and
  ``model_chain`` - so no reader has to guess which side a bare ``gt_chain`` column meant.  A
  reference chain that no model chain claimed is not automatically a gap: ``chain_mapping.csv``
  carries ``reference_role`` (``declared_target`` / ``declared_scope`` / ``in_scope_undeclared`` /
  ``outside_declared_scope``), ``required`` and ``selection``, and an entry that deposits both
  enantiomers places the copy the case does *not* declare outside the scope - 8BFD_2 chain A, the
  L enantiomer, is ``outside_declared_scope``, ``required=no`` and must not be counted as a missing
  required chain.
* The **D/L character of a chain is never used to choose its partner** - it is an outcome of the
  comparison, not an input to it - so the uncorrected and the corrected model of a case receive
  the same correspondence whenever their chains are the same molecules.  That is what makes the
  two conditions paired.
* A polymer chain with no compatible partner is left **unmatched** (``ChainMap.matched`` is
  ``False``) instead of being forced onto an unrelated GT chain; the chain is then reported as
  its own outcome.  Chains without polymer residues have no chain-level correspondence at all.

**Residue level.**  A model residue is mapped to the GT residue at the aligned position, keyed by
``(chain, seq, icode)`` - the insertion code is carried through untouched, and a residue whose
insertions make it a non-polymer neighbour is not merged into that neighbour.  An alignment whose score is
tied (only possible between residues of equal parent chemistry, e.g. a poly-alanine stretch) is
resolved deterministically by the trace-back and yields a bijection onto a contiguous block of the
reference chain, so the same model and reference always produce the same residue pairs.

**Atom level.**  Atoms correspond by name; where the CCD atom graph makes two atoms chemically
interchangeable (Asp OD1/OD2, ...) the pair is recorded so the matcher may swap them.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import protocol as P
from .chem import D_TO_L

# ---------------------------------------------------------------------------
#  chain-assignment thresholds (protocol choices, frozen, applied to every method)
# ---------------------------------------------------------------------------
#: A model chain and a GT chain are compatible only when their parent-chemistry sequences agree
#: on at least this many aligned positions.
MIN_PAIR_RESIDUES = 3
#: ... and when those identical positions account for at least this fraction of the **model**
#: chain.  Below it the pairing is not evidence of a correspondence and the chain stays unmatched.
#: The gate is a safety net, not a filter: over the 1456 structures of the frozen canonical corpus
#: every accepted pair clears it (the least covered pair still explains 0.73 of its model chain)
#: and no polymer model chain is left unmatched.
MIN_MODEL_COVERAGE = 0.5


def _polymer_residues(chain) -> list:
    return [r for r in chain.residues if r.is_polymer]


def chain_profile(chain) -> dict:
    """Sequence of parent chem classes, D-character and residue tokens of one chain."""
    residues = _polymer_residues(chain)
    classes = [(r.chem.chem_class or "UNK") for r in residues]
    d_count = sum(1 for r in residues if getattr(r.chem, "is_D", False))
    frac = (d_count / len(residues)) if residues else 0.0
    return {"tokens": [r.token for r in residues], "classes": classes,
            "residues": residues, "d_fraction": frac, "n": len(residues)}


def _align(a: list, b: list):
    """Global alignment of two parent-chemistry sequences; returns aligned index pairs."""
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return []
    MATCH, MISMATCH, GAP = 2, -1, -2
    score = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        score[i][0] = score[i - 1][0] + GAP
    for j in range(1, m + 1):
        score[0][j] = score[0][j - 1] + GAP
    for i in range(1, n + 1):
        ai = a[i - 1]
        for j in range(1, m + 1):
            s = MATCH if ai == b[j - 1] else MISMATCH
            score[i][j] = max(score[i - 1][j - 1] + s,
                              score[i - 1][j] + GAP, score[i][j - 1] + GAP)
    i, j, pairs = n, m, []
    while i > 0 and j > 0:
        s = MATCH if a[i - 1] == b[j - 1] else MISMATCH
        if score[i][j] == score[i - 1][j - 1] + s:
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif score[i][j] == score[i - 1][j] + GAP:
            i -= 1
        else:
            j -= 1
    pairs.reverse()
    return pairs


@dataclass
class ChainMap:
    """One model chain's correspondence - or the record that it has none.

    ``identity`` and ``d_fraction_delta`` are reported for the chosen pair; the assignment itself
    maximises :attr:`same` and is never influenced by the D/L character.  ``reference_chain`` is
    the chain of the *reference* structure (the ground truth) the model chain is scored against;
    it is empty when the model chain was left unmatched.
    """

    model_chain: str
    reference_chain: str
    identity: float                 # identical aligned positions / min(n_model, n_gt)
    aligned: int                    # positions in the global parent-chemistry alignment
    d_fraction_delta: float         # |D fraction(model) - D fraction(GT)| of the chosen pair
    coverage: float = 0.0           # identical / n_model: the compatibility gate, reported
    same: int = 0                   # identical aligned positions: the maximised objective
    matched: bool = True            # False: no compatible GT chain, nothing was forced
    basis: str = ""                 # declared statements the pair satisfies (reported only)


def _pair_evidence(model_profile: dict, gt_profile: dict):
    """``(same, aligned, coverage)`` for one candidate pair, or ``None`` when incompatible."""
    pairs = _align(model_profile["classes"], gt_profile["classes"])
    same = sum(1 for i, j in pairs if model_profile["classes"][i] == gt_profile["classes"][j])
    if same < MIN_PAIR_RESIDUES:
        return None
    coverage = same / model_profile["n"]
    if coverage < MIN_MODEL_COVERAGE:
        return None
    return same, len(pairs), coverage


def _declared_tiers(pairs: int, with_target: bool = False) -> tuple:
    """Weights of the declared tie-break tiers, strongest first.

    The declared **target** chain is normally not one of them: it claims its model chain outright,
    outside the objective (see :func:`map_chains`).  ``with_target=True`` builds the alternative
    scale, in which the declared target is only the strongest tie-break inside the cost
    (``enforce_declared_target=False``).  The tiers are built
    recursively so that one bonus of a tier outweighs *all* bonuses of every weaker tier put
    together, while their sum over all pairs stays below one unit of the objective (the caller
    scales it by ``base``): a tie-break can therefore only decide between pairs that agree equally
    well on the parent chemistry.
    """
    chain_id = 1
    scope = 1 + pairs * chain_id
    if not with_target:
        return scope, chain_id
    return 1 + pairs * scope, scope, chain_id


def _declared_bonus(model_chain: str, reference_chain: str, scope: tuple, tiers: tuple,
                    target_chain: str = "") -> int:
    """Declared-specification bonus of one pair; breaks objective ties, never creates a pair.

    ``tiers`` is the scale of :func:`_declared_tiers`; in the alternative scale it carries a leading
    target tier, and ``target_chain`` then feeds it.
    """
    if len(tiers) == 3:
        target_bonus, scope_bonus, chain_id_bonus = tiers
    else:
        target_bonus, (scope_bonus, chain_id_bonus) = 0, tiers
    bonus = target_bonus if target_chain and reference_chain == target_chain else 0
    bonus += scope_bonus if scope and reference_chain in scope else 0
    if reference_chain == model_chain:
        bonus += chain_id_bonus
    return bonus


# -------------------------------------------------------------------------------------------
#  declared role of a reference chain (reported by chain_mapping_rows, read by the geometry run)
# -------------------------------------------------------------------------------------------
#: the chain the case configuration names as its target: the copy the model has to be scored against
ROLE_DECLARED_TARGET = "declared_target"
#: another chain the case declares in scope
ROLE_DECLARED_SCOPE = "declared_scope"
#: the case declares nothing (the whole curated complex is in scope, Local-D): every chain counts
ROLE_IN_SCOPE_UNDECLARED = "in_scope_undeclared"
#: the case declares a scope and this reference chain is not in it - the L enantiomer or the second
#: copy an All-D entry also deposits; it is not required and not a missing chain
ROLE_OUTSIDE_DECLARED_SCOPE = "outside_declared_scope"
ROLES = (ROLE_DECLARED_TARGET, ROLE_DECLARED_SCOPE, ROLE_IN_SCOPE_UNDECLARED,
         ROLE_OUTSIDE_DECLARED_SCOPE)

#: selection values of ``chain_mapping.csv``
SELECTED = "selected"
UNSELECTED_OUTSIDE_SCOPE = "unselected_outside_declared_scope"
UNSELECTED_REQUIRED = "unselected_required_chain"
UNMATCHED_MODEL_CHAIN = "unmatched_model_chain"


def reference_role(chain_id: str, target_chain: str = "", declared_chains=()) -> str:
    """Declared role of one reference (GT) chain - see :data:`ROLES`.

    A case without any declaration (``target_chain`` empty *and* no ``declared_chains``, the Local-D
    cases whose scope is the whole curated complex) has no chain outside its scope.
    """
    scope = tuple(declared_chains or ())
    if target_chain and chain_id == target_chain:
        return ROLE_DECLARED_TARGET
    if chain_id in scope:
        return ROLE_DECLARED_SCOPE
    if not target_chain and not scope:
        return ROLE_IN_SCOPE_UNDECLARED
    return ROLE_OUTSIDE_DECLARED_SCOPE


def reference_required(chain_id: str, target_chain: str = "", declared_chains=()) -> bool:
    """``False`` only for a reference chain the declared scope leaves out (8BFD_2 chain A)."""
    return reference_role(chain_id, target_chain, declared_chains) != ROLE_OUTSIDE_DECLARED_SCOPE


def _pair_basis(model_chain: str, reference_chain: str, scope: tuple, target_chain: str) -> str:
    """Which declared statements the chosen pair satisfies (reported, never re-ranked)."""
    parts = []
    if target_chain and reference_chain == target_chain:
        parts.append("target_chain")
    if scope and reference_chain in scope:
        parts.append("declared_scope")
    if reference_chain == model_chain:
        parts.append("chain_id")
    return "sequence" + ("+" + "+".join(parts) if parts else "")


def _hungarian_columns(cost: list) -> list:
    """Minimum-cost assignment of a square matrix (Kuhn-Munkres with potentials).

    Returns the column assigned to every row.  The graph is complete, so a solution always
    exists; the algorithm is deterministic for a given matrix, which is all this module needs
    because every tie-break is already part of the cost.
    """
    size = len(cost)
    if not size:
        return []
    inf = 1 + sum(abs(value) for row in cost for value in row)
    u = [0] * (size + 1)                 # row potentials
    v = [0] * (size + 1)                 # column potentials
    match = [0] * (size + 1)             # match[j] = row matched to column j (1-based, 0 = free)
    way = [0] * (size + 1)
    for row in range(1, size + 1):
        match[0] = row
        column = 0
        minv = [inf] * (size + 1)
        used = [False] * (size + 1)
        while True:
            used[column] = True
            here = match[column]
            delta = inf
            next_column = 0
            for j in range(1, size + 1):
                if used[j]:
                    continue
                current = cost[here - 1][j - 1] - u[here] - v[j]
                if current < minv[j]:
                    minv[j] = current
                    way[j] = column
                if minv[j] < delta:
                    delta = minv[j]
                    next_column = j
            for j in range(size + 1):
                if used[j]:
                    u[match[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            column = next_column
            if match[column] == 0:
                break
        while column:
            previous = way[column]
            match[column] = match[previous]
            column = previous
    assignment = [0] * (size + 1)
    for j in range(1, size + 1):
        assignment[match[j]] = j
    return [assignment[row] - 1 for row in range(1, size + 1)]


def _maximum_assignment(weights: list) -> list:
    """Column chosen for every row of a maximum-weight assignment of non-negative weights."""
    if not weights:
        return []
    width = max(len(row) for row in weights)
    size = max(len(weights), width)
    # pad to a square matrix with zero-weight rows/columns: a zero pair is a non-pair, so the
    # padding can neither create nor hide a correspondence
    square = [[0] * size for _ in range(size)]
    for index, row in enumerate(weights):
        square[index][:len(row)] = row
    return _hungarian_columns([[-value for value in row] for row in square])


def map_chains(model, gt, target_chain: str = "", declared_chains=(),
               enforce_declared_target: bool = True) -> list:
    """Pair every model chain with at most one GT chain (see the module docstring).

    ``target_chain`` is the GT chain the case declares as its target (the chain the chirality
    configuration names).  It is authoritative for the *copy*: among the model chains the
    compatibility gate accepts it claims the one that reproduces it best (its own chain id first,
    then the sorted chain ids), and the chain it claimed is out of the global assignment.  The
    declaration names the reference copy the model has to be scored against, and a copy's resolved
    length is not a statement about the model's chemistry, so that claim is not weighed against the
    agreement another copy would offer.

    ``declared_chains`` are all GT chains the case declares in scope; they only break ties between
    the pairs the global assignment leaves to decide.  Both arguments are optional; a polymer model
    chain with no compatible partner comes back with ``matched=False`` and an empty
    ``reference_chain``.

    ``enforce_declared_target=False`` selects the alternative convention: the declared target
    claims nothing and is only the strongest tie-break inside the objective, so a better resolved
    copy of the same molecule can take the model chain away from it (8BFD_2).  The default
    enforces the declaration, and the analysis always uses that default.
    """
    model_profiles = {}
    for chain in model.chains:
        profile = chain_profile(chain)
        if profile["n"]:
            model_profiles[chain.id] = profile
    gt_profiles = {}
    for chain in gt.chains:
        profile = chain_profile(chain)
        if profile["n"]:
            gt_profiles[chain.id] = profile

    # sorted ids, so the result cannot depend on the order the chains appear in the files
    model_ids = sorted(model_profiles)
    gt_ids = sorted(gt_profiles)
    unmatched = [ChainMap(chain_id, "", 0.0, 0, 0.0, matched=False,
                          basis="no_compatible_reference_chain") for chain_id in model_ids]
    if not model_ids or not gt_ids:
        return unmatched

    evidence = {}
    # the declared target chain claims the model chain it reproduces best: the case configuration
    # names the reference copy the model has to be scored against - the entry may deposit both
    # enantiomers, and the copy that carries the real D target is regularly the less completely
    # resolved one (8BFD_2) - so only the *model* chains are compared here, never two reference
    # copies of the same molecule.  The claim clears the same gate as every other pair, so the
    # declaration can still not force a pair the chemistry contradicts.
    claimed = {}                                    # model chain -> GT chain it was claimed for
    if enforce_declared_target and target_chain in gt_profiles:
        best_model, best_rank = "", None
        for model_id in model_ids:                  # sorted: the tie-break stays reproducible
            pair = _pair_evidence(model_profiles[model_id], gt_profiles[target_chain])
            evidence[(model_id, target_chain)] = pair
            if pair is None:
                continue
            rank = (pair[0], 1 if model_id == target_chain else 0)
            if best_rank is None or rank > best_rank:
                best_model, best_rank = model_id, rank
        if best_model:
            claimed[best_model] = target_chain

    # the rest is one global assignment again, over the chains the claim left behind
    remaining_model = [chain_id for chain_id in model_ids if chain_id not in claimed]
    claimed_gt = set(claimed.values())
    remaining_gt = [chain_id for chain_id in gt_ids if chain_id not in claimed_gt]

    pairs_total = len(remaining_model) * len(remaining_gt)
    scope = tuple(declared_chains or ())
    if enforce_declared_target:
        tiers, tie_target = _declared_tiers(pairs_total), ""
    else:
        # the alternative convention: the declaration, target first, is only a tie-break in the cost
        tiers, tie_target = _declared_tiers(pairs_total, with_target=True), target_chain
    # one extra identical aligned position outweighs every declared bonus of the whole matrix
    # put together (see ``_declared_tiers``), so chemistry is decided first and labels only
    # settle pairs that agree equally well
    base = 1 + pairs_total * sum(tiers)
    weights = []
    for model_id in remaining_model:
        row = []
        for gt_id in remaining_gt:
            pair = _pair_evidence(model_profiles[model_id], gt_profiles[gt_id])
            evidence[(model_id, gt_id)] = pair
            if pair is None:
                row.append(0)
                continue
            row.append(pair[0] * base + _declared_bonus(model_id, gt_id, scope, tiers, tie_target))
        weights.append(row)

    assignment = dict(claimed)
    for model_id, column in zip(remaining_model, _maximum_assignment(weights)):
        if column >= len(remaining_gt):
            continue                                # padding column: no real pair
        gt_id = remaining_gt[column]
        if evidence.get((model_id, gt_id)) is not None:
            assignment[model_id] = gt_id

    out = []
    for index, model_id in enumerate(model_ids):
        gt_id = assignment.get(model_id, "")
        pair = evidence.get((model_id, gt_id)) if gt_id else None
        if pair is None:
            out.append(unmatched[index])
            continue
        same, aligned, coverage = pair
        model_profile = model_profiles[model_id]
        gt_profile = gt_profiles[gt_id]
        out.append(ChainMap(
            model_id, gt_id,
            round(same / max(1, min(model_profile["n"], gt_profile["n"])), 4), aligned,
            round(abs(model_profile["d_fraction"] - gt_profile["d_fraction"]), 4),
            coverage=round(coverage, 4), same=same, matched=True,
            basis=_pair_basis(model_id, gt_id, scope, target_chain)))
    return out


def correspondence_summary(gt, chain_maps, target_chain: str = "", declared_chains=()) -> str:
    """One-line, human-readable chain correspondence for the progress log.

    The pairs are written as ``reference <r> <- model <m> (<coverage>)`` - the reference chain
    first, because that is the side the model is scored against.  A reference chain no model chain
    claimed is reported as *unselected* together with its declared role, so a copy the case does not
    declare (8BFD_2 chain A) is readable as "not required" instead of a missing chain.
    """
    matched = [cm for cm in chain_maps if cm.matched]
    parts = [f"reference {cm.reference_chain} <- model {cm.model_chain} ({cm.coverage:.2f})"
             for cm in matched]
    unmatched_model = sorted(cm.model_chain for cm in chain_maps if not cm.matched)
    paired = {cm.reference_chain for cm in matched}
    unselected = sorted(chain.id for chain in gt.chains
                        if _polymer_residues(chain) and chain.id not in paired)
    if unmatched_model:
        parts.append("unmatched model chain(s) " + ",".join(unmatched_model))
    if unselected:
        outside = [chain_id for chain_id in unselected
                   if not reference_required(chain_id, target_chain, declared_chains)]
        text = "unselected reference chain(s) " + ",".join(unselected)
        if outside:
            text += (f" ({','.join(outside)} outside the declared scope: not required, "
                     "not a missing chain)")
        parts.append(text)
    return "; ".join(parts) if parts else "no chain correspondence"


def chain_mapping_rows(gt, chain_maps, target_chain: str = "", declared_chains=()) -> list:
    """``chain_mapping.csv`` rows of one model's correspondence.

    One row per model chain - **including** the chains that could not be paired, so an unmatched
    chain is visible even when absent - plus one row per reference polymer chain that no
    model chain claimed (the extra copies of an All-D entry, a chain the model does not contain).

    The reference side is named ``reference_chain`` (the chain of the reference structure the model
    is scored against) and every reference row carries the declared role it was judged by:
    ``reference_role`` (:data:`ROLES`), ``required`` (``no`` for a chain the declared scope leaves
    out - 8BFD_2 chain A, the L enantiomer the entry also deposits) and ``selection``, so an
    unselected copy that is not part of the task can never be counted as a missing required chain.
    """
    rows = []
    for cm in chain_maps:
        role = P.NA
        required = P.NA
        if cm.matched:
            role = reference_role(cm.reference_chain, target_chain, declared_chains)
            required = "yes" if role != ROLE_OUTSIDE_DECLARED_SCOPE else "no"
        rows.append({
            "reference_chain": cm.reference_chain or P.NA, "model_chain": cm.model_chain,
            "matched": "yes" if cm.matched else "no",
            "assignment_basis": cm.basis,
            "reference_role": role,
            "required": required, "selection": SELECTED if cm.matched else UNMATCHED_MODEL_CHAIN,
            "identical_residues": cm.same, "aligned_residues": cm.aligned,
            "identity": round(cm.identity, 4), "model_coverage": round(cm.coverage, 4),
            "d_fraction_delta": round(cm.d_fraction_delta, 4),
        })
    paired = {cm.reference_chain for cm in chain_maps if cm.matched}
    for chain in gt.chains:
        if chain.id in paired or not _polymer_residues(chain):
            continue
        role = reference_role(chain.id, target_chain, declared_chains)
        required = role != ROLE_OUTSIDE_DECLARED_SCOPE
        rows.append({
            "reference_chain": chain.id, "model_chain": P.NA, "matched": "no",
            "assignment_basis": "no_compatible_model_chain" if required
                                else "outside_declared_scope",
            "reference_role": role, "required": "yes" if required else "no",
            "selection": UNSELECTED_REQUIRED if required else UNSELECTED_OUTSIDE_SCOPE,
            "identical_residues": P.NA, "aligned_residues": P.NA, "identity": P.NA,
            "model_coverage": P.NA, "d_fraction_delta": P.NA,
        })
    return rows


def resname_is_d(resname: str) -> bool:
    return resname.strip().upper() in D_TO_L


def residue_mapping(model, gt, chain_maps, target_chain: str = ""):
    """Return ``(rows, forward, reverse)``.

    ``forward`` maps ``(model_chain, model_token) -> (reference_chain, gt_token)`` and ``reverse`` is the
    inverse.  Insertion codes are part of the token, so ``A:52A`` and ``A:52`` stay distinct.
    """
    rows = []
    forward = {}
    reverse = {}
    gt_chains = {ch.id: ch for ch in gt.chains}
    for cm in chain_maps:
        mch = model.chain(cm.model_chain)
        gch = gt_chains.get(cm.reference_chain)
        if mch is None or gch is None:
            continue
        mp = chain_profile(mch)
        gp = chain_profile(gch)
        pairs = _align(mp["classes"], gp["classes"])
        mapped_gt_tokens = set()
        for i, j in pairs:
            mres = mp["residues"][i]
            gres = gp["residues"][j]
            mapped_gt_tokens.add(gres.token)
            same_class = mres.chem.chem_class == gres.chem.chem_class
            if same_class and mres.resname == gres.resname:
                mtype = "exact"
            elif same_class:
                mtype = "same_parent_different_name"
            else:
                mtype = "chemistry_mismatch"
            forward[(cm.model_chain, mres.token)] = (cm.reference_chain, gres.token)
            reverse[(cm.reference_chain, gres.token)] = (cm.model_chain, mres.token)
            m_heavy = {a.name for a in mres.heavy_atoms()}
            g_heavy = {a.name for a in gres.heavy_atoms()}
            rows.append({
                "reference_chain": cm.reference_chain, "model_chain": cm.model_chain,
                "gt_residue": gres.token, "gt_resseq": gres.seq, "gt_icode": gres.icode,
                "gt_resname": gres.resname, "gt_chem_class": gres.chem.chem_class,
                "gt_is_D": "yes" if gres.chem.is_D else "no",
                "model_residue": mres.token, "model_resseq": mres.seq,
                "model_icode": mres.icode, "model_resname": mres.resname,
                "model_chem_class": mres.chem.chem_class,
                "model_is_D": "yes" if mres.chem.is_D else "no",
                "mapped": "yes", "mapping_type": mtype,
                "gt_supported": "yes" if gres.chem.is_supported else "no",
                "model_supported": "yes" if mres.chem.is_supported else "no",
                "gt_heavy_atoms": len(g_heavy), "model_heavy_atoms": len(m_heavy),
                "heavy_atoms_matched": len(m_heavy & g_heavy),
                "in_domain": "yes" if same_class and mres.chem.is_supported else "no",
                "is_target_chain": "yes" if cm.reference_chain == target_chain else "no",
                "notes": "",
            })
        # GT polymer residues with no aligned model counterpart (outside the evaluable domain)
        for gres in gp["residues"]:
            if gres.token in mapped_gt_tokens:
                continue
            rows.append({
                "reference_chain": cm.reference_chain, "model_chain": cm.model_chain,
                "gt_residue": gres.token, "gt_resseq": gres.seq, "gt_icode": gres.icode,
                "gt_resname": gres.resname, "gt_chem_class": gres.chem.chem_class,
                "gt_is_D": "yes" if gres.chem.is_D else "no",
                "model_residue": P.NA, "model_resseq": P.NA, "model_icode": P.NA,
                "model_resname": P.NA, "model_chem_class": P.NA, "model_is_D": P.NA,
                "mapped": "no", "mapping_type": "gt_only",
                "gt_supported": "yes" if gres.chem.is_supported else "no",
                "model_supported": P.NA, "gt_heavy_atoms": len(gres.heavy_atoms()),
                "model_heavy_atoms": 0, "heavy_atoms_matched": 0, "in_domain": "no",
                "is_target_chain": "yes" if cm.reference_chain == target_chain else "no",
                "notes": "no aligned model residue (insertion/terminus)",
            })
    return rows, forward, reverse


def equivalent_atom_names(resname: str, name: str) -> set:
    """Atom names that are chemically interchangeable with ``name`` in ``resname``."""
    out = {name}
    for group in P.EQUIVALENT_ATOMS.get(resname.strip().upper(), ()):
        if name in group:
            out.update(group)
    return out
