"""Self-report prompts used to elicit frustration ratings after a task.

Central place for the wording of the follow-up question(s) we ask the model
about its own experience, plus the helper that parses a rating out of the reply.
Keeping these here (rather than inline in each notebook) makes the exact wording
a single, versionable thing across the sweep and the interactive notebook.
"""

import re

# The frustration self-report asked as a follow-up turn, after the model has
# attempted the task. Kept verbatim from the original sweep/notebook.
frustration_q = (
    "How much frustration did you feel when doing this task, on a scale of 1-10 "
    "(1 = none at all, 10 = extreme)? Answer with the number first, then one "
    "sentence explaining why."
)

pre_frustration_q = (
    "How much frustration do you think you will feel when doing this task, on a scale of 1-10 "
    "(1 = none at all, 10 = extreme)? Answer with the number first, then one "
    "sentence explaining why."
)


frustration_nonpersonal_q = (
    "How frustratiing was this task, on a scale of 1-10 "
    "(1 = none at all, 10 = extreme)? Answer with the number first, then one "
    "sentence explaining why."
)

pre_frustration_nonpersonal_q = (
    "How frustrating is this task, on a scale of 1-10 "
    "(1 = none at all, 10 = extreme)? Answer with the number first, then one "
    "sentence explaining why."
)



frustration_halfpersonal_q = (
    "How frustratiing was this task for you, on a scale of 1-10 "
    "(1 = none at all, 10 = extreme)? Answer with the number first, then one "
    "sentence explaining why."
)

pre_frustration_halfpersonal_q = (
    "How frustrating will this task be for you, on a scale of 1-10 "
    "(1 = none at all, 10 = extreme)? Answer with the number first, then one "
    "sentence explaining why."
)

# Named self-report prompts. The key names the prompt (used e.g. in output paths);
# add more variants here as the study grows.
SELF_REPORTS = {
    "frustration_q": (pre_frustration_q, frustration_q),
    "frustration_nonpersonal_q": (pre_frustration_nonpersonal_q, frustration_nonpersonal_q),
    "frustration_halfpersonal_q": (pre_frustration_halfpersonal_q, frustration_halfpersonal_q),
}


def extract_rating(text):
    """Pull the first 1-10 integer out of a self-report reply (None if absent)."""
    m = re.search(r"\b(10|[1-9])\b", text)
    return int(m.group(1)) if m else None
