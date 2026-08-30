#!/usr/bin/env python3
"""
Random Trigonometry Question Generator
=======================================

Generates a randomised set of trigonometry questions spanning four topics
(exact values, equations, identities, and graph features) and writes out a
fully compilable LaTeX (.tex) document, including an answer key.

Every run produces a different question set. The script tracks a "signature"
for every question it creates and will not repeat a question unless the
requested number of questions exceeds the size of the possible question pool
(in which case it will warn you and start reusing signatures).

CONFIGURE THE SCRIPT BY EDITING THE VARIABLES BELOW, or override them from
the command line, e.g.:

    python3 generate_trig_questions.py --num-questions 15 --max-angle-multiple 3

Requires: sympy  (pip install sympy)
"""

import argparse
import random as rd
import warnings
from fractions import Fraction
from warnings import warn

import sympy as sp
from sympy import Rational, cos, latex, pi, sin, tan

# ---------------------------------------------------------------------------
# CONFIGURATION VARIABLES (defaults; can be overridden via CLI flags below)
# ---------------------------------------------------------------------------
NUM_QUESTIONS = 10  # how many questions to generate
MAX_ANGLE_MULTIPLE = 4  # max range for angles used in "exact value" /
# "equation" questions, expressed as a multiple of pi
# e.g. 4 means angles are drawn from [0, 4*pi)
SEED = None  # set an int here for reproducible runs, else None
OUTPUT_FILE = "trig_questions.tex"

# Denominators that yield "nice" (closed-form, textbook-standard) exact trig
# values when combined with pi. True arbitrary rational multiples of pi do
# NOT generally have closed-form exact values (this is a real mathematical
# limitation, not a script limitation), so exact-value/equation questions are
# drawn from this set of denominators, scaled up to MAX_ANGLE_MULTIPLE.
NICE_DENOMINATORS = [1, 2, 3, 4, 6]

MAX_GEN_ATTEMPTS_PER_Q = 300  # retries allowed before accepting a duplicate

TRIG_FUNCS = {"sin": (sin, "\\sin"), "cos": (cos, "\\cos"), "tan": (tan, "\\tan")}
TRIG_FUNC_NAMES = list(TRIG_FUNCS.keys())


# ---------------------------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------------------------
def genprob_exact_val(maxAngle, usedSigs, allowReps=False):
    for _ in range(MAX_GEN_ATTEMPTS_PER_Q):
        # Pick a trig function
        funcName = rd.choice(TRIG_FUNC_NAMES)
        func, funcLatex = TRIG_FUNCS[funcName]

        # Pick a denominator from nice list
        denom = rd.choice(NICE_DENOMINATORS)

        # Find acceptable numerator multiplier within domain
        numMax = denom * maxAngle
        if numMax < 0:
            warn("Maximum domain should be positive")

        # Generate numerator based on multiple in domain
        numer = rd.randint(-numMax, numMax)

        # Generate actual fraction to be used
        frac = Fraction(numer, denom)
        numerSim, denomSim = frac.numerator, frac.denominator

        # Reject duplicates
        signature = (funcName, numerSim, denomSim)
        if not allowReps and signature in usedSigs:
            continue

        # TODO:
        # Multiply fraction to make bigger (additional challenge)
        # numerMult = numer * randint
        # denomMult = denom * randint

        # Generate the angle
        angle = Rational(numerSim, denomSim) * pi
        value = sp.simplify(func(angle))

        # Add to signatures to prevent this one from being generated again
        usedSigs.add(signature)

        # LaTeX formatting
        angleTex = latex(Rational(numer, denom) * pi)
        # angleTex = latex(angle)
        questionText = f"Find the exact value of $\\displaystyle{funcLatex}\\left({angleTex}\\right)$"

        # Make sure answer is "undefined" if undefined
        if not value.is_finite:
            answerText = "Undefined"
        else:
            answerText = f"${latex(value)}$"

        return {
            "question": questionText,
            "answer": answerText,
        }

    # If too many questions where generated
    return None


# ---------------------------------------------------------------------------
# BUILD QUESTIONS
# ---------------------------------------------------------------------------
def build_question_set(nQuestions, maxAngle):
    questions = []
    warnMessages = []
    usedSigs = set()
    poolExhausted = False

    for _ in range(nQuestions):
        q = genprob_exact_val(maxAngle, usedSigs)

        if q is None:
            if not poolExhausted:
                warnMessages.append(
                    "Requested number of questions exceeds pool of unique exact-value questions given the max angle multiple. Repeats will occur..."
                )
                poolExhausted = True
            q = genprob_exact_val(maxAngle, usedSigs, poolExhausted)

        if q is None:
            # No question could be made for this angle
            warnMessages.append(
                "Could not generate a valid exact-value for the given max angle multiple; skipping..."
            )
            continue

        questions.append(q)

    return questions, warnMessages


# ---------------------------------------------------------------------------
# LATEX DOCUMENT ASSEMBLY
# ---------------------------------------------------------------------------

LATEX_PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage[margin=1in]{geometry}
\usepackage{enumitem}

\title{Trigonometry Practice Questions}
\author{}
\date{}

\begin{document}
\maketitle

\section*{Questions}
\begin{enumerate}[label=\arabic*.]
"""

LATEX_MIDDLE = r"""\end{enumerate}

\newpage
\section*{Answer Key}
\begin{enumerate}[label=\arabic*.]
"""

LATEX_END = r"""\end{enumerate}

\end{document}
"""


def build_latex_document(questions, warnMessages):
    parts = [LATEX_PREAMBLE]
    for q in questions:
        parts.append(f"  \\item {q['question']}\n")
        parts.append("  \\vfill\n")
    parts.append(LATEX_MIDDLE)
    for q in questions:
        parts.append(f"  \\item {q['answer']}\n")
    parts.append(LATEX_END)

    if warnings:
        comment_block = "\n".join(f"% WARNING: {w}" for w in warnMessages)
        parts.insert(0, comment_block + "\n")

    return "".join(parts)


# ---------------------------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------------------------


def main():
    # tempSigs = set()
    #
    # for i in range(10):
    #     qSet = genprob_exact_val(MAX_ANGLE_MULTIPLE, tempSigs)
    #
    #     if qSet is not None:
    #         print(f"Generated question {i}: {qSet['question']} -> {qSet['answer']}")

    parser = argparse.ArgumentParser(
        description="Generate randomised trig questions as LaTeX."
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=NUM_QUESTIONS,
        help="Number of questions to generate.",
    )
    parser.add_argument(
        "--max-angle-multiple",
        type=int,
        default=MAX_ANGLE_MULTIPLE,
        help="Max angle range as a multiple of pi (e.g. 4 -> [0, 4*pi)).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed for reproducible output (omit for a new random set each run).",
    )
    parser.add_argument(
        "--output", type=str, default=OUTPUT_FILE, help="Output .tex file path."
    )
    args = parser.parse_args()

    if args.seed is not None:
        rd.seed(args.seed)

    questions, warnings = build_question_set(
        args.num_questions, args.max_angle_multiple
    )

    document = build_latex_document(questions, warnings)

    with open(args.output, "w") as f:
        f.write(document)

    print(f"Wrote {len(questions)} questions to {args.output}")
    for w in warnings:
        print(f"WARNING: {w}")


if __name__ == "__main__":
    main()
