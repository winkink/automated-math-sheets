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

from sympy import (
    Eq,
    Rational,
    Symbol,
    acos,
    asin,
    atan,
    cos,
    expand,
    latex,
    pi,
    simplify,
    sin,
    sqrt,
    tan,
)

# ---------------------------------------------------------------------------
# CONFIGURATION VARIABLES (defaults; can be overridden via CLI flags below)
# ---------------------------------------------------------------------------
NUM_QUESTIONS = 10  # how many questions to generate
MAX_ANGLE_MULTIPLE = 4  # max range for angles used in "exact value" /
# "equation" questions, expressed as a multiple of pi
# e.g. 4 means angles are drawn from [0, 4*pi)
SEED = None  # set an int here for reproducible runs, else None
OUTPUT_FILE = "trig_questions.tex"

# EXACT SOLUTIONS QUESTIONS
# Denominators that yield "nice" (closed-form, textbook-standard) exact trig
# values when combined with pi. True arbitrary rational multiples of pi do
# NOT generally have closed-form exact values (this is a real mathematical
# limitation, not a script limitation), so exact-value/equation questions are
# drawn from this set of denominators, scaled up to MAX_ANGLE_MULTIPLE.
NICE_DENOMINATORS = [1, 2, 3, 4, 6]
MAX_GEN_ATTEMPTS_PER_Q = 300  # retries allowed before accepting a duplicate

# GENERAL SOLUTION QUESTIONS
MAX_COEFF_MULTIPLE = (
    3  # The maximum value the coefficient in front of the function can take
)
MAX_FACTOR_MULTIPLE = (
    5  # The maximum value the expression can be multiplied by on both sides
)

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------

TRIG_FUNCS = {"sin": (sin, "\\sin"), "cos": (cos, "\\cos"), "tan": (tan, "\\tan")}
TRIG_FUNC_NAMES = list(TRIG_FUNCS.keys())

# General solution specific constants
TRIG_FUNCS_GENSOL = {
    "sin": (
        sin,
        "\\sin",
        [0, Rational(1, 2), sqrt(2) / 2, sqrt(3) / 2, 1],
    ),
    "cos": (
        cos,
        "\\cos",
        [0, Rational(1, 2), sqrt(2) / 2, sqrt(3) / 2, 1],
    ),
    "tan": (
        tan,
        "\\tan",
        [0, sqrt(3) / 3, 1, sqrt(3)],
    ),
}

# Symbols reused by the "general solution" question generator: x is the
# unknown being solved for, n is the free integer parameter in the answer.
X = Symbol("x")
N = Symbol("n", integer=True)


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
        signature = ("exact", funcName, numerSim, denomSim)
        if not allowReps and signature in usedSigs:
            continue

        # Add to signatures to prevent this one from being generated again
        usedSigs.add(signature)

        # TODO:
        # Multiply fraction to make bigger (additional challenge)
        # numerMult = numer * randint
        # denomMult = denom * randint

        # Generate the angle
        angle = Rational(numerSim, denomSim) * pi
        value = simplify(func(angle))

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


def genprob_gen_solution(
    maxCoeffMultiple, maxFactorMultiple, usedSigs, allowReps=False
):
    for _ in range(MAX_GEN_ATTEMPTS_PER_Q):
        # Pick a trig function
        funcName = rd.choice(TRIG_FUNC_NAMES)
        func, _, possAns = TRIG_FUNCS_GENSOL[funcName]

        # Pick a final solution to equate to
        solSide = rd.choice(possAns)

        # Multiply coeff with random factor (keep positive for now)
        coeffFactor = rd.randint(1, maxCoeffMultiple)

        # Multiply whole expression by multiple to make more difficult
        extMultFactor = rd.randint(1, maxFactorMultiple)

        # Generate signature to reject duplicates
        signature = ("general", funcName, coeffFactor, solSide)
        if not allowReps and signature in usedSigs:
            continue

        # Add to signatures to prevent this one from being generated again
        usedSigs.add(signature)

        # Find reference angle (use sympy arc* functions)
        if funcName == "sin":
            refAngle = simplify(asin(solSide))
            answerText = f"$\\displaystyle x = {latex(expand((refAngle + 2 * pi * N) / coeffFactor))} \\text{{ or }} {latex(simplify((2 * N + 1) * pi / coeffFactor))} - {latex(simplify(refAngle / coeffFactor))}, n \\in \\mathbb{{Z}}$"
        elif funcName == "cos":
            refAngle = simplify(acos(solSide))
            answerText = f"$\\displaystyle x = {latex(simplify((2 * pi * N) / coeffFactor))} \\pm {latex(simplify(refAngle / coeffFactor))}, n \\in \\mathbb{{Z}}$"
        else:
            refAngle = simplify(atan(solSide))
            answerText = f"$\\displaystyle x = {latex(simplify((pi * N) / coeffFactor))} + {latex(simplify(refAngle / coeffFactor))}, n \\in \\mathbb{{Z}}$"

        # LaTeX formatting
        questionText = f"Find the general solution for: $\\displaystyle {latex(Eq(extMultFactor * func(coeffFactor * X), extMultFactor * solSide))}$"

        return {
            "question": questionText,
            "answer": answerText,
        }

    # If too many questions where generated
    return None


# ---------------------------------------------------------------------------
# BUILD QUESTIONS
# ---------------------------------------------------------------------------
def build_question_set(nQuestions, maxAngle, maxCoeffMultiple, maxFactorMultiple):
    questions = []
    warnMessages = []
    usedSigs = set()
    poolExhausted = False

    for _ in range(nQuestions):
        if rd.randint(0, 1) == 1:
            questionType = "genSolution"
            q = genprob_gen_solution(maxCoeffMultiple, maxFactorMultiple, usedSigs)
        else:
            questionType = "exactVal"
            q = genprob_exact_val(maxAngle, usedSigs)

        if q is None:
            if not poolExhausted:
                warnMessages.append(
                    "Requested number of questions exceeds pool of unique exact-value questions given the max angle multiple. Repeats will occur..."
                )
                poolExhausted = True
            if questionType == "genSolution":
                q = genprob_gen_solution(
                    maxCoeffMultiple, maxFactorMultiple, usedSigs, poolExhausted
                )
            elif questionType == "exactVal":
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

\begin{document}
\begin{center}
    {\LARGE\bfseries Exact Trigonometric Values}\\[0.5em]
    Give all answers in exact form. All angles are in radians.
\end{center}


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
        "--max-coeff-multiple",
        type=int,
        default=MAX_COEFF_MULTIPLE,
        help="Max coefficient value for general solution questions",
    )
    parser.add_argument(
        "--max-factor-multiple",
        type=int,
        default=MAX_FACTOR_MULTIPLE,
        help="Maximum factor to multiply both sides of expression by for general solution questions",
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
        args.num_questions,
        args.max_angle_multiple,
        args.max_coeff_multiple,
        args.max_factor_multiple,
    )

    document = build_latex_document(questions, warnings)

    with open(args.output, "w") as f:
        f.write(document)

    print(f"Wrote {len(questions)} questions to {args.output}")
    for w in warnings:
        print(f"WARNING: {w}")


if __name__ == "__main__":
    main()
