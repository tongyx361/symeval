# SymEval


<!-- WARNING: THIS FILE WAS AUTOGENERATED! DO NOT EDIT! -->

## Installation

For common users/developers, please just run the following command the
install the package:

``` shell
pip install "git+https://github.com/tongyx361/symeval.git"
```

## Quick Start

``` python
from symeval import *

evaluator = EvaluatorMathBatch()
```

`symeval` provides elaborate answer extraction and correctness judgement
pipelines based on regular expressions and SymPy symbolic calculation,
which is able to correctly process

- most **mathematical objects** such as matrices (vectors), intervals,
  symbols besides numbers,
- as well as some **special texts** like bool expressions, dates and
  times.

[`EvaluatorMath`](https://tongyx361.github.io/symeval/core.html#evaluatormath)
implements an elaborate evaluation pipeline for mathematical reasoning
tasks.

SymPy symbolic calculation causes risks of ex-long evaluation time.

To address this, we implement
[`EvaluatorMathBatch`](https://tongyx361.github.io/symeval/core.html#evaluatormathbatch)
to evaluate in batch with **timeout** but still efficiently.

``` python
test_eq(
    evaluator.batch_eq(ref_answers=["1/2", "1/2"], pred_answers=["0.5", "2/4"]),
    [True] * 2,
)
```

Here we provide a quick start guide. For more details, please refer to
the [API reference](https://tongyx361.github.io/symeval/core.html).

------------------------------------------------------------------------

<a
href="https://github.com/tongyx361/symeval/blob/main/symeval/core.py#LNone"
target="_blank" style="float:right; font-size:smaller">source</a>

### EvaluatorMathBatch

>  EvaluatorMathBatch (strict_extract:bool=True,
>                          include_percentage:bool=True, rel_tol:float=1e-09,
>                          abs_tol:float=1e-08, percent_rel_tol:float=0.001,
>                          ascii_only:bool=True, timeout:int=5, n_procs:int=2,
>                          use_tqdm:bool=True)

*Batch evaluator for math problems, capable of extracting answer segment
from complex resp and processing various mathematical objects
(e.g. fractions, symbolic expressions, matrices, vectors) and special
text (e.g. bool values).*

|  | **Type** | **Default** | **Details** |
|----|----|----|----|
| strict_extract | bool | True |  |
| include_percentage | bool | True | Whether to include percentage comparisons. |
| rel_tol | float | 1e-09 | The relative tolerance for numerical comparisons. |
| abs_tol | float | 1e-08 | The absolute tolerance for numerical comparisons. Necessary for precision issues. |
| percent_rel_tol | float | 0.001 | The absolute tolerance for percentage comparisons. |
| ascii_only | bool | True | Only allowing ASCII characters |
| timeout | int | 5 | The timeout for each evaluation. |
| n_procs | int | 2 |  |
| use_tqdm | bool | True |  |

#### Accurately Extracting Answer Strings

[`EvaluatorMath`](https://tongyx361.github.io/symeval/core.html#evaluatormath)
can:

1.  **extract** short answers from long responses rather **accurately**
2.  and **normalize** into a **mathematical** expression.

``` python
# MATH-style boxed answer
evaluator.extract_ans("Therefore, $1+1=\\boxed{2}$.")
```

``` python
# Answer around "answer"
evaluator.extract_ans(
    "Both $1$ and $11$ divide $11,$ so $\\boxed{11}=2$, and since $1,$ $2,$ $4,$ $5,$ $10,$ and $20$ divide $20,$ then $\\boxed{20}=6$. The inner expression, $\\boxed{11}\\times\\boxed{20}=2\\times6=12$. Finally, $\\boxed{12}=6$ because $1,$ $2,$ $3,$ $4,$ $6,$ and $12$ divide $12.$\n\nTherefore, $6$ is our answer. Please note that we have not boxed the correct answer as we normally do, as that would be especially confusing for this problem."
)
```

``` python
# Use the last number by default
evaluator.extract_ans(
    'First, we need to count the total number of letters in the word "CIRCLE". There are 6 letters.\n\nNext, we need to count the number of distinct letters. There are 6 distinct letters in the word "CIRCLE": C, I, R, L, E, and G.\n\nNow, let\'s consider the arrangements of the distinct letters. The number of ways to arrange n distinct items is n factorial (n!). So, we have 6! = 6 × 5 × 4 × 3 × 2 × 1 = 720 ways to arrange the distinct letters.\n\nHowever, the word "CIRCLE" has one letter that repeats (the letter \'C\' repeats twice). We have over-counted the number of distinct arrangements by including arrangements that are just rotations of each other (for example, "CIRCLE" and "LCIRCE" are considered different arrangements here, but they are the same word when read).\n\nTo correct for this, we divide the total number of arrangements by the number of ways to arrange the repeated letters. The number of ways to arrange 2 identical items is 2! = 2 × 1 = 2. So, we divide the total number of arrangements by 2 to get the correct number of distinct arrangements.\n\nTherefore, the number of ways to arrange the letters of the word "CIRCLE" is 720 ÷ 2 = 360.'
)
# More cases ...
```

``` python
# Normalize fraction
evaluator.extract_ans("The answer is 1/2")
```

``` python
# Normalize pmatrix
evaluator.extract_ans(
    "The answer is \\begin{pmatrix} 3 \\\\ \\frac{\\pi}{2} \\end{pmatrix}"
)
# More cases ...
```

More test cases:

<details class="code-fold">
<summary>Code</summary>

``` python
test_eq(evaluator.norm_ans_str("864 \\mbox{ inches}^2"), "864")
test_eq(evaluator.norm_ans_str("\\frac{270}7\\text{ degrees}"), "\\frac{270}7")
test_eq(evaluator.norm_ans_str(".0000672"), "0.0000672")
```

</details>

#### Correctly Processing Various Mathematical Objects / Special Text

[`EvaluatorMath`](https://tongyx361.github.io/symeval/core.html#evaluatormath),
based on regular expressions and [SymPy](https://www.sympy.org) symbolic
calculation, is able to correctly process

- most **mathematical objects** such as matrices (vectors), intervals,
  symbols besides numbers,
- as well as some **special texts** like bool expressions, dates and
  times.

``` python
evaluator.eq("x+y", "y+x") == True  # Expression
```

``` python
evaluator.eq("\\frac{1}{2}", "0.5") == True  # LaTeX
```

``` python
evaluator.eq(
    "\\begin{array}1\\\\2\\end{array}",
    "1,2",
)  # Matrix (Vector)
```

``` python
evaluator.eq("{1,2}", "{2,1}", compare_sets=True)  # Set
```

``` python
evaluator.eq("no", "false")  # Bool
# More mathematical objects and special texts ...
```

More test cases:

<details class="code-fold">
<summary>Code</summary>

``` python
test_eq(evaluator.eq("251,7\\\\ \\noindent", "0"), False)
test_eq(evaluator.eq("3.54*10^{-7}", "3.54e-07"), True)
test_eq(evaluator.eq(r"\frac{1}{2}", "0.5"), True)
test_eq(evaluator.eq("1", "100"), False)
test_eq(evaluator.eq("100", "1"), False)
test_eq(evaluator.eq("3.04", "0.0304", False), True)
test_eq(evaluator.eq(["0.0304", 0.0304], "3.04"), True)
test_eq(evaluator.eq("x<-1", "x>3"), False)
test_eq(
    evaluator.eq("(-\\infty,0)\\cup(0,\\infty)", "(-\\infty,0)\\cup(0,\\infty)"),
    True,
)
test_eq(evaluator.eq("1+2,2+1", "2+1,1+2"), True)
test_eq(evaluator.eq("5", "5"), True)
test_eq(evaluator.eq("0.1 + 0.2", "0.3"), True)  # `0.1 + 0.2 == 0.3` is `False`
test_eq(evaluator.eq("x + y", "y + x"), True)
test_eq(evaluator.eq("C", "C"), True)
test_eq(evaluator.eq("1,234", "1234"), True)
test_eq(evaluator.eq("12,34", "(12,34)"), True)

test_eq(evaluator.eq("\\$ 5", "5"), True)
test_eq(evaluator.eq("3 * \\sqrt{13}", "3\\sqrt{13}"), True)
test_eq(evaluator.eq("\\pi/2", "\\frac{\\pi}{2}"), True)
test_eq(evaluator.eq("(3,\\pi/2)", "(3,\\frac{\\pi}{2})"), True)
test_eq(evaluator.eq("23000", "\\$23{,}000"), True)
test_eq(evaluator.eq(r"\left(1,2\right)", r"\left(2,1\right)", compare_sets=True), True)
test_eq(evaluator.eq("White", "white"), True)
test_eq(evaluator.eq("[0,3)", "[0,1]"), False)
test_eq(evaluator.eq("[0,1]", "[0,3)"), False)
test_eq(evaluator.eq("1001.5", "1001"), False)
test_eq(evaluator.eq("\\frac{2003}{2}", "1001"), False)
```

</details>

``` python
test_eq(evaluator.eq("-2,1", "1,-2", compare_sets=True), True)
```

#### Normalized Majority Voting

``` python
maj_answers_list, norm_answers_list = evaluator.batch_get_maj_answers(
    [["", "", "1", "2", "2", "3", "3", "3"]]
)
print(f"{maj_answers_list = } <- {norm_answers_list = }")
```

### Parsing LaTeX

#### Interval

``` python
from symeval import latex2sympy_interval
```

``` python
latex2sympy_interval("(-11,-10)\\cup\\{-\\sqrt{110}\\}")
```

``` python
latex2sympy_interval("(-\\infty, 0) \\cup (0, \\infty)")
```

``` python
latex2sympy_interval("(a+b,b]")
```

#### Matrix / Vector

``` python
from symeval import EvaluatorMathBatch

evaluator = EvaluatorMathBatch()
```

``` python
evaluator.latex2matrix(r"\sqrt{400\cos^2(9\pi/44)},\frac{\pi}{4}")
```

``` python
evaluator.latex2matrix(
    r"\begin{pmatrix} \frac{1}{2} & 0 & -\frac{\sqrt{3}}{2} \\ 0 & 1 & 0 \\ \frac{\sqrt{3}}{2} & 0 & \frac{1}{2} \end{pmatrix}"
)
```

``` python
test_eq(
    evaluator.latex2matrix("\\begin{pmatrix}-18\\\\-49\\\\96\\end{pmatrix}"),
    Matrix([[-18, -49, 96]]),
)
test_eq(
    evaluator.latex2matrix("\\begin{pmatrix} 2 & 3 \\\\ 0 & -2 \\end{pmatrix}"),
    Matrix([[2, 3], [0, -2]]),
)
```

### Normalization

``` python
test_eq(evaluator.norm_math_str("251,7\\\\ \\noindent"), "251,7")
```

``` python
test_eq(fix_a_slash_b("(3/4)\\sqrt{3}"), "(\\frac{3}{4})\\sqrt{3}")
```

``` python
test_eq(evaluator.norm_pm("x\\pmy"), "x-y,x+y")
test_eq(evaluator.norm_pm("a\\mpb"), "a-b,a+b")
test_eq(evaluator.norm_pm("1\\pm\\sqrt{19}"), "1-\\sqrt{19},1+\\sqrt{19}")
test_eq(evaluator.norm_pm(r"\{1\pm\sqrt{5},-2\}"), "1-\\sqrt{5},1+\\sqrt{5},-2")
test_eq(
    evaluator.norm_pm("\\(\\frac{1\\pm\\sqrt{17}}{4}\\)"),
    "\\frac{1-\\sqrt{17}}{4},\\frac{1+\\sqrt{17}}{4}",
)
test_eq(
    evaluator.norm_pm(r"\frac{1\pm\sqrt{1-\frac{2}{\sqrt{3}}}}{1}"),
    "\\frac{1-\\sqrt{1-\\frac{2}{\\sqrt{3}}}}{1},\\frac{1+\\sqrt{1-\\frac{2}{\\sqrt{3}}}}{1}",
)
```

``` python
test_eq(norm_deg(r"20^\circ"), r"20")
test_eq(norm_deg(r"\sin 20^\circ"), r"\sin {20*\frac{\pi}{180}}")
```

``` python
test_eq(evaluator.norm_basic_fn(r"sinx"), r"\sin^{1}x")
test_eq(evaluator.norm_basic_fn(r"\sin^2x"), r"\sin^{2}x")
```

### Processing Sets

``` python
test_eq(evaluator.extract_set("{2,1}"), ["1", "2"])
```

``` python
test_eq(is_set("{2,1}"), True)
test_eq(is_set("orange"), False)
test_eq(is_set("x<-1orx>3"), True)
test_eq(is_set("(3/4)sqrt(3)"), False)
```

### Manipulating Strings

``` python
test_eq(evaluator.remove_first_paren_pair("{white}", "{"), "white")
```

## Contribution Guidelines

### Setup

For intended contributors, we recommend installing the package with the
`dev` extras and setting up the pre-commit hooks by running:

``` shell
git clone https://github.com/tongyx361/symeval.git
cd symeval
pip install ".[dev]"
pre-commit install
conda install quarto # For nbdev
```

### File Structure

    symeval
    ├── utils # Repository utilities
    ├── symeval # Package code for common utilities
    └── nbs # Notebooks and other files to run tests and generate documentation with https://nbdev.fast.ai

### Checklist Before Commit

Run the [`prepare-commit.sh`](utils/prepare-commit.sh) to clean the
notebooks and export scripts for pipeline notebooks, generate
documentation, run tests, render README if needed:

    bash utils/prepare-commit.sh
