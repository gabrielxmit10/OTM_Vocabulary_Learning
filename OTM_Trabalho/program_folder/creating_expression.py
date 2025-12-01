#!/usr/bin/env python
# coding: utf-8

# ### Importing and Configurations
# (config file variables and variables from the discover python files)

# In[18]:


# Importing Cell

# For simplifying expressions
from sympy import symbols, Eq, simplify
from sympy.logic import simplify_logic

# For analyzing time of some things 
# (for now just simplifying time)
import time


# For loading toml config files and 
# importing from other files (in LEXIMIZER)
import tomllib
from pathlib import Path
import sys

# For making deepcopy sometimes
import copy


# Add parent directory to sys.path
# for importing discover_words_to_learn
parent_dir = Path().resolve().parent
sys.path.insert(0, str(parent_dir))



# For z3
from z3 import Optimize, Solver, Int, Bool, If, Sum, Distinct, sat, unsat, And, Or, Not, unknown, is_true, is_false, Z3Exception
from z3 import simplify as z3_simplify




# In[19]:


# Load Config file variables (from config.toml)

# Load config.toml

if '__file__' in globals():
    # Running as imported module
    config_path = Path(__file__).parent.parent / "config.toml"
else:
    # Running as notebook
    config_path = Path("../config.toml")

with config_path.open("rb") as f:
    config = tomllib.load(f)
    
# config_path = Path("../config.toml")
# with config_path.open("rb") as f:
#     config = tomllib.load(f)


n_books = config["general"]["n_books"] # V
n_books_to_read = config["general"]["n_books_to_read"] # V
language = config["general"]["language"]
# vocab_known = config["general"]["vocab_known"]

books_path = config["paths"]["books_path"] # V # Used for listing book files, which will be replaced
# vocab_path = config["paths"]["vocab_path"] # V
output_path = config["paths"]["output_path"]

solver_time_limit = config["restrictions"]["solver_time_limit"]


# Directory containing config.toml
config_dir = config_path.parent
# Helper to resolve paths relative to config.toml
def resolve_config_path(raw_path):
    p = Path(raw_path)
    return (config_dir / p).resolve() if not p.is_absolute() else p.resolve()

books_dir = resolve_config_path(books_path) # Used for listing book files, which will be replaced
# vocab_dir = resolve_config_path(vocab_path)
# output_dir = resolve_config_path(output_path)

# books_dir


# In[20]:


# Get books from files (handles n_books_to_read)
# We handle some variables here even if they will be replaced or unused here
# (repetitive handling wont hurt, its for handling the variables early and telling the user right away if something is wrong)

# Variables from config treated here: n_books, n_books_to_read, books_path

# Function for this cell
def handle_book_variables(n_books, n_books_to_read, list_of_books):
    if list_of_books == []: # Throws error if no books found
        raise FileNotFoundError(f"No books found in the specified path: {books_path}. Check if the path is correct and if there are .txt files in it.")
    
    # Create new variables to return
    n_books_new = n_books
    n_books_to_read_new = n_books_to_read
    list_of_books_new = list_of_books.copy()

    # Adjust n_books and n_books_to_read based on the number of books found

    # Adjust n_books and list_of_books
    if n_books == -1 or n_books > len(list_of_books):
        n_books_new = len(list_of_books)
    elif 1 <= n_books <= len(list_of_books):
        list_of_books_new = list_of_books[:n_books]
    else:
        raise ValueError("n_books must be -1 or a value >= 1.")
    
    # Adjust n_books_to_read
    if n_books_to_read == -1 or n_books_to_read >= n_books:
        n_books_to_read_new = n_books
    elif 1 <= n_books_to_read < n_books:
        pass
    else:
        raise ValueError("n_books_to_read must be -1 or a value >= 1.")
    
    return n_books_new, n_books_to_read_new, list_of_books_new


# Takes books names and sorts them alphanumerically for a list
list_of_books = sorted(list(books_dir.glob("*.txt")))
# print(list_of_books_names)


# Handling for book variables (n_books, n_books_to_read, list_of_books)
book_arguments = handle_book_variables(n_books, n_books_to_read, list_of_books)
n_books, n_books_to_read, list_of_books = book_arguments

# n_books will be replaced (in this notebook)
# list_of_books will not be used (in this notebook)


# In[21]:


# Imports from discover_words_to_learn
import importlib
import time as time_module

print(f"Starting reload at {time_module.time()}")

# Remove module completely from cache
if 'discover_words_to_learn' in sys.modules:
    print("Module found in cache, deleting...")
    del sys.modules['discover_words_to_learn']
else:
    print("Module not in cache")

# Force fresh import (this will re-execute all module-level code)
print("Importing module...")
import discover_words_to_learn
print(f"Import completed at {time_module.time()}")

from discover_words_to_learn import (
    B, D, d, d_set, a,
    v_of_vocab_sets, set_of_ks,
    _false, _true, _and, _or,
    add_to_set_of_ks,
)

n_of_books_to_read = n_books_to_read
n_of_books = len(B)
n_of_unknown_words = len(d)


# In[22]:


# Handles variables some variables (mainly used in k function)
def handle_non_negatives(*args, **kwargs):
    # Check positional arguments
    for i, value in enumerate(args):
        if value < 0:
            raise ValueError(f"Positional argument {i} is negative: {value}")

    # Check keyword arguments
    for name, value in kwargs.items():
        if value < 0:
            raise ValueError(f"Argument '{name}' is negative: {value}")

def handle_a(b,s_d,w):
    if (b,s_d,w) not in a:
        raise ValueError(f"a[{b},{s_d},{w}] is not defined, but it is being accessed.")
    if a[b,s_d,w] >= s_d:
        raise ValueError(f"a[{b},{s_d},{w}] = {a[b,s_d,w]} is greater than or equal to sentence s_d = {s_d}.")
    if a[b,s_d,w] < 0 and a[b,s_d,w] != -1:
        raise ValueError(f"a[{b},{s_d},{w}] = {a[b,s_d,w]} is negative but not -1.")
    if w not in B[b][s_d]:
        raise ValueError(f"Word {w} is not in book {b}, sentence {a[b,s_d,w]}, even a[{b},{s_d},{w}] points to it.")


# ### Helper Functions
# (simplifying, string manipulation, vector cleaning, printing vectors etc.)

# In[23]:


def transform_x_to_p(s,p_to_x=0):
    # transforms a string with form "Eq(X[b],t)" to "Eq(P[t],b)" or back
    # if p_to_x is 0 or 1 respectively
    # if p_to_x is 2 we do "Eq(x[b],t)" to "Eq(p[t],b)"

    if "E" not in s: return s

    if p_to_x == 0:
        # Remove "Eq(X[" prefix and ")" suffix
        inner = s[len("Eq(X["):-1]   # gives: "i],j"

        # Split at "],"
        i, j = inner.split("],")

        # Build the new string
        return f"Eq(P[{j}],{i})"
    
    elif p_to_x == 1:
        # Remove "Eq(P[" prefix and ")" suffix
        inner = s[len("Eq(P["):-1] # gives: "i],j"
        i, j = inner.split("],") # Split at "],"
        return f"Eq(X[{j}],{i})"
    
    elif p_to_x == 2:
        # Remove "Eq(x[" prefix and ")" suffix
        inner = s[len("Eq(x["):-1]   # gives: "i],j"

        # Split at "],"
        i, j = inner.split("],")

        # Build the new string
        return f"Eq(p[{j}],{i})"
        

        


# In[24]:


def tokenize(expr):
    tokens = []
    i = 0
    n = len(expr)

    while i < n:
        c = expr[i]

        # skip spaces
        if c.isspace():
            i += 1
            continue

        # Standalone parenthesis
        if c == '(' or c == ')':
            tokens.append(c)
            i += 1
            continue

        # Identifiers / words (AND, OR, f, foo, x23...)
        if c.isalpha():
            start = i
            while i < n and (expr[i].isalnum() or expr[i] == '_'):
                i += 1
            name = expr[start:i]

            # If next char is "(" → function call
            if i < n and expr[i] == '(':
                depth = 0
                start2 = i
                while i < n:
                    if expr[i] == '(':
                        depth += 1
                    elif expr[i] == ')':
                        depth -= 1
                        if depth == 0:
                            i += 1
                            break
                    i += 1
                tokens.append(name + expr[start2:i])  # full: f(2,3)
            else:
                tokens.append(name)
            continue

        # Numbers
        if c.isdigit():
            start = i
            while i < n and expr[i].isdigit():
                i += 1
            tokens.append(expr[start:i])
            continue

        # Commas or other punctuation
        tokens.append(c)
        i += 1

    # Piecing together l, [, w, ], [, t, ] into l[w][t]
    tokens2 = []
    i = 0
    
    while i < len(tokens):
        # Check for the pattern: l [ x ] [ y ]
        if (tokens[i] == "l" or tokens[i] == "L") and i + 6 < len(tokens) :
            # Build the merged token
            merged = tokens[i] + "[" + tokens[i+2] + "][" + tokens[i+5] + "]"
            tokens2.append(merged)
            i += 7  # Skip everything used
        else:
            tokens2.append(tokens[i])
            i += 1

    return tokens2


# In[25]:


def prefix_to_infix_function(tokens):
    return prefix_to_infix(tokens)[1:-1]
def prefix_to_infix(tokens):
    """
    Converts prefix vector like:
        ["Or","(", "A", ",", "And","(", "B", ",", "C", ")", ")"]
    into STRICTLY parenthesised infix:
        ( A OR ( B AND C ) )
    """

    # ----------------------------------------------------------
    # PARSER (same idea as before)
    # ----------------------------------------------------------

    def parse_expr(i):
        t = tokens[i]

        if t in ("Or", "And"):
            op = t
            assert tokens[i+1] == "("
            i += 2

            args = []
            while True:
                arg, i = parse_expr(i)
                args.append(arg)

                if tokens[i] == ",":
                    i += 1
                    continue
                elif tokens[i] == ")":
                    i += 1
                    break
                else:
                    raise ValueError(f"Unexpected token {tokens[i]} in {op}")

            return (op, args), i

        return t, i + 1

    ast, pos = parse_expr(0)
    if pos != len(tokens):
        raise ValueError("Extra tokens after end of expression.")


    # ----------------------------------------------------------
    # INFIX SERIALISER (strict parentheses)
    # ----------------------------------------------------------

    OP_MAP = {
        "And": "AND",
        "Or": "OR",
    }

    def to_infix(expr):
        # atomic
        if isinstance(expr, str):
            return [expr]

        op, args = expr
        op_inf = OP_MAP[op]

        # Convert each argument recursively
        arg_vectors = [to_infix(a) for a in args]

        # Combine with operator, fully parenthesised
        result = ["("]
        for i, vec in enumerate(arg_vectors):
            result.extend(vec)
            if i < len(arg_vectors) - 1:
                result.append(op_inf)
        result.append(")")

        return result

    return to_infix(ast)


def infix_to_prefix_function(tokens): # Makes AND, OR into And(), Or() # Created by Copilot
    precedence = {_or: 1, _and: 2}

    def op_to_name(op):
        return op.capitalize()

    output = []
    ops = []

    def apply_op():
        op = ops.pop()
        right = output.pop()
        left = output.pop()
        output.append((op, left, right))

    i = 0
    while i < len(tokens):
        t = tokens[i]

        # treat any non-operator, non-paren token as an operand
        if t not in (_and, _or, "(", ")"):
            output.append(t)

        elif t in (_and, _or):
            while (ops and ops[-1] in (_and, _or) and
                   precedence[ops[-1]] >= precedence[t]):
                apply_op()
            ops.append(t)

        elif t == "(":
            ops.append(t)

        elif t == ")":
            while ops and ops[-1] != "(":
                apply_op()
            ops.pop()

        i += 1

    while ops:
        apply_op()

    ast = output[0]

    def build_vector(node):
        # Leaf node (operand)
        if isinstance(node, str):
            return [node]

        op, left, right = node
        name = op.capitalize()

        # Collect all arguments for this operator as a list of argument-token-lists
        args = []

        def collect(n):
            # If same operator, flatten its children (so AND(a,AND(b,c)) -> AND(a,b,c))
            if isinstance(n, tuple) and n[0] == op:
                collect(n[1])
                collect(n[2])
            else:
                # keep each argument as its own token-list
                args.append(build_vector(n))

        collect(left)
        collect(right)

        # Build the final flattened vector inserting commas only BETWEEN arguments
        vec = [name, "("]
        for idx, a_tokens in enumerate(args):
            vec.extend(a_tokens)          # insert the whole argument token sequence
            if idx < len(args) - 1:
                vec.append(",")
        vec.append(")")
        return vec

    return build_vector(ast)


# In[26]:


def eq_to_eqeq(s): # transforms string "Eq(p[t],b)" to "p[t] == b" # Made by Copilot
    
    # If not in this format returns the string back:
    if s[0] != 'E': return s

    inner = s[3:-1]  # contents between the parentheses
    # find the top-level comma (ignore commas inside bracket pairs)
    depth = 0
    split_idx = -1
    for i, ch in enumerate(inner):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
        elif ch == "," and depth == 0:
            split_idx = i
            break

    if split_idx == -1:
        return s  # couldn't find a top-level comma -> leave unchanged

    left = inner[:split_idx].strip()
    right = inner[split_idx + 1 :].strip()

    return f"{left} == {right}"


# In[27]:


# Cleaning Vector Function
def clean_vector_sympy(v, t_max, w_max, b_max):
    L = [[[] for t in range(t_max+1)] for w in range(w_max)]
    X = [[] for b in range(b_max)]

    for t in range(t_max+1):
        for w in range(w_max):
            L[w][t] = symbols(f"L[{w}][{t}]")
    
    for b in range(b_max):
        X[b] = symbols(f"X[{b}]")
    

    # Making 0 into False and 1 into True
    # Replacing x with X, so we use X[] and not x[] by accident
    # Replacing l with L, so we use L[][] and not l[][] by accident
    
    # Vector for the translation of v to the sympy symbols    
    v_for_sympy = []

    # Translating from our version to SymPys
    i = 0
    while i < len(v):
        if v[i] == _false: v_for_sympy.append('False'); i+=1
        elif v[i] == _true: v_for_sympy.append('True'); i+=1
        elif v[i] == _and: v_for_sympy.append('&'); i+=1
        elif v[i] == _or: v_for_sympy.append('|'); i+=1
        else: 
            # turns all x to X and l to L
            v_i_translation = v[i].translate(str.maketrans({"x": "X", "l": "L"}))
            v_for_sympy.append(v_i_translation); i+=1


    # Putting all together in a string
    v_as_string = " ".join(v_for_sympy)

    # Creating actual SymPy expression
    expr = eval(v_as_string)

    # Simplify expression
    simplified_expr = simplify_logic(simplify(expr),form='dnf')

    # Convert back to the vector format
    simplified_v = convert_sympy_str_to_vector(str(simplified_expr))

    return simplified_v

def convert_sympy_str_to_vector(s):
    
    # its for the cases of "Eq(X[w], 0)" to turn into "Eq(X[w],0)"
    s2 = s.replace(", ", ",")
    # separate string into different tokens
    # ("(Eq(X[w],0))" -> ["(","Eq(X[w],0)",")"])
    v = tokenize(s2)

    # Vector for the translation of v from the sympy symbols
    v_from_sympy = []

    # Translating from SymPys version to ours
    i = 0
    while i < len(v):
        if v[i] == 'False': v_from_sympy.append(_false); i+=1
        elif v[i] == 'True': v_from_sympy.append(_true); i+=1
        elif v[i] == '&': v_from_sympy.append(_and); i+=1
        elif v[i] == '|': v_from_sympy.append(_or); i+=1
        else: 
            # turns all X to x and L to l
            v_i_translation = v[i].translate(str.maketrans({"X": "x", "L": "l"}))
            v_from_sympy.append(v_i_translation); i+=1

    return v_from_sympy




def clean_vector_sympy2(v, t_max, w_max):
    L = [[[] for t in range(t_max+1)] for w in range(w_max)]
    P = [[] for t in range(t_max+1)]

    for t in range(t_max+1):
        for w in range(w_max):
            L[w][t] = symbols(f"L[{w}][{t}]")
        P[t] = symbols(f"P[{t}]")  
    

    # Making 0 into False and 1 into True
    # Replacing x with X, so we use X[] and not x[] by accident
    # Replacing l with L, so we use L[][] and not l[][] by accident
    
    # Vector for the translation of v to the sympy symbols
    v_for_sympy = []

    # Translating from our version to SymPys
    i = 0
    while i < len(v):
        if v[i] == _false: v_for_sympy.append('False'); i+=1
        elif v[i] == _true: v_for_sympy.append('True'); i+=1
        elif v[i] == _and: v_for_sympy.append('&'); i+=1
        elif v[i] == _or: v_for_sympy.append('|'); i+=1
        else: 
            # turns all x to X and l to L, then turns all X to P
            v_i_translation = v[i].translate(str.maketrans({"x": "X", "l": "L"}))
            v_i_translation = transform_x_to_p(v_i_translation,0)
            v_for_sympy.append(v_i_translation); i+=1

    # Putting all together in a string
    v_as_string = " ".join(v_for_sympy)

    # Creating actual SymPy expression
    expr = eval(v_as_string)

    # Simplify expression
    simplified_expr = simplify_logic(simplify(expr),form='dnf') 
    # is the one we were doing before (first one we did when problem was working), but idk if it's the best. 
    # Maybe we dont need to simplify that much cause the solver will deal with things better. 
    # So we'll have to see the tradeoff between solver time and simplification time later (adding both and seeing in which case it goes better considering bigger examples with actual books)

    # simplified_expr = simplify_logic(simplify(expr))
    # simplified_expr = simplify_logic(expr)
    # simplified_expr = simplify(expr)
    # -------- Change Simplify ---------

    # Convert back to the vector format
    simplified_v = convert_sympy_str_to_vector2(str(simplified_expr))

    return simplified_v

def convert_sympy_str_to_vector2(s):
    
    # its for the cases of "Eq(P[w], 0)" to turn into "Eq(P[w],0)"
    s2 = s.replace(", ", ",")
    # separate string into different tokens
    # ("(Eq(P[w],0))" -> ["(","Eq(P[w],0)",")"])
    v = tokenize(s2)

    # Vector for the translation of v from the sympy symbols
    v_from_sympy = []

    # Translating from SymPys version to ours
    i = 0
    while i < len(v):
        if v[i] == 'False': v_from_sympy.append(_false); i+=1
        elif v[i] == 'True': v_from_sympy.append(_true); i+=1
        elif v[i] == '&': v_from_sympy.append(_and); i+=1
        elif v[i] == '|': v_from_sympy.append(_or); i+=1
        else: 
            # turns all P to X, then X to x and L to l
            v_i_translation = transform_x_to_p(v[i],1)
            v_i_translation = v_i_translation.translate(str.maketrans({"X": "x", "L": "l"}))
            v_from_sympy.append(v_i_translation); i+=1

    return v_from_sympy



def clean_vector_z3(v_prime,t_max,w_max, is_l=0):


    domain = []

    L = [[[] for t in range(t_max+1)] for w in range(w_max)]
    P = [[] for t in range(t_max+1)]

    P = [Int(f"P[{t}]") for t in range(t_max+1)]

    for t in range(t_max+1):
        for w in range(w_max):
            L[w][t] = Bool(f"L[{w}][{t}]")
    
    for P_i in P:
        domain.append(P_i >= 0)
        domain.append(P_i < n_of_books)

    # Adding constraint of P[] != P[]
    domain.append(Distinct(P))


    # Turning x to X, than X to P
    expr = [ transform_x_to_p(i.replace('x','X'),0) for i in v_prime ]
    # Turning l to L, than Eq() into ==
    expr = [ eq_to_eqeq(i.replace('l','L')) for i in expr]
    # Turning 0 into False and 1 into True
    expr = ['False' if i == _false else 'True' if i == _true else i for i in expr]
    # Turning AND and OR into And() and Or() on vector expr
    expr = infix_to_prefix_function(expr)


    expr = eval(' '.join(expr))
    

    if is_always_true(expr, domain):
        return [_true]
    elif is_always_false(expr, domain):
        return [_false]
    else:
        return v_prime
    

def is_always_true(e, domain):
    s = Solver()
    s.add(domain)
    s.add(Not(e))
    return s.check() == unsat

def is_always_false(e, domain):
    s = Solver()
    s.add(domain)
    s.add(e)
    return s.check() == unsat



# In[28]:


# clean vector functions with @lru_cache

from functools import lru_cache

# Cache for clean_vector_sympy
@lru_cache(maxsize=5000)
def clean_vector_sympy_cached(expr_tuple, t_max, w_max, b_max):
    """Cached version - accepts tuple instead of list"""
    return tuple(clean_vector_sympy(list(expr_tuple), t_max, w_max, b_max))

# Cache for clean_vector_sympy2
@lru_cache(maxsize=5000)
def clean_vector_sympy2_cached(expr_tuple, t_max, w_max):
    """Cached version - accepts tuple instead of list"""
    return tuple(clean_vector_sympy2(list(expr_tuple), t_max, w_max))

# Cache for clean_vector_z3 (if you want)
@lru_cache(maxsize=5000)
def clean_vector_z3_cached(expr_tuple, t_max, w_max, is_l=0):
    """Cached version - accepts tuple instead of list"""
    return tuple(clean_vector_z3(list(expr_tuple), t_max, w_max, is_l))


# In[29]:


def print_vector(vec): # prints vectors values with " " in between them
    for x in vec:
        print(x, end=" ")
    print()  # final newline


# In[30]:


# Function that simplifies if alone in dnf structure (might not be in dnf, so we have to be careful) (uses vectors in z3 form (prefix)) (just removes in first Or, so not recursive)
def simplify_or(tokens):
    """
    tokens: list[str] representing one big expression like:
        ["Or", "(", "l[35][1]", ",", "Eq(x[10],2)", ",", ... , ")"]
    Returns a simplified token list.
    """

    # ----------------------------------------------------------
    # 1. PARSER — very small recursive parser for this structure
    # ----------------------------------------------------------

    def parse_expr(i):
        token = tokens[i]

        # Case 1: operation: Or(...) or And(...)
        if token in ("Or", "And"):
            op = token
            assert tokens[i+1] == "("
            i += 2  # jump over 'Op ('

            args = []
            while True:
                # parse argument
                arg, i = parse_expr(i)
                args.append(arg)

                if tokens[i] == ",":
                    i += 1
                    continue
                elif tokens[i] == ")":
                    i += 1
                    break
                else:
                    raise ValueError(f"Unexpected token {tokens[i]} while parsing {op}")

            return (op, args), i

        # Case 2: atomic token: l[35][1], Eq(x,2), etc.
        else:
            return token, i + 1

    # ----------------------------------------------------------
    # 2. SERIALISER — turns AST back into token vector
    # ----------------------------------------------------------

    def to_tokens(expr):
        if isinstance(expr, tuple):
            op, args = expr
            out = [op, "("]
            for k, a in enumerate(args):
                out.extend(to_tokens(a))
                if k < len(args) - 1:
                    out.append(",")
            out.append(")")
            return out
        else:
            return [expr]

    # ----------------------------------------------------------
    # 3. SIMPLIFICATION
    # ----------------------------------------------------------

    def simplify(expr):
        if not isinstance(expr, tuple):
            return expr  # atomic

        op, args = expr
        args = [simplify(a) for a in args]

        if op == "Or":
            # collect top-level atoms of Or
            top_atoms = set()
            for a in args:
                if not isinstance(a, tuple):  # atomic
                    top_atoms.add(a)

            new_args = []
            for a in args:
                if isinstance(a, tuple) and a[0] == "And":
                    # If any argument of this And is in top_atoms: remove the And entirely
                    and_args = a[1]
                    if any((not isinstance(x, tuple) and x in top_atoms) for x in and_args):
                        continue  # skip whole And
                new_args.append(a)

            return ("Or", new_args)

        return (op, args)

    # ----------------------------------------------------------
    # Apply all stages
    # ----------------------------------------------------------

    ast, pos = parse_expr(0)
    if pos != len(tokens):
        raise ValueError("Parsing ended early, malformed token stream")

    ast = simplify(ast)
    return to_tokens(ast)
def simplify_or_full(tokens):
    v = infix_to_prefix_function(tokens)
    v = simplify_or(v)
    v = prefix_to_infix_function(v)
    return v


# In[31]:


def andify(vector_of_ands): # vector_of_ands is a vector of vectors
    v = []
    and_symbol_string = _and
    # if at least one element is _false, the whole AND is false
    for element in vector_of_ands:
        if element == [_false] or element == _false:
            return [_false]
    if (all([element == [_true] or element == _true for element in vector_of_ands])):
        return [_true]


    for i in range(len(vector_of_ands)):
        v.append("(")
        if vector_of_ands[i] == []:
            # if empty its the same as AND 1
            v.append('1')
        else:
            v.extend(vector_of_ands[i])
        v.append(")")
        if i != len(vector_of_ands)-1: 
            # if not the last, append an "AND" after
            v.append(and_symbol_string)

    return v

def orify(vector_of_ors): # vector_of_ors is a vector of vectors
    v = []
    or_symbol_string = _or
    # if at least one element is _true, the whole OR is true
    for element in vector_of_ors:
        if element == [_true] or element == _true:
            return [_true]
    if (all([element == [_false] or element == _false for element in vector_of_ors])):
        return [_false]

    for i in range(len(vector_of_ors)):
        v.append("(")
        if vector_of_ors[i] == []: 
            # if empty its the same as OR 0
            v.append(_false)
        else:
            v.extend(vector_of_ors[i])
        v.append(")")
        if i != len(vector_of_ors)-1: 
            # if not the last, append an "OR" after
            v.append(or_symbol_string)
    return v

def vector_x_eq_int(b, t):
    # x_eq_int = ["(",f"Eq(x[{b}],{t})",")"]
    x_eq_int = [f"Eq(x[{b}],{t})"]
    return x_eq_int


# ### Creating Expressions (main part)
# (Using/Creating: K, k, l[w][t], m[w][t] etc.)

# In[32]:


def K_reread(b, w, t): # Can I learn word w from book b at time t?
    # This K is the older version, which allows rereads, 
    # but hopefully can provide more aggregation, leaving the reread to be dealt by another constraint
    
    # Check if we know K (we registered in imported set_of_ks)
    if t == 0:
        base_K = f"K_reread({b},{w},{t})"
        if base_K in set_of_ks:
            return set_of_ks[base_K]
    
    # check if book b has w
    if w not in v_of_vocab_sets[b]:
        # w is not to be found in book b

        if t == 0:
            # we read no books yet (t = 0) and 
            # book b doesnt have w in it

            # So, we cant learn it through book b, and return 0
            return [_false]
        
        # maybe we could have learned it in a past book
        # So, we check if we knew it at t-1 (l[w][t-1])

        # Check if they are True or False or has size 1 (is Eq(x[b],t)) 
        # so we can replace them already instead of a string of l[w][t-1]
        if ( l[w][t-1] == [_false] or l[w][t-1] == [_true] 
             or len(l[w][t-1]) == 1):
            return l[w][t-1]

        v_prime = [f"l[{w}][{t-1}]"]
        # v_prime = l[w][t-1]
        # ------ Change l ------

        return v_prime
    
        
    
    # At this point, we know:
    #   1. Word w is in b
    #   2. Word w is not alone in a sentence in b (else it would be in vocab)

    v_prime = []

    v_of_ors = []

    for s_d in range(len(B[b])):        
        if w in B[b][s_d]:

            # This 'if' will never be entered because of "At this point, we know:" point 2
            if len(B[b][s_d]) == 1:
                # w is alone in s_d, so we learned w in this book
                # So, we can return a 1, cause we will learn w in this book b
                return [_true]

            # if w is not alone, we need to know
            # all other words omega
            v_of_ands = []
            for omega in B[b][s_d]:

                if omega == w:
                    continue
                
                v_of_ands.append(k(b, s_d, omega, t))

            v_of_ands = andify(v_of_ands) 
            # Make the AND of all ks of the words in s_d

            v_of_ors.append(v_of_ands)
    v_prime = orify(v_of_ors)
    # Make the OR of knowing w in all sentences in book b

    return v_prime


# In[33]:


# def k(b, s_d, omega, t): # Do I know word omega in sentence s_d of b at time t?

from collections import OrderedDict

class LRUCache:
    def __init__(self, max_size=1000):
        self.cache = OrderedDict()
        self.max_size = max_size

    def get(self, key):
        if key not in self.cache:
            return None
        # Move it to the top (end)
        self.cache.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            # Update + move to top
            self.cache.move_to_end(key)
        self.cache[key] = value

        # If too big → remove least recently used
        if len(self.cache) > self.max_size:
            self.cache.popitem(last=False)  # pop oldest

# Set that serves as Cache for k() results
set_of_k_tests = LRUCache(max_size=3500)

def k(b, s_d, omega, t): # Do I know word omega in sentence s_d of b at time t?

    handle_non_negatives(b=b, s_d=s_d, omega=omega, t=t)

    base_k = f"k({b},{s_d},{omega},{t})"

    # Check if k(b,s_d,omega,t) was already calculated 
    # before and return its value if it was
    cached_k = set_of_k_tests.get(base_k)
    if cached_k is not None:
        return cached_k

    
    if len(B[b][s_d]) == 1 and B[b][s_d][0] == omega:
        # omega is the only word in sentence s_d of book b

        # Add before returning
        # add_to_set_of_ks(base_k,[_true])
        # -------------- NEW --------------
        # set_of_k_tests.put(base_k, [_true])

        # So, we learn omega reading book b at s_d. 
        # Hence, returning 1 for value of k
        return [_true]
    
    if omega not in B[b][s_d]: # Error check
        # omega is not in sentence s_d of book b
        # So, we shouldn't have called it
        raise ValueError(f"Word {omega} is not in B[{b}][{s_d}], but k({b},{s_d},{omega},{t}) was called.")
    
    # From here on, we know that:
    #   1. omega is in B[b][s_d]
    #   2. omega is not alone in B[b][s_d]

    if s_d == 0: # starting sentence
        if t == 0: # starting book
            # if we are at the starting s_d and starting book
            # and s_d has more than 1 word, we cant learn it
            # so we return [0]

            # Add before returning
            # add_to_set_of_ks(base_k,[_false])
            # -------------- NEW --------------
            # set_of_k_tests.put(base_k, [_false])

            return [_false]
        
        
        # if first sentence, we can only look back at t-1
        
        # Check if they are True or False or has size 1 (is Eq(x[b*],t*) or l[w*][t*])
        # so we can replace them already instead of a string of l[omega][t-1]
        if ( l[omega][t-1] == [_false] or l[omega][t-1] == [_true] 
             or len(l[omega][t-1]) == 1):
            # if l[omega][t-1] is a single value, 
            # we can return it as the value to simplify later

            # Add before returning
            # -#-# add_to_set_of_ks(base_k,l[omega][t-1]) 
            # -------------- NEW --------------
            set_of_k_tests.put(base_k, l[omega][t-1])
            
            return l[omega][t-1]

        # Add before returning
        # add_to_set_of_ks(base_k,[f"l[{omega}][{t-1}]"]) # -#-# 
        # add_to_set_of_ks(base_k,l[omega][t-1])
        # --- Change l ---
        set_of_k_tests.put(base_k, [f"l[{omega}][{t-1}]"])

        return [f"l[{omega}][{t-1}]"]
        # return l[omega][t-1]
        # --- Change l ---
    

    # From here on, we know that:
    #   1. omega is in B[b][s_d]
    #   2. omega is not alone in B[b][s_d]
    #   3. s_d > 0

    v_prime = []

    v_of_ors = []

    
    # EITHER we knew omega in the last sentence omega showed up at in b
    # (or if didnt, on last book, t-1)
    if (b,s_d,omega) in a: # means it showed up in b before s_d
        # So we call k on the last appearance a of omega in b from this sentence s_d
        handle_a(b,s_d,omega)


        v_of_ors.append(k(b, a[b,s_d,omega], omega, t))

    else: # means it didnt show up in b before s_d
        # So we rely on knowing it from previous book

        if t == 0:
            # we are in the first book, and omega didnt show up before s_d in b
            # So, we cant know it, returning 0
            v_of_ors.append([_false])

        elif l[omega][t-1] == [_false] or l[omega][t-1] == [_true] or len(l[omega][t-1]) == 1:
            # if l[omega][t-1] is a single value, 
            # we can return it as the value to simplify later
            v_of_ors.append(l[omega][t-1])

        else: # is not single value, so we return the string
            v_of_ors.append( [f"l[{omega}][{t-1}]"] )


    # OR if (omega is in s_d) we know ALL the other words w 
    # in the same s_d of book b ( make an AND of "knowing it before" )
    # If (omega not in s_d) we can just rely on knowing it on last book t-1, 
    # which we did above in v_of_ors
    if omega in B[b][s_d]:
        v_of_ands = []

        for w in B[b][s_d]:
            if w == omega:
                continue

            if (b,s_d,w) in a: # means it showed up in b before s_d
                # So we call k on the last appearance a of w in b from this sentence s_d
                handle_a(b,s_d,w)
                v_of_ands.append(k(b, a[b,s_d,w], w, t))
            else: # means it didnt show up in b before s_d
                # So we rely on knowing it from previous book
                if t == 0:
                    # we are in the first book, and omega didnt show up before s_d in b
                    # So, we cant know it, returning 0
                    v_of_ands.append([_false])
                
                elif l[w][t-1] == [_false] or l[w][t-1] == [_true] or len(l[w][t-1]) == 1:
                    # if l[w][t-1] is a single value, 
                    # we can return it as the value to simplify later
                    v_of_ands.append(l[w][t-1])

                else:
                    v_of_ands.append( [f"l[{w}][{t-1}]"] )

        v_of_ands = andify(v_of_ands)
        v_of_ors.append(v_of_ands)
    
    v_prime = orify(v_of_ors)


    if cached_k is not None:
        raise ValueError(f"Logic error: cached_k ({cached_k}) should be None, if it wasn't we should have used it's value with the get at the start of k({b},{s_d},{w},{t}).")
    
    # Simplify to add to the cache
    if not (v_prime == [_false] or v_prime == [_true] or len(v_prime) == 1):
        # Just simplify with z3 if its not obvious its 0, 1 or a single value
        v_prime = clean_vector_z3(v_prime,t,n_of_unknown_words)
    v_prime = clean_vector_sympy(v_prime,t,n_of_unknown_words,n_of_books)
    v_prime = clean_vector_sympy2(v_prime,t,n_of_unknown_words)
    set_of_k_tests.put(base_k, v_prime)

    return v_prime


# In[34]:


l = [[[] for t in range(n_of_books_to_read)] for w in range(n_of_unknown_words)]
m = [[[] for t in range(n_of_books_to_read)] for w in range(n_of_unknown_words)]

print(" ------------------------- Creating expressions for l[w][t] variables and symplifying them ------------------------- ")

# Dictionary to store canonical variables for each unique expression
# Key: ' '.join(l[w][t]) expression string
# Value: the first l[w][t] variable name that had this expression
expression_to_canonical = {}
# Dictionary to map duplicate variables to their canonical version
# Key: duplicate variable name (e.g., "l[235][0]")
# Value: canonical variable name (e.g., "l[234][0]")
duplicate_mapping = {}


# simplification_cache = LRUCache(max_size=3000)


# For measuring total time for all expressions to be simplified
total_time = 0.0

counter = 0
# for t in range(0,1): # for testing purposes, limit to first book only
for t in range(n_of_books_to_read): # -#-# 
    # verify run time
    vector_time_t = []
    # for w in range(50): # for testing purposes, limit to first 50 words
    for w in range(n_of_unknown_words): # -#-#
        
        v_of_ors1 = []

        # Create "l[w][t-1] OR"
        if t != 0:
            if l[w][t-1] != [_false]:
                
                # Check if l[w][t-1] == [1], 
                # if yes we can say that any l[w][t*>t-1] == 1
                # So we just return [1]
                if l[w][t-1] == [_true]:
                    l[w][t] = [_true]
                    if t == n_of_books_to_read-1: m[w][t] = True
                    else: m[w][t] = False
                    continue

                # Check if has size 1 (is Eq(x[b*],t*) or l[w*][t*])
                if (len(l[w][t-1]) == 1):
                    v_of_ors1.append(l[w][t-1])

                # If not, just add string version
                else:    
                    v_of_ors1.append([f"l[{w}][{t-1}]"])
                    # v_of_ors1.append(l[w][t-1])
                    # --- Change l ---

        
        v_prime = []
        v_of_ors2 = []

        # Create ORs of each book b ((x[b] == t) AND K_reread(b,w,t))
        for b in range(n_of_books):

            v_of_ands = []

            # Create "(x[b] == t) AND"
            v_of_ands.append(vector_x_eq_int(b,t))
            
            # Create "K_reread(b,w,t)
            v_of_ands.append(K_reread(b,w,t))

            v_of_ands = andify(v_of_ands)
            v_of_ors2.append(v_of_ands)
        
        v_of_ors2 = orify(v_of_ors2)
        v_of_ors1.append(v_of_ors2)

        v_prime = orify(v_of_ors1)

        l[w][t] = v_prime
        


        start = time.perf_counter() # verify run time


        # Cleaning with z3 (simple 0 or 1)
        if not (l[w][t] == [_false] or l[w][t] == [_true] or len(l[w][t]) == 1):
            # Just simplify with z3 if its not obvious its 0, 1 or a single value
            # After all it returns only if its always 0 or 1 or neither

            l[w][t] = clean_vector_z3(l[w][t],t,n_of_unknown_words,1)


        # Cleaning with SymPy - two passes

        # This cleans with the perspective of Eq(P[t],b), 
        # simplifying SOME (not all) cases of reading two books at once
        l[w][t] = clean_vector_sympy2(l[w][t],t,n_of_unknown_words)

        # This cleans with the perspective of Eq(X[b],t), 
        # simplifying SOME (not all) cases of rereading
        l[w][t] = clean_vector_sympy(l[w][t],t,n_of_unknown_words,n_of_books)

        




        # Replacing each 'l[w][t]' in the expression that is 
        # a duplicate from an already created canonical variable l
        for i in range(len(l[w][t])):
            element = l[w][t][i]
            if element[0] != 'l': continue # ignore elements that arent l variables
            if element in duplicate_mapping:
                # Replace duplicate with canonical variable
                l[w][t][i] = duplicate_mapping[element]

        # Saving expression string for duplicate detection 
        # (only do if size is > 1, so we dont save values like 0, 1, or Eq(x[b],t))
        if len(l[w][t]) > 1:
            l_var_name = f'l[{w}][{t}]'
            expr_str = ' '.join(l[w][t])

            if expr_str in expression_to_canonical:
                # There is another l with equivalent expression
                canonical_var = expression_to_canonical[expr_str]
                duplicate_mapping[l_var_name] = canonical_var


                # print(f"Duplicate found: {l_var_name} -> {canonical_var} (expression: {expr_str[:50]}...)")


                # This turns l[w][t] into another l[w][t] with the same expression
                # Now l[w][t] wont turn into a variable, 
                # since m[w][t] = False when len(l[w][t]) == 1
                l[w][t] = [canonical_var]
            else:
                # This l is the first one of its kind
                # So we add its expr to point back to it if it repeats later
                expression_to_canonical[expr_str] = l_var_name


            
        # Simplifies simple OR structures (like A OR (A AND B) = A)
        # Many structures are like this, so helps a lot
        if len(l[w][t]) > 1:
            # Further simplify OR structures with custom simplifier
            l[w][t] = simplify_or_full(l[w][t])

        end = time.perf_counter() # verify run time


        # Creating m[w][t] to indicate if l[w][t]
        # should be a variable or not
        if len(l[w][t]) != 1 or t == n_of_books_to_read-1:
            m[w][t] = True
        else: m[w][t] = False



        if __name__ == "__main__": # Print only in notebook
            print(f"{counter}. l[{w}][{t}] = ", end='')
            print_vector(l[w][t])
            print()
            counter += 1



        
        # verify run time
        elapsed = end - start
        total_time += elapsed
        vector_time_t.append(f"{elapsed:.5f}")


    print(f" ------------------------- Round complete for t = {t}. So, variables l[*][{t}] created. ------------------------- ")
    # print(f"l[*][{t}] time = {vector_time_t}") # verify run time
    print(f"l[*][{t}] total time to simplify = {sum(map(float, vector_time_t)):.5f}\n") # verify run time


print(f"Total time: {round(total_time,5)}") # verify run time


# In[35]:


# Copying l and m to immutable structures (tuples of tuples)
# For when you change l[w][t] along the way and want their initial values
def deep_tuple(obj):
    """Recursively convert lists to tuples"""
    if isinstance(obj, list):
        return tuple(deep_tuple(item) for item in obj)
    return obj

# Convert to immutable structure
l_copy = deep_tuple(l)
m_copy = deep_tuple(m)


# In[36]:


intermediate_amount_of_variables = 0
sizes_of_intermediate_variables = []
for t in range(n_of_books_to_read):
    for w in range(n_of_unknown_words):
        if m[w][t]:
            intermediate_amount_of_variables += 1
            size_l = len([i for i in l[w][t] if i not in ('(',')','AND','OR')])



            # print(f"l[{w}][{t}] ({size_l}) = ", end='')
            # print_vector(l[w][t])
            # print()


            
print(f"Amount of inter vars l[w][t]: {intermediate_amount_of_variables}")


# In[37]:


# Results Display (shows all l[w][t] vectors) 
# for t in range(n_of_books_to_read): # n_of_books_to_read
#     for w in range(n_of_unknown_words): # n_of_unknown_words


#         print(f"l[{w}][{t}] = ",end='')
#         print_vector(l[w][t])
        
#         print()


# ### Preparing for Solver and Solving
# (Changes in structure to pass to solver, creating model, and solving with z3)

# In[38]:


# Definitions for Translating

# range of values of p (max value = rp)
rp = n_of_books-1


# In[39]:


def l_brackets_to_underline(s): # transforms string "l[w][t]" to "l_w_t" # Made by Copilot

    if not isinstance(s, str): 
        return s
    if not s.startswith('l'):
        return s

    try:
        i1 = s.find('[')
        j1 = s.find(']', i1)
        if i1 == -1 or j1 == -1:
            return s
        w = s[i1+1:j1].strip()

        i2 = s.find('[', j1)
        j2 = s.find(']', i2)
        if i2 == -1 or j2 == -1:
            return s
        t = s[i2+1:j2].strip()

        if w == "" or t == "":
            return s

        return f"l_{w}_{t}"
    except Exception:
        return s


# In[40]:


def transform_for_create_l_variable(s):

    if s[0] == 'E': return eq_to_eqeq(s)
    elif s[0] == 'l': return l_brackets_to_underline(s)
    else: return s


# In[41]:


from z3 import BoolVal # REMOVE, put on top with other imports

def create_l_variable(v,w,t): # Creates intermediate variable l_w_t and its constraint
    # just doing for z3 now, so solver isnt used


    # for each string s we change whats needed in terms of format
    v_transformed = [transform_for_create_l_variable(s) for s in v]

    # Create Boolean variable l_w_t in the global scope 
    # (not just in this funtion)
    l_name = f"l_{w}_{t}"
    exec(f"{l_name} = Bool('{l_name}')", globals())

    # This is the variable we just created 
    # (picked up from global scope)
    l_var_global = globals()[l_name] 

    # Create value of l_w_t through the expression
    # (represented by its constraint)
    
    # Build Z3 expression by parsing the transformed tokens
    expr = build_z3_expression(v_transformed)

    # Simplify one last time
    simplified_expr = z3_simplify(expr)

    # Add constraint
    opt.add(l_var_global == simplified_expr)

    return

def build_z3_expression(tokens): # Builds Z3 expression from vectors # Made by Copilot
    """Convert token list to Z3 expression"""
    # Join tokens and replace function names for Z3
    expr_str = " ".join(tokens)
    
    # Create a safe namespace with only Z3 objects
    safe_globals = {
        'And': And,
        'Or': Or,
        'Bool': Bool,
        'p': p,  # Your p list
        'opt': opt
    }
    
    # Add all l_* variables that exist
    for key in list(globals().keys()):
        if key.startswith('l_'):
            safe_globals[key] = globals()[key]

    result = eval(expr_str, {"__builtins__": {}}, safe_globals)

    if isinstance(result, bool):
        return BoolVal(result)
    
    return result


# In[42]:


# Translating (with z3 in mind for now)

print(f" ------------------------- Translating Expressions for Solving ------------------------- ")

# Initializing the Optimizer
opt = Optimize()

# Vector of If(l_w_t, 1, 0) for each w and t where m[w][t] is True
# Helps in Objective Function creation (f()) later
f_terms = []


# Creating p variables
p = [Int(f"p_{i}") for i in range(n_of_books_to_read)]
for p_i in p:
    opt.add(p_i >= 0, p_i <= rp)

# Creating intermediate variables l_w_t 
# (their constraints are also created in create_l_variable)
for t in range(n_of_books_to_read):
    for w in range(n_of_unknown_words):           

        # m[w][t] means variable l_w_t must be created
        if m[w][t]:

            # Replacing _true and _false by True and False
            # This only happens when l[w][t] is l[w][n_of_books_to_read-1], 
            # cause its the only that has m[w][t] at True even with len() of 1
            if l[w][t] == [_true]:
                l[w][t] = ['True']
            if l[w][t] == [_false]:
                l[w][t] = ['False']

            # Turning "Eq(x[b],t)" into "Eq(p[t],b)"
            l[w][t] = [ transform_x_to_p(i,2) for i in l[w][t] ]
            
            # Makes AND, OR into And(), Or() on vector of strings l[w][t]
            l[w][t] = infix_to_prefix_function(l[w][t])



            # Making each l_w_t variable needed
            create_l_variable(l[w][t],w,t)

            # For the last t, we add them to f_terms to sum them later
            if t == n_of_books_to_read-1:
                # Since we want to maximize the learning of words at the last time t
                f_terms.append(If(eval(f"l_{w}_{t}"), 1, 0))

# Creating constraints (not the l_w_t ones)

# Constraint of multiple reads at the same time
opt.add(Distinct(p))

# Creating Objective Function f()

# We use f_terms from before to create f
f = Sum(f_terms)
# And say we wanna Maximize it
obj = opt.maximize(f)
None


# In[43]:


# Solving (with z3 in mind for now)
import z3 # (will leave here cause idk if sth up there will be messed up by this)

print(f" ------------------------- Starting the Solving Process ------------------------- ")


# --- Set timeout if needed ---
if solver_time_limit != -1:
    # solver_time_limit is seconds; Z3 expects milliseconds
    z3.set_param("timeout", solver_time_limit * 1000)


# --- Solve ---
result = opt.check()
print("Result:", result)

# Try to see bounds
try:
    print("Lower bound:", opt.lower(obj))
    print("Upper bound:", opt.upper(obj))
except Z3Exception:
    print("Bounds unavailable.")



if result == sat:
    model = opt.model()

elif result == unknown:
    print("Timeout hit or optimality unknown. Best-so-far model returned.")
    model = opt.model()

else:  # unsat
    print("Unsatisfiable problem. Didn't find a solution.")
    model = None


# If we found a solution print it
if model is not None:
    # Print position of each book to read
    for p_i in p:
        print(f"{p_i} =", model[p_i])

    # Print boolean variable assignments
    for w in range(n_of_unknown_words):
        print(f"l_{w} =", model[eval(f"l_{w}_{n_of_books_to_read-1}")], f"-> {d[w]}")

    print("Objective value =", model.evaluate(f))




# In[44]:


# Storing the solution

print(f" ------------------------- Storing the Solution ------------------------- ")

solution_learned_ws = []
solution_not_learned_ws = []
solution_order = []

if result == sat:
    model = opt.model()
    
    # Extract reading order
    solution_order = [model[p[i]].as_long() for i in range(n_of_books_to_read)]
    
    # Classify words
    for w in range(n_of_unknown_words):
        l_var = globals()[f"l_{w}_{n_of_books_to_read-1}"]
        if model[l_var]:
            solution_learned_ws.append(w)
        else:
            solution_not_learned_ws.append(w)


# In[45]:


# Displaying Results of Solution

print(f"The best reading order is: {solution_order}")
print(f"    In the books names it is:\n\n    ",end="")
for i in range(len(solution_order)):
    b = solution_order[i]
    print(f"{D[b]} ({b}) -> {i+1}",end="")
print()

print(f"The Words learned ({len(solution_learned_ws)} words):\n\n",end="")
for w in solution_learned_ws:
    print(f"    ✓ - {d[w]} ({w})")
print()
print()

print(f"The Words not learned ({len(solution_not_learned_ws)} words):\n\n",end="")
for w in solution_not_learned_ws:
    print(f"    ✗ - {d[w]} ({w})")


