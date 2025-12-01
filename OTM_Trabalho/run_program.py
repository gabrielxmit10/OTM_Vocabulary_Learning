# This file is responsible for running the program. Running the notebook by importing it as a module,
# and then outputting the results to a text file in output_results.txt in the output folder.






# -------------------------------- CODE STARTS HERE --------------------------------

# "C:\Users\55219\Documents\UFRJ\6° Período\OTM\.venv\Scripts\python.exe" run_program.py


# Importing libraries

import subprocess
import sys
import importlib.util
from pathlib import Path
import tomllib




# Importing from config

config_path = Path(__file__).parent / "config.toml"
with config_path.open("rb") as f:
    config = tomllib.load(f)

# n_books = config["general"]["n_books"]
# n_books_to_read = config["general"]["n_books_to_read"]
vocab_known = config["general"]["vocab_known"]
language = config["general"]["language"]
# books_path = config["paths"]["books_path"]
# vocab_path = config["paths"]["vocab_path"]
output_path = config["paths"]["output_path"]

# Directory containing config.toml
config_dir = config_path.parent
# Helper to resolve paths relative to config.toml
def resolve_config_path(raw_path):
    p = Path(raw_path)
    return (config_dir / p).resolve() if not p.is_absolute() else p.resolve()
# books_dir = resolve_config_path(books_path)
# vocab_dir = resolve_config_path(vocab_path)
output_dir = resolve_config_path(output_path)




# Importing from notebook (making it .py first)

def import_notebook(nb_path):
    import subprocess, sys, importlib.util
    from pathlib import Path
    
    nb_path = Path(nb_path)
    py_path = nb_path.with_suffix(".py")
    
    subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "script", str(nb_path)], check=True)
    
    spec = importlib.util.spec_from_file_location(nb_path.stem, str(py_path))
    mod = importlib.util.module_from_spec(spec)
    
    # Set the module's __file__ attribute so relative paths work correctly
    sys.modules[spec.name] = mod
    
    spec.loader.exec_module(mod)
    return mod

creating_expression = import_notebook("program_folder/creating_expression.ipynb")

B = creating_expression.B
D = creating_expression.D
d = creating_expression.d
d_set = creating_expression.d_set
solution_learned_ws = creating_expression.solution_learned_ws
solution_not_learned_ws = creating_expression.solution_not_learned_ws
solution_order = creating_expression.solution_order
n_of_books = creating_expression.n_of_books
n_of_books_to_read = creating_expression.n_of_books_to_read
solver_time_limit = creating_expression.solver_time_limit





# Outputting the results to a file in output folder

output_file = output_dir / "output_results.txt"

# Get books not read (if applicable)
books_not_read = []
if n_of_books_to_read < n_of_books:
    books_read_indices = set(solution_order)
    books_not_read = [i for i in range(n_of_books) if i not in books_read_indices]

# Sort words alphabetically
words_learned_sorted = sorted([d[w] for w in solution_learned_ws])
words_not_learned_sorted = sorted([d[w] for w in solution_not_learned_ws])

with output_file.open("w", encoding="utf-8") as f:
    # Header
    f.write(f"You have {n_of_books} books and want to read {n_of_books_to_read}.\n")
    f.write("Here are the results for your choice.\n\n")
    
    # Summary with time limit note if applicable
    f.write(f"You can learn up to {len(solution_learned_ws)} words out of {len(d)} unknown words found in these {n_of_books} books.\n")
    if solver_time_limit != -1:
        f.write(f"(We aren't sure if this is the maximum possible, but it's what our solver found, given your time limit = {solver_time_limit} seconds).\n\n")
    else:
        f.write("\n")
    
    f.write(2*"------------------------------------------------------------" + "\n\n")
    
    # Reading order
    f.write("Order to read the books in:\n\n")
    f.write(f"    Order with numbers: {solution_order}\n\n")
    f.write("    Order with names:\n")
    for i, book_idx in enumerate(solution_order):
        f.write(f"        {D[book_idx]} (book {book_idx}) -> Position {i}\n")
    f.write("\n")
    
    f.write(2*"------------------------------------------------------------" + "\n\n")
    
    # Words learned
    f.write(f"With this order you will learn {len(solution_learned_ws)} words in all books. These words are:\n\n")
    f.write("    " + " ".join(words_learned_sorted) + "\n\n")
    
    f.write(2*"------------------------------------------------------------" + "\n\n")
    
    # Words NOT learned
    f.write(f"With this order you will NOT learn {len(solution_not_learned_ws)} words. These words are:\n\n")
    f.write("    " + " ".join(words_not_learned_sorted) + "\n\n")
    
    f.write(2*"------------------------------------------------------------" + "\n\n")
    
    # Books not read (only if applicable)
    if books_not_read:
        f.write("Books not read:\n\n")
        books_not_read_names = [f"{D[b]} (book {b})" for b in books_not_read]
        f.write("    " + ", ".join(books_not_read_names) + "\n\n")
        f.write(2*"------------------------------------------------------------" + "\n\n")
    
    # Note about vocabulary
    f.write("Note:\n\n")
    f.write("The program might not have considered some words given by your vocabulary file.\n")
    f.write("This is because they might be names, made up words, or words that don't fit the language model in the given language.\n")
    f.write("Look into the file output_words_in_vocab.txt for more information.\n")


print()
print(f"Output of run_program.py written to: {output_file}")