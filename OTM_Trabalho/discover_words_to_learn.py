# This file is used to discover words the program will consider for learning.
# It will create a file in the output folder called "output_words_to_learn.txt"
# The file will contain all the words to learn in alphabetical order.

# This is DEPENDENT on your vocabulary file and your book files.
# This file also uses the files discover_words_books.py and discover_words_vocabulary.py as it calls them directly.

# The output file created is with the purpose of helping the User understand what the program is considering,
# before running it with undesidered words, considered or unconsidered.
# (ex: user forgot to include some words in their vocabulary that show in the books, so they can add them before running the main program)

# The only parameters from config.toml that are used here are:
# - n_books (indirectly)
# - language (directly)
# - vocab_known (directly)
# - books_path (indirectly)
# - vocab_path (indirectly)
# - output_path (directly)

# Observation 1: This file is also called in the main notebook of the program, as it creates some important structures that are used later on.



# -------------------------------- CODE STARTS HERE --------------------------------

# ------------------------- REMOVE START -------------------------
# "C:\Users\55219\Documents\UFRJ\6° Período\OTM\.venv\Scripts\python.exe" discover_words_to_learn.py
# ------------------------- REMOVE END -------------------------


# Importing necessary libraries

from pathlib import Path
import tomllib


# Load config.toml

config_path = Path(__file__).parent / "config.toml"
with config_path.open("rb") as f:
    config = tomllib.load(f)

language = config["general"]["language"]
vocab_known = config["general"]["vocab_known"] 
output_path = config["paths"]["output_path"]
# These are the only we need directly from the config file. 
# The rest we will import from discover_words_books.py and discover_words_vocab.py

# Directory containing config.toml
config_dir = config_path.parent
# Helper to resolve paths relative to config.toml
def resolve_config_path(raw_path):
    p = Path(raw_path)
    return (config_dir / p).resolve() if not p.is_absolute() else p.resolve()

output_dir = resolve_config_path(output_path)

supported_languages = {
    "en" : "en",
    "pt" : "pt",
    # "de" : "de",
}

supported_languages_for_writing = {
    "en" : "English (en)",
    "pt" : "Portuguese (pt)",
    # "de" : "German (de)",
}

supported_languages_map_for_spacy = {
    "en": "en_core_web_sm",   # English
    "pt": "pt_core_news_sm",   # Portuguese
    # "de": "de_core_news_sm",   # German
}

supported_languages_map_for_nltk = {
    "en": "english",   # English
    "pt": "portuguese",   # Portuguese
    # "de": "german",   # German
}

# Some Definitions of Symbols (0,1,AND,OR)
# Will be imported in the notebook later
_false = '0'
_true = '1'
_and = "AND"
_or = "OR"


# Functions for language treatment
# Function to treat unsupported language
def handle_unsupported_language_error(language):
    if language not in supported_languages:
        raise ValueError(f"Unsupported language: {language}. Supported languages are: {list(supported_languages_for_writing.keys())}")
# Function to handle vocab_known variable
def handle_vocab_known_variable(vocab_known):
    if vocab_known not in [True, False]:
        raise ValueError(f"Invalid vocab_known value: {vocab_known}. It must be either 'true' or 'false'.")

# Function to map word to its index in d
def map_word_to_ds(word):
    if word in d_set: # already mapped
        word_index = d_set[word]
    else: # not mapped yet (so add to d and d_set)
        word_index = len(d)
        d_set[word] = word_index
        d.append(word)
    return word_index
# Function to add to set_of_ks
def add_to_set_of_ks(base_k,v_prime):
    # adds to set_of_ks with base_k as key and v_prime as value
    if base_k not in set_of_ks:
        set_of_ks[base_k] = v_prime
        return True # Indicates that was added
    return False # Indicates that wasnt added (was already there)


# Handling config variables
handle_unsupported_language_error(language)
handle_vocab_known_variable(vocab_known)

# Importing from the other files

# Imports from discover_words_books
import sys
import importlib
import time as time_module

print(f"Starting reload at {time_module.time()}")

# Remove module completely from cache
if 'discover_words_books' in sys.modules:
    print("Module found in cache, deleting...")
    del sys.modules['discover_words_books']
else:
    print("Module not in cache")

# Force fresh import (this will re-execute all module-level code)
print("Importing module...")
import discover_words_books
print(f"Import completed at {time_module.time()}")


from discover_words_books import n_books, D, B_w_strings as B_strings, size_considered
from discover_words_vocab import considered_words_list as vocab_words_list

# Convert to set cause from the file we get the ordered list,
# but here its more helpful to have it as a set for faster checking
vocab_words_set = set(vocab_words_list)


# ----------------------- REMOVE START -----------------------
# Assuming vocab_known is False
def f(order):
    n_learned = 0
    learned_words = set()
    for b in range(len(B_strings)):
        for s_d in range(len(B_strings[b])):
            for word in B_strings[b][s_d]:
                if vocab_known:
                    if word in vocab_words_set:
                        learned_words.add(word)
                else:
                    if word not in vocab_words_set:
                        learned_words.add(word)


    B_strings_copy = [[] for _ in range(len(B_strings))]
    for b in order:
        for s_d in range(len(B_strings[b])):
            B_strings_copy[b].append(sorted(list(set(B_strings[b][s_d]))))

            unknown_words = []
            for word in B_strings_copy[b][s_d]:
                if word not in learned_words:
                    unknown_words += [word]
                    continue
                else: continue # word is known
            if len(unknown_words) == 1:
                
                word = unknown_words[0]
                if word == "equipe":
                    print(f"Learned word '{word}' in book {b}, sentence {s_d} -> {B_strings_copy[b][s_d]}")


                learned_words.add(word)
                n_learned += 1
    return n_learned

from itertools import permutations

def k_permutations(n: int, k: int):
    """
    Generate all permutations of length k from numbers 0..n-1 (no repetition).
    Returns an iterator of tuples.
    """
    if not (0 <= k <= n):
        raise ValueError("k must be between 0 and n inclusive")
    return permutations(range(n), k)

# Example:
# n=6, k=3 -> tuples of 3 positions using numbers 0..5 without repeats
# for order in k_permutations(6, 3):
#     words_learned_in_order = f(order)
#     print(f"Results with order of books {order}: {words_learned_in_order}")


# print(f"Results with order of books [2,1,0]: {f([2,1,0])}")
# print(f"Results with order of books [0,2,3]: {f([0,2,3])}")
# print(f"Results with order of books [4,0,3]: {f([4,0,3])}")
# print(f"Results with order of books [1,2,0]: {f([1,2,0])}")
# print(f"Results with order of books [2,3,4]: {f([2,3,4])}")
# print(f"Results with order of books [3,1,2]: {f([3,1,2])}")
# print(f"Results with order of books [3,1,5]: {f([3,1,5])}")
# print(f"Results with order of books [5,4,1]: {f([5,4,1])}")



# ----------------------- REMOVE END -----------------------








# Creating B (list of books, sentences, words)
B = []

# Creating d (mapping of int to word)
d = []
# Creating d_set (dict that maps word to its int in d) 
# (opposite of d)
d_set = {}

# Creating v_of_vocab_sets (list of sets of vocab words in each book)
v_of_vocab_sets = []

# Creating v_of_alone_sets (list of sets of vocab words alone in a sentence in each book)
# Will be used to display which words will be learned for sure in each book
v_of_alone_sets = []

# Creating set_of_ks (set that stores values of ks, like K(b,w,t,bv) and k(b,s_d,w,t))
set_of_ks = {}

# Creating a (set that represents appearance of words in books)
# if a[b,s_d,w] exists, then its value is the number of the sentence where w last showed up before sentence s_d in b. 
# If doesnt exist, w didnt show up before s_d
a = {}


# Building B, d, d_set, v_of_vocab_sets, v_of_alone_sets, set_of_ks, a
for b in range(n_books):
    A = {} # represents last appereance of word w in book b, or -1 if was alone in sentence
    b_set_of_alone_words = set() # set of vocab words that were alone in a book sentence, and will therefore be learned
    # (we'll add to v_of_alone_sets later)
    b_vocab = set() # set of vocab words in book b
    # (we'll add to v_of_vocab_sets later)
    b_sentence_list = [] # list of sentences in book b
    # (we'll add to B later)

    empty_sentence_count = 0 # for adding value of 


    for s_d in range(len(B_strings[b])):
        s_d_word_set = set() # set of ws in sentence s_d of book b
        # (we'll add to b_sentence_list later)

        # sort words in sentence and eliminate duplicates
        B_strings[b][s_d] = sorted(list(set(B_strings[b][s_d])))


        # Removing known words (by vocab or by showing alone before)
        B_strings_sentence_copy = B_strings[b][s_d].copy() # we'll modify this copy and attribute after all removals
        for word in B_strings[b][s_d]:
            if vocab_known:
                if word in vocab_words_set:
                    # remove if word is known
                    B_strings_sentence_copy.remove(word)
                else:
                    # remove if word showed up alone before
                    if A.get(word,0) == -1: B_strings_sentence_copy.remove(word)
            else:
                if word not in vocab_words_set: 
                    # remove if word is known
                    B_strings_sentence_copy.remove(word)
                else:
                    # remove if word showed up alone before
                    if A.get(word,0) == -1: B_strings_sentence_copy.remove(word)
        B_strings[b][s_d] = B_strings_sentence_copy



        # Turned to set for cases of same word twice in sentence
        for word in B_strings[b][s_d]:

            # Map word to w_index
            w_index = map_word_to_ds(word)

            # Add w to the vocab set of book b
            b_vocab.add(w_index)

            # Add w_index to sentence s_d (if hasn't appeared alone before, eliminates redundancy)
            # Create values of a[b,s_d,w_index] and A[w_index] when necessary
            if len(B_strings[b][s_d]) == 1:
                if(word in A and A[word] == -1): 
                    # w already showed up before alone in a sentence
                    # we dont need to add it to the sentence
                    continue
                else: 
                    # showed up alone, for the first time
                    A[word] = -1 
                    s_d_word_set.add(w_index)
            else: # sentence has more than 1 word
                if s_d == 0:
                    if w_index not in A:
                        # showed up in first sentence, and is not alone
                        A[word] = 0 # register appearance
                        s_d_word_set.add(w_index) # add to sentence
                    else: # word showed up multiple times in first sentence 
                        # (should not happen cause of set, but just in case)
                        continue # was already added to A and sentence, so skip
                else: # s_d != 0 and sentence has more than 1 word
                    if word in A:
                        if A[word] == -1: 
                            continue # showed up alone before, so skip
                        else:
                            # showed up before, not alone
                            # Have to do -empty_sentence_count cause empty sentences 
                            # dont show in end count but show in s_d indexing
                            s_d_corrected = s_d - empty_sentence_count
                            a[b,s_d_corrected,w_index] = A[word] # create a[b,s_d,w_index]
                            A[word] = s_d_corrected # register new appearance
                            s_d_word_set.add(w_index) # add to sentence
                    else: # first appearance of w in book b
                        s_d_corrected = s_d - empty_sentence_count
                        A[word] = s_d_corrected # register appearance
                        s_d_word_set.add(w_index) # add to sentence


            
        # Creating list out of set for sentence s_d
        s_d_word_list = sorted(list(s_d_word_set))
        # Adding what we created in the word loop    
        # Add sentence if not empty (empty means all words were known)
        if s_d_word_list != []:
            b_sentence_list.append(s_d_word_list) # Add sentence s_d to B[b]

            # Word is alone in sentence
            if len(s_d_word_list) == 1:
                w = s_d_word_list[0]
                # add to set of alone words for book b
                b_set_of_alone_words.add(w) 
                
                # Create K_reread(b,w,0)
                base_K = f"K_reread({b},{w},0)"
                add_to_set_of_ks(base_K,[_true])
        else: empty_sentence_count += 1
    
    # Adding what we created in the sentence loop
    B.append(b_sentence_list) # Add book b to B
    v_of_vocab_sets.append(b_vocab) # Add book b's vocab set to v_of_vocab_sets
    v_of_alone_sets.append(b_set_of_alone_words) # Add book b's alone vocab set to v_of_alone_sets



# Now, we have:
# - B (for the notebook, same name)
# - d, d_set (for the notebook, if the output part is there)
# - v_of_vocab_sets (for the notebook, same name) (for outputting here)
# - v_of_alone_sets (for outputting here)
# - set_of_ks (for the notebook, same name)
# - D (for the notebook, same name)


# Lets use these to create variables for the output file

# list of lists of words to learn per book
words_to_learn_per_book = [] 
# Building words_to_learn_per_book
for b in range(n_books):
    words_to_learn_in_b = []
    for w in v_of_vocab_sets[b]:
        word = d[w]
        words_to_learn_in_b.append(word)
    words_to_learn_per_book.append(sorted(words_to_learn_in_b))

# list of lists of words learned for sure per book
words_learned_for_sure_per_book = []
# Building words_learned_for_sure_per_book
for b in range(n_books):
    words_learned_for_sure_in_b = []
    for w in v_of_alone_sets[b]:
        word = d[w]
        words_learned_for_sure_in_b.append(word)
    words_learned_for_sure_per_book.append(sorted(words_learned_for_sure_in_b))

# list of words to learn for all books together
words_to_learn_united = []
# Building words_to_learn_united
all_words_set = set()
for b in range(n_books):
    for w in v_of_vocab_sets[b]:
        all_words_set.add(d[w])
words_to_learn_united = sorted(list(all_words_set))

# list of words learned for sure for all books together (for the case we read all)
words_learned_for_sure_united = []
# Building words_learned_for_sure_united
all_alone_words_set = set()
for b in range(n_books):
    for w in v_of_alone_sets[b]:
        all_alone_words_set.add(d[w])
words_learned_for_sure_united = sorted(list(all_alone_words_set))

# Now, we have:
# - words_to_learn_per_book
# - words_learned_for_sure_per_book
# - words_to_learn_united
# - words_learned_for_sure_united


# ----------------------- REMOVE START -----------------------
# # # Prints sentences from books with words to learn as strings
# # for b in range(len(B)):
# #     print(f"Book {b} - {D[b]}" + "-------------------------------------------")
# #     for s_d in range(len(B[b])):
# #         print([d[w] for w in B[b][s_d]])

# Prints B structure
for b in range(len(B)):
    print(f"Book {b} - {D[b]} ({len(B[b])} sents): \n")
    for s_d in range(len(B[b])):
        print("     ",s_d, ": ", B[b][s_d])
    print()
print()
# print()
# # Prints set_of_ks
# print("\nSet of ks:")
# for k in set_of_ks:
#     print(f"    {k} : {set_of_ks[k]}")

# Prints a
# print("\nAppearences (a):")
# for key in a:
#     print(f"    a{key} : {a[key]}")

# # Prints how many times word d[w] appears in books
# d_2 = [[0 for b in range(n_books)] for _ in range(len(d))]
# for b in range(n_books):
#     for s_d in range(len(B[b])):
#         for w in B[b][s_d]:
#             d_2[w][b] += 1
# print("\nWord appears in book (d_2):")
# # for w in range(len(d)):
# #     print(f"    {d[w]} : {d_2[w]}")
# # Shows only the top words that appear the most in books
# top = 30
# word_appearances = []
# for w in range(len(d)):
#     total_appearances = sum(d_2[w])
#     word_appearances.append((total_appearances, w))
# word_appearances.sort(reverse=True)
# print(f"\nTop {top} words that appear the most in books:")
# for i in range(top):
#     # print(f"    {d[w]} : {d_2[w]}")
#     total_appearances, w = word_appearances[i]
#     print(f"    {i+1}. {d[w]} : {d_2[w]}, Total = {total_appearances}, Avg = {total_appearances/n_books:.2f}")



# print()
# print()
# maximum_in_d_2 = max([max(d_2[w]) for w in range(len(d))])
# print(f"Maximum times a word appears in a book: {maximum_in_d_2}")
# print()


# # Prints 5 biggest sizes of sentences in books and what sentence and book they are from)
# print("\n5 Biggest sentence sizes in books:")
# biggest_sentence_sizes = []
# for b in range(n_books):
#     for s_d in range(len(B[b])):
#         sentence_size = len(B[b][s_d])
#         biggest_sentence_sizes.append((sentence_size, b, s_d))
# biggest_sentence_sizes.sort(reverse=True)
# print()
# print()
# for i in range(5):
#     size, b, s_d = biggest_sentence_sizes[i]
#     print(f"    Size ({i+1}): {size}\n      Book {b} - Sentence {s_d}:")
#     print("        ", [w for w in B[b][s_d]])
#     print("        ", [d[w] for w in B[b][s_d]])

# set_of_k_tests[b,w] = (sent,value_of_k_test) for key 
set_of_k_tests = {}
# # Prints path from number through a's til finds start
# def k_test(b,s_d,w,first=False):

#     if set_of_k_tests.get((b,w),None) is not None:
#         if set_of_k_tests[b,w][0] == s_d:
#             print(f"We used stored k_test({b},{s_d},{w})")
#             return set_of_k_tests[b,w][1]
        

#     path = []
#     v_of_ands = []
#     v_of_ors = []
#     # current_s_d = s_d
#     # while current_s_d >= 0:
#     #     path.append((b,current_s_d,w))
#     #     if (b,current_s_d,w) in a:
#     #         current_s_d = a[b,current_s_d,w]
#     #     else:
#     #         break
#     if len(B[b][s_d]) == 1 and B[b][s_d][0] == w:
#         # print(f"({b},{s_d},{w})")
#         return _true

#     if w not in B[b][s_d]:
#         raise ValueError(f"Error in k_test: word {w} not in B[{b}][{s_d}]")
#     else:
#         if a.get((b,s_d,w),None) is not None:
#             if a[b,s_d,w] >= s_d:
#                 raise ValueError(f"Error in a structure: a[{b},{s_d},{w}] = {a[b,s_d,w]} >= {s_d} = s_d")
#             # print(f'I (k_test({b},{s_d},{w})) am calling k_test({b},{a[b,s_d,w]},{w})')
#             # print()
#             v_of_ors.append(k_test(b,a[b,s_d,w],w))
#         else:
#             # NOT
#             # print(f"({b},{s_d},{omega})")
#             v_of_ors.append((b,s_d,w))
#         if first: print(f"WORD W {w} DONE --------------------------------------------\n")


#     for omega in B[b][s_d]:
#         if omega == w: continue
#         if a.get((b,s_d,omega),None) is not None:
#             if a[b,s_d,omega] >= s_d:
#                 raise ValueError(f"Error in a structure: a[{b},{s_d},{omega}] = {a[b,s_d,omega]} >= {s_d} = s_d")
            
#             # print(f'I (k_test({b},{s_d},{w})) am calling k_test({b},{a[b,s_d,omega]},{omega})')
#             print()
#             v_of_ands.append(k_test(b,a[b,s_d,omega],omega))
#         else:
#             # NOT
#             # print(f"({b},{s_d},{omega})")
#             v_of_ands.append((b,s_d,omega))
#         if first: print(f"OMEGA {omega} DONE --------------------------------------------\n")
    
#     if any([isinstance(item, tuple) or [isinstance(item, tuple)] or item == [] for item in v_of_ands]): v_of_ands = []
#     if all([item == _true for item in v_of_ands]): v_of_ands = _true

#     path = [v_of_ors,v_of_ands]
#     if any([item == _true or [_true] for item in path]): path = _true
#     if all([isinstance(item, tuple) or [isinstance(item, tuple)] or item == [] for item in v_of_ors]): path = []

#     print(f"k_test({b},{s_d},{w}) = {path}".replace(" ", ""))
#     print()

#     if set_of_k_tests.get((b,w),None) is None:
#         set_of_k_tests[b,w] = (s_d,path)
#     elif set_of_k_tests[b,w][0] < s_d:
#         set_of_k_tests[b,w] = (s_d,path)
#     elif set_of_k_tests[b,w][0] == s_d:
#         pass # keep existing
#     # elif set_of_k_tests[b,w][0] > s_d:
#     #     raise ValueError(f"There is a sentence {set_of_k_tests[b,w][0]} stored in set_of_k_tests({b},{w}) that is greater than current sentence {s_d}, which should not happen.")
#     #     My Bad, this is actually possible

#     if len(set_of_k_tests) > 1500:
#         raise ValueError(f"Warning: set_of_k_tests has grown too large ({len(set_of_k_tests)} entries). Stopping further k_test calls to prevent memory issues.")

#     return path







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



# set_of_k_tests[b,s_d,w,t] will be cache for the ks
set_of_k_tests = LRUCache(max_size=1500)

# # Prints path from number through a's til finds start
# def k_test(b,s_d,w,first=False):

#     cached_k = set_of_k_tests.get((b,s_d,w,0))
#     if cached_k is not None:
#         print(f"We used stored k_test({b},{s_d},{w},0)")
#         return cached_k
        

#     path = []
#     v_of_ands = []
#     v_of_ors = []
#     # current_s_d = s_d
#     # while current_s_d >= 0:
#     #     path.append((b,current_s_d,w))
#     #     if (b,current_s_d,w) in a:
#     #         current_s_d = a[b,current_s_d,w]
#     #     else:
#     #         break
#     if len(B[b][s_d]) == 1 and B[b][s_d][0] == w:
#         # print(f"({b},{s_d},{w})")
#         return _true

#     if w not in B[b][s_d]:
#         raise ValueError(f"Error in k_test: word {w} not in B[{b}][{s_d}]")
#     else:
#         if a.get((b,s_d,w),None) is not None:
#             if a[b,s_d,w] >= s_d:
#                 raise ValueError(f"Error in a structure: a[{b},{s_d},{w}] = {a[b,s_d,w]} >= {s_d} = s_d")
#             # print(f'I (k_test({b},{s_d},{w})) am calling k_test({b},{a[b,s_d,w]},{w})')
#             # print()
#             v_of_ors.append(k_test(b,a[b,s_d,w],w))
#         else:
#             # NOT
#             # print(f"({b},{s_d},{omega})")
#             v_of_ors.append((b,s_d,w))
#         if first: print(f"WORD W {w} DONE --------------------------------------------")

#     # Goes inverted over sentence B, for testing
#     for i in range(len(B[b][s_d])-1,-1,-1):
#         omega = B[b][s_d][i]
#         if omega == w: continue
#         if a.get((b,s_d,omega),None) is not None:
#             if a[b,s_d,omega] >= s_d:
#                 raise ValueError(f"Error in a structure: a[{b},{s_d},{omega}] = {a[b,s_d,omega]} >= {s_d} = s_d")
            
#             # print(f'I (k_test({b},{s_d},{w})) am calling k_test({b},{a[b,s_d,omega]},{omega})')
#             # print()
#             v_of_ands.append(k_test(b,a[b,s_d,omega],omega))
#         else:
#             # NOT
#             # print(f"({b},{s_d},{omega})")
#             v_of_ands.append((b,s_d,omega))
#         if first: print(f"OMEGA {omega} DONE --------------------------------------------")
    
#     if any([isinstance(item, tuple) or [isinstance(item, tuple)] or item == [] for item in v_of_ands]): v_of_ands = []
#     if all([item == _true for item in v_of_ands]): v_of_ands = _true

#     path = [v_of_ors,v_of_ands]
#     if any([item == _true or [_true] for item in path]): path = _true
#     if all([isinstance(item, tuple) or [isinstance(item, tuple)] or item == [] for item in v_of_ors]): path = []

#     print(f"k_test({b},{s_d},{w}) = {path}".replace(" ", ""))
#     # print()

    
#     if cached_k is None:
#         set_of_k_tests.put((b,s_d,w,0),path)
#     else:
#         pass # keep existing

#     # if len(set_of_k_tests) > 2000:
#     #     raise ValueError(f"Warning: set_of_k_tests has grown too large ({len(set_of_k_tests)} entries). Stopping further k_test calls to prevent memory issues.")

#     return path

# k_test(0,108,4,True)
# k_test(0,19,64)
# ----------------------- REMOVE END -----------------------




# Outputting the results to a file in output folder
output_file = output_dir / "output_words_to_learn.txt"
with output_file.open("w", encoding="utf-8") as f:
    f.write(f"These are all the words for learning for your given language, {supported_languages_for_writing[language]}, based on your vocabulary and {n_books} book files.\n")
    f.write("You chose that the words in your vocabulary file are considered as ")
    if vocab_known: f.write("KNOWN.\n\n")
    else: f.write("UNKNOWN.\n\n")
    f.write(f"In total, we found {len(words_to_learn_united)} to learn out of the {size_considered} considered words in these books:\n\n")

    f.write(2*"------------------------------------------------------------" + "\n\n")

    # place a for loop here to write all words in each book
    for b in range(n_books):
        f.write(f"Book {b} - {D[b]} ({len(words_to_learn_per_book[b])} words):\n\n")
        f.write("    " + f"These are the {len(words_to_learn_per_book[b])} words to learn in this book:\n\n")
        f.write("        " + " ".join(words_to_learn_per_book[b]) + "\n\n")

        f.write("    " + f"These are the {len(words_learned_for_sure_per_book[b])} words you will learn for sure if you read this book:\n\n")
        f.write("        " + " ".join(words_learned_for_sure_per_book[b]) + "\n\n")
        f.write(2*"------------------------------------------------------------" + "\n\n")
    
    f.write(f"For all books together ({len(words_to_learn_united)} words):\n\n")
    f.write("    " + f"These are the {len(words_to_learn_united)} words to learn in all the {n_books} books:\n\n")
    f.write("        " + " ".join(words_to_learn_united) + "\n\n")

    f.write("    " + f"These are the {len(words_learned_for_sure_united)} words you will learn for sure if you read all {n_books} books:\n\n")
    f.write("        " + " ".join(words_learned_for_sure_united) + "\n\n")
    f.write(2*"------------------------------------------------------------" + "\n\n")

print()
print(f"Output of discover_words_to_learn.py written to: {output_file}")



# árvore, galho, ali, azul
# print("árvore, galho, ali, azul")
