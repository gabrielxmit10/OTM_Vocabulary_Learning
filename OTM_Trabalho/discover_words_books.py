# This file is used to discover words in the books.
# It will create a file in the output folder called "output_words_in_books.txt"
# The file will contain all the words that are in the books in alphabetical order.

# This is INDEPENDENT of your vocabulary file.

# The file is created with the purpose of helping the User create their vocabulary file.

# The only parameters from config.toml that are used here are:
# - n_books
# - language
# - paths.books_path
# - paths.output_path


# -------------------------------- CODE STARTS HERE --------------------------------


# --------------------------------- REMOVE START ----------------------------
def print_vector(vec):
    """Print vector elements with quotes, separated by spaces"""
    print(" ".join(repr(x) for x in vec))
    # print()
# --------------------------------- REMOVE END ----------------------------


# "C:\Users\55219\Documents\UFRJ\6° Período\OTM\.venv\Scripts\python.exe" discover_words_books.py

# Importing necessary libraries

from pathlib import Path
import tomllib
import spacy
from wordfreq import zipf_frequency, word_frequency, top_n_list
import math


hyphens = [
        "\xad",  # soft hyphen
    ]


# Load config.toml

config_path = Path(__file__).parent / "config.toml"
with config_path.open("rb") as f:
    config = tomllib.load(f)

language = config["general"]["language"]
books_path = config["paths"]["books_path"]
output_path = config["paths"]["output_path"]
n_books = config["general"]["n_books"]
# Directory containing config.toml
config_dir = config_path.parent
# Helper to resolve paths relative to config.toml
def resolve_config_path(raw_path):
    p = Path(raw_path)
    return (config_dir / p).resolve() if not p.is_absolute() else p.resolve()
books_dir = resolve_config_path(books_path)
output_dir = resolve_config_path(output_path)

supported_languages = {
    "en" : "en",
    "pt" : "pt",
    # "de" : "de",
}

supported_languages_map_for_wordfreq = {
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

# Functions for language treatment
# Function to treat unsupported language
def handle_unsupported_language_error(language):
    if language not in supported_languages:
        raise ValueError(f"Unsupported language: {language}. Supported languages are: {list(supported_languages_for_writing.keys())}")
# Returns lemma, pos (type) of word OR the word and -1 if not in morphological dictionary
def lemmatize_word(word, language): # DEPENDENT ON THE LANGUAGE AND MORPHOLOGICAL DICTIONARY (CREATE FOR GERMAN LATER)
    if language == 'pt':
        if word not in LEMMA_DICT:
            return word, -1  # Return the word itself and -1 for Failure
        
        lemma = LEMMA_DICT[word]["lemma"]
        pos = LEMMA_DICT[word]["type"]
        return lemma, pos
    
    if language == 'en':
        lemma = wn.morphy(word)
        list_of_lemmas = []
        for pos in ['n', 'v', 'a', 'r']: # noun, verb, adj, adv
            lemma = wn.morphy(word, pos)

            # print(word, lemma, pos, 0)
            # --- PRINTING ---
            
            if lemma is not None: list_of_lemmas.append(lemma)

        # Choose the shortest lemma if multiple found
        if list_of_lemmas != []: lemma = min(list_of_lemmas,key=len) 
        # print(word, lemma, "FINAL, for now") 
        # print()
        # --- PRINTING ---
        
        
        # lemma = wn_lemmatizer.lemmatize(word)
        

        if lemma is None: # word not found in WordNet

            return word, -1  # Return the word itself and -1 for Failure
        
        # To ensure the lemma is valid in WordNet
        # print(word, lemma, "=", wn.synsets(word))


        # if not all([([] != syn.instance_hypernyms()) for syn in wn.synsets(word)]): # All synsets are instance synsets (is proper noun)
        #     print(word, lemma, "=", wn.synsets(word))
        #     print(word, lemma, "=", [([] != syn.instance_hypernyms()) for syn in wn.synsets(word)])
        # print(word)


        return lemma, 0  # We dont use pos here, so we return 0
    
    if language == 'de':
        None # TBD: not done yet    
    
    else:
        raise ValueError(f"Unsupported language: {language}")  
# Function to check if a word is a proper noun only (ONLY USED FOR ENGLISH in create_vocab_lists)
def is_proper_noun_only(word):
    # WordNet is case-sensitive for proper nouns
    candidates = {word, word.capitalize()}

    # Collect all synsets for ALL POS
    all_synsets = []
    for w in candidates:
        all_synsets.extend(wn.synsets(w))

    if not all_synsets:
        return False  # unknown word -> treat as not proper



    # if word == 'was': print(all_synsets)
    # REMOVE LATER, COMMENT FOR DEBUG

    # For each synset:
    for syn in all_synsets:
        if syn.pos() != 'n':

            # Any non-noun sense = NOT a proper noun only
            return False
        
        # It's a noun sense; check if it's an instance noun
        if not syn.instance_hypernyms():
            # A noun without instance hypernyms = common noun
            return False

    # If we got here, EVERY sense is an instance noun
    return True
# Function to check if ALL NOUN senses of word are proper nouns (ONLY USED FOR ENGLISH in create_vocab_lists)
def all_noun_senses_are_proper(word):
    # WordNet is case-sensitive for proper nouns
    candidates = {word, word.capitalize()}

    noun_synsets = []
    for w in candidates:
        noun_synsets.extend(wn.synsets(w, pos='n'))

    if not noun_synsets:
        return False  # no noun meanings -> cannot say all noun meanings are proper

    for syn in noun_synsets:
        # If a noun sense does NOT have instance hypernyms → it's a common noun
        if not syn.instance_hypernyms():
            return False

    return True  # all noun senses are proper noun senses


# Separates book text into different sentences based on punctuation
def split_text_into_sentences(text, language):
    sentences = sent_tokenize(text, language=supported_languages_map_for_nltk[language])
    # Correcting some cases of new lines and hyphens
    for s in range(len(sentences)):
        sentences[s] = sentences[s].replace("\n", " ")
        for hyphen in hyphens:
            sentences[s] = sentences[s].replace(hyphen, "-")
    return sentences
# Separates sentence into different words/lemmas using Spacy and WordNet
# Returns list of words/lemmas and list of booleans indicating if word is considered
def split_sentence_into_words_lemmas(sentence, language):
    # Returns list of words or lemmas in the sentence AND a list of booleans indicating if word is considered
    # Technically, we could just return the considered words, but this is better for debugging later
    doc = nlp(sentence)
    
    # Creating raw_words_list and lemmas_list direct from Spacy
    raw_words_list = [w.text for w in doc]
    lemmas_list = [w.lemma_ for w in doc]

    # -------------------- REMOVE START ----------------
    # for w in range(len(raw_words_list)-1): 
    #     word = raw_words_list[w]
    #     word2 = raw_words_list[w+1]
    #     lemma = lemmas_list[w]
    #     lemma2 = lemmas_list[w+1]
    #     if word[0:3] == "vir": print("DEBUG WORD:", repr(word), repr(word2), repr(lemma), repr(lemma2))
    # print_vector(raw_words_list)
    # -------------------- REMOVE END ----------------

    word_or_lemma_list = [] # list of words or lemmas to be returned
    considered_bools = [] # list of booleans indicating if word is considered

    for i in range(len(raw_words_list)):
        word = raw_words_list[i]
        lemma = lemmas_list[i]
        end_word_form, considered_bool = decide_consideration(word,lemma,language)
        considered_bools.append(considered_bool)
        word_or_lemma_list.append(end_word_form)

    return word_or_lemma_list, considered_bools


# Function to handle book variables (n_books, list_of_books)
def handle_book_variables(n_books, list_of_books):
    if list_of_books == []: # Throws error if no books found
        raise FileNotFoundError(f"No books found in the specified path: {books_path}. Check if the path is correct and if there are .txt files in it.")
    
    # Create new variables to return
    n_books_new = n_books
    list_of_books_new = list_of_books.copy()

    # Adjust n_books and n_books_to_read based on the number of books found

    # Adjust n_books and list_of_books
    if n_books == -1 or n_books > len(list_of_books):
        n_books_new = len(list_of_books)
    elif 1 <= n_books <= len(list_of_books):
        list_of_books_new = list_of_books[:n_books]
    else:
        raise ValueError("n_books must be -1 or a value >= 1.")
    
    return n_books_new, list_of_books_new
# Function that decides if a word/lemma should be considered or unconsidered (DEPENDENT ON THE LANGUAGE)
# Returns word, boolean (indicating the word to be considered or not AND if it is considered)
def decide_consideration(word_input,lemma_input,language): 

    if language == 'pt':
        word = word_input.lower()
        lemma, pos = lemmatize_word(word, language)

        if pos == -1: # is NOT in morphological dictionary
            return word, False
        
        else: # is in morphological dictionary
            return lemma, True

    elif language == 'en':
        # We'll only use lemma_input from Spacy as word
        # word_input is for returning the original word if unconsidered
        word_lower = word_input.lower()
        word = lemma_input.lower()
        lemma, pos = lemmatize_word(word, language)

        if pos == -1: # is NOT in morphological dictionary
            return word_lower, False
        
        else: # is in morphological dictionary
            if is_proper_noun_only(word):
                # print("u", word, "proper noun")
                return word_lower, False
            elif all_noun_senses_are_proper(word): 
                # Chooses whichever lemma exists that 
                # is not noun in order VERB, ADJ, ADV
                for pos_ordered in ['v', 'a', 'r']:
                    lemma2 = wn.morphy(word, pos_ordered) 

                    if lemma2 is not None: 
                        # return first found
                        return lemma2, True
                        # print("c", word, lemma, pos_ordered, 1)
                # All were None, so proper noun (will never reach here if is_proper_noun_only works well)
                return word_lower, False
            else:
                # Passed all proper noun filters
                return lemma, True

    else:
        raise ValueError(f"Unsupported language: {language}") 
         





# Handling unsupported language
handle_unsupported_language_error(language)

# Importing Morphological Dictionary for given language
if language == 'pt':
    from others_folder.morphobr_loader import LEMMA_DICT
    from nltk.tokenize import sent_tokenize
elif language == 'en':
    from nltk.corpus import wordnet as wn
    from nltk.stem import WordNetLemmatizer
    from nltk.tokenize import sent_tokenize
    wn_lemmatizer = WordNetLemmatizer()
# elif language == 'de':
#     None # TBD: not done yet

nlp = spacy.load(supported_languages_map_for_spacy[language])



# Takes books names and sorts them alphanumerically for a list
list_of_books = sorted(list(books_dir.glob("*.txt"))) # Has the Path objects to each book
list_of_books_names = [p.name for p in list_of_books] # Just the names of the books as strings
# print(list_of_books_names)


# Handling for book variables (n_books, n_books_to_read, list_of_books)
book_arguments = handle_book_variables(n_books, list_of_books)
n_books, list_of_books = book_arguments

# Creation of D mapping
D = list_of_books_names[:len(list_of_books)]

# Opening each book and reading its context
# Adding them
book_full_text_list = []
for book_file in list_of_books:
    with book_file.open("r", encoding="utf-8") as f:
        book_full_text = f.read()
        book_full_text_list.append(book_full_text)
        # print(book_full_text)

    # print(f"Book: {book_file.name}")
    # print(book_full_text)
    # print()




# Creating B structure but with sentences as strings
#   B[b][s_d] is sentence s_d (string) in book b
B_sd_strings = []

# Building B_sd_strings with split sentences
for book_full_text in book_full_text_list:
    sentences = split_text_into_sentences(book_full_text, language)
    # print(sentences)
    B_sd_strings.append(sentences)

# ----------------------------- REMOVE START -----------------------------
# # Outputting the results to the console for checking
# for b in range(n_books):
#     print(f"Book {b} - {D[b]}: ")
#     for s_d in range(len(B_sd_strings[b])):
#         print(f"  Sentence {s_d}: {B_sd_strings[b][s_d]}")
#         print()
# print(len(B_sd_strings))
# ----------------------------- REMOVE END -----------------------------


# Creating B structure but with strings instead of the integers of words
#   B[b][s_d][w] is word w (string) in sentence s_d in book b
B_w_strings = [[] for b in range(n_books)]

# Creating v_of_vocab_sets (list of sets of vocab words in book b)
v_of_vocab_sets = [set() for b in range(n_books)]
# Creating v_of_unc_vocab_sets (list of sets of unconsidered vocab words in book b)
v_of_unc_vocab_sets = [set() for b in range(n_books)]

# Building B_w_strings with split and filtering of words/lemmas
# Building v_of_vocab_sets and v_of_unc_vocab_sets
for b in range(n_books): # For each book b
    # print(f"Book {b} - {D[b]}: ")
    # print()
    for s_d in range(len(B_sd_strings[b])): # For each sentence s_d in book b
        # Separate sentence into words
        word_list, considered_bools = split_sentence_into_words_lemmas(B_sd_strings[b][s_d], language)
        
        # print(f"Sentence {s_d}:")
        # print_vector(word_list)
        # print(considered_bools)
        # print()

        # list of words in the sentence to be added to B_w_strings[b]
        sentence_word_list = [] 

        # Add to sentence list only the considered words
        for i in range(len(considered_bools)):
            # If word was considered
            if considered_bools[i]: 
                sentence_word_list.append(word_list[i])
                v_of_vocab_sets[b].add(word_list[i])
            # If word was unconsidered
            else: 
                v_of_unc_vocab_sets[b].add(word_list[i])

        B_w_strings[b].append(sentence_word_list)




# Creating considered and unconsidered vocab lists UNITED for all books
considered_words_set_united = set()
unconsidered_words_set_united = set()
for book_set in v_of_vocab_sets: considered_words_set_united.update(book_set)
for book_set in v_of_unc_vocab_sets: unconsidered_words_set_united.update(book_set)

considered_words_list_united = sorted([w for w in considered_words_set_united if w.strip()])
unconsidered_words_list_united = sorted([w for w in unconsidered_words_set_united if w.strip()])

# Creating considered and unconsidered vocab lists per book b
# Does .strip() to remove " ", "\n", "\t", etc.
considered_words_list = [sorted([w for w in v_of_vocab_sets[b] if w.strip()]) for b in range(n_books)]
unconsidered_words_list = [sorted([w for w in v_of_unc_vocab_sets[b] if w.strip()]) for b in range(n_books)]




# Outputting the results to a file in output folder
output_file = output_dir / "output_words_in_books.txt"
with output_file.open("w", encoding="utf-8") as f:
    f.write(f"These are all the words CONSIDERED and UNCONSIDERED for learning for your given language, {supported_languages_for_writing[language]}, based on your book files:\n\n")
    f.write("- Considered words are mainly nouns, verbs, adjectives and adverbs.\n")
    f.write("- Unconsidered words are proper nouns, punctuation, made up words, errors in the language model, not well formatted words, etc.\n\n")
    f.write(2*"------------------------------------------------------------" + "\n\n")

    for b in range(n_books):
        f.write(f"Book {b} - {D[b]} ({len(considered_words_list[b]) + len(unconsidered_words_list[b])} words):\n\n")
        f.write("    " + f"These are the {len(considered_words_list[b])} words CONSIDERED for learning:\n\n")
        f.write("        " + " ".join(considered_words_list[b]) + "\n\n")

        f.write("    " + f"These are the {len(unconsidered_words_list[b])} words UNCONSIDERED for learning:\n\n")
        f.write("        " + " ".join(unconsidered_words_list[b]) + "\n\n")
        f.write(2*"------------------------------------------------------------" + "\n\n")
    
    f.write(f"For all books together, these are the {len(considered_words_list_united)+len(unconsidered_words_list_united)} words with alphabetical ordering:\n\n")
    f.write("    " + f"These are the {len(considered_words_list_united)} words CONSIDERED for learning:\n\n")
    f.write("        " + " ".join(considered_words_list_united) + "\n\n")

    f.write("    " + f"These are the {len(unconsidered_words_list_united)} words UNCONSIDERED for learning:\n\n")
    f.write("        " + " ".join(unconsidered_words_list_united) + "\n\n")
    f.write(2*"------------------------------------------------------------" + "\n\n")


    f.write(f"For all books together, these are the {len(considered_words_list_united)} considered words with frequency ordering:\n\n")
    f.write("    " + f"These are the {len(considered_words_list_united)} words CONSIDERED for learning (from more to less frequent):\n\n")
    considered_by_freq = sorted(considered_words_list_united, key=lambda w: -zipf_frequency(w, supported_languages_map_for_wordfreq[language]))
    freqs_in_order = [zipf_frequency(word, supported_languages_map_for_wordfreq[language]) for word in considered_by_freq]
    for i in range(math.ceil(freqs_in_order[0]), -1, -1):
        higher_int = i
        smaller_int = i - 1
        if higher_int == 0: break
        # f.write("        " + f"Zipf frequencies between {higher_int} and {smaller_int}:" + "\n\n")
        
        # Get words in this frequency range (sorted by frequency, highest first)
        words_in_range = [word for word, freq in zip(considered_by_freq, freqs_in_order) 
                        if smaller_int <= freq < higher_int]
        
        # Write all words in one line, separated by spaces
        if words_in_range:
            f.write("        " + " ".join(words_in_range) + "\n")
        
        f.write("\n")

    # f.write("        " + " ".join(considered_by_freq) + "\n\n")
    f.write(2*"------------------------------------------------------------" + "\n\n")

print()
print(f"Output of discover_words_books.py written to: {output_file}")

# Used for the file that will import this (discover_words_to_learn.py and notebook later)
size_considered = len(considered_words_list_united)


# --------------------- REMOVE START ------------------
# considered_by_freq = sorted(considered_words_list_united, key=lambda w: -zipf_frequency(w, supported_languages_map_for_wordfreq[language]))
# freqs_in_order = [zipf_frequency(word, supported_languages_map_for_wordfreq[language]) for word in considered_by_freq]
# # ten words per line with their frequencies like "(word, freq);"
# # Format words with frequencies, 10 per line

# print()
# print()
# words_per_line = 10
# formatted_lines = []
# for i in range(0, len(considered_by_freq), words_per_line):
#     line_words = considered_by_freq[i:i+words_per_line]
#     line_freqs = freqs_in_order[i:i+words_per_line]
#     line = " ".join(f"({word}, {freq:.2f});  " for word, freq in zip(line_words, line_freqs))
#     formatted_lines.append(line)

# # Print or write to file
# for line in formatted_lines:
#     print(line)

# --------------------- REMOVE END ------------------
