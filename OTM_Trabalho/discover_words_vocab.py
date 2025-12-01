# This file is used to discover words in the vocabulary file.
# It will create a file in the output folder called "output_words_in_vocab.txt"
# The file will contain all the words in the vocabulary file in their base form in alphabetical order,
#   showing which are CONSIDERED or UNCONSIDERED by the language model.

# This is INDEPENDENT of your book files.

# The file is created with the purpose of helping the User edit their vocabulary file,
#   seeing what is or not considered by the language model.

# We recommend the User to write the words in their BASE FORM (LEMMA) in the vocabulary file. 
#   That is, writing singular instead of plural for nouns, writing the infinitive form for verbs, etc.
#   This will avoid scenarios in which the lack of context makes it hard for the language model to find the correct form of the word.
#   (One example is the English word "legs", which will not take the lemma "leg", but instead assume the WordNet entry for legs meaning "staying power")

# The only parameters from config.toml that are used here are:
# - language
# - vocab_path
# - output_path

# Observation 1: Words will show up in lowercase in the output file.
# Observation 2: The words considered are mostly Nouns, Verbs, Adjectives and Adverbs. 
#   We tried our best to filter out proper nouns, punctuation, made up words, and others, but there may be some errors.

# -------------------------------- CODE STARTS HERE --------------------------------

# "C:\Users\55219\Documents\UFRJ\6° Período\OTM\.venv\Scripts\python.exe" discover_words_vocab.py

# Importing necessary libraries

from pathlib import Path
import tomllib
import fasttext
import spacy
from wordfreq import zipf_frequency, word_frequency, top_n_list



# ------------------------------- REMOVE --------------------------------
# Display the most 1000 most frequent words according to wordfreq that are not recognized by our lemmatize function
# language = 'pt'

# words = top_n_list(language, 10000)

# if language == 'pt':
#     from others_folder.morphobr_loader import LEMMA_DICT

# v=[]

# for word in words:
#     word2 = str(word).lower()
#     if language == 'pt':
#         if word2.lower() not in LEMMA_DICT:
#             freq = zipf_frequency(word2, language)
#             v.append([word2,freq])
        
#         # lemma = LEMMA_DICT[word]["lemma"]
#         # pos = LEMMA_DICT[word]["type"]
#         # return lemma, pos
# for i in v:
#     print(i[0], i[1], end=",  ")
# # print(v)
# ------------------------------- REMOVE --------------------------------









# Load config.toml

config_path = Path(__file__).parent / "config.toml"
with config_path.open("rb") as f:
    config = tomllib.load(f)

language = config["general"]["language"]
vocab_path = config["paths"]["vocab_path"]
output_path = config["paths"]["output_path"]
# Directory containing config.toml
config_dir = config_path.parent
# Helper to resolve paths relative to config.toml
def resolve_config_path(raw_path):
    p = Path(raw_path)
    return (config_dir / p).resolve() if not p.is_absolute() else p.resolve()
vocab_dir = resolve_config_path(vocab_path)
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
    
    elif language == 'en':
        lemma = wn.morphy(word)
        list_of_lemmas = []
        for pos in ['n', 'v', 'a', 'r']: # noun, verb, adj, adv
            lemma = wn.morphy(word, pos)

            # print(word, lemma, pos, 0)
            # --- PRINTING ---
            
            if lemma is not None: list_of_lemmas.append(lemma)

        # Choose the shortest lemma if multiple found
        if list_of_lemmas != []: lemma = min(list_of_lemmas,key=len) 

        if lemma is None: # word not found in WordNet
            return word, -1  # Return the word itself and -1 for Failure


        return lemma, 0  # We dont use pos here, so we return 0
    
    # elif language == 'de':
    #     None # TBD: not done yet    
    
    else:
        raise ValueError(f"Unsupported language: {language}")  
# Creates considered and unconsidered sorted vocab lists
def create_vocab_lists(vocab_set, language): # DEPENDENT ON THE LANGUAGE AND MORPHOLOGICAL DICTIONARY (CREATE FOR GERMAN LATER)
    # freq_threshold = { # words with pos = -1 and freq below this are unconsidered
    #     'pt': 4.2,
    #     'en': 3.7,
    #     # 'de': 4.2,
    #     } 
    considered, unconsidered = set(), set()

    for word in vocab_set:
        lemma, pos = lemmatize_word(word.lower(), language)

        if language == 'en':

            if pos == -1: # is NOT in morphological dictionary
                unconsidered.add(word)
                # print("u", word, "not in dict")
                continue
            else: # is in morphological dictionary

                # Test if word is only proper noun
                if is_proper_noun_only(word):
                    unconsidered.add(word)
                    # print("u", word, "proper noun")
                    continue

                # Is not only proper noun, but all noun senses are proper nouns
                elif all_noun_senses_are_proper(word): 
                    # Chooses whichever lemma exists that 
                    # is not noun in order VERB, ADJ, ADV
                    for pos_ordered in ['v', 'a', 'r']:
                        lemma = wn.morphy(word, pos_ordered) 
                        # Here we'll use morphy cause it returns None when pos doesn't exist for the word

                        if lemma is not None:
                            # Can add multiple times if word has multiple non-noun senses,
                            # but set will take care of duplicates 
                            considered.add(lemma)
                            continue
                    continue
                
                # If reached here, after the filter, word is in morphological dictionary 
                # but not a proper noun. So we add its lemma to considered
                else:
                    considered.add(lemma)
                    continue
        
        if language == 'pt':
            if pos == -1: # is NOT in morphological dictionary
                unconsidered.add(word)
                # print("c", word, "not in dict but considered")
                continue

            else: # is in morphological dictionary
                considered.add(lemma)
                continue

        # if language == 'de':
    
    # If considered and unconsidered have some overlap, remove from unconsidered
    # Hopefully if its wrong, it will be filtered correctly when in the book part
    unconsidered = unconsidered - considered

    considered_sorted_list = sorted(list(considered))
    unconsidered_sorted_list = sorted(list(unconsidered))

    return considered_sorted_list, unconsidered_sorted_list
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



# Handling unsupported language
handle_unsupported_language_error(language)

# Importing Morphological Dictionary for given language
if language == 'pt':
    from others_folder.morphobr_loader import LEMMA_DICT
elif language == 'en':
    from nltk.corpus import wordnet as wn
    from nltk.stem import WordNetLemmatizer
    wn_lemmatizer = WordNetLemmatizer()
# elif language == 'de':
#     None # TBD: not done yet


# Opening the vocab file and reading its content
vocab_file = vocab_dir / "vocab.txt"
with vocab_file.open("r", encoding="utf-8") as f:
    vocab_full_text = f.read()

# Creating the set of vocab words in a raw version (not lemmatized yet)
vocab_set = set(vocab_full_text.split())
# print(f"Initial vocab set: {vocab_set}")
# print()

# vocab_set.add("hipopótamos já dormiam tranquilos. Só o pequeno hipopótamo que não. Ele tentava fechar os olhos, se acomodar, mas depois de um tempo abria os olhos de novo e andava nervoso pelo cercado. Simplesmente não conseguia dormir.")



considered_words_list = []
unconsidered_words_list = []
# Creating the considered and unconsidered vocab lists
considered_words_list, unconsidered_words_list = create_vocab_lists(vocab_set, language)

# Outputting the results to a file in output folder
output_file = output_dir / "output_words_in_vocab.txt"
with output_file.open("w", encoding="utf-8") as f:
    f.write(f"These are all the words CONSIDERED for learning for your given language, {supported_languages_for_writing[language]}, based on your vocabulary file:\n\n")
    f.write("    " + " ".join(considered_words_list) + "\n\n")
    
    f.write(f"These are the words UNCONSIDERED for learning (punctuation, made up words, proper nouns, etc.) for your given language, {supported_languages_for_writing[language]}, based on your vocabulary file:\n\n")
    f.write("    " + " ".join(unconsidered_words_list) + "\n")

print()
print(f"Output of discover_words_vocab.py written to: {output_file}")



# print()
# print(language)
# print("Considered words:", considered_words_list)
# print("Unconsidered words:", unconsidered_words_list)


# -------------------------------- USE TO BUILD VOCAB_PATH HANDLING --------------------------------

# # Gets vocab words from files (No filter made yet, just listing all words)
# # This not only gets the path to the ONLY file in vocab_path, but also reads it
# # Variables from config treated here: vocab_path

# # Load vocab from file
# files_in_vocab_dir = list(vocab_dir.glob("*.txt"))
# if files_in_vocab_dir == []:
#     raise FileNotFoundError(f"No vocabulary file found in the specified path: {vocab_path}")
# elif len(files_in_vocab_dir) > 1:
#     raise FileExistsError(f"Multiple vocabulary files found in the specified path: {vocab_path}. Please ensure only one vocabulary file is present.")
# else: vocab_file = files_in_vocab_dir[0]

# with vocab_file.open("r", encoding="utf-8") as vf:
#     vocab_text = vf.read()

# # Separates by whitespace (space, \n, tab, multiple spaces)
# vocab_set = set(vocab_text.split())  # Using set to avoid duplicates
# vocab_set
