# This file is used to discover words in the books.
# It will create a file in the output folder called "output_discovered_words.txt"
# The file will contain all the words that are in the books in alphabetical order.
# This is independent of your vocabulary file.

# The file is created with the purpose of helping the user to create their vocabulary file.

# The only parameters from config.toml that are used here are:
# - n_books
# - language
# - paths.books_path
# - paths.output_path







# Description for me:
    # Read the config file to get the parameters (n_books, language, paths)
    # Read the books from the books folder (up to n_books) (configure cases of their different values)
    # For each book:
        # Read the book and extract ALL words
            # Pass each of them through the language model to filter out some words
                # Add Unknown words to an unknown words structure (think set is good, cause i dont need alphabetical order for these)
                # Add Known words to a known words structure (NEEDs alphabetical order)
                    # Add them to some structure (set, dictionary or list). Whatever is better to pick them later and print them in alphabetical order.
    # Write these discovered words to "output_discovered_words.txt", using the format in "output_discovered_format.txt"