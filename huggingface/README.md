This directory is to maintain the 🤗 support of METL.

Herein are a few files to facilitate uploading the wrapper to 🤗. First, combine_files.py takes all of the files in the METL directory, barring files that have test or _.py (think, innit.py here) and combines them into a single file. combine_files.py also appends the huggingface wrapper code itself (stored in huggingface_code.py) onto the bottom of the script.

This script then gets auto-updated to 🤗 after formatting it by running the push_to_hub.py script. Some additional small comments are included in the top of each file repeating these responsibilities.