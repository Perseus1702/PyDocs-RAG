import os


def rename_files_in_folder(folder_path, old_substr, new_substr):
    try:
        files = os.listdir(folder_path)

        for filename in files:
            if old_substr in filename:
                new_filename = filename.replace(old_substr, new_substr)
                old_filepath = os.path.join(folder_path, filename)
                new_filepath = os.path.join(folder_path, new_filename)
                os.rename(old_filepath, new_filepath)
                print(f"File '{filename}' renamed to '{new_filename}'.")
    except FileNotFoundError:
        print(f"Folder '{folder_path}' not found.")

# Specify the folder path containing the files
folder_path = "input/intent/utterances/"

# Specify the substring to be removed and the new substring
old_substring = "-utterances"
new_substring = ""

# Call the function to rename files in the folder
rename_files_in_folder(folder_path, old_substring, new_substring)
