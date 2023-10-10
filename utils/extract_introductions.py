import os

# Define the directory where the txt files are located
source_directory = '/common/home/hg343/Research/LLM_ProblemConversion/datasets/problems/ajp_articles'  # Change this to the path of your directory

# Define the directory to save the introductions
intro_directory = os.path.join(source_directory, 'introductions')
if not os.path.exists(intro_directory):
    os.makedirs(intro_directory)

# Iterate through all files in the directory
for filename in os.listdir(source_directory):
    if filename.endswith('.txt'):
        with open(os.path.join(source_directory, filename), 'r', encoding='utf-8') as file:
            content = file.read()
            # Define patterns to look for the introduction start and end
            # These patterns might need to be adjusted based on the actual structure of your papers
            start_keywords = ['I']
            end_keywords = ['II']

            start_idx = None
            end_idx = None

            for keyword in start_keywords:
                start_idx = content.find(keyword)
                if start_idx != -1:
                    print(f'found start idx, {start_idx} , {start_idx}')
                    break

            for keyword in end_keywords:
                end_idx = content.find(keyword, start_idx if start_idx is not None else 0)
                if end_idx != -1:
                    print(f'found end idx, {end_idx} , {start_idx}')
                    break

            if start_idx is not None and end_idx is not None and start_idx < end_idx:
                introduction = content[start_idx:end_idx]
                with open(os.path.join(intro_directory, "intro_" + filename), 'w', encoding='utf-8') as intro_file:
                    intro_file.write(introduction)
                    print(f'{filename} has been written')
            else:
                print(f"No introduction found for {filename}, start_idx {start_idx}, end_idx {end_idx}")

print("Extraction completed!")

