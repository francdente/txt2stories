import pandas as pd

# Load the CSV file
df_g01 = pd.read_csv("paper/data_annotation/g01_user_stories.csv", sep=";")
df_g04 = pd.read_csv("paper/data_annotation/g04_user_stories.csv", sep=";")

# Randomly sample 5 stories
sampled_stories_g01 = df_g01["User Story"].sample(n=5, random_state=42)
sampled_stories_g04 = df_g04["User Story"].sample(n=5, random_state=42)

# Print them
print("Randomly selected user stories for g01:\n")
for i, story in enumerate(sampled_stories_g01, 1):
    print(f"{i}. {story}")

print("Randomly selected user stories for g04:\n")
for i, story in enumerate(sampled_stories_g04, 1):
    print(f"{i}. {story}")
