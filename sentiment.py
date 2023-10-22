from transformers import pipeline
import topics

classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)
sentence = "That loud bang? Tee hee~ You were gone for so long I was getting bored so I let a shot loose~"
model_outputs = classifier(sentence)
emotion_list = []
# produces a list of dicts for each of the labels

for i in range(len(model_outputs[0])):
    emotion_list.append(model_outputs[0][i])

# Initialize variables to keep track of the maximum score and its associated label
max_score = 0  # Start with a low value that you know is less than any score
max_label = None  # Initialize to None

# Iterate through the list of dictionaries to find the maximum score and its label
for entry in emotion_list:
    if entry['score'] > max_score:
        max_score = entry['score']
        max_label = entry['label']

# Print the maximum score and its associated label
print(topics.extract_topic(sentence))
print(max_label + "(" + str(max_score) + ")")