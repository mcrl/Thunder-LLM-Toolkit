import re
import math

def calculate_score_fullscale(docs, results):

    raw_output = results[0]
    lines = raw_output.split("\n")
    user = {}
    for line in lines:
        match = re.match(r"([\w가-힣]+):\s*(\d+)", line.strip())
        if match:
            emotion, score = match.groups()
            user[emotion] = int(score)

    # Reference answer
    reference = eval(docs["reference_answer_fullscale"])

    # Check if all 4 emotions are returned
    if len(user.items()) != 4:
        return {"eqbench": 0, "percent_parseable": 0}

    # Validate emotions against the reference
    emotions_dict = {}
    for emotion, user_emotion_score in user.items():
        for i in range(1, 5):
            if emotion == reference[f"emotion{i}"]:
                emotions_dict[emotion] = True

    if len(emotions_dict) != 4:
        return {"eqbench": 0, "percent_parseable": 0}

    # Calculate score differences
    difference_tally = 0
    for emotion, user_emotion_score in user.items():
        for i in range(1, 5):
            if emotion == reference[f"emotion{i}"]:
                d = abs(
                    float(user_emotion_score) - float(reference[f"emotion{i}_score"])
                )
                if d == 0:
                    scaled_difference = 0
                elif d <= 5:
                    scaled_difference = 6.5 * (1 / (1 + math.e ** (-1.2 * (d - 4))))
                else:
                    scaled_difference = d
                difference_tally += scaled_difference

    # Invert the difference tally to calculate the final score
    adjust_const = 0.7477
    final_score = 10 - (difference_tally * adjust_const)
    final_score_percent = final_score * 10

    return {"eqbench": final_score_percent, "percent_parseable": 100}
