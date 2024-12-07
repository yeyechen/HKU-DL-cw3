import json
import os
import glob

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def calculate_accuracy(predictions, true_answers):
    correct_predictions = 0
    total_questions = len(true_answers)

    true_answers_dict = {item['id']: item['answerKey'] for item in true_answers}

    for prediction in predictions:
        question_id = prediction['id']
        predicted_answer = prediction['predicted']
        true_answer = true_answers_dict.get(question_id)

        if predicted_answer == true_answer:
            correct_predictions += 1

    accuracy = correct_predictions / total_questions
    return accuracy

def main():
    true_answers_file = 'data/test.jsonl'
    true_answers = load_jsonl(true_answers_file)

    predictions_path = 'predictions'
    predictions_files = glob.glob(os.path.join('predictions', '*.jsonl'))

    for pred in predictions_files:
      predictions = load_jsonl(pred)
      accuracy = calculate_accuracy(predictions, true_answers)
      print(f"{pred}, Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()