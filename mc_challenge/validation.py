def validate(sections, model, vocabulary):
    num_correct = 0
    total = 0
    for section in sections:
        num_correct += validate_section(section, model, vocabulary)
        total += len(section.questions)
    return num_correct / total

def validate_section(section, model, vocabulary):
    # tokenize and convert to indices
    text = vocabulary(
        section.text,
        tokenization=True)

    num_correct = 0
    for question in section.questions:
        question_idx = vocabulary(
            question.question,
            tokenization = True)
        answers_idx = [vocabulary(
            ans,
            tokenization = True) for ans in question.answers]
        prediction = model.predict(text, question_idx, answers_idx)

        if prediction == question.answer:
            num_correct += 1

    return num_correct
