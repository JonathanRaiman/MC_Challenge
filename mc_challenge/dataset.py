import os


class Questions(object):
    """
    Small class for holding question objects
    from machine comprehension dataset.
    Holds `question`, and a list of `answers`
    as strings. Also has a `question_type` field
    to store whether question is single or
    multiple-choice.
    """
    __slots__ = ["question", "answers", "question_type", "answer"]
    @classmethod
    def build(cls, qs):
        qtype = None
        questions = []
        current_q = None
        current_answers = []
        for q in qs:
            if q.startswith("multiple: ") or q.startswith("one: "):
                if current_q is not None:
                    questions.append(cls(current_q, current_answers, qtype))
                    current_answers = []
                    current_q = None
                qtype, current_q = q.split(": ", 2)
            else:
                current_answers.append(q)

        if current_q is not None:
            questions.append(cls(current_q, current_answers, qtype))
        return questions

    def __init__(self, question, answers, question_type, answer=0):
        self.question      = question
        self.answer        = answer
        self.answers       = answers
        self.question_type = question_type

    def __str__(self):
        return "<Question \"%s\" answers=%r qtype=%s>" % (self.question, self.answers, self.qtype)

    def _repr_html_ (self):
        return "<div><h4>%s <small>(%s)</small></h4><div><ol style='list-style-type: upper-alpha;'>%s</ol></div>" % (
                self.question,
                self.question_type,
                "\n".join(["<li %s>%s</li>" % ("style='color:red'" if k == self.answer else "", ans) for k, ans in enumerate(self.answers)])
        )

class Section(object):
    """
    Section object contains the reading `text`
    and a list of `questions` (see Question above).
    """
    def __init__(self, parsed_text):
        self.section_name, self.turk_info, self.text, *qs =  parsed_text.split("\t")
        self.questions = Questions.build(qs)
        self.text = self.text.replace(r"\newline", "\n")

    def __str__(self):
        return "<Section text=\"%s\" questions=%r>" % (
            self.text if len(self.text) < 50 else self.text[0:25] + "..." + self.text[-25:],
            self.questions)

    def add_answers(self, answers):
        for question, ans in zip(self.questions, answers):
            if ans == 'A':
                question.answer = 0
            elif ans == 'B':
                question.answer = 1
            elif ans == 'C':
                question.answer = 2
            elif ans == 'D':
                question.answer = 3
            else:
                raise ValueError("The answer can only be [A-D] (was %r)" % (ans))

    def _repr_html_ (self):
        return """
            <div>
                <h2>%s</h2>
                %s
                <hl/>
                %s
            </div>""" % (
                self.section_name,
                "\n".join(["<p>" + t + "</p>" for t in self.text.split(r'\newline')]),
                "\n".join([q._repr_html_() for q in self.questions])
                )

def load_mc(data_path, validation_fraction=0.1):
    ans_sets = []
    tsv_sets = []
    dataset_names = set()
    for path in os.listdir(data_path):
        if path.endswith(".ans") or path.endswith(".tsv"):
            path, extension = os.path.splitext(path)
            dataset_names.add(path)

    for name in dataset_names:
        answer_path = os.path.join(data_path, name + ".ans")
        question_path = os.path.join(data_path, name + ".tsv")
        ans_sets.extend(
            [ans.split("\t") for ans in open(answer_path, "rt").read().split("\n") if len(ans) > 0]
        )
        tsv_sets.extend([Section(sec) for sec in open(question_path, "rt").read().split("\n") if len(sec) > 0])

    for ans_set, section in zip(ans_sets, tsv_sets):
        section.add_answers(ans_set)

    test = []
    training = []
    for section in tsv_sets:
        if 'test' in section.section_name:
            test.append(section)
        else:
            training.append(section)

    num_validation = int(validation_fraction * len(training))

    validation = training[:num_validation]
    training = training[num_validation:]
    return training, validation, test

class NumericalSection(object):
    def __init__(self, section, vocabulary):
        self.text = vocabulary(section.text,
                   tokenization=True)
        self.questions = []
        for q in section.questions:
            ans_idx = [vocabulary(ans, True) for ans in q.answers]
            q_idx = vocabulary(q.question, True)
            self.questions.append(
                (q_idx,
                ans_idx,
                q.answer)
            )

def convert_to_idx(sections, vocabulary):
    idx_sections = []
    for section in sections:
        idx_sections.append(NumericalSection(section, vocabulary))
    return idx_sections
