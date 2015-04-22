import re
from lxml import html
import cssselect

def is_intro_to_passage(line):
    """
    Utility for detecting whether a line
    marks the beginning of a reading comprehension
    passage.
    """
    if line.startswith("<small"):
        return 0
    score = 0
    for a in ["based", "on", "following", "passage"]:
        if a in line:
            score += 1
    return score

class Question(object):
    __slots__ = ["question", "number", "correct_answer", "answers"]
    def __init__(self, question, number):
        self.question = question
        self.number   = number
        self.correct_answer = None
        self.answers = []
    def __str__(self):
        return "<Question question=%r number=%s>" % (self.question, self.number)

    def __repr__(self):
        return str(self)

section_finder        = re.compile("Section (\d+)")
exercise_a_key_finder = re.compile("Exercise (\d+)")
exercise_finder       = re.compile("EXERCISE (\d+)")
level_finder          = re.compile("Level ([A-Z]+)")
answer_finder         = re.compile(
    '<[^>]+> *(?P<number>\d+)\. *<span class="bold">(?P<answer>[A-Z])</span>'
)

def extract_questions(read_passage):
    question_number = re.compile("(?P<number>\d+)\. +(?P<question>[^\(].+)")
    answer_identifier = re.compile("\((?P<key>[A-Z])\)? +(?P<answer>.+)")
    answer_roman_identifier = re.compile("(?P<numeral>[A-Z]+)\. +(?P<answer>.+)")
    passage, question_section = read_passage
    page = html.fromstring("<html><head></head><body>" + "\n".join(question_section) + "</body></html>")

    p_tags = page.cssselect("p")

    questions  = []
    answer_set = []
    answers    = []

    for k, paragraph in enumerate(p_tags):
        if len(paragraph.cssselect("a")) > 0:
            # is a question
            if len(answer_set) > 0:
                assert(len(answer_set) == 5), "Not 5 answers in this answer set"
                assert(len(questions) > 0), "Answers to non-existing question"
                assert(len(questions[-1].answers) == 0), "Duplicate answers to last question"
                questions[-1].answers = answer_set
            answer_set = []
            q_text = paragraph.text_content().strip()
            matches = question_number.match(q_text)
            if matches is not None:
                questions.append(Question(
                        question = matches.group("question"),
                        number   = int(matches.group("number"))))
            else:
                continue
        else:
            a_text = paragraph.text_content().strip()
            if len(a_text) > 0:
                matches = answer_identifier.match(a_text)
                if matches is not None:
                    answer_set.append(matches.group("answer"))
                else:
                    continue
    if len(answer_set) > 0:
        assert(len(answer_set) == 5), "Not 5 answers in this answer set"
        assert(len(questions[-1].answers) == 0), "Duplicate answers to last question"
        questions[-1].answers = answer_set

    del page
    joint_passage = "\n".join(passage)
    line_breaks = re.compile("<br[^>]+>")
    joint_passage = line_breaks.sub(" ", joint_passage)

    page_numbers = re.compile('<span class="italic">\(\d+\)</span>')
    joint_passage = page_numbers.sub(" ", joint_passage)

    page = html.fromstring("<html><head></head><body>" + joint_passage + "</body></html>")
    segments = []
    passage = "\n".join([p.strip() for p in page.itertext()])

    return (passage, questions)

class Reader(object):
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.in_passage = False
        self.in_q_section = False
        self.in_answer_key = False
        self.passages = {}
        self.answer_key      = {}
        self.current_passage = []
        self.current_questions = []
        self.current_section_num = None
        self.current_level       = None
        self.current_a_key_level = None
        self.linenumber = 0
        self.in_answer_sheet = False
        self.exams   = []
        self.current_section = None

    def reset(self):
        if len(self.answer_key) > 0:
            for key in self.passages.keys():
                assert(key in self.answer_key), "Question section missing from answer key"

            transcribed_passages = {}

            for key in self.passages.keys():
                cleaned_passages =  [extract_questions(read_passage)
                                          for read_passage in self.passages[key]]

                for passage, questions in cleaned_passages:
                    for q in questions:
                        assert(q.number in self.answer_key[key]), "Missing answer to question %d (\"%s\")from section %r" % (
                            q.number,
                            q.question,
                            key)
                        q.correct_answer = self.answer_key[key][q.number]

                transcribed_passages[key] = cleaned_passages
            self.exams.append((self.answer_key, transcribed_passages))
        elif len(self.answer_key) == 0 and len(self.passages) > 0:
            total_missing = 0
            for value in self.passages.values():
                total_missing += len(value)
            if self.verbose and total_missing > 0:
                print("answersheet without answers to:")
                print(self.passages)

        self.in_answer_key       = False
        self.current_section     = None
        self.answer_key          = {}
        self.passages            = {}
        self.current_level       = None
        self.current_a_key_level = None
        self.current_section_num = None

        self.current_passage   = []
        self.current_questions = []

    def update_answer_key(self, line):
        if "Analysis of Test Results" in line or "Answer Explanations" in line:
            if self.verbose:
                print("</answerkey> (linenumber = %d)" % (self.linenumber))
            self.reset()
        else:
            if section_finder.search(line):
                self.current_section = int(section_finder.findall(line)[0])
                if self.current_section in self.answer_key:
                    raise Exception("Duplicate section in answer key")
                self.answer_key[self.current_section] = {}
            elif level_finder.search(line):
                self.current_a_key_level = level_finder.findall(line)[0]
            elif exercise_a_key_finder.search(line):
                if self.current_a_key_level is None:
                    raise Exception("Answers to Exercise before declaring level")
                self.current_section = self.current_a_key_level + "_" + exercise_a_key_finder.findall(line)[0]
                # if self.verbose: print("Switching to \"%s\" (linenumber = %d)" % (self.current_section, self.linenumber))
                if self.current_section in self.answer_key:
                    raise Exception("Duplicate section in answer key")
                self.answer_key[self.current_section] = {}

            else:
                if line.startswith("<td") and '<span class="bold">' in line:
                    matches = answer_finder.match(line)
                    if int(matches.group("number")) in self.answer_key[self.current_section]:
                        raise Exception("Duplicate answer found")
                    self.answer_key[self.current_section][int(matches.group("number"))] = matches.group("answer")

    def update_passage(self, line):
        if '<span class="bold">Section' in line:
            self.current_section_num = int(section_finder.findall(line)[0])
            if self.verbose: print("Switching to \"%d\" (linenumber = %d)" % (self.current_section_num, self.linenumber))
            if self.current_section_num in self.passages:
                raise Exception(
                    "Previously existing section (%d) rediscovered %r (linenumber = %d)" % (
                        self.current_section_num,
                        self.passages,
                        self.linenumber))
            self.passages[self.current_section_num] = []
        elif '<span class="bold">Level ' in line:
            self.current_level = level_finder.findall(line)[0]
            if self.verbose: print("Switching to \"%s\" (linenumber = %d)" % (self.current_level, self.linenumber))
        elif '<p class="calibre9"><span class="calibre3">EXERCISE ' in line:
            if self.current_level is None:
                raise Exception("Level not declared before seeing exercise")
            self.current_section_num = self.current_level + "_" + exercise_finder.findall(line)[0]
            if self.verbose:
                print("Switching to \"%s\" (linenumber = %d)" % (
                        self.current_section_num,
                        self.linenumber))
            if self.current_section_num in self.passages:
                raise Exception("Previously existing section (%r) rediscovered %r" % (self.current_section_num, self.passages))
            self.passages[self.current_section_num] = []

        intro_to_passage_score = is_intro_to_passage(line)
        if intro_to_passage_score > 2:
            # skip next line since it's the continuation of the intro
            if intro_to_passage_score != 4:
                pass


            if self.in_passage:
                if len(self.current_questions) > 0 and len(self.current_passage) > 0:
                    if self.current_section_num not in self.passages:
                        print(line)
                        print(self.linenumber)
                        print(self.passages)
                        raise Exception("missing section intro %r" % (self.current_section_num))
                    self.passages[self.current_section_num].append((self.current_passage, self.current_questions))
                self.current_passage   = []
                self.current_questions = []
                self.in_q_section      = False
            else:
                self.in_passage = True
        elif self.in_passage and ('class="calibre39"' in line):
            self.in_passage = False
            self.in_q_section = False
            if len(self.current_questions) > 0 and len(self.current_passage) > 0:
                self.passages[self.current_section_num].append(
                    (self.current_passage, self.current_questions))
            self.current_passage = []
            self.current_questions = []
        elif self.in_passage:
            if self.in_q_section:
                self.current_questions.append(line)
            else:
                if 'class="calibre53"' in line or 'class="calibre52"' in line:
                    self.in_q_section = True
                    self.current_questions.append(line)
                else:
                    self.current_passage.append(line)

def extract_answer_key(lines, verbose = False):
    """
    Extract answer key from ebook lines.
    Looks for "Answer Key" title page and
    then for each sub table in this answer
    key captures the section # and the
    question # with associated correct
    answer A-E

    Inputs
    ------

    list<str> lines : input epub as lines of text

    Ouputs
    ------

    list<dict> : answer keys for each exam. Each dict
    has multiple sections, each of which are also dict
    with question number as subkey and as value the
    correct answer letters A-Z

    """

    # dealing with answer keys:
    section_finder        = re.compile("Section (\d+)")
    exercise_a_key_finder = re.compile("Exercise (\d+)")
    exercise_finder       = re.compile("EXERCISE (\d+)")
    level_finder          = re.compile("Level ([A-Z]+)")
    answer_finder         = re.compile(
        '<[^>]+> *(?P<number>\d+)\. *<span class="bold">(?P<answer>[A-Z])</span>'
    )
    # dealing with passages & questions:
    reader = Reader(verbose=verbose)

    while reader.linenumber < len(lines):
        line = lines[reader.linenumber]

        if "ANSWER SHEET" in line:
            # ignore answersheet contents
            if reader.verbose:
                print("<answersheet>")
            reader.reset()
            reader.in_answer_sheet = True
        else:
            if reader.in_answer_sheet:
                # leave answersheet
                if "mbppagebreak" in line:
                    reader.in_answer_sheet = False
                    if reader.verbose:
                        print("</answersheet>")

            else:
                # note that a section just ended
                if "Answer Key" in line or "Answers to Passage-Based" in line:
                    reader.in_answer_key = True
                    reader.in_passage = False
                    reader.in_q_section = False
                    if reader.verbose:
                        print("<answerkey> (linenumber = %d)" % (
                                reader.linenumber,))
                if reader.in_answer_key:
                    # parse answer key
                    reader.update_answer_key(line)
                else:

                    # collect information from passage
                    reader.update_passage(line)
        reader.linenumber+=1
    return reader.exams

import json

def save_tests(exams, path):
    with open(path, "wt") as out_file:
        out_file.write("[\n    ")
        for exam_number, (answer_key, exam) in enumerate(exams):
            keys = list(exam.keys())
            for key_num, section_key in enumerate(keys):
                for question_number, (passage, questions) in enumerate(exam[section_key]):
                    el = {
                        "passage": passage,
                        "questions": [],
                        "exam_number": "%d" % (exam_number),
                        "section_name" : "%s" % (str(section_key))
                    }
                    for q in questions:
                        el["questions"].append({
                            "question": q.question,
                            "answers": q.answers,
                            "correct_answer": q.correct_answer
                            })
                    out_json = json.dumps(el)
                    out_json = out_json.replace("\\u201d", "\\\"")
                    out_json = out_json.replace("\\u201c", "\\\"")
                    out_json = out_json.replace("\\u2019", "'")
                    out_json = out_json.replace("\\u2018", "'")
                    out_json = out_json.replace("\\u2013", "-")
                    out_json = out_json.replace("\\u2014", "-")
                    out_json = out_json.replace("\\u2026", "…")
                    out_json = out_json.replace("\\u00e9", "é")
                    out_json = out_json.replace("\\u00f1", "ñ")
                    out_file.write("    " + out_json)
                    if (
                        (question_number < len(exam[section_key]) - 1) or
                        (key_num < len(keys) -1) or
                        (exam_number < len(exams) - 1)
                        ):
                        out_file.write(",\n    ")
        out_file.write("\n]")

from epub_conversion.utils import open_book, convert_epub_to_lines

if __name__ == "__main__":
    book = open_book("Barrons/Barrons.epub")
    lines = convert_epub_to_lines(book)
    exams = extract_answer_key(lines, verbose=False)
    save_tests(exams, "barrons.json")
