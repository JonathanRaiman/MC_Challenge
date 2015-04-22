from mc_challenge.theano_utils import quadratic_form, create_shared
import theano, theano.tensor as T
from theano_lstm import Layer, Embedding, create_optimization_updates
from collections import OrderedDict

class QFormModel(object):
    def __init__(self, hidden_size, vocab_size, num_answers):
        self.embedding = Embedding(vocab_size, hidden_size)
        self.q_form    = create_shared("tensor",
                                       1,
                                       hidden_size,
                                       hidden_size)

        self.params = self.embedding.params + [self.q_form]

        # create a triplet scoring function:
        sentence = T.ivector()
        question = T.ivector()
        answer   = T.ivector()
        self.score_triplet = theano.function([sentence, question, answer],
            self.get_score(sentence, question, answer),
            allow_input_downcast=True)


        # create an error function
        answers = [T.ivector() for i in range(num_answers)]
        targets = [T.fscalar() for i in range(num_answers)]
        answer_targets = []
        for a, t in zip(answers, targets):
            answer_targets.extend([a, t])

        error = self.get_error(
                sentence,
                question,
                *answer_targets)
        self.error_fun = theano.function([
            sentence,
            question] + answer_targets,
            error, allow_input_downcast=True)

        gparams = T.grad(error, self.params)
        updates = OrderedDict()

        self.gradient_caches = [theano.shared(param.get_value(True, True) * 0.0, borrow=True, name=param.name + "_grad")
                        for param in self.params]

        for gparam_cache, gparam in zip(self.gradient_caches, gparams):
            updates[gparam_cache] = gparam_cache + gparam

        self.update_gradient = theano.function([
                sentence,
                question] + answer_targets,
                error,
                updates=updates, allow_input_downcast=True)

        # create a training function:
        true_updates, gsums, xsums, lr, max_norm = create_optimization_updates(
            None,
            self.params,
            method="sgd",
            gradients=self.gradient_caches
            )
        self.lr = lr

        for gparam_cache in self.gradient_caches:
            true_updates[gparam_cache] = T.zeros_like(gparam_cache)

        self.apply_gradient = theano.function(
            inputs  = [],
            outputs = [],
            updates = true_updates)

    def embed(self, sequence):
        return self.embedding.activate(sequence).mean(axis=0)

    def question_representation(self, sentence, question):
        """
        Take the mean embedding of the words in the text and
        the mean of the question, and the mean of
        use that to predict an answer
        """
        return self.embed(sentence) * self.embed(question)

    def answer_score(self, question_rep, answer_rep):
        """
        Score an answer based on its interaction with the embedding of
        a question and text.
        """
        return T.nnet.sigmoid(quadratic_form(
            self.q_form,
            answer_rep,
            question_rep))

    def get_score(self, s, q, a):
        q_rep = T.tanh(self.question_representation(s, q))
        a_rep = self.embed(a)
        return self.answer_score(q_rep, a_rep)

    def get_error(self, s, q, *answer_target):
        q_rep = T.tanh(self.question_representation(s, q))
        error = 0.0
        i = 0
        while i < len(answer_target):
            a_rep = self.embed(answer_target[i])
            score = self.answer_score(q_rep, a_rep)
            error += T.nnet.binary_crossentropy(score[0], answer_target[i+1])
            i+=2
        return error
