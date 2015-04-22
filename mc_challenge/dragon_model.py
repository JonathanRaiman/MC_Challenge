from mc_challenge.theano_utils import quadratic_form, create_shared, facebook_quadratic_form
import theano, theano.tensor as T
from theano_lstm import Layer, Embedding, create_optimization_updates
from collections import OrderedDict
import numpy as np

class DragonModel(object):
    def __init__(self,
            hidden_size,
            internal_features,
            intermediate_size,
            vocab_size,
            num_answers,
            tensor=True,
            method="sgd"):
        self.text_embedding = Embedding(vocab_size, hidden_size)
        self.question_embedding = Embedding(vocab_size, hidden_size)
        self.answer_embedding = Embedding(vocab_size, hidden_size)
        self.params = self.text_embedding.params + self.question_embedding.params + self.answer_embedding.params
        self.tensor = tensor
        if tensor:
            self.q_form_U  = create_shared("question_answer_tensor",
                                           intermediate_size,
                                           internal_features,
                                           3 * hidden_size)

            self.q_form_V  = create_shared("question_answer_tensor",
                                           intermediate_size,
                                           internal_features,
                                           3 * hidden_size)


            self.params.append(self.q_form_U)
            self.params.append(self.q_form_V)

        # here are the affine parameters
        self.bias           = create_shared("bias", intermediate_size)
        self.projection_mat = create_shared("projection_mat",
                                           intermediate_size,
                                           3 * hidden_size)

        self.scoring_mat    = create_shared("scoring_mat",
                                            1,
                                            intermediate_size)

        self.params += [
            self.bias,
            self.projection_mat
        ]

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
            error,
            allow_input_downcast=True)

        gparams = T.grad(error, self.params,
            disconnected_inputs='ignore')
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
        true_updates, self.gsums, self.xsums, lr, max_norm = create_optimization_updates(
            None,
            self.params,
            method=method,
            gradients=self.gradient_caches
            )
        self.lr = lr

        for gparam_cache in self.gradient_caches:
            true_updates[gparam_cache] = T.zeros_like(gparam_cache)

        self.apply_gradient = theano.function(
            inputs  = [],
            outputs = [],
            updates = true_updates)

    def embed(self, sequence, embedding):
        return embedding.activate(sequence).mean(axis=0)

    def reset_caches(self):
        for param in self.gsums + self.xsums:
            if param is not None:
                param.set_value(param.get_value(True, True) * 0.0)

    def get_score(self, s, q, a):
        s_embed = self.embed(s, self.text_embedding)
        q_embed = self.embed(q, self.question_embedding)
        a_embed = self.embed(a, self.answer_embedding)

        obs = T.concatenate([s_embed, q_embed, a_embed])

        before_tanh = self.bias + self.projection_mat.dot(obs)
        if self.tensor:
            before_tanh += facebook_quadratic_form(self.q_form_U, self.q_form_V, obs, obs)

        return T.nnet.sigmoid(
                self.scoring_mat.dot(
                    T.tanh(before_tanh)
                )
            )

    def predict(self, text, question_idx, answers_idx):
        return np.array(
            [self.score_triplet(text, question_idx, a) for a in answers_idx]
        ).argmax()

    def get_error(self, s, q, *answer_target):
        error = 0.0
        i = 0
        while i < len(answer_target):
            score = self.get_score(s, q, answer_target[i])
            error += T.nnet.binary_crossentropy(score[0], answer_target[i+1])
            i+=2
        return error
