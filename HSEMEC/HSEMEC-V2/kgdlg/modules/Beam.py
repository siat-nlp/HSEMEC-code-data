from __future__ import division

import random

import torch


class Beam(object):
    """
    Class for managing the internals of the beam search process.
    Takes care of beams, back pointers, and scores.
    Args:
       size (int): beam size
       pad, bos, eos (int): indices of padding, beginning, and ending.
       n_best (int): nbest size to use
       cuda (bool): use gpu
       global_scorer (:obj:`GlobalScorer`)
    """

    def __init__(self, size, unk, pad, bos, eos,
                 n_best=1, cuda=False,
                 global_scorer=None,
                 min_length=0):

        self.size = size
        self.tt = torch.cuda if cuda else torch

        # The score for each translation on the beam.
        self.scores = self.tt.FloatTensor(size).zero_()
        self.all_scores = []

        # The backpointers at each time-step.
        self.prev_ks = []

        # The outputs at each time-step.
        self.next_ys = [self.tt.LongTensor(size)
                            .fill_(pad)]
        self.next_ys[0][0] = bos

        # Has EOS topped the beam yet.
        self._eos = eos
        self._unk = unk
        self.eos_top = False

        # The attentions (matrix) for each time.
        self.attn = []

        # Time and k pair for finished.
        self.finished = []
        self.n_best = n_best

        # Information for global scoring.
        self.global_scorer = global_scorer
        self.global_state = {}

        # Minimum prediction length
        self.min_length = min_length

    def get_current_state(self):
        "Get the outputs for the current timestep."
        return self.next_ys[-1]

    def get_current_origin(self):
        "Get the backpointers for the current timestep."
        return self.prev_ks[-1]

    # def advance(self, word_probs, attn_out):
    def advance(self, word_probs, cur_words):
        """
        Given prob over words for every last beam `wordLk` and attention
        `attn_out`: Compute and update the beam search.
        Parameters:
        * `word_probs`- probs of advancing from the last step (K x words)
        * `attn_out`- attention at the last step
        Returns: True if beam search is complete.
        """

        # wrod_probs = [10, 40004]
        # print('word_probs ', word_probs.size())
        num_words = word_probs.size(1)

        cur_len = len(self.next_ys)

        if cur_len < self.min_length:
            # force the output to be longer than self.min_length
            for k in range(len(word_probs)):
                word_probs[k][self._eos] = -1e20

        for k in range(len(word_probs)):
            word_probs[k][self._unk] = -1e20

        # Sum the previous scores.
        # print('len(self.prev_ks) ', len(self.prev_ks))
        # print('self.next_ys[-1].size(0): ',self.next_ys[-1].size(0))
        if len(self.prev_ks) > 0:
            beam_scores = word_probs + \
                          self.scores.unsqueeze(1).expand_as(word_probs)

            # Don't let EOS have children.
            for i in range(self.next_ys[-1].size(0)):
                if self.next_ys[-1][i] == self._eos:
                    beam_scores[i] = -1e20
        else:
            beam_scores = word_probs[0]
        # print(self.next_ys)

        use_diverse_dec = True
        if not use_diverse_dec:
            flat_beam_scores = beam_scores.view(-1)
            best_scores, best_scores_id = flat_beam_scores.topk(self.size, 0, True, True)

            self.scores = best_scores
            self.all_scores.append(self.scores)
            prev_k = best_scores_id / num_words
            self.prev_ks.append(prev_k)
            self.next_ys.append((best_scores_id - prev_k * num_words))
        else:
            if len(self.prev_ks) > 0:
                best_scores = []
                best_scores_id = []
                prev_k = []
                for i in range(self.size):
                    top1 = True
                    if top1:
                        score, id = beam_scores[i].topk(1, 0, True, True)
                    else:
                        score, id = beam_scores[i].topk(2, 0, True, True)
                        if random.random() < 0.5:
                            select = 0
                        else:
                            select = 1
                        score = score[select]
                        id = id[select]

                    prob_decay = False
                    if prob_decay:
                        # print(id)
                        # print(cur_words)
                        # print(id in cur_words[i])
                        if id in cur_words[i]:
                            decay_rate = 50
                            word_probs[i, id] = word_probs[i, id] * (decay_rate ** cur_words[i].count(id))
                            beam_scores = word_probs + self.scores.unsqueeze(1).expand_as(word_probs)
                            if top1:
                                score, id = beam_scores[i].topk(1, 0, True, True)
                            else:
                                score, id = beam_scores[i].topk(2, 0, True, True)
                                if random.random() < 0.5:
                                    select = 0
                                else:
                                    select = 1
                                score = score[select]
                                id = id[select]
                    best_scores.append(score.item())
                    best_scores_id.append(id.item())
                    prev_k.append(i)
                # print('best_scores: ', best_scores)
                # print('id: ', best_scores_id)
                # print('prev_k: ', prev_k)
                # print('after sort')
                for i, id in enumerate(best_scores_id):
                    cur_words[i].append(id)

                # temp = zip(best_scores, best_scores_id, prev_k)
                # temp = sorted(temp, key=lambda x: -x[0])
                # best_scores, best_scores_id, prev_k = list(zip(*temp))
                best_scores = torch.tensor(best_scores).cuda()
                best_scores_id = torch.tensor(best_scores_id).cuda()
                prev_k = torch.tensor(prev_k).cuda()

                self.scores = best_scores
                self.all_scores.append(self.scores)
                self.prev_ks.append(prev_k)
                self.next_ys.append(best_scores_id)
                # print('best_scores: ', best_scores)
                # print('id: ', best_scores_id)
                # exit()
            else:
                flat_beam_scores = beam_scores.view(-1)
                best_scores, best_scores_id = flat_beam_scores.topk(self.size, 0, True, True)

                for i, id in enumerate(best_scores_id):
                    # print(i, id.item())
                    cur_words[i] = [id.item()]
                # print('first_word: ', cur_words)

                self.scores = best_scores
                self.all_scores.append(self.scores)
                # best_scores_id is flattened beam x word array, so calculate which
                # word and beam each score came from
                prev_k = best_scores_id / num_words
                # print('best_scores: ', best_scores)
                # print('id: ', best_scores_id)
                # print('prev_k: ', prev_k)
                # print('prev_k', prev_k, ' len：', len(prev_k))
                self.prev_ks.append(prev_k)
                self.next_ys.append((best_scores_id - prev_k * num_words))
                # print('pre ', self.prev_ks[-1])

        if self.global_scorer is not None:
            self.global_scorer.update_global_state(self)

        for i in range(self.next_ys[-1].size(0)):
            if self.next_ys[-1][i] == self._eos:
                sentence_len = len(self.next_ys) - 1
                s = self.scores[i] / (sentence_len ** 0)
                if self.global_scorer is not None:
                    global_scores = self.global_scorer.score(self, self.scores)
                    s = global_scores[i]
                self.finished.append((s, len(self.next_ys) - 1, i))

        # End condition is when top-of-beam is EOS and no global score.
        if self.next_ys[-1][0] == self._eos:
            # self.all_scores.append(self.scores)
            self.eos_top = True

    def done(self):
        return self.eos_top and len(self.finished) >= self.n_best

    def sort_finished(self, minimum=None):
        # print('in sort_finished======')
        # print('len(self.finished)', len(self.finished))
        # print('minimum', minimum)
        # print('len(self.finished) ',len(self.finished))
        # print('self.finished ',self.finished)
        if minimum is not None:
            i = 0
            # Add from beam until we have minimum outputs.
            while len(self.finished) < minimum:
                s = self.scores[i]
                if self.global_scorer is not None:
                    global_scores = self.global_scorer.score(self, self.scores)
                    s = global_scores[i]
                self.finished.append((s, len(self.next_ys) - 1, i))

        self.finished.sort(key=lambda a: -a[0])
        scores = [sc for sc, _, _ in self.finished]
        ks = [(t, k) for _, t, k in self.finished]
        # print("in sort_finished============: \n")
        # print("len scores:", len(scores))
        # print("len ks:", len(ks))

        return scores, ks

    def get_hyp(self, timestep, k):
        """
        Walk back to construct the full hypothesis.
        """
        hyp, attn = [], []
        for j in range(len(self.prev_ks[:timestep]) - 1, -1, -1):
            hyp.append(self.next_ys[j + 1][k])
            # attn.append(self.attn[j][k])
            k = self.prev_ks[j][k]
        return hyp[::-1]  # , torch.stack(attn[::-1])


class GNMTGlobalScorer(object):
    """
    NMT re-ranking score from
    "Google's Neural Machine Translation System" :cite:`wu2016google`
    Args:
       alpha (float): length parameter
       beta (float):  coverage parameter
    """

    def __init__(self, alpha, beta):
        self.alpha = alpha
        self.beta = beta

    def score(self, beam, logprobs):
        "Additional term add to log probability"
        cov = beam.global_state["coverage"]
        pen = self.beta * torch.min(cov, cov.clone().fill_(1.0)).log().sum(1)
        l_term = (((5 + len(beam.next_ys)) ** self.alpha) /
                  ((5 + 1) ** self.alpha))
        return (logprobs / l_term) + pen

    def update_global_state(self, beam):
        "Keeps the coverage vector as sum of attens"
        if len(beam.prev_ks) == 1:
            beam.global_state["coverage"] = beam.attn[-1]
        else:
            beam.global_state["coverage"] = beam.global_state["coverage"] \
                .index_select(0, beam.prev_ks[-1]).add(beam.attn[-1])
