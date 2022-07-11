import numpy
import torch
import kgdlg
import kgdlg.IO as IO
from torch.autograd import Variable
import kgdlg.utils.print_utils as print_utils
import sys


class Inferer(object):
    """
    Uses a model to translate a batch of sentences.
    Args:
       model (:obj:`onmt.modules.NMTModel`):
          NMT model to use for translation
       fields (dict of Fields): data fields
       beam_size (int): size of beam to use
       n_best (int): number of translations produced
       max_length (int): maximum length output to produce
       global_scores (:obj:`GlobalScorer`):
         object to rescore final translations
       beam_trace (bool): trace beam search for debugging
    """

    def __init__(self, model, fields,
                 beam_size, opt, n_best=1,
                 max_length=100,
                 global_scorer=None,
                 beam_trace=False, min_length=0, use_mmi=0,
                 tgt_vocab=None):
        self.model = model
        # Set model in eval mode.
        self.model.eval()
        self.fields = fields
        self.n_best = n_best
        self.max_length = max_length
        self.global_scorer = global_scorer
        self.beam_size = beam_size
        self.min_length = min_length
        self.use_mmi = use_mmi
        self.opt = opt

        self.padding_idx = tgt_vocab.stoi[kgdlg.IO.PAD_WORD]
        weight = torch.ones(len(tgt_vocab))
        weight[self.padding_idx] = 0
        weight[kgdlg.IO.UNK] = 0
        self.criterion = torch.nn.NLLLoss(weight.cuda(), size_average=False)

        # for debugging
        self.beam_accum = None
        if beam_trace:
            self.beam_accum = {
                "predicted_ids": [],
                "beam_parent_ids": [],
                "scores": []}

    def inference_batch(self, batch):
        """
        Translate a batch of sentences.
        Mostly a wrapper around :obj:`Beam`.
        Args:
           batch (:obj:`Batch`): a batch from a dataset object
           data (:obj:`Dataset`): the dataset object
        Todo:
           Shouldn't need the original dataset.
        """

        # (0) Prep each of the components of the search.
        # And helper method for reducing verbosity.

        # src = batch.src[0][:-1][1:]
        #        src_lengths = batch.src[1] - 2
        #        src_lengths = src_lengths.tolist()
        src = batch.src[0]
        src_lengths = batch.src[1].tolist()
        batch_size = len(src_lengths)
        src_lengths = torch.LongTensor(src_lengths)

        src_emo = batch.src_emo  # [batch_size]
        tgt_emo = batch.tgt_emo

        if self.opt.use_cuda:
            src_lengths = src_lengths.cuda()
        else:
            src = src.cpu()

        if self.opt.vae_type in [6, 7, 8]:
            # @mark
            tgt = batch.tgt[0][:-1]
            tgt_lengths = batch.tgt[1] - 1
            tgt_lengths = torch.LongTensor(tgt_lengths.tolist())
            if self.opt.use_cuda:
                tgt_lengths = tgt_lengths.cuda()
            else:
                tgt = tgt.cpu()

        beam_size = self.beam_size
        vocab = self.fields["tgt"].vocab

        if self.use_mmi == 0:
            # print('use Beam.py')
            beam = [kgdlg.Beam(beam_size, n_best=self.n_best,
                               cuda=self.opt.use_cuda,
                               global_scorer=self.global_scorer,
                               unk=vocab.stoi[IO.UNK_WORD],
                               pad=vocab.stoi[IO.PAD_WORD],
                               eos=vocab.stoi[IO.EOS_WORD],
                               bos=vocab.stoi[IO.BOS_WORD],
                               min_length=self.min_length)
                    for __ in range(batch_size)]
        elif self.use_mmi == 1:
            # print('use Beam_for_MMI.py')
            beam = [kgdlg.Beam_for_MMI(beam_size, n_best=self.n_best,
                                       cuda=self.opt.use_cuda,
                                       global_scorer=self.global_scorer,
                                       unk=vocab.stoi[IO.UNK_WORD],
                                       pad=vocab.stoi[IO.PAD_WORD],
                                       eos=vocab.stoi[IO.EOS_WORD],
                                       bos=vocab.stoi[IO.BOS_WORD],
                                       diverse_rate=0.1,
                                       min_length=self.min_length)
                    for __ in range(batch_size)]
        else:
            print('param use_mmi error!')
            exit()

        # Help functions for working with beams and batches
        def var(a):
            return a

        def rvar(a):
            return var(a.repeat(1, beam_size, 1))

        def bottle(m):
            return m.view(batch_size * beam_size, -1)

        def unbottle(m):
            return m.view(beam_size, batch_size, -1)

        def ppl_bottle(v):
            return v.view(-1, v.size(2))

        # (1) Run the encoder on the src.

        # Encoder
        # print("src:", src)
        # print("src_len:", src_lengths)
        if self.opt.vae_type in [0, 1, 2, 3, 4]:
            context, enc_states = self.model.c_encode(src, src_lengths)
        elif self.opt.vae_type in [5]:
            context, enc_states = self.model.x_encode(src, src_lengths)
            # context, enc_states = self.model.x_encode(tgt, tgt_lengths) # TODO use tgt and tgt_lengths after testing
        elif self.opt.vae_type in [6, 7, 8]:
            c_context, c_enc_states = self.model.c_encode(src, src_lengths)
            x_context, x_enc_states = [None, None]
            original_x_context, original_x_enc_states = self.model.x_encode(tgt, tgt_lengths)

            print_utils.save_latent_Z_with_text(src, \
                                                c_enc_states[0], tgt, original_x_enc_states[0], \
                                                self.opt, 6, sys.stdout, "src_tgt")

            if 0 == self.opt.cluster_num:  # Hard Coding: if use cluster, forbid x_encode
                x_enc_states = original_x_enc_states

        # Predict Emotion
        pred_emo, emo_pred_loss, correct = [0, 0, 0]
        if self.opt.use_emo_pred:
            pred_emo, emo_pred_loss, correct = self.model.emo_predictor(c_enc_states[0], src_emo, tgt_emo)
        else:
            pred_emo = tgt_emo

        # Search Memory
        if 0 < self.opt.cluster_num and self.opt.vae_type in [6, 7, 8]:
            if self.opt.close_emo_mem:
                memory_target_z = self.model.data_cluster.forward(c_enc_states[0].detach(), tgt_emo*0)
            else:
                memory_target_z = self.model.data_cluster.forward(c_enc_states[0].detach(), pred_emo)
            # x_enc_states[0] = memory_target_z # !!! important
            # x_enc_states = c_enc_states # just copy its shape # todo
            x_enc_states = Variable(torch.zeros(c_enc_states.shape))  # just copy its shape
            x_enc_states[0] = memory_target_z
            if 1 == self.opt.cluster_param_in_cuda:
                x_enc_states = x_enc_states.cuda()

        # Latent
        if self.opt.vae_type in [1, 2]:
            latent_vector_z = self.model.cvae.inference(enc_states)
        elif self.opt.vae_type in [3, 4]:
            latent_vector_z, P_c_given_x, gmm_loss = self.model.gmm_net(enc_states, src, None)
        elif self.opt.vae_type in [6, 7]:
            if 0 == self.opt.src_tgt_latent_merge_type:
                latent_vector_z = torch.cat((c_enc_states, x_enc_states), 2)
                assert (self.opt.use_src_or_tgt_attention in [3])
            elif 1 == self.opt.src_tgt_latent_merge_type:
                latent_vector_z = c_enc_states + x_enc_states
            elif 2 == self.opt.src_tgt_latent_merge_type:
                if self.opt.vae_type in [6]:
                    latent_vector_z = self.model.latent_net.merge_src_tgt_forword(c_enc_states, x_enc_states)
                elif self.opt.vae_type in [7]:
                    latent_vector_z = self.model.latent_net.merge_src_tgt_forword_memory(c_enc_states, x_enc_states)
        elif self.opt.vae_type in [8]:
            if 1 == self.opt.inference_by_posterior:
                latent_vector_z = self.model.latent_net.inference_by_posterior(c_enc_states, original_x_enc_states,
                                                                               x_enc_states)
            else:  # Normal Branch
                latent_vector_z = self.model.latent_net.inference(c_enc_states, x_enc_states)

        ''' compute ppl by cross_entroy_loss'''
        if self.opt.use_emo_emb:
            tgt_inputs = [tgt, pred_emo]
        else:
            tgt_inputs=tgt
        dec_outputs, dec_hiddens, attn = self.model.decode(tgt_inputs, c_context, latent_vector_z)
        pred_x = self.model.generator(ppl_bottle(dec_outputs))
        x = batch.tgt[0][1:].view(-1)
        loss = self.criterion(pred_x, x)
        loss_data = loss.item()
        non_padding = x.ne(self.padding_idx)
        n_words = non_padding.sum().item()

        # Decoder
        if self.opt.vae_type in [1, 2, 3, 4, 6, 7, 8]:
            dec_states = latent_vector_z
        if self.opt.vae_type in [0, 5]:
            dec_states = enc_states
        # if self.opt.vae_type in [6]: # only for debug
        #    dec_states = x_enc_states

        # (2) Repeat src objects `beam_size` times.

        if self.opt.vae_type in [6, 7, 8]:
            if 0 == self.opt.use_src_or_tgt_attention:
                context = c_context
            elif 1 == self.opt.use_src_or_tgt_attention:
                context = x_context
            elif 2 == self.opt.use_src_or_tgt_attention:
                context = c_context + x_context
            elif 3 == self.opt.use_src_or_tgt_attention:
                context = torch.cat((c_context, x_context), 2)
        context = rvar(context.data)  # ?? data VS no data???
        if not isinstance(dec_states, tuple):  # GRU
            dec_states = dec_states.data.repeat(1, beam_size, 1)
        else:  # LSTM
            dec_states = (
                dec_states[0].data.repeat(1, beam_size, 1),
                dec_states[1].data.repeat(1, beam_size, 1),
            )

        # (3) run the decoder to generate sentences, using beam search.
        # print(f'max_length: {self.max_length}')

        cur_words = {}
        for i in range(self.max_length):
            # print('----------------------------------------')
            if all((b.done() for b in beam)):
                break

            # Construct batch x beam_size nxt words.
            # Get all the pending current beam words and arrange for forward.

            inp = var(torch.stack([b.get_current_state() for b in beam])
                      .t().contiguous().view(1, -1))
            # print('~~~ ', i)
            # print(vocab.itos[:10])
            # print('inp: ', inp)
            # print('inp: ', inp.size())
            # Temporary kludge solution to handle changed dim expectation
            # in the decoder
            # inp = inp.unsqueeze(2)

            # Run one step.
            if self.opt.use_emo_emb:
                # print('in inferer.py')
                # print('inp: ', inp.size())
                # print('tgt_emo: ', tgt_emo.size())
                if not pred_emo is None:
                    tgt_emo = pred_emo
                inp = [inp, tgt_emo]
            dec_out, dec_states, attn = self.model.decode(inp, context, dec_states)
            # print('in inference_batch=================')
            # print('dec_out: ',dec_out)
            # print('dec_out.shape: ',dec_out.shape)
            # print('dec_states: ',dec_states)
            # print('dec_states.shape: ',dec_states.shape)

            if not isinstance(dec_states, tuple):  # GRU
                dec_states = [
                    dec_states
                ]
            else:
                dec_states = [
                    dec_states[0],
                    dec_states[1]
                ]
            dec_out = dec_out.squeeze(0)
            # dec_out: beam x rnn_size

            # (b) Compute a vector of batch*beam word scores.
            out = self.model.generator(dec_out).data
            out = unbottle(out)
            # print('generator(dec_out): ', out)
            # print('generator(dec_out).shape: ', out.shape)
            # [beam_size, batch_size, vocab_size] = [10, 1, 40004]

            # (c) Advance each beam.

            for j, b in enumerate(beam):
                b.advance(out[:, j], cur_words)  # out[:, j] = [10, 40004]
                self.beam_update(j, b.get_current_origin(), beam_size, dec_states)

            if len(dec_states) == 1:  # GRU
                dec_states = dec_states[-1]
            else:
                dec_states = (
                    dec_states[0],
                    dec_states[1]
                )
                # (4) Extract sentences from beam.
        ret = self._from_beam(beam)

        ret['pred_emo'] = pred_emo
        # print(f"ret['pred_emo'] = {pred_emo}")
        # print('ret["predictions"]', ret["predictions"])
        # print('ret["scores"]', ret["scores"])
        # print('shape ret["predictions"][0]',numpy.array(ret['predictions'][0]).shape)
        # print('shape ret["scores"][0]',numpy.array(ret['scores'][0]).shape)
        return ret, loss_data, n_words, correct

    def beam_update(self, idx, positions, beam_size, states):
        for e in states:
            a, br, d = e.size()
            sent_states = e.view(a, beam_size, br // beam_size, d)[:, :, idx]
            sent_states.data.copy_(
                sent_states.data.index_select(1, positions))

    def _from_beam(self, beam):
        ret = {"predictions": [],
               "scores": []}
        for b in beam:
            if self.beam_accum:
                self.beam_accum['predicted_ids'].append(torch.stack(b.next_ys[1:]).tolist())
                self.beam_accum['beam_parent_ids'].append(torch.stack(b.prev_ks).tolist())
                self.beam_accum['scores'].append(torch.stack(b.all_scores).tolist())

            n_best = self.n_best
            scores, ks = b.sort_finished(minimum=n_best)
            # print('scores ',scores)

            hyps = []


            for i, (times, k) in enumerate(ks[:n_best]):
                hyp = b.get_hyp(times, k)
                hyps.append(hyp)
            # print('hyps ', hyps)
            ret["predictions"].append(hyps)
            ret["scores"].append(scores)
        # print('in _from_beam:==========')
        # print("ret[predictions]=",ret["predictions"])
        # print("ret[scores]=",ret["scores"])
        return ret
