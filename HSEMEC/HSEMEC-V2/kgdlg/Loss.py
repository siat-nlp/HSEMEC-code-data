import torch
import torch.nn as nn
from torch.nn import functional
from torch.autograd import Variable
from kgdlg.Trainer import Statistics
import kgdlg.IO
import kgdlg.utils.misc_utils as misc_utils


class NMTLossCompute(nn.Module):
    """
    Standard NMT Loss Computation.
    """

    def __init__(self, model, tgt_vocab, opt):
        super(NMTLossCompute, self).__init__()
        self.generator = model.generator  # the generator from model
        self.embedding = model.decoder.embedding
        self.tgt_vocab = tgt_vocab
        self.opt = opt
        self.padding_idx = tgt_vocab.stoi[kgdlg.IO.PAD_WORD]
        weight = torch.ones(len(tgt_vocab))
        weight[self.padding_idx] = 0
        weight[kgdlg.IO.UNK] = 0
        self.criterion = nn.NLLLoss(weight, size_average=False)
        self.emo_criterion = nn.CrossEntropyLoss(size_average=False)
        self.emo_cls = model.emo_cls

    def make_shard_state(self, batch, output):
        """ See base class for args description. """
        return {
            "output": output,
            "target": batch.tgt[0],
        }

    def gaussian_kld(self, recog_mu, recog_logvar, prior_mu, prior_logvar):
        recog_mu = recog_mu.view(-1, recog_mu.size(-1))
        recog_logvar = recog_logvar.view(-1, recog_logvar.size(-1))
        prior_mu = prior_mu.view(-1, prior_mu.size(-1))
        prior_logvar = prior_logvar.view(-1, prior_mu.size(-1))
        kld = -0.5 * torch.sum(1 + (recog_logvar - prior_logvar)
                               - torch.div(torch.pow(prior_mu - recog_mu, 2), torch.exp(prior_logvar))
                               - torch.div(torch.exp(recog_logvar), torch.exp(prior_logvar)), 1)
        return kld.sum(0)

    def compute_loss(self, recon_x, x,
                     recog_mu, recog_logvar,
                     prior_mu, prior_logvar,
                     gmm_loss, cluster_loss, recon_x_memory,
                     kl_loss_anneal_weight,
                     emo_pred_loss,
                     pred_correct,
                     tgt_emo=None,
                     emo_cls_out=None):

        # scores = self.generator(self.bottle(output))
        # target = target.view(-1)
        # loss = self.criterion(scores,target)


        NLL_memory_data, gmm_data, cluster_loss_mean_data, KLD_data, emo_cls_loss_data, emo_pred_loss_data = [0, 0, 0,
                                                                                                              0, 0, 0]
        # reconstruction loss
        # print('recon_x: ', recon_x.size())
        # print('x: ', x.size())
        if self.opt.no_recons:
            NLL = self.criterion(recon_x, x)  # Rec Loss
            # print('loss: ',NLL)
            NLL_data = NLL.item()
            loss = NLL*0
        else:
            NLL = self.criterion(recon_x, x)  # Rec Loss
            # print('loss: ',NLL)
            NLL_data = NLL.item()
            loss = NLL
        # print('rec_loss: ', NLL_data)
        # print('x: ', x.size())

        # emo pred loss
        if self.opt.use_emo_pred:
            loss += emo_pred_loss * self.opt.lambda_for_emo_pred_loss
            emo_pred_loss_data = emo_pred_loss.item()

        # emo_cls loss
        if self.opt.use_emo_cls:
            if not recon_x_memory is None:
                if not emo_cls_out is None and not tgt_emo is None:
                    emo_cls_loss = self.emo_criterion(emo_cls_out, tgt_emo)
                    emo_cls_loss_data = emo_cls_loss.item()
                    loss += emo_cls_loss * self.opt.lambda_for_emo_cls_loss
                    # print('logsoftmax: ', nn.functional.log_softmax(emo_cls_out, dim=1))
                    # print('emo_cls_out', emo_cls_out)
                    # print('tgt_emo', tgt_emo)
                    # print('emo_cls_loss: ', emo_cls_loss)
                    # print('emo_cls_loss_data: ', emo_cls_loss_data)
                else:
                    print('emo_cls loss error')

        # GMM loss
        if self.opt.vae_type in [3, 4]:
            gmm_loss_mean = torch.mean(gmm_loss)
            loss += gmm_loss_mean
            gmm_data = gmm_loss_mean.item()

        # KL loss
        if self.opt.no_recons:
            pass
        else:
            if self.opt.vae_type in [1, 2] or \
                    (self.opt.vae_type in [8] and self.opt.variance_memory_type in [0, 1, 4, 5]):
                if not (recog_mu is None or recog_logvar is None \
                        or prior_mu is None or prior_logvar is None):
                    KLD = self.gaussian_kld(recog_mu, recog_logvar, prior_mu, prior_logvar)
                    loss += KLD * self.opt.lambda_for_kl_loss * kl_loss_anneal_weight
                    KLD_data = KLD.item()
                    # if not (self.opt.vae_type in [8] and self.opt.variance_memory_type in [4,5]):
                    #    stats = self.stats(NLL_data, KLD_data, recon_x, x)

        # cluster loss
        if self.opt.vae_type in [0, 5, 6, 8]:
            if (not cluster_loss is None) and self.opt.cluster_num > 0:
                cluster_loss_mean = torch.mean(cluster_loss)
                loss += cluster_loss_mean * self.opt.lambda_for_nn_and_kmeans
                cluster_loss_mean_data = cluster_loss_mean.item()
                # if not self.opt.vae_type in [8]:
                #    stats = self.stats(NLL_data, cluster_loss_mean_data, recon_x, x)
            # else:
            # if not self.opt.vae_type in [8]:
            #    stats = self.stats(NLL_data, 0, recon_x, x)

        # predict loss
        if self.opt.vae_type in [7] or \
                (self.opt.vae_type in [8] and self.opt.variance_memory_type in [2, 3, 4, 5]):
            if not recon_x_memory is None:
                NLL_memory = self.criterion(recon_x_memory, x)  # pred_loss
                NLL_memory_data = NLL_memory.item()
                # print('pred_loss: ', NLL_memory_data)
                # exit()
                loss += NLL_memory * self.opt.lambda_for_memory_loss
                # stats = self.stats(NLL_data, NLL_memory_data, recon_x, x)
            # else:
            # stats = self.stats(NLL_data, 0, recon_x, x)
        if self.opt.vae_type in [1] and self.opt.cvae_print_reconstruct_loss == 1:
            NLL_memory = self.criterion(recon_x_memory, x)
            NLL_memory_data = NLL_memory.item()

        if self.opt.vae_type in [3, 4]:
            stats = self.stats(NLL_data, NLL_memory_data, gmm_data, cluster_loss_mean_data, recon_x, x,
                               emo_cls_loss_data, emo_pred_loss_data, pred_correct)
        elif self.opt.vae_type in [0, 1, 2, 5, 6, 7, 8]:
            stats = self.stats(NLL_data, NLL_memory_data, KLD_data, cluster_loss_mean_data, recon_x, x,
                               emo_cls_loss_data, emo_pred_loss_data, pred_correct)

        return loss, stats

    def compute_train_loss(self, batch, output,  # Main Step
                           recog_mu, recog_logvar,
                           prior_mu, prior_logvar,
                           gmm_loss, cluster_loss,
                           output_memory, kl_loss_anneal_weight,
                           emo_pred_loss, pred_correct, pred_emo):
        """
        Compute the loss in shards for efficiency.
        """

        batch_stats = Statistics()

        # print('output: ',output.size())
        # [batch_size * sentence_len, hidden_size] * [hidden_size, vocab_size]
        # -> [batch_size * sentence_len, vocab_size]
        # output = [sentence_len, batch_size, hidden_size]
        recon_x = self.generator(self.bottle(output))  # recon_x = [batch_size * sentence_len, vocab_size]

        recon_x_memory = None
        if not output_memory is None:
            recon_x_memory = self.generator(self.bottle(output_memory))  # the generator from model
            if self.opt.use_emo_cls:
                pred_res = self.unbottle(recon_x_memory, batch.batch_size)
                # print('pred_res: ', pred_res.size())
                # print('pred_res: ', pred_res[:5, :2, -10:])
                pred_res = torch.exp(pred_res)  # [sen_len, batch_size, vocab_size]
                # print('exp-pred_res: ', pred_res[:5, :2, -10:])
                # print('sum_prob: ', torch.sum(pred_res, dim=2))
                emb_weight = self.embedding.embedding.weight.data
                # print('x_encoder.embedding.weight.data: ', emb_weight.size())
                ewe = torch.matmul(pred_res, emb_weight)  # expected word embedding
                ewe = torch.mean(ewe, dim=0)
                # print('ewe after mean: ', ewe.size())
                emo_cls_out = self.emo_cls(ewe)  # [batch_size, emo_num]
                # print('emo_cls_out: ', emo_cls_out.size())
                # print('emo_cls_out: ', emo_cls_out)

        x = batch.tgt[0][1:].view(-1)
        # print('recon_x: ', recon_x.size())
        # print('x: ', x.size())
        if self.opt.use_emo_cls and (not output_memory is None):
            loss, stats = self.compute_loss(recon_x, x, \
                                            recog_mu, recog_logvar, prior_mu, prior_logvar, \
                                            gmm_loss, cluster_loss, recon_x_memory, kl_loss_anneal_weight, \
                                            emo_pred_loss, pred_correct, \
                                            pred_emo, emo_cls_out)
        else:
            loss, stats = self.compute_loss(recon_x, x, \
                                            recog_mu, recog_logvar, prior_mu, prior_logvar, \
                                            gmm_loss, cluster_loss, recon_x_memory, kl_loss_anneal_weight, \
                                            emo_pred_loss, pred_correct,
                                            pred_emo)
        stats.kld = stats.kld / batch.batch_size
        stats.loss_per_sample = stats.loss_per_sample / batch.batch_size
        stats.memory_loss_per_sample = stats.memory_loss_per_sample / batch.batch_size
        stats.cluster_loss = stats.cluster_loss / batch.batch_size
        stats.emo_cls_loss_per_sample = stats.emo_cls_loss_per_sample / batch.batch_size
        stats.emo_pred_loss_per_sample = stats.emo_pred_loss_per_sample / batch.batch_size
        stats.pred_correct_per_sample = stats.pred_correct_per_sample / batch.batch_size
        loss.div(batch.batch_size).backward()
        batch_stats.update(stats)

        return batch_stats

    def compute_valid_loss(self, batch, output):
        """
        Compute the loss monolithically, not dividing into shards.
        """

        shard_state = self.make_shard_state(batch, output)
        _, batch_stats = self.compute_loss(batch, **shard_state)

        return batch_stats

    def stats(self, loss, memory_loss_data, KLD_data, cluster_loss_data, scores, target, emo_cls_loss_data,
              emo_pred_loss_data, pred_correct):
        """
        Compute and return a Statistics object.
        Args:
            loss(Tensor): the loss computed by the loss criterion.
            scores(Tensor): a sequence of predict output with scores.
        """
        pred = scores.max(1)[1]
        non_padding = target.ne(self.padding_idx)
        num_correct = pred.eq(target) \
            .masked_select(non_padding) \
            .sum()
        return Statistics(loss, memory_loss_data, KLD_data, cluster_loss_data, non_padding.sum().item(),
                          num_correct.item(), emo_cls_loss_data, emo_pred_loss_data, pred_correct)

    def bottle(self, v):
        return v.view(-1, v.size(2))

    def unbottle(self, v, batch_size):
        return v.view(-1, batch_size, v.size(1))
