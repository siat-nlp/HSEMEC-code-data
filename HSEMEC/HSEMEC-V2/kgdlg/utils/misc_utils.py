import yaml
import torch.utils.data as data
import codecs
import math
import os
import sys, time
import kgdlg
class HParams(object):
    def __init__(self, **entries): 
        self.__dict__.update(entries)   


def load_hparams(config_file):
    with codecs.open(config_file, 'r', encoding='utf-8') as f:
        configs = yaml.load(f)
        hparams = HParams(**configs)
        return hparams

        
def safe_exp(value):
    """Exponentiation with catching of overflow error."""
    try:
        ans = math.exp(value)
    except OverflowError:
        ans = float("inf")
    return ans


def latest_checkpoint(model_dir):
    
    cnpt_file = os.path.join(model_dir,'checkpoint')
    try:
        cnpt = open(cnpt_file,'r').readline().strip().split(':')[-1]
    except:
        return None
    cnpt = os.path.join(model_dir,cnpt)
    return cnpt



class ShowProcess():
    i = 1
    max_steps = 0
    max_arrow = 50

    def __init__(self, max_steps):
        self.max_steps = max_steps
        self.i = 1

    def show_process(self, i=None):
        if i is not None:
            self.i = i
        num_arrow = int(self.i * self.max_arrow / self.max_steps)
        num_line = self.max_arrow - num_arrow
        percent = self.i * 100.0 / self.max_steps
        process_bar = '[' + '>' * num_arrow + '-' * num_line + ']'\
                      + '%.2f' % percent + '%' + '\r'
        sys.stdout.write(process_bar)
        sys.stdout.flush()
        self.i += 1

    def close(self, words='done'):
        print('')
        print(words)
        self.i = 1
