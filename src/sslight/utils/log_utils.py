import os, sys, atexit, builtins, decimal, logging, functools, simplejson, subprocess, pickle, json
from collections import deque
from fvcore.common.file_io import PathManager


"""
Utility functions for logging.
Mostly copy-paste from the dino repo:
https://github.com/facebookresearch/dino/blob/main/utils.py
"""


def get_sha():
    ''' Get the git sha of the current repo '''
    cwd = os.path.dirname(os.path.abspath(__file__))

    def _run(command):
        return subprocess.check_output(command, cwd=cwd).decode('ascii').strip()
    sha = 'N/A'
    diff = "clean"
    branch = 'N/A'
    try:
        sha = _run(['git', 'rev-parse', 'HEAD'])
        subprocess.check_output(['git', 'diff'], cwd=cwd)
        diff = _run(['git', 'diff-index', 'HEAD'])
        diff = "has uncommited changes" if diff else "clean"
        branch = _run(['git', 'rev-parse', '--abbrev-ref', 'HEAD'])
    except Exception:
        pass
    message = f"sha: {sha}, status: {diff}, branch: {branch}"
    return message


def _suppress_print():
    """
    Suppresses printing from the current process.
    """

    def print_pass(*objects, sep=" ", end="\n", file=sys.stdout, flush=False):
        pass

    builtins.print = print_pass


@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    io = PathManager.open(filename, "a", buffering=1024)
    atexit.register(io.close)
    return io


def setup_logging(rank, output_dir=None):
    """
    Sets up the logging for multiple processes. Only enable the logging for the
    master process, and suppress logging for the non-master processes.
    """
    # Set up logging format.
    _FORMAT = "[%(levelname)s: %(filename)s: %(lineno)4d]: %(message)s"

    if rank == 0:
        # Enable logging for the master process.
        logging.root.handlers = []
    else:
        # Suppress logging for non-master processes.
        if False:
            _suppress_print()

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    plain_formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(filename)s: %(lineno)3d: %(message)s",
        datefmt="%m/%d %H:%M:%S",
    )

    if rank==0:
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(plain_formatter)
        logger.addHandler(ch)

    if output_dir is not None and rank==0:
        filename = os.path.join(output_dir, "stdout.log")
        fh = logging.StreamHandler(_cached_log_stream(filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(plain_formatter)
        logger.addHandler(fh)


def get_logger(name):
    """
    Retrieve the logger with the specified name or, if name is None, return a
    logger which is the root logger of the hierarchy.
    Args:
        name (string): name of the logger.
    """
    return logging.getLogger(name)


def log_json_stats(stats):
    """
    Logs json stats.
    Args:
        stats (dict): a dictionary of statistical information to log.
    """
    stats = {
        k: decimal.Decimal("{:.5f}".format(v)) if isinstance(v, float) else v
        for k, v in stats.items()
    }
    json_stats = simplejson.dumps(stats, sort_keys=True, use_decimal=True)
    logger = get_logger(__name__)
    logger.info("json_stats: {:s}".format(json_stats))


###########################################################
## Functions for saving and loading data
###########################################################

def pickle_load(path):
    with open(path, 'rb') as fid:
        data_ = pickle.load(fid)
    return data_


def pickle_save(path, data):
    with open(path, 'wb') as fid:
        pickle.dump(data, fid)


def json_load(path):
    with open(path, 'r') as fid:
        data_ = json.load(fid)
    return data_


def json_save(path, data):
    with open(path, 'w') as fid:
        json.dump(data, fid, indent=4, sort_keys=True)


###########################################################
## Accumulating metrics 
###########################################################

class AverageMeter():
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        if self.count == 0:
            return str(self.val)

        return f'{self.val:.4f} ({self.avg:.4f})'


class MovingAverageMeter(object):
    def __init__(self, window):
        self.window = window
        self.reset()

    def reset(self):
        self.history = deque()
        self.avg = 0
        self.sum = None
        self.val = None

    @property
    def count(self):
        return len(self.history)

    @property
    def isfull(self):
        return len(self.history) == self.window

    def __getstate__(self):
        state = self.__dict__.copy()
        state['history'] = np.array(state['history'])
        return state

    def __setstate__(self, state):
        state['history'] = deque(state['history'])
        self.__dict__.update(state)

    def update(self, val, n=1):
        if n == 1:
            self.update_one_sample(val)
            return 

        self.history.extend([val] * n)
        if self.sum is None:
            self.sum = val * n
        else:
            self.sum += val * n
        while len(self.history) > self.window:
            self.sum -= self.history.popleft()
        self.val = val
        self.avg = self.sum / self.count
    
    def update_one_sample(self, val):
        self.history.append(val)
        if self.sum is None:
            self.sum = val
        else:
            self.sum += val
        if len(self.history) > self.window:
            self.sum -= self.history.popleft()
        self.val = val
        self.avg = self.sum / self.count

    def __str__(self):
        if self.count == 0:
            return str(self.val)

        return f'{self.val:.4f} ({self.avg:.4f})'

    def __repr__(self):
        return "<MovingAverageMeter of window {} with {} elements, val {}, avg {}>".format(
            self.window, self.count, self.val, self.avg)