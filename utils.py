from colorama import Fore, Back, Style, init
init(autoreset=True)

class AverageMeter(object):
    """From https://github.com/pytorch/examples/blob/master/imagenet/main.py"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def __repr__(self):
        return f'{self.avg:.2e}'

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class dotdict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"{name} not found")

def log_info(msg):
    print(Fore.YELLOW + "[INFO] " + Style.RESET_ALL + msg)

def log_success(msg):
    print(Fore.GREEN + "[SUCCESS] " + Style.RESET_ALL + msg)

def log_warning(msg):
    # Orange isn't available, so we simulate it with a combination
    print(Fore.LIGHTRED_EX + "[WARNING] " + Style.RESET_ALL + msg)

def log_error(msg):
    print(Fore.RED + "[ERROR] " + Style.RESET_ALL + msg)
