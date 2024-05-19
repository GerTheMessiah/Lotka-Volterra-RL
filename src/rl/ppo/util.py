import sys


def print_progress(iteration, total, prefix='', suffix='', decimals=2, bar_length=100):
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = '█' * filled_length + '-' * (bar_length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percents}% {suffix}')
    if iteration == total:
        sys.stdout.write('\r\n')
    sys.stdout.flush()
