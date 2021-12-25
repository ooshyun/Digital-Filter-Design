"""Utility for tracking and debugging
"""
import sys, time


def track_time(func):
    """Track the function time
    """

    def new_func(*args, **kwargs):
        start_time = time.perf_counter()
        func(*args, **kwargs)
        end_time = time.perf_counter()
        exec_time = end_time - start_time
        print(f"Execute Time: {exec_time:.6f}")

    return new_func


def print_progress(iteration,
                   total,
                   prefix='',
                   suffix='',
                   decimals=1,
                   barLength=100):
    """Progress Bar
        This library provides an easy progress bar implementation for refence in the
        terminal. This function should be called everytime the value needs to be 
        updated, since it only provides the graphical part of the progress bar

        Parameters
            - iteration: the current state (n or i) of the progress
            - total: the total number of interactions
            - prefix: what will be shown before the progress bar (in the same line)
            - suffix: what will be shown after the progress bar (in the same line)
            - decimals: the number of fractional digits shown as progress
            - barLength: the number of characters used in the progress bar

        NOTE: You can edit the function for different characters
    """
    formatstr = "{0:." + str(decimals) + "f}"
    percent = formatstr.format(100 * (iteration / float(total)))
    filledLength = int(round(barLength * iteration / float(total)))
    bar = '#' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percent, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\n')
    sys.stdout.flush()


if __name__ == '__main__':
    """Time tracking example"""

    @track_time
    def hello(name):
        print(f"Hello Guys, I'm {name}")

    hello("Cesar")
    """Progress bar example"""
    total = 10e3
    for i in range(int(total)):
        print_progress(i, total)
