""" Explaination and Example for Concurrency in Python
"""
import logging
import os, math
import threading
import multiprocessing as mp
import concurrent.futures
import queue

"""
    Method
        1. Process / Thread, indicate directly
        2. In process, Pool
        3. concurrent.futures library

    Concept
        1. Multiprocessing
            (In personal opinion,)  
            Method 1. Pool    : If there's itereation, it was convenient. (iter)
            Method 2. Process : It was convenient 
                                if there's no corrleation between operations.
            
            Children type: 'fork', 'spawn' or 'forkserver'.

            Memory: Queue, Pipe
            Shared memory: Value, Array

            TODO:
                Mananger
                Proxy
                Lock
                Semaphore
                Async
                Listener and Client

            Reference
                https://docs.python.org/3/library/multiprocessing.html
        
    2. Thread
        It has two methods:
            1. Threading
            2. Concurrent library

        Property
            Condition
                acquire()
                release()
                wait()
                wait_for()
                notify()
                notify_all()
            Semaphore
                acquire()
                release()
            BoundedSemaphore
            Event
                is_set()
                set()
                clear()
                wait()
            Timer
            Barrier

        Reference
            https://realpython.com/intro-to-python-threading/#daemon-threads


    * concurrent.futures
        Reference
            concurrent concept: https://realpython.com/python-concurrency/
            example: https://docs.python.org/3/library/concurrent.futures.html

    The concept of Operation System: https://www.notion.so/ooshyun/906ced560be4405596aeddbbf17ba5bd

"""
import time
import random
import sys

""" Logging for multiprocessing """
logger = mp.log_to_stderr()
logger.setLevel(logging.INFO)
# set the logger for debug
format = "%(asctime)s: %(message)s"
logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")
logging.getLogger().setLevel(logging.DEBUG)

""" Functions used by test code """


def thread_function(name):
    logging.info("Thread %s: starting", name)
    time.sleep(2)
    logging.info("Thread %s: finishing", name)


class FakeDatabase:
    def __init__(self):
        self.value = 0
        self._lock = threading.Lock()

    def update(self, name):
        # Race condition
        logging.info("Thread %s: starting update", name)
        local_copy = self.value
        local_copy += 1
        time.sleep(0.1)
        self.value = local_copy
        logging.info("Thread %s: finishing update", name)

    def locked_update(self, name):
        logging.info("Thread %s: starting update", name)
        logging.debug("Thread %s about to lock", name)
        with self._lock:  # same as self._lock.acquire()
            logging.debug("Thread %s has lock", name)
            local_copy = self.value
            local_copy += 1
            time.sleep(0.1)
            self.value = local_copy
            logging.debug("Thread %s about to release lock", name)
        # same as self._lock.release()

        logging.debug("Thread %s after release", name)
        logging.info("Thread %s: finishing update", name)


def calculate(func, args):
    result = func(*args)

    return "%s says that %s%s = %s" % (
        mp.current_process().name,
        func.__name__,
        args,
        result,
    )


def worker(input, output):
    for func, args in iter(input.get, "STOP"):
        result = calculate(func, args)
        output.put(result)


def calculatestar(args):
    return calculate(*args)


def mul(a, b):
    time.sleep(0.5 * random.random())
    return a * b


def plus(a, b):
    time.sleep(0.5 * random.random())
    return a + b


def f(x):
    return 1.0 / (x - 5.0)


PRIMES = [
    112272535095293,
    112582705942171,
    112272535095293,
    115280095190773,
    115797848077099,
    1099726899285419,
]


def is_prime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False

    sqrt_n = int(math.floor(math.sqrt(n)))
    for i in range(3, sqrt_n + 1, 2):
        if n % i == 0:
            return False
    return True


""" Test code """


def test_multiprocessing_pool():
    """ 1. Multiprocess, Use Pool
        Reference. https://docs.python.org/3/library/multiprocessing.html#multiprocessing-programming
    """
    PROCESSES = 4
    print("Creating pool with %d processes\n" % PROCESSES)

    with mp.Pool(PROCESSES) as pool:
        #
        # Tests
        #

        TASKS = [(mul, (i, 7)) for i in range(10)] + [(plus, (i, 8)) for i in range(10)]

        results = [pool.apply_async(calculate, t) for t in TASKS]
        results.append(pool.apply_async(calculate, (mul, (10, 7))))

        imap_it = pool.imap(calculatestar, TASKS)
        imap_unordered_it = pool.imap_unordered(calculatestar, TASKS)

        print("Ordered results using pool.apply_async():")
        for r in results:
            print("\t", r.get())  # Call get() means run the function
        print()

        print("Ordered results using pool.imap():")
        for x in imap_it:
            print("\t", x)
        print()

        print("Unordered results using pool.imap_unordered():")
        for x in imap_unordered_it:
            print("\t", x)
        print()

        print("Ordered results using pool.map() --- will block till complete:")
        for x in pool.map(calculatestar, TASKS):
            print("\t", x)
        print()

        #
        # Test error handling
        #

        print("Testing error handling:")

        try:
            print(pool.apply(f, (5,)))
        except ZeroDivisionError:
            print("\tGot ZeroDivisionError as expected from pool.apply()")
        else:
            raise AssertionError("expected ZeroDivisionError")

        try:
            print(pool.map(f, list(range(10))))
        except ZeroDivisionError:
            print("\tGot ZeroDivisionError as expected from pool.map()")
        else:
            raise AssertionError("expected ZeroDivisionError")

        try:
            print(list(pool.imap(f, list(range(10)))))
        except ZeroDivisionError:
            print("\tGot ZeroDivisionError as expected from list(pool.imap())")
        else:
            raise AssertionError("expected ZeroDivisionError")

        it = pool.imap(f, list(range(10)))
        for i in range(10):
            try:
                x = next(it)
            except ZeroDivisionError:
                if i == 5:
                    pass
            except StopIteration:
                break
            else:
                if i == 5:
                    raise AssertionError("expected ZeroDivisionError")

        assert i == 9
        print("\tGot ZeroDivisionError as expected from IMapIterator.next()")
        print()

        # Testing timeouts

        print("Testing ApplyResult.get() with timeout:", end=" ")
        res = pool.apply_async(calculate, TASKS[0])
        while 1:
            sys.stdout.flush()
            try:
                sys.stdout.write("\n\t%s" % res.get(0.02))
                break
            except mp.TimeoutError:
                sys.stdout.write(".")
        print()
        print()

        print("Testing IMapIterator.next() with timeout:", end=" ")
        it = pool.imap(calculatestar, TASKS)
        while 1:
            sys.stdout.flush()
            try:
                sys.stdout.write("\n\t%s" % it.next(0.02))
            except StopIteration:
                break
            except mp.TimeoutError:
                sys.stdout.write(".")
        print()
        print()


def test_multiprocessing_process():
    """2. Multiprocess, Use Process
        Reference. https://docs.python.org/3/library/multiprocessing.html#multiprocessing-programming

        run    : after start, process run the instruction including this 'run' function
        start  : after start, process is alive (is_alive() == True)
        join   : 
            1. When join() is called in a process, then the process waits for all of processes 
                already called join() method
            2. If timeout is positive, then process block time which is timeout value.
        daemon : child process will shut down immedately after main process finish.
        name, is_alive(), pid, close()
    """
    procs = []

    NUMBER_OF_PROCESSES = 4

    TASKS1 = [(mul, (i, 7)) for i in range(10)]
    TASKS2 = [(plus, (i, 8)) for i in range(10)]

    # If use queue in queue library, it raise "TypeError: cannot pickle '_thread.lock' object"
    task_queue = mp.Queue()
    done_queue = mp.Queue()

    for task in TASKS1:
        task_queue.put(task)

    for i in range(NUMBER_OF_PROCESSES):
        proc = mp.Process(target=worker, args=(task_queue, done_queue), daemon=True)
        procs.append(proc)
        proc.start()

    # Get and print results
    print("Unordered results:")
    for i in range(len(TASKS1)):
        print("\t", done_queue.get())

    # Add more tasks using `put()`
    for task in TASKS2:
        task_queue.put(task)

    # Get and print some more results
    for i in range(len(TASKS2)):
        print("\t", done_queue.get())

    # Tell child processes to stop
    for i in range(NUMBER_OF_PROCESSES):
        task_queue.put("STOP")

    # In this place, join is useless because the operation in worker end only when receive STOP
    # If there's no STOP, then always on.
    # for proc in procs:
    #     proc.join()


def test_multiprocessing_concurrent():
    NUM_PROCESS_NUMBERS = 4
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=NUM_PROCESS_NUMBERS
    ) as executor:
        for number, prime in zip(PRIMES, executor.map(is_prime, PRIMES)):
            print("%d is prime: %s" % (number, prime))


def test_threading_thread():
    threads = list()
    for index in range(3):
        logging.info("Main    : create and start thread %d.", index)
        x = threading.Thread(target=thread_function, args=(index,))
        threads.append(x)
        x.start()

    for index, thread in enumerate(threads):
        logging.info("Main    : before joining thread %d.", index)
        thread.join()
        logging.info("Main    : thread %d done", index)


def test_concurrent_thread():
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        executor.map(thread_function, range(3))


def test_threading_race_condition():
    database = FakeDatabase()
    logging.info("Testing update. Starting value is %d.", database.value)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        for index in range(2):
            # race condition
            # executor.submit(database.update, index)
            # solve race condition
            executor.submit(database.locked_update, index)
    logging.info("Testing update. Ending value is %d.", database.value)


def inc(x):
    x += 1


if __name__ == "__main__":
    # Check the number of process through os library
    print("This computer has %d CPUs" % os.cpu_count())
    print("Child process is %d" % os.getpid())

    # Check the number of process through multiprocessing library
    print("This computer has %d CPUs" % mp.cpu_count())
    print(f"Child alive process is {mp.active_children()}")

    """ 1. Multiprocessing """

    # Calling freeze_support() has no effect when invoked on any operating system other than Windows.
    # In addition, if the module is being run normally by the Python interpreter
    # on Windows (the program has not been frozen), then freeze_support() has no effect.
    # mp.freeze_support()

    # test_multiprocessing_pool()
    # test_multiprocessing_process()
    # test_multiprocessing_concurrent()

    """ 2. Threading """

    # test_threading_thread()
    # test_threading_race_condition()
    # test_concurrent_thread()

    """ Details """
    
    # import dis
    # print(dis.dis(inc))
