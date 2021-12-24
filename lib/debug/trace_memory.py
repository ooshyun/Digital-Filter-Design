"""Memory tracking
    Not proved yet...
"""
import tracemalloc
from flask import Flask

app = Flask(__name__)
tracemalloc.start()
my_snapshot = None

@app.route("/karyogram", methods=["POST"])
def karyogram():
    print(f"hello memory tracking world")
    
    # Save to global var for using track
    global my_snapshot
    if not my_snapshot:
        # Save the initial state of memory
        my_snapshot = tracemalloc.take_snapshot()
    else:
        lines = []
        # Get the statistics to comparison between current and initial memory state
        top_stats = tracemalloc.take_snapshot().compare_to(my_snapshot, 'lineno')
        # Print Top 10 memory usage
        for stat in top_stats[:10]:
            lines.append(str(stat))
        print('\n'.join(lines), flush=True)

@app.route("/infer", methods=["POST"])
def infer():
    print(f"hello memory tracking world")
    # Get the statistics in current memory state and Print TOP 5 
    snapshot = tracemalloc.take_snapshot()
    for idx, stat in enumerate(snapshot.statistics('lineno')[:5], 1):
        print(str(stat), flush=True)

    # Print the Detail about the most memory usage
    traces = tracemalloc.take_snapshot().statistics('traceback')
    for stat in traces[:1]:
        print("memory_blocks=", stat.count, "size_kB=", stat.size / 1024, flush=True)
        for line in stat.traceback.format():
            print(line, flush=True)


if __name__=='__main__':
    pass