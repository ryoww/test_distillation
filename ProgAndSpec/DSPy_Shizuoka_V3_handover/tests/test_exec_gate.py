import threading
import time

from src import exec_gate


def test_exec_slot_limits_concurrent_holders():
    exec_gate.set_exec_concurrency(2)
    active, peak, lock = [0], [0], threading.Lock()

    def work():
        with exec_gate.exec_slot():
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            time.sleep(0.05)
            with lock:
                active[0] -= 1

    threads = [threading.Thread(target=work) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    exec_gate.set_exec_concurrency(None)
    assert peak[0] == 2


def test_no_limit_by_default():
    exec_gate.set_exec_concurrency(0)
    with exec_gate.exec_slot():
        pass
