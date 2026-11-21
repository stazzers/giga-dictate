"""Serialize native audio control without ever blocking AppKit or process exit."""
import threading
from concurrent.futures import Future
from queue import Empty, Queue


class AudioExecutor:
    def __init__(self):
        self.queue = Queue()
        self.lock = threading.Lock()
        self.closed = False
        self.thread = threading.Thread(target=self._run, name="GigaAudioControl", daemon=True)
        self.thread.start()

    def submit(self, function, *args):
        future = Future()
        with self.lock:
            if self.closed:
                future.set_exception(RuntimeError("Audio control is closed"))
            else:
                self.queue.put((future, function, args))
        return future

    def _run(self):
        while True:
            task = self.queue.get()
            if task is None:
                return
            future, function, args = task
            if future.set_running_or_notify_cancel():
                try:
                    future.set_result(function(*args))
                except BaseException as error:
                    future.set_exception(error)

    def shutdown(self, wait=False, cancel_futures=False):
        with self.lock:
            self.closed = True
            if cancel_futures:
                while True:
                    try:
                        task = self.queue.get_nowait()
                        if task:
                            task[0].cancel()
                    except Empty:
                        break
            self.queue.put(None)
        if wait:
            self.thread.join(timeout=.2)
