from collections import deque
import threading

class ThreadSafeDeque:
    def __init__(self, iterable=(), maxlen = None):
        self.deque = deque(iterable, maxlen)
        self.maxlen = self.deque.maxlen
        self.lock = threading.Lock()
        self.non_empty = threading.Condition(self.lock)

    # if the queue is empty, the thread will wait. 
    def pop(self):
        with self.non_empty:
            while not self.deque:
                self.non_empty.wait()
            return self.deque.pop()
    
    def popleft(self):
        with self.non_empty:
            while not self.deque:
                self.non_empty.wait()
            return self.deque.popleft()

    def append(self, item):
        with self.non_empty:
            self.deque.append(item)
            self.non_empty.notify()
    
    def clear(self):
        with self.lock:
            self.deque.clear()
    
    def remove(self, item):
        with self.lock:
            self.deque.remove(item)

    def __len__(self):
        with self.lock:
            return len(self.deque)
    
    def __str__(self):
        with self.lock:
            return self.deque.__str__()
    
    def lazy_pop(self):
        '''
        pop nothing if the queue is empty, do not wait
        '''
        with self.non_empty:
            while not self.deque:
                return None
            return self.deque.pop()
    
    