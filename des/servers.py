"""Models a collection of servers.

A collection of servers listen for events and process the events from a queue
based on their availability.
The collection of servers can be of many types, for example, FIFO, priorityQ,
LIFO etc.

Extend BaseServers to model your own server priority.

Since, when events are being served in a queue - we cannot "schedule" the
events beforehand, they will be  executed by the server.
The server needs two events to execute - one at the start of the service
and the second at the end of the service.
"""
import abc
import heapq
from typing import Text, List, Optional, Union, Tuple

from .simulator import Event, Process, Simulator


class BaseManagers(abc.ABC):
    """Models a collection of servers that are managed together.

    This class maintains the main queue by which tasks are assigned to the
    servers. Implementations of this class specify how tasks are added/removed
    from the queue.
    """
    def __init__(self, name: Text,
                 sim: Simulator) -> None:
        self.name: Text = name
        self.servers: List[BaseServer] = []
        self.sim: Simulator = sim
        self.queue: List[Tuple[Union[int, float], int, Tuple[Event, Event]]] \
            = []
        self._events_pushed: int = 0

    def add_server(self, server: 'BaseServer'):
        assert server.currently_running is False, "Cannot provide a server " \
                                                  "whose process is already " \
                                                  "running."
        self.servers.append(server)
        # start the service for this server.
        server.start(0)

    def request_service(self, event_at_start_of_service: Event,
                        event_at_end_of_service: Event) -> \
            None:
        """An event requests to use the server.

        If there are available servers, then the event is accepted; else it
        is put in a queue.

        Args:
            event_at_start_of_service: The event to execute at the start
            event_at_end_of_service: The event to execute at the end of service.
        """
        # add the work to the queue.
        self.push((event_at_start_of_service, event_at_end_of_service))
        # if any server is available, they should pick up the latest work
        available_server = self._get_next_available_server()
        if available_server is not None:
            # Start the serving process with the available server immediately.
            available_server.start()

    def _get_next_available_server(self) -> Optional['BaseServer']:
        """Gets the next available server

        Returns: The next available server if any.
        """
        available_server = None
        try:
            available_server = next((server for server in self.servers
                                     if server.is_not_busy))
        except StopIteration:
            pass
        return available_server

    def has_work(self) -> bool:
        return len(self.queue) > 0

    @abc.abstractmethod
    def push(self, events: Tuple[Event, Event]) -> None:
        """Pushes work to the queue."""
        pass

    @abc.abstractmethod
    def pull(self) -> Optional[Tuple[Event, Event]]:
        """Pulls work from the queue."""
        pass


class BaseServer(Process, abc.ABC):
    """Models a single server & its service process.

    Implement the get_processing_time method which tells the server the time
    taken to process a given event. This can access any state of the server
    (modeled by the user) and the current event w/ the server (stored as
    _current_event). The current event is actually a tuple of events -
    signifying the event that occurs at the start of service and the event that
    occurs at the end of the service.
    """

    def __init__(self, name: Text, manager: BaseManagers, sim: Simulator):
        super().__init__(name, sim)
        self.name: Text = name
        self.sim: Simulator = sim
        self._busy: bool = False
        self._next_event_time: Optional[float] = None
        self._current_event: Optional[Tuple[Event, Event]] = None
        self._current_event_start_time: Optional[Union[int, float]] = None
        self._events: List[
            Tuple[Event, Union[int, float], Union[int, float]]] = []
        self.manager: BaseManagers = manager

    # The process related functions
    def execute(self):
        """
        The server process executes based on the state of the system.
        """
        if self._current_event is None:
            # not currently working on a job - so start a new job.
            if self.can_continue():
                # take the next available job and assign it to yourself.
                self._current_event = self.manager.pull()
                # start the current event
                start_event = self._current_event[0]
                start_event.execute()
                self._current_event_start_time = self.sim.now
                # mark yourself busy
                self._busy = True
                # This particular server has the next event to be scheduled
                # at the end of service
                self._next_event_time = self.get_processing_time()
        else:
            # This is the end of the event.
            end_event = self._current_event[1]
            end_event.execute()
            self._events.append((self._current_event,
                                 self._current_event_start_time,
                                 self.sim.now))
            # clear all the events and set yourself free
            self._current_event = None
            self._current_event_start_time = None
            self._busy = False
            # The next event in the service process is immediate.
            self._next_event_time = 0.0

    def can_continue(self):
        if self._current_event is not None:
            # if currently working on an event, then the process continues...
            return True
        else:
            # otherwise  the process finishes when the system has no more work.
            return self.manager.has_work()

    def get_next_event_time(self):
        return self._next_event_time

    @property
    def is_busy(self) -> bool:
        return self._busy

    @property
    def is_not_busy(self) -> bool:
        return not self._busy

    @abc.abstractmethod
    def get_processing_time(self) -> Union[float, int]:
        """Returns the time needed to process the event.
        This method tells the server the time
        taken to process a given event. This can access any state of the server
        (modeled by the user) and the current event w/ the server (stored as
        _current_event). The current event is actually a tuple of events -
        signifying the event that occurs at the
        """
        pass


class FIFOManagers(BaseManagers):
    """Basic Queue - Work is dispatched in first come first serve order"""

    def __init__(self, name: Text, sim: Simulator):
        super().__init__(name, sim)

    def push(self, events: Tuple[Event, Event]):
        # First in based on arrival time!
        heapq.heappush(self.queue, (self.sim.now,  self._events_pushed, events))
        self._events_pushed += 1

    def pull(self) -> Optional[Tuple[Event, Event]]:
        try:
            _, _, events = heapq.heappop(self.queue)
            return events
        except IndexError:
            return None
