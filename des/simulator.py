"""Simulator class runs the discrete event simulation.

All simulations are managed via the simulation environment (simulator)
provided by this class. This environment keeps track of all the events
and executes them according to the schedule.

The simulator sequentially executes Events.
"""
import abc
import dataclasses
import heapq
from typing import Optional, List, Text, Tuple, Union


@dataclasses.dataclass
class EventID(object):
    """Unique name for an event.

    Consists of:
    time the event is scheduled,
    its priority
    and its unique event id.

    """
    scheduled_time: float
    priority: int
    event_id: int


class Event(abc.ABC):
    """Models an event.

    An event consists of two methods:
    1. schedule: Schedules the event on the simulator.
    2. execute: Executes the event.

    All events in a discrete event simulation are extensions of this base
    event class - with responsibility to specify the execute method. Any state
    that the event should care about must be contained within the event, so that
    execute() can access the state.
    """
    def __init__(self, sim: 'Simulator') -> None:
        """Constructor.

        Args:
            sim: The simulator to which the event is attached.
        """
        self._sim: 'Simulator' = sim
        self._scheduled: bool = False
        self._completed: bool = False
        self._event_id: Optional['EventID'] = None

    def schedule(self, delay: Union[int, float], priority: int) -> None:
        """Schedule the event.

        Args:
            delay: The delay after which the event should be executed.
            priority: The priority of the event (0 is the highest priority)
        """
        self._event_id = self._sim.schedule(self, delay, priority)
        self._scheduled = True

    @property
    def sim(self):
        return self._sim

    @abc.abstractmethod
    def execute(self):
        """Abstract method that describes the actions to be taken when an
        event is executed."""
        pass

    def can_execute(self, event_id: 'EventID') -> bool:
        """Does a check if the event ID being executed matches the event ID
        assigned to the event when it was scheduled.

        Args:
            event_id: The Event ID that the simulator is executing
        """
        assert self._scheduled is True, "Cannot execute an event that is not " \
                                        "scheduled"

        return self._event_id == event_id and not self._completed

    def set_completed(self):
        self._completed = True

    @property
    def completed(self):
        return self._completed


class _ProcessEvent(Event):
    """A Process Event manages the process by scheduling the next event in
    the process.
    """

    def __init__(self, sim: 'Simulator', process: 'Process') -> None:
        """Constructor.

        Args:
          sim: The simulator object
          process: The process that we are simulating
        """
        super().__init__(sim)
        self.process = process

    def execute(self):
        self.process.execute()
        if self.process.can_continue():
            self.process.resume()
        else:
            self.process.mark_completed()


class Process(abc.ABC):
    """Models a process.

    A process is defined by 3 user provided functions:
    (i) can_continue? A boolean function that tells if the process continues
    or not.
    (ii) execute: What needs to be done when an event from the process occurs.
    (iii) get_next_event_time: When should the next event in the process
    generating function be scheduled.
    """
    def __init__(self, name: Text, sim: 'Simulator') -> None:
        """Constructor.

        Args:
            sim: Simulator
            name: Name of the process
        """
        self.name = name
        self.sim = sim
        self.currently_running = False

    @abc.abstractmethod
    def execute(self) -> None:
        """Implements the actions that should take place when an event from
        the process takes place."""
        pass

    @abc.abstractmethod
    def can_continue(self) -> bool:
        """Tells if there are more events in the process or not."""
        pass

    @abc.abstractmethod
    def get_next_event_time(self) -> Union[int, float]:
        """Tells when the next event from the process must be scheduled."""
        pass

    def start(self, delay: [Union[int, float]] = 0, priority: Optional[
        int] = 0) -> None:
        """Starts the process.

         Args:
             delay: The delay with which to start the process
             priority: The priority of the process
         """
        assert self.currently_running is False, "Cannot start a process that " \
                                                "is already running."
        _ProcessEvent(self.sim, self).schedule(delay, priority)

    def mark_completed(self):
        """Once a process is marked completed, it can start again."""
        self.currently_running = False

    def resume(self) -> None:
        """Schedules the next event of the process.
        """
        next_event_at = self.get_next_event_time()
        _ProcessEvent(self.sim, self).schedule(next_event_at, 0)


class Simulator(object):

    def __init__(self, name: Text, initial_time: float = 0) -> None:
        """Initialize the simulator.

        Args:
            name: A name for the simulator
            initial_time: the strting time (absolute) for the simulator
        """
        self.name = name
        self._events: List[Tuple[float, int, int, 'Event']] = []
        self._now = initial_time
        self._event_id = 0

    def schedule(self, event: 'Event', delay: float = 0,
                 priority: int = 0) -> EventID:
        """Schedules the event.

        Args:
            event: The event to schedule
            delay: The delay after which to schedule the event.
            priority: The priority for the event. 0 is the highest priority.
        """
        scheduled_time = self._now + delay
        event_id = EventID(scheduled_time, priority, self._event_id)
        heapq.heappush(self._events,
                       (scheduled_time, priority, self._event_id, event))
        self._event_id += 1
        return event_id

    @property
    def now(self):
        return self._now

    def run(self, run_until: Optional[float] = None) -> None:
        """Simulates the process until the given time.

        Args:
            run_until: The time up to which we wish to simulate the process.
        """
        if run_until:
            assert run_until > self._now, "simulate until must be greater " \
                                          "than current clock time"
        while True:
            try:
                self._now, priority, event_num, event = heapq.heappop(
                    self._events)
                event_id = EventID(self._now, priority, event_num)
                if event.can_execute(
                        event_id):  # check if event is not completed and has
                    # the same eventId
                    event.execute()
                    event.set_completed()
                # otherwise, do nothing - we don't have to execute this event
                # as something has
                # changed since it was first scheduled.
                if run_until and self._now > run_until:
                    break
            except IndexError:
                break  # end of simulation
        return None
