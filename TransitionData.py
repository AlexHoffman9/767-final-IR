# Class for Transition data for the buffer
from enum import Enum

class TransitionComponent(Enum):
    state = 0
    action = 1
    reward = 2
    next_state = 3
    ratio = 4

def extract_transition_components(transitions_list, transition_component):
    return [item[transition_component.value] for item in transitions_list]
