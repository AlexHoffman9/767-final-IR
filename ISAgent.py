import numpy as np
from OffPolicyAgent import OffPolicyAgent

class ISAgent(OffPolicyAgent):
    def __init__(self):
        print("Overridden Constr")
    
    def train(self):
        print("Overridden")


a = ISAgent()
a.train()
