from .connect import Agent


class Evaluator:
    def __init__(self, input, output, agent: Agent):
        self.input = input
        self.outputs = output
        self.agent = agent

    def evaluate(self, data):
        return self.agent.client.chat.completions(data)
