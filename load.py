from detecto import core


def init():
    model = core.Model.load('model_weights.pth', ['rbc'])
    return model
