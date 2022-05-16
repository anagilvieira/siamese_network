MODELS = {}

def register_model(cls):
    MODELS[cls.__name__] = cls
    return cls
