NAME_TRACKER: dict[str, int] = {}
REGISTERED_OBJECTS: dict[str, object] = {}


def to_unique(*name):
    name = "_".join([str(n) for n in name if n is not None])
    if name not in NAME_TRACKER:
        NAME_TRACKER[name] = 0
    NAME_TRACKER[name] += 1
    return f"{name}_{NAME_TRACKER[name]}"


def register_object(obj, name: str):
    if name in REGISTERED_OBJECTS:
        raise ValueError(f"Object with name {name} is already registered.")
    REGISTERED_OBJECTS[name] = obj


def get_registered_object(name: str, default=None):
    return REGISTERED_OBJECTS.get(name, default)


def get_registered_objects():
    return REGISTERED_OBJECTS
