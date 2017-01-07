from settings import PRINTED_BOARD_SQUARE_CHAR

# Inspired by http://stackoverflow.com/a/4037979
class cached_property(object):
    def __init__(self, factory):
        self.attr_name = factory.__name__
        self.factory = factory

    def __get__(self, instance, cls):
        attr = self.factory(instance)
        setattr(instance, self.attr_name, attr)
        return attr


def stringify_grid(grid):
    string = ["|" + ''.join([PRINTED_BOARD_SQUARE_CHAR if c == 1 else " " for c in r]) + "|" for r in grid]
    string = ["-" * (grid.shape[1] + 2)] + string + ["-" * (grid.shape[1] + 2)]
    return '\n'.join(string)


# Inspired by http://stackoverflow.com/questions/5980042
def get_verbose_print_func(verbose=False):
    return print if verbose else (lambda * a, ** k: None)
