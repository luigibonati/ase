"""
Exciting Input
"""


class ExcitingInput:
    """
    Base class for exciting inputs
    """

    def attributes_to_dict(self) -> dict:
        """

        """
        dictionary = self.__dict__
        # TODO(Alex) Strip all entries apart from attributes from dictionary
        return dictionary


def query_exciting_version(exciting_root):
    """
    Query the exciting version

    TODO Parse me from src/version.f90, post-build
    """
    return 'Not implemented'
