import os

PRIVATE_LOC = "DATALAB_PRIVATE_LOC"


def has_private_loc() -> bool:
    return PRIVATE_LOC in os.environ


def replace_private_loc(url_or_filename: str) -> str:
    if PRIVATE_LOC in url_or_filename:
        if not has_private_loc():
            raise ValueError(
                "Attempting to use private dataset, but "
                f"{PRIVATE_LOC} environmental variable not found"
            )
        url_or_filename = url_or_filename.replace(PRIVATE_LOC, os.environ[PRIVATE_LOC])
    return url_or_filename
