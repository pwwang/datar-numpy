
def pytest_sessionstart(session):
    from datar import options

    options(backends="numpy")
