
def incrementCount(start):
    count = start
    while True:
        yield count
        count += 1
