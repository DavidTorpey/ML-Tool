from uuid import uuid4

class Utils(object):

    def __init__(self):
        pass

    def generateFileName(self, extension):
        return str(uuid4()).replace('-','') + extension