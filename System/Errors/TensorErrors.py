class TensorMatMulException(Exception):
    def __init__(self, message, code=None):
        super().__init__(message)
        self.code = code

    def __str__(self):
        if self.code:
            return f"{self.args[0]} (Error code: {self.code})"
        return self.args[0]
