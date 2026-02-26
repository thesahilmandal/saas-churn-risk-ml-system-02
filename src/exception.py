import sys
from src.logging import logging


class CustomerChurnException(Exception):
    """
    Custom exception class to capture detailed error context
    (file name, line number, and error message).
    """

    def __init__(self, error_message: Exception, error_details: sys):
        super().__init__(error_message)

        _, _, exc_tb = error_details.exc_info()

        if exc_tb:
            self.file_name = exc_tb.tb_frame.f_code.co_filename
            self.lineno = exc_tb.tb_lineno
        else:
            self.file_name = "Unknown"
            self.lineno = -1

        self.error_message = error_message

    def __str__(self) -> str:
        return (
            f"Error occurred in python script "
            f"[{self.file_name}] at line number "
            f"[{self.lineno}] with error message "
            f"[{self.error_message}]"
        )


if __name__ == "__main__":
    try:
        logging.info("Entered the try block")
        a = 1 / 0
    except Exception as e:
        logging.error("Exception occurred", exc_info=True)
        raise CustomerChurnException(e, sys)