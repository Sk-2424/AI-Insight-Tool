import sys
import logger

def error_message_detail(error, error_details:sys):
    """
    This function return error message with it's source script, line no. and error message
    """
    _,_,exc_tb = error_details.exc_info()  # it return three objects execption type, exception value & exception object
    file_name = exc_tb.tb_frame.f_code.co_filename # extract the filename where execption has occured
    error_message = f"Error occured in the python scripts in {file_name} at line no. {exc_tb.tb_lineno} error message {str(error)}"
    return error_message

class CustomException(Exception):
    def __init__(self,error_message,error_details:sys):
        super().__init__(error_message)
        self.error_message=error_message_detail(error_message,error_details=error_details)
    
    def __str__(self):
        return self.error_message


if __name__ == '__main__':  
    def division(a,b):  ## testing
        try:
            return a/b
        except Exception as e:
            logger.logging.info(CustomException(e, sys))
            raise CustomException(e, sys)
    
    print(division(5,0))
    