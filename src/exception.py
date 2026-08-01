import sys

def error_detail_message(error,error_detail:sys):
    _,_,traceback=error_detail.exc_info()
    filename=traceback.tb_frame.f_code.co_filename
    lineno=traceback.tb_lineno
    error_message=f'Error occured in file name: {filename}\nline number: {lineno}\nError Message: {str(error)}'
    return error_message

class CustomException(Exception):
    def __init__(self,error_message,error_detail:sys):
        super().__init__(error_message)
        self.error_message=error_detail_message(
            error=error_message,
            error_detail=error_detail)
        
    def __str__(self):
        return self.error_message