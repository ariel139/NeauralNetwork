class Error:
    @staticmethod
    def type_error(data, expected_type, ):
        if not isinstance(data, expected_type):
            raise TypeError(f'None Valid Data Type!\ngot {type(data)} instead of a {expected_type}')


