

def test_year_string_to_list_parsing():
    from rtml.utils.utils import year_string_to_list
    inputs = [
        ('1990', [1990]),
        ('1990-1991', [1990, 1991]),
        ('1990-91', [1990, 1991]),
        ('90-91', [1990, 1991]),
        ('1989-1991', [1989, 1990, 1991]),
        ('1990-1991+2003-2005', [1990, 1991, 2003, 2004, 2005]),
        ('1990+1999+2005-06', [1990, 1999, 2005, 2006]),
        ('1990+1999', [1990, 1999]),
    ]
    for i, (string, expected) in enumerate(inputs):
        actual = year_string_to_list(string)
        err_msg = f"Input {i+1}: Expected {expected}, but {actual} was returned."
        assert all([a == b for a, b in zip(actual, expected)]), err_msg
