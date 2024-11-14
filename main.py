import traceback
from tokenizer import full_tokenize
from parser import parse_expression
from executor import execute_tree
from errors import MathError

def execute(expression: str):
    return execute_tree(parse_expression(full_tokenize(expression)))

if __name__ == '__main__':
    while True:
        broken = False
        prompt = 'λ>'
        expression = ""
        while True:
            try:
                expression += input(f'{prompt} ')
            except KeyboardInterrupt:
                broken = True
                break
            if not expression.strip():
                break
            if not expression.endswith(';'):
                prompt = '..'
                continue
            try:
                result = execute(expression[:-1])
                print(result)
                break
            except (MathError, AssertionError) as e:
                traceback.print_exception(e)
                break
            except Exception as e:
                print(' '.join(traceback.format_exception_only(e)), end="")
                break
        if broken:
            break
