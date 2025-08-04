import ast
import numpy as np
code="""
def nStarDiamond(n: int) -> None:
    # Write your code here.
    for i in range(n):
        for j in range(n-i-1):
            print('',end=' ')
        for j in range(2*i+1):
            print('*',end='')
        print()
    for i in range(n,0,-1):
        for j in range(n-i):
            print('',end=' ')
        for j in range(2*i-1):
            print('*',end='')
        print()
nStarDiamond(3)
"""
print(ast.dump(ast.parse(code),indent=2))
