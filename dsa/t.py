#print('hello,world',sep=',',end='pp',flush=False)
def nNumberTriangle(n: int) -> None:
    # Write your solution here.
    for i in range(n,1,-1):
        for j in range(1,i):
            print(j,end=' ')
        print()

nNumberTriangle(5)
def nStarTriangle(n: int) -> None:
    # Write your code here.
    for i in range(n,0,-1):
        for j in range(n-i):
            print('',end=' ')
        for j in range(2*i-1):
            print('*',end='')
        for j in range(n-i):
            print('',end=' ')
        print()
nStarTriangle(3)

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
def nStarTriangle(n: int) -> None:
    for i in range(2*n):
        stars=i
        if i>n:
            stars=2*n-i
        for j in range(stars):
            print('*',end=' ')
        print()
nStarTriangle(3)

