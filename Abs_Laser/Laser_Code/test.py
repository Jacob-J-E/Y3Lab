def y(x,*coe):
    args = list(coe)
    print(type(coe))
    print(coe[-1])
    print(coe[-2])
    print(coe[:-2])
    print(x)

y(3,1,2,3,4,5,6,7,8,9,10)