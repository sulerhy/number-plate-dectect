
def isDigitPresent(x, d):

    # Breal loop if d is present as digit
    while (x > 0):

        if (x % 10 == d):
            break

        x = x / 10


    # If loop broke
    return (x > 0)
results= []
for i in range(1 ,30001):
    if ( i %3 == 0) or isDigitPresent(i, 3):
        if not(( i %3 == 0) and isDigitPresent(i, 3)):
            results.append(i)
            print(i)

print(sum(results))
