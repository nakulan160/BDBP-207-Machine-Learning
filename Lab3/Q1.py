a=int(input("Enter the value of a: "))
b=int(input("Enter the value of b: "))
c=int(input("Enter the value of c: "))
a_dif_b=abs(a-b)
a_dif_c=abs(a-c)
b_dif_c=abs(b-c)
print(a_dif_b,a_dif_c,b_dif_c)
if (a_dif_b<a_dif_c) and (a_dif_b<b_dif_c):
    print("The closest of the three numbers are", a,b)
elif (a_dif_c<b_dif_c) and (a_dif_c<a_dif_b):
    print("The closest of the three numbers are",a,c)
else:
    print("The closest of the three numbers are",b,c)