
def cookie(flavour = "chocolate chip", shaepe = "round"):

    return f"Here is your {shaepe} {flavour} cookie!"

class Cookie:
    def __init__(self, flavour = "chocolate chip", shape = "round"):
        self.flavour = flavour
        self.shape = shape

    def __str__(self):
        return f"Here is your {self.shape} {self.flavour} cookie!"


if __name__ == "__main__":
    print(cookie())
    print(cookie("oatmeal raisin", "square"))
    cookie1 = Cookie()
    cookie2 = Cookie("oatmeal raisin", "square")
    

