### This collection of functions is the backbone of the feasibility check for the SAIL Hackathon
### These methods should help finding feasible query points for the Challenge

def is_feasible(x1, x2, x3, x4, x5, x6, x7, x8):
    
    if not is_feasible0(x1):
        return False
    if not is_feasible1(x1, x2):
        return False
    if not is_feasible2(x1, x2, x3):
        return False
    if not is_feasible3(x1, x2, x4):
        return False
    if not is_feasible4(x1, x2, x5):
        return False
    if not is_feasible5(x1, x2, x6):
        return False
    if not is_feasible6(x1, x2, x7):
        return False
    if not is_feasible7(x1, x2, x8):
        return False    
    return True


# Feasibility function for Engine speed
# Simple unique lower and upper bound
def is_feasible0(x1):
    return 650 <= x1 <= 2250

# Feasibility function for Engine load
# Unique lower bound with piecewise linear upper bound dependent on Engine speed

def is_feasible1(x1, x2):
    if 650 <= x1 <= 2250:
        if x1 < 1200:
            y_upper = (x1 - 650) * 70 / 550 + 100
            return x2 <= y_upper
        elif x1 < 1600:
            return x2 <= 170
        else:
            y_upper = 170 + (x1 - 1600) * (-35 / 650)
            return x2 <= y_upper
    return False

### All functions from here on are dependent on the Engine speed and Engine load 
### and therefore have two subfunctions

# Feasibility function for Railpressure
def is_feasible2(x1, x2, x3):
    def speed_feasibility(x1, x3):
        if 650 <= x1 < 1400:
            y_upper = (x1 - 650) * 1200 / 1150 + 1500
            y_lower = 500
        elif 1400 <= x1 < 1800:
            y_lower = (x1 - 1400) * 500 / 850 + 500
            y_upper = (x1 - 650) * 1200 / 1150 + 1500
        elif 1800 <= x1 <= 2250:
            y_lower = (x1 - 1400) * 500 / 850 + 500
            y_upper = 2700
        else:
            return False
        return x3 >= y_lower and x3 <= y_upper

    def load_feasibility(x2, x3):
        if 0 <= x2 < 25:
            return x3 >= 500 and x3 <= 2700
        elif 25 <= x2 <= 175:
            y_upper = 2700
            y_lower = (x2 - 25) * 500 / 150 + 500
            return x3 <= y_upper and x3 >= y_lower
        return False

    return speed_feasibility(x1, x3) and load_feasibility(x2, x3)

# Feasibility function for Air supply
def is_feasible3(x1, x2, x4):
    def load_feasibility(x2, x4):
        if 0 <= x2 < 25:
            y_lower = x2 * 40 / 25 + 50
            return x4 >= y_lower and x4 <= 1400
        elif 25 <= x2 < 100:
            y_lower = (x2 - 25) * 90 / 75 + 90
            return x4 >= y_lower and x4 <= 1400
        elif 100 <= x2 <= 175:
            y_lower = (x2 - 100) * 370 / 75 + 180
            return x4 >= y_lower and x4 <= 1400
        return False

    def speed_feasibility(x1, x4):
        if 650 <= x1 <= 2250:
            y_upper = (x1 - 650) * 900 / 1600 + 500
            y_lower = (x1 - 650) * 150 / 1600 + 50
            return x4 <= y_upper and x4 >= y_lower
        return False

    return load_feasibility(x2, x4) and speed_feasibility(x1, x4)

# Feasibility function for crank angle
def is_feasible4(x1, x2, x5):
    def speed_feasibility(x1, x5):
        if 650 <= x1 < 1400:
            return -10 <= x5 <= 7.5
        elif 1400 <= x1 <= 2250:
            y_lower = -10
            y_upper = (x1 - 1400) * 7.5 / 850 + 7.5
            return y_lower <= x5 <= y_upper
        return False

    def load_feasibility(x2, x5):
        if 0 <= x2 < 25:
            y_upper = 12
            y_lower = (x2 - 0) * (-6) / 25 - 4
            return y_lower <= x5 <= y_upper
        elif 25 <= x2 < 150:
            return -10 <= x5 <= 12
        elif 150 <= x2 <= 175:
            y_upper = 12
            y_lower = (x2 - 150) * 10 / 25 - 10
            return y_lower <= x5 <= y_upper
        return False

    return speed_feasibility(x1, x5) and load_feasibility(x2, x5)

# Feasibility function for Intake pressure
def is_feasible5(x1, x2, x6):
    def speed_feasibility(x1, x6):
        if 650 <= x1 < 2250:
            y_upper = 3200
            y_lower = (x1 - 650) * 100 / 1600 + 900
            return x6 <= y_upper and x6 >= y_lower
        return False

    def load_feasibility(x2, x6):
        if 0 <= x2 < 75:
            y_lower = x2 * (1000 - 900) / (75) + 900
            return y_lower <= x6 <= 3000
        elif 75 <= x2 < 100:
            y_lower = (x2 - 75) * (1500 - 1000) / (125 - 75) + 1000
            return y_lower <= x6 <= 3000
        elif 100 <= x2 < 125:
            y_upper = (x2 - 100) * (3300 - 3000) / (125 - 100) + 3000
            y_lower = (x2 - 75) * (1500 - 1000) / (125 - 75) + 1000
            return y_lower <= x6 <= y_upper
        elif 125 <= x2 <= 175:
            y_upper = 3300
            y_lower = (x2 - 125) * (2500 - 1500) / (175 - 125) + 1500
            return x6 <= y_upper and x6 >= y_lower
        return False

    return speed_feasibility(x1, x6) and load_feasibility(x2, x6)

# Feasibility function for Back pressure
def is_feasible6(x1, x2, x7):
    def speed_feasibility(x1, x7):
        if 650 <= x1 <= 2250:
            y_upper = 4000
            y_lower = (x1 - 650) * (250) / (1600) + 900
            return x7 <= y_upper and x7 >= y_lower
        return False

    def load_feasibility(x2, x7):
        if 0 <= x2 < 75:
            y_lower = (x2) * (100) / (75) + 900
            y_upper = 4000
            return x7 <= y_upper and x7 >= y_lower
        elif 75 <= x2 < 175:
            y_lower = (x2 - 75) * (500) / (50) + 1000
            y_upper = 4000
            return x7 <= y_upper and x7 >= y_lower
        else:
            return False

        return x7 <= y_upper and x7 >= y_lower

    return speed_feasibility(x1, x7) and load_feasibility(x2, x7)

# Feasibility function for Intake temperature
def is_feasible7(x1, x2, x8):
    def speed_feasibility(x1, x8):
        if 650 <= x1 <= 2250:
            y_upper = (90 - 70)* (x1 - 650) / (2250 - 650) + 70
            y_lower = 38
            return y_lower <= x8 <= y_upper
        
        return False

    def load_feasibility(x2, x8):
        if 0 <= x2 < 75:
            y_lower = 38
            y_upper = (80 - 70) * (75 - x2) / 75 + 70
        elif 75 <= x2 < 110:
            y_lower = 38
            y_upper = (90 - 70) * (x2 - 75) / (110 - 75) + 70
        elif 110 <= x2 < 125:
            y_lower = 38
            y_upper = 90
        elif 125 <= x2 <= 150:
            y_lower = 38
            y_upper = (70 - 90) * (x2 - 125) / (175 - 125) + 90 
        elif 150 <= x2 <= 175:
            y_lower = (60 - 38) * (x2 - 150) / (175 - 150) + 38
            y_upper = (70 - 90) * (x2 - 125) / (175 - 125) + 90
        else:
            return False

        return x8 <= y_upper and x8 >= y_lower

    return speed_feasibility(x1, x8) and load_feasibility(x2, x8)
