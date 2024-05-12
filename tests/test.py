from math import ceil
import pickle
import random
from pulp import *
import pandas as pd
import sys
import datetime

path_to_cplex = r'C:\Program Files\IBM\ILOG\CPLEX_Studio2211\cplex\bin\x64_win64\cplex.exe'
solver = CPLEX_CMD(path=path_to_cplex)

def now():
        return datetime.datetime.now()
    
def format_delta(begin, end):
    delta = end - begin
    return f"{delta.seconds}.{delta.microseconds//1000} seconds"

class Timer:
    
    def __enter__(self):
        self.begin = now()

    def __exit__(self, type, value, traceback):
        print("Loding time", format_delta(self.begin, now()))

def grouped(iterable, n):
    return [(iterable[i], iterable[i+1]) for i in range(len(iterable)-1)]

def gcd(p, q):
    #  Euclid's algorithm to find the GCD.
    while q != 0:
        p, q = q, p % q
    return p

def is_coprime(x, y):
    # Check if the GCD of 'x' and 'y' is equal to 1.
    return gcd(x, y) == 1

file_name = sys.argv[1]

with Timer() as loading_time: 

    with open(file_name + ".json") as f:
        input_data = json.load(f)
        f.close()

    '''
    Optimize the pivoting with the incumbent not necessary in test phase

    try:   
        with open('scheduling_results.pkl', 'rb') as f:  # open a text file
            y_feti = pickle.load(f) # serialize the list
    except:
        y_feti = {}
    '''
    y_feti = {}

    #parameters input
    E = input_data["E"] # 9 directed link
    print("E: ", E)
    F = input_data["F"] # 6 flows
    print("F: ", F)
    SF_duration = input_data["SF_duration"] #us about 1ms
    print("SF_duration: ", SF_duration)
    tau = input_data["tau"] #bit 64Kbit = 8 KB
    print("tau: ", tau)
    E_f = input_data["E_f"]
    print("E_f: ", E_f)
    P_f = input_data["P_f"]
    print("P_f: ", P_f)
    # delay and jitter constraints
    DELAY_f = input_data["DELAY_f"]
    print("DELAY_f: ", DELAY_f)
    JITTER_f = input_data["JITTER_f"]
    print("JITTER_f: ", JITTER_f)
    B_e = input_data["B_e"]
    print("B_e: ", B_e)

    # pre-proccessing
    T_f = [ ceil(SF_duration/P_f[f-1]) for f in F]
    print("T_f: ", T_f)
    tau_e = [tau/B_e[e-1] for e in E]
    print("tau_e: ", tau_e)
    tau_min = min(tau_e)
    print("tau_min: ", tau_min)
    t_max = int(2 * SF_duration/tau_min) # doubled the SF
    print("t_max: ", t_max)

    SF = range(1,t_max+1)

    T_e = [ [t for t in range(1,t_max+1,int(tau_e[e-1]/tau_min))] 
            for e in E ]

    max_tf = max(T_f)
    I = range(1,max_tf+1)

    # check a-priori infeasibility
    timeslots = sum([len(T_e[e-1]) for e in E])/2
    required_timeslots = sum([len(E_f[f-1]) for f in F for _ in range(1,T_f[f-1] + 1)])
    print("required_timeslots: ", required_timeslots)
    print("timeslots: ", timeslots)

    for b1,b2 in grouped(B_e,2):
        if is_coprime(b1,b2):
            print(f"Links are co-prime in interface speed")

    for f in F:
        # check co-prime
        if is_coprime(P_f[f-1], SF_duration):
            print(f"Flow {f} is infeasible, P_f and SF_duration are co-prime")
            exit(1)

        if P_f[f-1] > SF_duration:
            print(f"Flow {f} is infeasible, P_f is greater than SF_duration")
            exit(1)
        if P_f[f-1] < tau_min:
            print(f"Flow {f} is infeasible, P_f is less than tau_min")
            exit(1)

    if timeslots < required_timeslots:
        print("infeasible, not enough timeslots for all flows")
        exit(1)

    prob = LpProblem("Scheduling Problem", LpMinimize)

    # Decision variables
    # space reduction of 4D array
    feti_opt_space = [(f,e,t,i) for f in F
                                for e in E_f[f-1]
                                    for t in T_e[e-1]
                                        for i in range(1,T_f[f-1] + 1)]
    fi_opt_space = [(f,i) for f in F for i in range(1,T_f[f-1] + 1)]
    et_opt_space = [(e,t) for e in E for t in T_e[e-1]]

    x_feti = LpVariable.dicts("x_feti", feti_opt_space, cat="Binary")
    c_et   = LpVariable.dicts("c_et", et_opt_space, cat="Binary")
    d_fi   = LpVariable.dicts("d_fi", fi_opt_space, cat="Continuous")
    d      = LpVariable("d", cat="Continuous")
    j_f    = LpVariable.dicts("j_f", (F), cat="Continuous")
    #j      = LpVariable("j", cat="Continuous")

    # start from the incumbent configuration
    for (f,e,t,i) in y_feti.keys():
        if f in F and e in E_f[f-1] and t in T_e[e-1] and i in range(1,T_f[f-1] + 1):
            x_feti[f,e,t,i].setInitialValue(y_feti[f,e,t,i])

    for f in F:
        for e in E_f[f-1]:
            for i in range(1,T_f[f-1] + 1):
                #constraint 4
                prob += lpSum([x_feti[f, e, t, i] for t in T_e[e-1]]) <= 1
                    
        for i in range(1,T_f[f-1] + 1 ):

            for e in E_f[f-1]:
                # constraint 13 
                prob += lpSum([x_feti[f, e, t, i] for t in T_e[e-1]]) == 1

            for e1,e2 in grouped(E_f[f-1],2):
                for t_start in T_e[e1-1]:
                    # constraint 11
                    prob += lpSum([x_feti[f,e2,t,i] for t in T_e[e2-1] if t >= (t_start + int(tau_e[e1-1]/tau_min)) ]) >= x_feti[f,e1,t_start,i]
            
            # constraint 6
            e_dest = E_f[f-1][-1]
            prob += d_fi[f,i] + (P_f[f-1] * (i-1)) == lpSum((t-1) * tau_min * x_feti[f,e_dest,t,i] for t in T_e[e_dest-1]) 
            
            for t in SF: 
                # constraint 5
                prob += lpSum([x_feti[f,e,t,i] for e in E_f[f-1] if t in T_e[e-1]]) <= 1
                
            for i2 in range(1,T_f[f-1] + 1 ):
                # constraint 7
                i1 = i 
                if d_fi[f,i1] >= d_fi[f,i2]:
                    prob += j_f[f] >= d_fi[f,i1] - d_fi[f,i2]
            # constraint 8
            prob += d_fi[f,i] <=  DELAY_f[f-1]

        # constraint 9
        prob += j_f[f] <= JITTER_f[f-1]

        # contraint 12
        #prob += j >= j_f[f]

    for e in E:
        for t in T_e[e-1]:
            # constraint 3
            prob += lpSum([x_feti[f,e,t,i] for f in F for i in range(1,T_f[f-1] + 1) if e in E_f[f-1]]) <= 1

    for f in F:
        e_src = E_f[f-1][0]
        if T_f[f-1] > 1:
        
            for i in range(1,T_f[f-1] + 1):
                # constraint 16
                prob += lpSum([tau_min * t * x_feti[f,e_src,t,i] for t in T_e[e_src-1]]) >= P_f[f-1] * (i-1)
            
            for i1,i2 in grouped(range(1,T_f[f-1] + 1),2):
                for t1 in T_e[e_src-1]:
                    # constraint 14
                    T2 = [t for t in T_e[e_src-1] if t > t1]
                    prob += lpSum([x_feti[f,e_src,t2,i2] for t2 in T2]) >= x_feti[f,e_src,t1,i1]

    for f in F:
        for i in range(1,T_f[f-1] + 1):
            # constraint 15
            prob += d_fi[f,i] <= d

    for e in E:
        for t in T_e[e-1]:
            if t > int(0.5 * t_max):
                # constraint 17
                prob += lpSum([x_feti[f,e,t,i] + x_feti[f,e,t-int(0.5 * t_max),i] for f in F for i in range(1,T_f[f-1] + 1) if e in E_f[f-1]]) <= 1

    for (f,e,t,i) in y_feti.keys():
        if f in F and e in E_f[f-1] and t in T_e[e-1] and i in range(1,T_f[f-1] + 1):
            prob += c_et[e,t] + y_feti[f,e,t,i] >= x_feti[f,e,t,i] 
            prob +=  x_feti[f,e,t,i] + y_feti[f,e,t,i] + c_et[e,t]  <= 2
            prob +=  x_feti[f,e,t,i] + y_feti[f,e,t,i] >= c_et[e,t] 
            prob += c_et[e,t] + x_feti[f,e,t,i] >= y_feti[f,e,t,i] 
                                
    # The objective function is added: minimize the number of changes and the maximum delay
    prob += d + lpSum(c_et[e,t] for e in E for t in T_e[e-1])

prob.solve(solver)

if LpStatus[prob.status] == "Infeasible":
    print("Infeasible")
    exit(1)

print("Status:", LpStatus[prob.status])

#check the configuration 
for f in F:
    iteration_time = {}
    for i in range(1,T_f[f-1] + 1):
        path_l = []
        for e in E_f[f-1]:
            for t in T_e[e-1]:
                if value(x_feti[f,e,t,i]) == 1:
                    path_l.append((e,t))

        print(f"Flow {f}_{i} path: ", path_l)
        # all scheduled
        if len(path_l) != len(E_f[f-1]):
            print(f"Flow {f}_{i} is not scheduled")
            exit(1)
        
        e_src,t0 = path_l.pop(0)
        t_start = t0
        if e_src != E_f[f-1][0]:
            print(f"Flow {f}_{i} is not started at the source interface")
            exit(1)
        
        while path_l:
            e,t = path_l.pop(0)
            if t - t0 < int(tau_e[e_src-1]//tau_min):
                print(f"Flow {f}_{i} is not scheduled with the minimum transmission delay interval")
                exit(1)
            e_src,t0 = e,t
        t_end = t0

        if e_src !=  E_f[f-1][-1]:
            print(f"Flow {f}_{i} is not ended at the destination interface")
            exit(1)

        if t_end < t_start : 
            print(f"Flow {f}_{i} is not scheduled with the correct transmission delay")
            exit(1)

        if int(((t_end-1)  * tau_min) - (P_f[f-1] * (i-1))) != int(value(d_fi[f,i])):
            print(f"Flow {f}_{i} is not scheduled with the correct evaluated transmission delay, {value(d_fi[f,i])} != {int(((t_end-1)  * tau_min) - (P_f[f-1] * (i-1)))}")
            exit(1)

        iteration_time[i] = (t_start,t_end)
        
        if value(d_fi[f,i]) > DELAY_f[f-1]:
            print(f"Flow {f}_{i} is not scheduled with the maximum delay constraint")
            exit(1)

        print(f"Flow {f}_{i} check passed")

    if value(j_f[f]) > JITTER_f[f-1]:
        print(f"Flow {f} is not scheduled with the jitter constraint")
        exit(1)

    s,e = 0,0
    
    for s_i,e_i in iteration_time.values():
        
        if s <= s_i: s = s_i
        else:
            print(f"Flow {f} is not scheduled with the correct time precedence")
            exit(1)

        if e <= e_i: e = e_i
        else:
            print(f"Flow {f} is not scheduled with the correct time precedence")
            exit(1)

        if e_i > int(0.5 * t_max) and s_i < int(0.5 * t_max):
            t_correspondent = e_i - int(0.5 * t_max)
            # check the circular correspondent
            if value(x_feti[f,E_f[f-1][-1],t_correspondent,i]) != 0:
                print(f"Circular correspondent for flow {f} uncorrect")
                exit(1)
    
    print(f"Flow {f} check passed")

# post-processing
# random color list
def hex_random_color():
    r = lambda: random.randint(0,255)
    return format('%02X%02X%02X' % (r(),r(),r()))

colors = {}

for f in F:
    for i in range(1,T_f[f-1] + 1):
        colors[f"{f}_{i}'"] = hex_random_color()

def cell_style(s):
    print(s)
    if s not in colors.keys(): return None
    color = colors[s]
    return f"color: {color};" 

df = pd.DataFrame(columns=SF, index=E)
for f in F:
    for i in range(1,T_f[f-1] + 1):
        for e in E_f[f-1]:
            for t in T_e[e-1]:
                if value(x_feti[f,e,t,i]) == 1:
                    for j in range(0,int(tau_e[e-1]//tau_min)):
                        df.loc[e,(t+j)] = f"{f}_{i}"

df.style.apply(cell_style)


df_opt = pd.DataFrame(columns=range(1,int(t_max*0.5) +1), index=E)
for f in F:
    for e in E_f[f-1]:
        for t in T_e[e-1]:    
            for i in range(1,T_f[f-1] + 1):
                if value(x_feti[f,e,t,i]) == 1:

                    # post-processing shifting
                    if t > int(0.5 * t_max): 
                        t -= int(0.5 * t_max)

                    for j in range(0,int(tau_e[e-1]//tau_min)):
                        df_opt.loc[e,(t+j)] = f"{f}_{i}"        

df_kpi = pd.DataFrame(columns=I, index=F)

for f in F:
    variances = []
    for i in range(1,T_f[f-1] + 1):
        df_kpi.loc[f,i] = value(d_fi[f,i])
        variances.append(value(d_fi[f,i]))
    df_kpi.loc[f,max(I)+1] = max(variances) - min(variances)

# save the variable x_feti
y_feti = {}
for f in F:
    for e in E_f[f-1]:
        for t in T_e[e-1]:    
            for i in range(1,T_f[f-1] + 1):
                y_feti[f,e,t,i] = int(value(x_feti[f,e,t,i]))

with open(file_name+'_results.pkl', 'wb') as f:  # open a text file
    pickle.dump(y_feti, f) # serialize the list

with pd.ExcelWriter( file_name + ".xlsx") as writer:
    df.to_excel(writer, sheet_name="Scheduling")
    df_opt.to_excel(writer, sheet_name="SchedulingOptimized")
    df_kpi.to_excel(writer, sheet_name="KPI")

print("Optimization completed")