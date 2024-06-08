# test the factor f
from pulp import *
import json
from problem.milp_model import ILP
from problem.gurobi_ilp import GurobiILP

with open('tests/scenario8.json') as f:
    data = json.load(f)
    
    instance = GurobiILP(data)
    instance.pre_processing(debug=True)
    
    if instance.is_apriori_feasible():
        if instance.solve()["status"] != "Infeasible":
            
            #instance.post_processing()
            stat =  instance.stat()
            print(stat)
            

            instance.generate_gantt()
            
            """
            instance.export_results("results")
            if instance.is_valid():
                instance.store_data_plane("results")
                instance.generate_gantt()
            """
        del instance
    