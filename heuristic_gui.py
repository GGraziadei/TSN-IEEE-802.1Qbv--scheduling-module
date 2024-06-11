import copy
from problem.GRASP_model import GRASP
from problem.flow import Flow
from problem.greedy_model import Greedy
import json

SOLVER = "greedy"

# instance_generator/factor_f/instance_11.json
with open('instance_generator/instances/instance_C1_lan/instance.json', 'r') as f:
    data = json.load(f)

    # constructive solution
    instance = Greedy(network_parameeter=data)
    instance.pre_processing()

    sol = instance.solve()

    if sol:
        j = max(instance.j_f.values())
        d = max(instance.d_fi.values())
        print("---",SOLVER,"---")
        print("Constructive solution with maximum jitter", j, "maximum delay", d)
        #instance.ls()

    
    if SOLVER == "grasp":
        instance = GRASP(network_parameeter=data)
        instance.set_alpha(0.1)
        instance.set_execution_time(60)
    
        
    if SOLVER == "grasp":

        if sol: 
            j_f = [j]
            d_fi = [d]
            x_feti = {}
        else:
            j_f = None
            d_fi = None
            x_feti = None

        instance.pre_processing()
        sol= instance.solve(d_fi=d_fi,j_f=j_f,x_feti=x_feti)
        if sol:
            j = max(j_f)
            d = max(d_fi)
            print("GRASP solution with maximum jitter", j, "maximum delay", d)
            exit() 
    
    if sol:

        print("---",SOLVER,"---")
        print("Constructive solution with maximum jitter", j, "maximum delay", d)

        #instance.export_results("heuristic_results_pre")
        #print(instance.stat())
        instance.new_request(Flow(-1,[1,2], 400, 1000, 400, 100))
        instance.new_request(Flow(-1,[1,2], 400, 1000, 400, 100))
        instance.new_request(Flow(-1,[1,2], 400, 1000, 400, 100))
        
        instance.generate_gantt()

        """
        # new request 2
        flow = Flow(-1,[1,3], 100, 256, 128, 200)
        instance.flow_prepare(flow)
        if instance.new_request(flow):
            instance.export_results("heuristic_results_post_2")
            

        
        # new request 3
        flow = Flow(-1,[1,2,3], 50, 256, 128, 800)
        instance.flow_prepare(flow)
        if instance.new_request(flow):
            instance.export_results("heuristic_results_post_3")
            

        # new request 4
        flow = Flow(-1,[1,2,3], 50, 256, 128, 1000)
        instance.flow_prepare(flow)
        if instance.new_request(flow):
            instance.export_results("heuristic_results_post_4")
            

        # new request 5
        flow = Flow(-1,[1,2,3], 50, 256, 128, 100)
        instance.flow_prepare(flow)
        if instance.new_request(flow):
            instance.export_results("heuristic_results_post_5")
            

        # new request 6
        flow = Flow(-1,[1,2,3], 50, 256, 128, 100)
        instance.flow_prepare(flow)
        if instance.new_request(flow):
            instance.export_results("heuristic_results_post_6")
            

        # new request 7
        flow = Flow(-1,[1,2,3], 50, 256, 128, 100)
        instance.flow_prepare(flow)
        if instance.new_request(flow):
            instance.export_results("heuristic_results_post_7")
            
        
        # new request 8
        flow = Flow(-1,[1,2,3], 50, 256, 128, 100)
        instance.flow_prepare(flow)
        if instance.new_request(flow):
            instance.export_results("heuristic_results_post_8")

        """

        #instance.generate_gantt()
        #print(instance.network.links_stats())

    else:
        print("No solution found")




