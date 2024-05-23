from argparse import ArgumentParser
import csv

from matplotlib import pyplot as plt


parser = ArgumentParser()
parser.add_argument("-f", "--filename", dest="filename", help="Output filename.", required=True)

args = parser.parse_args()

with open(f'{args.filename}.csv', 'r') as f:
    from pandas import read_csv
    df = read_csv(f)
    summary = df.describe()
    print(summary)
    df["interface_id"] = df["interface"].astype(str)

    # reduce to the most used interface
    df = df[df['traffic_troughput'] > 100]   

    interfaces = df['interface_id']
    traffic_troughput = df['traffic_troughput']
    traffic_network = df["network_troughput"]
    traffic_wasted = df["waste_troughput"] 
    plt.scatter(interfaces, traffic_network, marker='x')
    plt.scatter(interfaces, traffic_troughput, marker='*')
    plt.plot(interfaces, traffic_wasted, marker='o')
    plt.ylabel("troughput (Kbitps)")
    plt.xlabel("Interface_id")
    plt.yscale('log')
    plt.legend(["Network Troughput", "Signal Traffic Troughput", "Wasted Troughput % "])
    
    # write all the tick to x
    
    plt.title("Troughput of interfaces with signal traffic > 100 Kbitps")

    # x is not a linspace
    plt.xticks(interfaces)

    # add a box in red for the first 3 interfaces
    plt.axvspan(-1, 3, color='green', alpha=0.2)
    plt.text(0, 1000, "Optical interfaces", fontsize=12, color='green')

    plt.show()