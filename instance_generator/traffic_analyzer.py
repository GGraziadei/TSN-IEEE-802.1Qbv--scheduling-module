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
    df = df[df['traffic_troughput'] > 110]   

    interfaces = df['interface_id']
    traffic_troughput = df['traffic_troughput']
    network_troughput = df["network_troughput"]
    df["waste_troughput"] = df["network_troughput"] - df["traffic_troughput"]

    plt.plot(interfaces, network_troughput, alpha=0.5)
    plt.scatter(interfaces, traffic_troughput, marker='x')
    plt.plot(interfaces, df["waste_troughput"], marker='o')
    plt.ylabel("Troughput (Kbitps)")
    plt.xlabel("Interface_id")
    plt.yscale('log')
    plt.legend(["Network Troughput", "Signal Traffic Troughput", "Wasted Troughput"])
    
    # write all the tick to x
    
    plt.title("Troughput of interfaces with signal traffic > 110 Kbitps")

    # x is not a linspace
    plt.xticks(interfaces)

    plt.tight_layout()
    plt.savefig(f'{args.filename}.png')

    df.to_excel(f'{args.filename}.xlsx')