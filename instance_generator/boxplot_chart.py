from argparse import ArgumentParser
from matplotlib import pyplot as plt

# data
parser = ArgumentParser()
parser.add_argument("-f", "--filename", dest="filename", help="Input filename.", required=True)
args = parser.parse_args()

if not args.filename:
    print("Filename is required")
    exit()

if not args.filename.endswith(".xlsx"):
    print("Filename must be a xlsx file")
    exit()

import pandas as pd

df = pd.read_excel(args.filename)

# plot
print(df)

# filter data by quantile
df = df[df['size'] < df['size'].quantile(0.95)]
df = df[df['size'] > df['size'].quantile(0.05)]

df.boxplot(column='size', by='category')
plt.title('Contiguos available time by section')

plt.axhline(y=0.8 * 960 / 10**3, color='r', linestyle='--')

plt.ylabel('Available time (us)')
plt.xlabel('Cycle section')

# x axis sequence
plt.xticks(rotation=45, ha='right')
# y sclae 0 60 
plt.tight_layout()

# remove title
plt.suptitle("")

plt.savefig(f'{args.filename}.png')
