
data = []
c = 0
with open("out.txt", "r" ) as f:
    for i in f.readlines():
        # from string extract data
        # remove space
        i = i.strip()
        # Split the string by '--' to extract different parts
        parts = i.split("--")

        # Extracting Flow ID, Flow Start, and Flow End
        if len(parts) != 4:
            continue
        flow_id = parts[0].strip()  # Extracting "1_282" from "Flow 1_282"
        jitter = int(parts[1].strip())  # Converting "12" to an integer
        delay = int(parts[2].strip())  # Converting "91" to an integer
        c +=1
        data.append({
            "flow_id": c,
            "jitter": jitter,
            "delay": delay
        })
        
# draw a grpah with jitte rand delay
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
delays = [d["delay"] for d in data]
jitters = [d["jitter"] for d in data]
n = [d["flow_id"] for d in data]
ax.scatter(n, delays, label="delay")
ax.scatter(n, jitters, label="jitter")
ax.legend()
plt.show()
