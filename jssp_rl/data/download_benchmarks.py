"""download_benchmarks.py
======================
Parse the OR-Library format file (from thomasWeise/jsspInstancesAndResults)
and produce pickle files for evaluation:
  - benchmark_ft.pkl (mt06, mt10, mt20)
  - benchmark_taillard.pkl (ta01-ta80)
  - benchmark_dmu.pkl (dmu01-dmu80)
  - benchmark_abz.pkl (abz5-abz9)
  - benchmark_la.pkl (la01-la40)
  - benchmark_orb.pkl (orb01-orb10)
  - benchmark_swv.pkl (swv01-swv20)
  - benchmark_yn.pkl (yn1-yn4)

Each pickle is a list of dicts with name, times, machines, jobs, machines, bks.
"""

import os
import sys
import json
import pickle
import re
import torch

# ================================================================
# Best Known Solutions (BKS) gathered from thomasWeise repo
# ================================================================
BKS = {
    # Fisher-Thompson / MT
    "ft06": 55, "mt06": 55, "ft10": 930, "mt10": 930, "ft20": 1165, "mt20": 1165,
    # Taillard (15x15 to 100x20)
    "ta01": 1231, "ta02": 1244, "ta03": 1218, "ta04": 1175, "ta05": 1224,
    "ta06": 1238, "ta07": 1227, "ta08": 1217, "ta09": 1274, "ta10": 1241,
    "ta11": 1357, "ta12": 1367, "ta13": 1342, "ta14": 1345, "ta15": 1339,
    "ta16": 1360, "ta17": 1462, "ta18": 1396, "ta19": 1332, "ta20": 1348,
    "ta21": 1642, "ta22": 1600, "ta23": 1557, "ta24": 1644, "ta25": 1595,
    "ta26": 1643, "ta27": 1680, "ta28": 1603, "ta29": 1625, "ta30": 1584,
    "ta31": 1764, "ta32": 1784, "ta33": 1791, "ta34": 1829, "ta35": 2007,
    "ta36": 1819, "ta37": 1771, "ta38": 1673, "ta39": 1795, "ta40": 1669,
    "ta41": 2005, "ta42": 1937, "ta43": 1846, "ta44": 1979, "ta45": 2000,
    "ta46": 2004, "ta47": 1889, "ta48": 1941, "ta49": 1961, "ta50": 1923,
    "ta51": 2760, "ta52": 2756, "ta53": 2717, "ta54": 2839, "ta55": 2679,
    "ta56": 2781, "ta57": 2943, "ta58": 2885, "ta59": 2655, "ta60": 2723,
    "ta61": 2868, "ta62": 2869, "ta63": 2755, "ta64": 2702, "ta65": 2725,
    "ta66": 2845, "ta67": 2825, "ta68": 2784, "ta69": 3071, "ta70": 2995,
    "ta71": 5464, "ta72": 5181, "ta73": 5568, "ta74": 5339, "ta75": 5392,
    "ta76": 5342, "ta77": 5436, "ta78": 5394, "ta79": 5358, "ta80": 5183,
    # DMU / Mehta & Uzsoy
    "dmu01": 2563, "dmu02": 2706, "dmu03": 2731, "dmu04": 2669, "dmu05": 2749,
    "dmu06": 3244, "dmu07": 3046, "dmu08": 3188, "dmu09": 3092, "dmu10": 2984,
    "dmu11": 3430, "dmu12": 3492, "dmu13": 3681, "dmu14": 3394, "dmu15": 3343,
    "dmu16": 3751, "dmu17": 3814, "dmu18": 3844, "dmu19": 3765, "dmu20": 3710,
    "dmu21": 4380, "dmu22": 4725, "dmu23": 4668, "dmu24": 4648, "dmu25": 4164,
    "dmu26": 4647, "dmu27": 4848, "dmu28": 4692, "dmu29": 4691, "dmu30": 4732,
    "dmu31": 5640, "dmu32": 5927, "dmu33": 5728, "dmu34": 5385, "dmu35": 5635,
    "dmu36": 5621, "dmu37": 5851, "dmu38": 5713, "dmu39": 5747, "dmu40": 5577,
    "dmu41": 3248, "dmu42": 3390, "dmu43": 3441, "dmu44": 3475, "dmu45": 3272,
    "dmu46": 4035, "dmu47": 3939, "dmu48": 3763, "dmu49": 3710, "dmu50": 3729,
    "dmu51": 4156, "dmu52": 4311, "dmu53": 4390, "dmu54": 4362, "dmu55": 4270,
    "dmu56": 4941, "dmu57": 4663, "dmu58": 4708, "dmu59": 4619, "dmu60": 4739,
    "dmu61": 5172, "dmu62": 5251, "dmu63": 5323, "dmu64": 5240, "dmu65": 5190,
    "dmu66": 5717, "dmu67": 5779, "dmu68": 5765, "dmu69": 5709, "dmu70": 5889,
    "dmu71": 6223, "dmu72": 6463, "dmu73": 6153, "dmu74": 6196, "dmu75": 6189,
    "dmu76": 6807, "dmu77": 6792, "dmu78": 6770, "dmu79": 6952, "dmu80": 6673,
    # Adams, Balas, Zawack (abz)
    "abz5": 1234, "abz6": 943, "abz7": 656, "abz8": 648, "abz9": 678,
    # Lawrence (la01-40)
    "la01": 666, "la02": 655, "la03": 597, "la04": 590, "la05": 593,
    "la06": 926, "la07": 890, "la08": 863, "la09": 951, "la10": 958,
    "la11": 1222, "la12": 1039, "la13": 1150, "la14": 1292, "la15": 1207,
    "la16": 945, "la17": 784, "la18": 848, "la19": 842, "la20": 902,
    "la21": 1046, "la22": 927, "la23": 1032, "la24": 935, "la25": 977,
    "la26": 1218, "la27": 1235, "la28": 1216, "la29": 1152, "la30": 1355,
    "la31": 1784, "la32": 1850, "la33": 1719, "la34": 1721, "la35": 1888,
    "la36": 1268, "la37": 1397, "la38": 1196, "la39": 1233, "la40": 1222,
    # Applegate & Cook (orb)
    "orb01": 1059, "orb02": 888, "orb03": 1005, "orb04": 1005, "orb05": 887,
    "orb06": 1010, "orb07": 397, "orb08": 899, "orb09": 934, "orb10": 944,
    # Storer, Wu, Vaccari (swv)
    "swv01": 1407, "swv02": 1475, "swv03": 1398, "swv04": 1464, "swv05": 1424,
    "swv06": 1630, "swv07": 1513, "swv08": 1671, "swv09": 1633, "swv10": 1663,
    "swv11": 2983, "swv12": 2972, "swv13": 3104, "swv14": 2968, "swv15": 2885,
    "swv16": 2924, "swv17": 2794, "swv18": 2852, "swv19": 2843, "swv20": 2823,
    # Yamada & Nakano (yn)
    "yn1": 884, "yn2": 870, "yn3": 859, "yn4": 929,
}


def parse_or_library_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    blocks = re.split(r'\n\s*instance\s+', content)
    instances = {}
    for block in blocks[1:]:
        lines = block.strip().split('\n')
        name = lines[0].strip().lower()
        
        data_start = -1
        num_jobs = num_machines = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if not stripped or stripped.startswith('+'): continue
            if any(p in stripped for p in ["Fisher", "Taillard", "Demirkol", "Adams", "Lawrence", "Applegate", "Storer", "Yamada", "from"]): continue
            
            parts = stripped.split()
            if len(parts) == 2:
                try:
                    nj, nm = int(parts[0]), int(parts[1])
                    if 2 <= nj <= 500 and 2 <= nm <= 500:
                        num_jobs, num_machines = nj, nm
                        data_start = i + 1
                        break
                except ValueError: continue
        
        if data_start < 0: continue
        
        times = []
        machines = []
        job_count = 0
        for i in range(data_start, len(lines)):
            stripped = lines[i].strip()
            if not stripped or stripped.startswith('+'): continue
            parts = stripped.split()
            if len(parts) < 2 * num_machines: continue
            
            job_times = []
            job_machines = []
            for k in range(num_machines):
                m = int(parts[2 * k])
                t = int(parts[2 * k + 1])
                job_machines.append(m)
                job_times.append(float(t))
            
            times.append(job_times)
            machines.append(job_machines)
            job_count += 1
            if job_count >= num_jobs: break
        
        if job_count == num_jobs:
            instances[name] = {
                "name": name,
                "times": torch.tensor(times, dtype=torch.float32),
                "machines": torch.tensor(machines, dtype=torch.long),
                "num_jobs": num_jobs,
                "num_machines": num_machines,
                "bks": BKS.get(name, -1),
            }
    return instances


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    saved_dir = os.path.join(base_dir, "saved")
    os.makedirs(saved_dir, exist_ok=True)
    
    cached = "/tmp/jssp_instance_data.txt"
    remote_url = "https://raw.githubusercontent.com/thomasWeise/jsspInstancesAndResults/master/data-raw/instance-data/instance_data.txt"
    
    if "--fetch" in sys.argv or not os.path.exists(cached):
        import urllib.request
        urllib.request.urlretrieve(remote_url, cached)
    
    all_instances = parse_or_library_file(cached)
    print(f"[OK] Parsed {len(all_instances)} instances")
    
    # Organize into sets
    instance_sets = {
        "benchmark_ft.pkl": ["ft", "mt"],
        "benchmark_taillard.pkl": ["ta"],
        "benchmark_dmu.pkl": ["dmu"],
        "benchmark_abz.pkl": ["abz"],
        "benchmark_la.pkl": ["la"],
        "benchmark_orb.pkl": ["orb"],
        "benchmark_swv.pkl": ["swv"],
        "benchmark_yn.pkl": ["yn"],
    }
    
    for filename, prefixes in instance_sets.items():
        subset = []
        for name, inst in sorted(all_instances.items()):
            if any(name.startswith(p) for p in prefixes):
                subset.append(inst)
        
        if not subset:
            continue
            
        def sort_key(inst):
            num = re.findall(r'\d+', inst["name"])
            return int(num[0]) if num else 0
        subset.sort(key=sort_key)
        
        path = os.path.join(saved_dir, filename)
        with open(path, "wb") as f:
            pickle.dump(subset, f)
        print(f"  -> {filename}: {len(subset)} instances")
    
    bks_path = os.path.join(saved_dir, "benchmark_bks.json")
    with open(bks_path, "w") as f:
        json.dump(BKS, f, indent=2)
    print(f"  -> benchmark_bks.json: {len(BKS)} entries")


if __name__ == "__main__":
    main()
