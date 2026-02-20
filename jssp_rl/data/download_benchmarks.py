"""download_benchmarks.py
======================
Parse the OR-Library format file (from thomasWeise/jsspInstancesAndResults)
and produce three pickle files for evaluation:
  - saved/benchmark_ft.pkl      (FT06, FT10, FT20)
  - saved/benchmark_taillard.pkl (Ta01-Ta80)
  - saved/benchmark_dmu.pkl     (DMU01-DMU80)

Each pickle is a list of dicts:
  {"name": str, "times": Tensor, "machines": Tensor,
   "num_jobs": int, "num_machines": int, "bks": int}

Also writes saved/benchmark_bks.json with all Best Known Solutions.

Usage:
  python data/download_benchmarks.py          # uses cached /tmp file
  python data/download_benchmarks.py --fetch  # re-downloads from GitHub
"""

import os
import sys
import json
import pickle
import re
import torch

# ================================================================
# Best Known Solutions (BKS) from thomasWeise repo (Jan 2026)
# ================================================================
BKS = {
    # Fisher-Thompson
    "ft06": 55, "ft10": 930, "ft20": 1165,
    # Taillard Ta01-Ta10 (15x15)
    "ta01": 1231, "ta02": 1244, "ta03": 1218, "ta04": 1175, "ta05": 1224,
    "ta06": 1238, "ta07": 1227, "ta08": 1217, "ta09": 1274, "ta10": 1241,
    # Taillard Ta11-Ta20 (20x15)
    "ta11": 1357, "ta12": 1367, "ta13": 1342, "ta14": 1345, "ta15": 1339,
    "ta16": 1360, "ta17": 1462, "ta18": 1396, "ta19": 1332, "ta20": 1348,
    # Taillard Ta21-Ta30 (20x20)
    "ta21": 1642, "ta22": 1600, "ta23": 1557, "ta24": 1644, "ta25": 1595,
    "ta26": 1643, "ta27": 1680, "ta28": 1603, "ta29": 1625, "ta30": 1584,
    # Taillard Ta31-Ta40 (30x15)
    "ta31": 1764, "ta32": 1784, "ta33": 1791, "ta34": 1829, "ta35": 2007,
    "ta36": 1819, "ta37": 1771, "ta38": 1673, "ta39": 1795, "ta40": 1669,
    # Taillard Ta41-Ta50 (30x20)
    "ta41": 2005, "ta42": 1937, "ta43": 1846, "ta44": 1979, "ta45": 2000,
    "ta46": 2004, "ta47": 1889, "ta48": 1941, "ta49": 1961, "ta50": 1923,
    # Taillard Ta51-Ta60 (50x15)
    "ta51": 2760, "ta52": 2756, "ta53": 2717, "ta54": 2839, "ta55": 2679,
    "ta56": 2781, "ta57": 2943, "ta58": 2885, "ta59": 2655, "ta60": 2723,
    # Taillard Ta61-Ta70 (50x20)
    "ta61": 2868, "ta62": 2869, "ta63": 2755, "ta64": 2702, "ta65": 2725,
    "ta66": 2845, "ta67": 2825, "ta68": 2784, "ta69": 3071, "ta70": 2995,
    # Taillard Ta71-Ta80 (100x20)
    "ta71": 5464, "ta72": 5181, "ta73": 5568, "ta74": 5339, "ta75": 5392,
    "ta76": 5342, "ta77": 5436, "ta78": 5394, "ta79": 5358, "ta80": 5183,
    # DMU01-DMU05 (20x15)
    "dmu01": 2563, "dmu02": 2706, "dmu03": 2731, "dmu04": 2669, "dmu05": 2749,
    # DMU06-DMU10 (20x20)
    "dmu06": 3244, "dmu07": 3046, "dmu08": 3188, "dmu09": 3092, "dmu10": 2984,
    # DMU11-DMU15 (30x15)
    "dmu11": 3430, "dmu12": 3492, "dmu13": 3681, "dmu14": 3394, "dmu15": 3343,
    # DMU16-DMU20 (30x20)
    "dmu16": 3751, "dmu17": 3814, "dmu18": 3844, "dmu19": 3765, "dmu20": 3710,
    # DMU21-DMU25 (40x15)
    "dmu21": 4380, "dmu22": 4725, "dmu23": 4668, "dmu24": 4648, "dmu25": 4164,
    # DMU26-DMU30 (40x20)
    "dmu26": 4647, "dmu27": 4848, "dmu28": 4692, "dmu29": 4691, "dmu30": 4732,
    # DMU31-DMU35 (50x15)
    "dmu31": 5640, "dmu32": 5927, "dmu33": 5728, "dmu34": 5385, "dmu35": 5635,
    # DMU36-DMU40 (50x20)
    "dmu36": 5621, "dmu37": 5851, "dmu38": 5713, "dmu39": 5747, "dmu40": 5577,
    # DMU41-DMU45 (20x15, harder)
    "dmu41": 3248, "dmu42": 3390, "dmu43": 3441, "dmu44": 3475, "dmu45": 3272,
    # DMU46-DMU50 (20x20, harder)
    "dmu46": 4035, "dmu47": 3939, "dmu48": 3763, "dmu49": 3710, "dmu50": 3729,
    # DMU51-DMU55 (30x15, harder)
    "dmu51": 4156, "dmu52": 4311, "dmu53": 4390, "dmu54": 4362, "dmu55": 4270,
    # DMU56-DMU60 (30x20, harder)
    "dmu56": 4941, "dmu57": 4663, "dmu58": 4708, "dmu59": 4619, "dmu60": 4739,
    # DMU61-DMU65 (40x15, harder)
    "dmu61": 5172, "dmu62": 5251, "dmu63": 5323, "dmu64": 5240, "dmu65": 5190,
    # DMU66-DMU70 (40x20, harder)
    "dmu66": 5717, "dmu67": 5779, "dmu68": 5765, "dmu69": 5709, "dmu70": 5889,
    # DMU71-DMU75 (50x15, harder)
    "dmu71": 6223, "dmu72": 6463, "dmu73": 6153, "dmu74": 6196, "dmu75": 6189,
    # DMU76-DMU80 (50x20, harder)
    "dmu76": 6807, "dmu77": 6792, "dmu78": 6770, "dmu79": 6952, "dmu80": 6673,
}


def parse_or_library_file(filepath):
    """Parse the combined OR-Library format file into a list of instances.
    
    Format:
      instance <name>
      ...description lines...
      <num_jobs> <num_machines>
      <machine_id> <time> <machine_id> <time> ...  (one line per job)
    """
    with open(filepath, 'r') as f:
        content = f.read()

    # Split by "instance " markers
    blocks = re.split(r'\n\s*instance\s+', content)
    
    instances = {}
    for block in blocks[1:]:  # skip first (header)
        lines = block.strip().split('\n')
        name = lines[0].strip()
        
        # Find the line with "num_jobs num_machines"
        data_start = -1
        num_jobs = num_machines = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped and not stripped.startswith('+') and not stripped.startswith('lower') and not stripped.startswith('Fisher') and not stripped.startswith('Taillard') and not stripped.startswith('from ') and not stripped.startswith('Demirkol') and not stripped.startswith('Adams') and not stripped.startswith('Lawrence') and not stripped.startswith('Applegate') and not stripped.startswith('Storer') and not stripped.startswith('Yamada'):
                parts = stripped.split()
                if len(parts) == 2:
                    try:
                        nj, nm = int(parts[0]), int(parts[1])
                        if 2 <= nj <= 500 and 2 <= nm <= 500:
                            num_jobs, num_machines = nj, nm
                            data_start = i + 1
                            break
                    except ValueError:
                        continue
        
        if data_start < 0:
            continue
        
        # Parse job data: machine_id time machine_id time ...
        times = []
        machines = []
        job_count = 0
        for i in range(data_start, len(lines)):
            stripped = lines[i].strip()
            if not stripped or stripped.startswith('+'):
                continue
            parts = stripped.split()
            if len(parts) < 2 * num_machines:
                continue
            
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
            if job_count >= num_jobs:
                break
        
        if job_count == num_jobs:
            instances[name] = {
                "name": name,
                "times": torch.tensor(times, dtype=torch.float32),
                "machines": torch.tensor(machines, dtype=torch.long),
                "num_jobs": num_jobs,
                "num_machines": num_machines,
                "bks": BKS.get(name, -1),
            }
        else:
            print(f"[Warning] Incomplete instance {name}: got {job_count}/{num_jobs} jobs")
    
    return instances


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    saved_dir = os.path.join(base_dir, "saved")
    os.makedirs(saved_dir, exist_ok=True)
    
    # Determine source file
    cached = "/tmp/jssp_instance_data.txt"
    remote_url = "https://raw.githubusercontent.com/thomasWeise/jsspInstancesAndResults/master/data-raw/instance-data/instance_data.txt"
    
    if "--fetch" in sys.argv or not os.path.exists(cached):
        print(f"[Fetch] Downloading from GitHub...")
        import urllib.request
        urllib.request.urlretrieve(remote_url, cached)
        print(f"[OK] Downloaded to {cached}")
    else:
        print(f"[Cache] Using cached file: {cached}")
    
    # Parse all instances
    print("[Parse] Parsing OR-Library format...")
    all_instances = parse_or_library_file(cached)
    print(f"[OK] Parsed {len(all_instances)} instances total")
    
    # Split into benchmark sets
    ft_instances = []
    ta_instances = []
    dmu_instances = []
    
    for name, inst in sorted(all_instances.items()):
        if name.startswith("ft"):
            ft_instances.append(inst)
        elif name.startswith("ta"):
            ta_instances.append(inst)
        elif name.startswith("dmu"):
            dmu_instances.append(inst)
    
    # Sort by instance number
    def sort_key(inst):
        num = re.findall(r'\d+', inst["name"])
        return int(num[0]) if num else 0
    
    ft_instances.sort(key=sort_key)
    ta_instances.sort(key=sort_key)
    dmu_instances.sort(key=sort_key)
    
    # Save pickle files
    def save_pkl(instances, filename):
        path = os.path.join(saved_dir, filename)
        with open(path, "wb") as f:
            pickle.dump(instances, f)
        print(f"  -> {filename}: {len(instances)} instances")
        # Print summary by size
        sizes = {}
        for inst in instances:
            s = f"{inst['num_jobs']}x{inst['num_machines']}"
            sizes[s] = sizes.get(s, 0) + 1
        for s, c in sorted(sizes.items()):
            print(f"       {s}: {c}")
    
    print(f"\n[Save] Writing benchmark pickle files to {saved_dir}/")
    save_pkl(ft_instances, "benchmark_ft.pkl")
    save_pkl(ta_instances, "benchmark_taillard.pkl")
    save_pkl(dmu_instances, "benchmark_dmu.pkl")
    
    # Also save the combined BKS as JSON for easy access from other scripts
    bks_path = os.path.join(saved_dir, "benchmark_bks.json")
    with open(bks_path, "w") as f:
        json.dump(BKS, f, indent=2)
    print(f"  -> benchmark_bks.json: {len(BKS)} entries")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"BENCHMARK SUMMARY")
    print(f"{'='*60}")
    print(f"  Fisher-Thompson (FT):   {len(ft_instances)} instances")
    print(f"  Taillard (Ta01-80):     {len(ta_instances)} instances")
    print(f"  Demirkol (DMU01-80):    {len(dmu_instances)} instances")
    print(f"  TOTAL:                  {len(ft_instances) + len(ta_instances) + len(dmu_instances)} instances")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
