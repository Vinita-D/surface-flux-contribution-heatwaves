#import os
#from pathlib import Path
#import ic_rec_from_particle_output as fs


#FLEXPART_DIR = Path(os.environ["flexpart_dir"])

#pi_path = next(FLEXPART_DIR.glob("partinit_*.nc"))
#po_path = next(FLEXPART_DIR.glob("partoutput_*_init.nc"))

#csv_path = fs.run_rec_init_to_csv(pi_path, po_path)
#print("dataframe file saved in", csv_path)


import os
from pathlib import Path
import ic_rec_from_particle_output2 as calc


def main():
    flexpart_dir = Path(os.environ["flexpart_dir"])

    partinit_file = next(flexpart_dir.glob("partinit_*.nc"))

    partoutput_files = list(flexpart_dir.glob("partoutput_*_init.nc"))
    partoutput_file = partoutput_files[0] if partoutput_files else None

    n_workers = int(os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 1)

    calc.run_to_csv(
        pi_path=partinit_file,
        po_path=partoutput_file,
        out_dir=flexpart_dir,
        n_workers=n_workers,
        batch_size=64,
    )

    print("release-level file written")
    print("hourly file written")


if __name__ == "__main__":
    main()
