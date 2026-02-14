from glob import glob
import os

if __name__ == "__main__":
    mojo_files = glob("mmm_audio/*.mojo")
    for mojo_file in mojo_files:
        print(f"Running {mojo_file}...")
        os.system(f"mojo build --emit object {mojo_file}")