#!/usr/bin/env python3
import os
import re
import glob

# Replace these imports with your actual generator modules
from thermometer_gen import ThermometerGen
from popcnt_gen import PopCountGenerator
from groupsum_gen import GroupSumGenerator
from lutlayer_gen import LutLayerGenerator
from wnn_gen import WnnGenerator

def read_mapping_file(filename):
    """Read mapping file and return a dictionary of LUTs."""
    lut_map = []
    with open(filename, "r") as f:
        # It's just one line, but it is seperated by ";"
        line = f.readline().strip()
        for lut in line.split(";"):
            lut_map.append(lut)
    return lut_map

def read_lut_data(filename):
    """Read LUT data from a file and return it as a list of strings."""
    with open(filename, "r") as f:
        return [line.strip() for line in f]


def read_model_config(filename):
    """
    Read model configuration from a file and return (luts_num, luts_inp_num).
    Raises if not found.
    """
    luts_num = -1
    luts_inp_num = -1
    thermometer_bins = -1
    
    with open(filename, "r") as f:
        for line in f:
            if "luts_num" in line:
                luts_num = int(line.split("=", 1)[1].strip())
            elif "luts_inp_num" in line:
                luts_inp_num = int(line.split("=", 1)[1].strip())
            elif "thermometer_bins" in line:
                thermometer_bins = int(line.split("=", 1)[1].strip())
    if luts_num < 0 or luts_inp_num < 0:
        raise ValueError("Missing luts_num or luts_inp_num in config")
    return luts_num, luts_inp_num, thermometer_bins


class VHDLEntity:
    """
    Parse a single entity out of a VHDL file, generate component declaration,
    and query port datatype & width.
    """
    ENTITY_RE     = re.compile(r'entity\s+(\w+)\s+is', re.IGNORECASE)
    PORT_BLOCK_TMPL = r'entity\s+{name}\s+is.*?port\s*\(\s*(.*?)\s*\);\s*end\s+entity'
    PORT_RE       = re.compile(
        r'(\w+)\s*:\s*(in|out)\s*'
        r'(std_logic_vector\s*\(\s*(\d+)\s+downto\s*(\d+)\s*\)|std_logic)',
        re.IGNORECASE
    )

    def __init__(self, vhdl_path: str):
        self.vhdl_path = vhdl_path
        self.text = open(vhdl_path, "r").read()
        self.name = None
        # ports: list of (name, direction, type_decl, width)
        self.ports = []

    def parse(self, entity_pattern: str = None) -> bool:
        """
        Find and parse the first entity whose name matches entity_pattern (regex),
        or else the first entity in the file. Returns True if parsed.
        """
        for m in self.ENTITY_RE.finditer(self.text):
            candidate = m.group(1)
            if entity_pattern and not re.search(entity_pattern, candidate):
                continue

            block_re = re.compile(
                self.PORT_BLOCK_TMPL.format(name=re.escape(candidate)),
                re.IGNORECASE | re.DOTALL
            )
            block_m = block_re.search(self.text)
            if not block_m:
                continue

            self.name = candidate
            ports_block = block_m.group(1)
            for pm in self.PORT_RE.finditer(ports_block):
                pname, direction, tdecl, msb, lsb = pm.group(1), pm.group(2).lower(), pm.group(3).lower(), pm.group(4), pm.group(5)
                if "std_logic_vector" in tdecl:
                    width = abs(int(msb) - int(lsb)) + 1
                else:
                    width = 1
                self.ports.append((pname, direction, tdecl, width))
            return True

        return False

    def to_component(self) -> str:
        """
        Render a VHDL component declaration for the parsed entity.
        """
        if not self.name:
            raise RuntimeError("No entity parsed")
        lines = [f"component {self.name} is", "  port ("]
        for i, (pname, direction, tdecl, _) in enumerate(self.ports):
            sep = ";" if i < len(self.ports)-1 else ""
            lines.append(f"    {pname} : {direction} {tdecl}{sep}")
        lines.append("  );")
        lines.append(f"end component {self.name};")
        return "\n".join(lines)

    def get_port_info(self, port_name: str):
        """
        Return (type_decl, width) for a given port_name.
        Raises KeyError if not found.
        """
        for pname, _, tdecl, width in self.ports:
            if pname == port_name:
                return tdecl, width
        raise KeyError(f"Port '{port_name}' not in entity '{self.name}'")


def main():
    # 1) Cleanup and create output directories
    if os.path.exists("output"):
        os.system("rm -rf output")
    for sub in ["lutlayer", "popcount", "groupsum", "wnn"]:
        os.makedirs(os.path.join("output", sub), exist_ok=True)

    # 2) Read LUT data and model config
    lut_data   = read_lut_data("luts_data.txt")
    lut_count, lut_n, thermometer_bins = read_model_config("model_config.txt")
    num_classes = 5
    lut_mapping = read_mapping_file("/home/mmecik/repositories/DWN/src/hw_gen/mapping.txt")
    
    print(f"lut_count: {lut_count}, lut_n: {lut_n}")
    print(f"num_classes: {num_classes}")
    
    # Wait for enter
    input()
    
    ThermometerGen.generate(
        bins=thermometer_bins,
        scale_factor=100000,
        dataset_str="jsc",
        output_dir="output/thermometer",
    )

    # 3) Generate VHDL modules
    LutLayerGenerator.generate(
        lut_n=lut_n,
        lut_count=lut_count,
        lut_data_list=lut_data,
        output_dir="output/lutlayer",
    )

    PopCountGenerator.generate(
        num_inputs=lut_count//num_classes,
        output_dir="output/popcount",
        generate_figures=0,
        signed_in=0,
        msb_in=0,
        compression="optimalMinStages"
    )
    
    entity = VHDLEntity("output/popcount/popcnt.vhdl")
    if not entity.parse(entity_pattern=r"^FixMultiAdder_"):
        raise RuntimeError("Failed to parse FixMultiAdder entity")
    print(f"Parsed entity: {entity.name}")
    #print(f"Ports: {entity.ports}")
    
    tdecl, r_width = entity.get_port_info("R")
    print(f"Port 'R': type={tdecl}, width={r_width}")

    GroupSumGenerator.generate(
        population_width=r_width,
        num_classes=num_classes,
        output_dir="output/groupsum",
    )
    
    WnnGenerator.generate_tb(
      num_classes=num_classes,
      num_neurons=lut_count,
      num_inputs=lut_n,
      dataset_file="/home/mmecik/repositories/DWN/src/hw_gen/dataset.txt",
      pred_file="/home/mmecik/repositories/DWN/src/hw_gen/predictions.txt",
      vhdl_files=[
        "/home/mmecik/repositories/DWN/src/hw_gen/output/thermometer/thermometer.vhdl",
        "/home/mmecik/repositories/DWN/src/hw_gen/output/popcount/popcnt.vhdl",
        "/home/mmecik/repositories/DWN/src/hw_gen/output/groupsum/groupsum.vhdl",
        "/home/mmecik/repositories/DWN/src/hw_gen/output/lutlayer/lutlayer.vhdl"
      ],
      inputs_wnn=16,
      lut_mapping=lut_mapping,
      output_dir="tb"
    )

    print("Generation and entity parsing complete.")


if __name__ == "__main__":
    main()