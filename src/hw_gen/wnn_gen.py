#!/usr/bin/env python3
import os
import re
import shutil
from typing import List

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
        lines = [f"component {self.name}", "  port("]
        for i, (pname, direction, tdecl, _) in enumerate(self.ports):
            sep = ";" if i < len(self.ports)-1 else ""
            lines.append(f"    {pname} : {direction} {tdecl}{sep}")
        lines.append("  );")
        lines.append(f"end component;")
        return "\n".join(lines)
      
    def create_instance_map_full_port(self, src_ports: List[str], dst_ports: List[str]) -> str:
        """
        Create a VHDL instance map for the parsed entity.
        :param port_map: List of port names to be mapped.
        :return: VHDL instance map string.
        """
        if not self.name:
            raise RuntimeError("No entity parsed")
        lines = [f"  UUT_{self.name}: {self.name}"]
        lines.append("    port map (")
        for i, (src_port, dst_port) in enumerate(zip(src_ports, dst_ports)):
            sep = "," if i < len(src_ports)-1 else ""
            lines.append(f"      {src_port} => {dst_port}{sep}")
        lines.append("    );")
        return "\n".join(lines)
      
    def create_multiple_instances_partial_port(self, src_ports: List[str], dst_ports: List[str], num_instances: int, portmap: List[str]) -> str:
        """
        Create multiple VHDL instance maps for the parsed entity.
        :param port_map: List of port names to be mapped.
        :param num_instances: Number of instances to create.
        :return: VHDL instance map string.
        """
        if not self.name:
            raise RuntimeError("No entity parsed")
        lines = []
        for i in range(num_instances):
            lines.append(f"  UUT_{self.name}_{i}: {self.name}")
            lines.append("    port map (")
            for j, (src_port, dst_port) in enumerate(zip(src_ports, dst_ports)):              
                sep = "," if j < len(portmap[i])-1 else ""
                lines.append(f"      {src_port} => {dst_port}({portmap[i][j]}){sep}")
            lines.append("    );")
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
      
    def get_port_names(self):
        """
        Return a list of port names for the parsed entity.
        """
        return [pname for pname, _, _, _ in self.ports]

class WnnGenerator:
  
  @staticmethod
  def generate_tb(num_classes, num_neurons, num_inputs, dataset_file, pred_file, vhdl_files, output_dir):
    
    vhdl_component_fixmult = VHDLEntity("/home/mmecik/repositories/DWN/src/hw_gen/output/popcount/popcnt.vhdl")
    vhdl_component_groupsum = VHDLEntity("/home/mmecik/repositories/DWN/src/hw_gen/output/groupsum/groupsum.vhdl")
    vhdl_component_lutlayer = VHDLEntity("/home/mmecik/repositories/DWN/src/hw_gen/output/lutlayer/lutlayer.vhdl")
    
    vhdl_component_groupsum.parse(entity_pattern=r"^GroupSum_")
    vhdl_component_lutlayer.parse(entity_pattern=r"^LUTLayer_")
  
    if not vhdl_component_fixmult.parse(entity_pattern=r"^FixMultiAdder_"):
      raise RuntimeError("Failed to parse VHDL component")
    
    if not vhdl_component_groupsum.parse():
      raise RuntimeError("Failed to parse VHDL component")
    
    if not vhdl_component_lutlayer.parse():
      raise RuntimeError("Failed to parse VHDL component")

    r_width_popcnt = vhdl_component_fixmult.get_port_info("R")[1]
    
    print(f"R width: {r_width_popcnt}")
    print(vhdl_component_lutlayer.name)
    
    
    src_ports_fixmult = vhdl_component_fixmult.get_port_names()
    dst_ports_fixmult = ["R_all"] + ["data_out" for src_port in src_ports_fixmult[1:]]
    
    print(f"Source ports for FixMultiAdder: {src_ports_fixmult}")
    print(f"Destination ports for FixMultiAdder: {dst_ports_fixmult}")
    

    # Create the port mapping for FixMultiAdder
    portmap_fixmult = []
    
    for i in range(num_classes):
      
      mapping_per_fixmult = []
      mapping_per_fixmult.append(f"{(r_width_popcnt - 1 + (i * r_width_popcnt))} downto {i * r_width_popcnt}")
      offset = i * (len(src_ports_fixmult) - 1)
      
      for j in range(len(src_ports_fixmult) - 1):
        bit_address = (len(src_ports_fixmult) - 1) * num_classes - j - offset - 1
        mapping_per_fixmult.append(f"{bit_address} downto {bit_address}")
      
      portmap_fixmult.append(mapping_per_fixmult)

    print(f"Port mapping for FixMultiAdder: {portmap_fixmult}")
    
    input("Press Enter to continue...")

    
    
    vhdl_str = f"""
library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.textio.all;
use ieee.std_logic_textio.all;

entity WNN_tb is
end entity;

architecture Behavioral of WNN_tb is
  constant clk_period : time := 10 ns;
  
  -- Signals
  signal clk        : std_logic := '0';
  signal reset      : std_logic := '0';
  signal data_in    : std_logic_vector({(num_neurons * num_inputs)-1} downto 0) := (others => '0');
  signal data_out   : std_logic_vector({(num_neurons)-1} downto 0);
  signal R_all      : std_logic_vector({(r_width_popcnt * num_classes)-1} downto 0);
  signal prediction : std_logic_vector({num_classes-1} downto 0);
  
    -- File handles
  file data_file : text open read_mode is "dataset.txt";
  file pred_file : text open read_mode is "predictions.txt";
  
    -- Component declarations
  component LUTLayer
    port(
      data_in  : in  std_logic_vector({(num_neurons * num_inputs) - 1} downto 0);
      data_out : out std_logic_vector({(num_neurons)-1} downto 0)
    );
  end component;

  {vhdl_component_fixmult.to_component()}
  
  component GroupSum
    port(
      data_in  : in  std_logic_vector({(r_width_popcnt * num_classes)-1} downto 0);
      data_out : out std_logic_vector({num_classes-1} downto 0)
    );
  end component;
  
  begin
    ----------------------------------------------------------------
    -- DUT instantiations
    ----------------------------------------------------------------
  {vhdl_component_lutlayer.create_instance_map_full_port(vhdl_component_lutlayer.get_port_names(), ["data_in", "data_out"])}
  
  {vhdl_component_fixmult.create_multiple_instances_partial_port(src_ports_fixmult, dst_ports_fixmult, num_classes, portmap_fixmult)}
  
  {vhdl_component_groupsum.create_instance_map_full_port(vhdl_component_groupsum.get_port_names(), ["R_all", "prediction"])}

  ----------------------------------------------------------------
  -- Clock generation
  ----------------------------------------------------------------
  clk_gen: process
  begin
    clk <= '0';
    wait for clk_period/2;
    clk <= '1';
    wait for clk_period/2;
  end process;

  ----------------------------------------------------------------
  -- Stimulus + Accuracy Computation
  ----------------------------------------------------------------
  stim_proc: process
    variable line_dat      : line;
    variable line_prd      : line;
    variable tmp_dat       : std_logic_vector({(num_neurons * num_inputs)-1} downto 0);
    variable exp_int       : integer;
    variable exp_slv       : std_logic_vector(4 downto 0);
    variable got_int       : integer;
    variable total_count   : integer := 0;
    variable correct_count : integer := 0;
    variable pct_correct   : integer;
    variable sample_idx    : integer := 0;
  begin
    -- Apply reset
    reset <= '1';
    wait for 2*clk_period;
    reset <= '0';
    wait for clk_period;

    -- Loop through all vectors
    while not endfile(data_file) loop
      -- Read input vector
      readline(data_file, line_dat);
      read(line_dat, tmp_dat);
      data_in <= tmp_dat;

      -- Read expected output (integer)
      readline(pred_file, line_prd);
      read(line_prd, exp_int);
      exp_slv := std_logic_vector(to_unsigned(exp_int, 5));

      -- Drive and sample
      wait until rising_edge(clk);
      wait for clk_period/4;

      -- Decode one-hot prediction to integer
      got_int := -1;
      for i in prediction'range loop
        if prediction(i) = '1' then
          got_int := i;
          exit;
        end if;
      end loop;

      -- Update counters
      total_count := total_count + 1;
      if got_int = exp_int then
        correct_count := correct_count + 1;
        report "Sample " & integer'image(sample_idx)
               & ": Correct (" & integer'image(got_int)
               & ", vec=" & to_string(prediction) & ")"
               severity note;
      else
        report "Sample " & integer'image(sample_idx)
               & ": Mismatch got " & integer'image(got_int)
               & " (vec=" & to_string(prediction)
               & ") expected " & integer'image(exp_int)
               severity error;
      end if;

      sample_idx := sample_idx + 1;
    end loop;

    -- Compute and report accuracy
    if total_count > 0 then
      pct_correct := (correct_count * 100) / total_count;
    else
      pct_correct := 0;
    end if;

    report "Total samples:   " & integer'image(total_count)
           & ", Correct samples: " & integer'image(correct_count)
           & ", Accuracy: " & integer'image(pct_correct) & "%"
           severity note;
    report "Simulation complete." severity note;

    wait;  -- end simulation
  end process;

end architecture Behavioral;

"""

# Write the VHDL testbench to a file
    with open(os.path.join(output_dir, "wnn_tb.vhdl"), "w") as f:
        f.write(vhdl_str)

    # Copy the dataset and prediction files to the output directory
    shutil.copyfile(dataset_file, os.path.join(output_dir, "dataset.txt"))
    shutil.copyfile(pred_file, os.path.join(output_dir, "predictions.txt"))
    
    # Copy vhdl files to the output directory
    for vhdl_file in vhdl_files:
        shutil.copyfile(vhdl_file, os.path.join(output_dir, os.path.basename(vhdl_file)))

    print(f"Testbench generated in {output_dir}/WNN_tb.vhd")
    print(f"Dataset copied to {output_dir}/dataset.txt")
    print(f"Predictions copied to {output_dir}/predictions.txt")

if __name__ == "__main__":

    WnnGenerator.generate_tb(
      num_classes=5,
      num_neurons=15,
      num_inputs=3,
      dataset_file="/home/mmecik/repositories/DWN/src/hw_gen/dataset.txt",
      pred_file="/home/mmecik/repositories/DWN/src/hw_gen/predictions.txt",
      vhdl_files=[
        "/home/mmecik/repositories/DWN/src/hw_gen/output/wnn/popcnt.vhdl",
        "/home/mmecik/repositories/DWN/src/hw_gen/output/wnn/groupsum.vhdl",
        "/home/mmecik/repositories/DWN/src/hw_gen/output/wnn/lutlayer.vhdl"
      ],
      output_dir="tb"
    )