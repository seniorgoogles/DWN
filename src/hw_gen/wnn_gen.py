#!/usr/bin/env python3
import os
import re
import shutil
from typing import List

class WnnGenerator:
    """
    1) Generates the WNN top‐level VHDL entity (for synthesis).
    2) Generates a matching WNN_tb.vhdl testbench.
    3) Extracts component declarations from arbitrary VHDL files.
    """

    @staticmethod
    def create_inputs_outputs(input_num: int, output_num: int) -> str:
        return f"""
    data_in  : in  std_logic_vector({input_num-1} downto 0);
    data_out : out std_logic_vector({output_num-1} downto 0)"""

    @staticmethod
    def create_component_string(
        component_name: str,
        generics: str,
        ports: str
    ) -> str:
        gen_block = f"    generic (\n{generics}\n    );\n" if generics else ""
        return (
            f"component {component_name}\n"
            f"{gen_block}"
            f"    port (\n{ports}\n    );\n"
            f"end component {component_name};"
        )

    @staticmethod
    def create_component_from_vhdl(vhdl_file: str) -> str:
        """
        Reads a VHDL file, extracts the first 'entity ... is' + 'port(...)',
        and returns a component declaration string.
        """
        text = open(vhdl_file).read()
        # pull out entity name
        m_ent = re.search(r'entity\s+(\w+)\s+is', text, re.IGNORECASE)
        if not m_ent:
            raise ValueError(f"No entity found in {vhdl_file}")
        name = m_ent.group(1)

        # pull generics (optional)
        m_gen = re.search(r'generic\s*\(\s*(.*?)\);\s*port', text, re.S | re.IGNORECASE)
        generics = m_gen.group(1).strip() if m_gen else ""

        # pull ports
        m_port = re.search(r'port\s*\(\s*(.*?)\);\s*', text, re.S | re.IGNORECASE)
        if not m_port:
            raise ValueError(f"No port block found in {vhdl_file}")
        ports = "\n".join("        " + line.strip()
                          for line in m_port.group(1).splitlines())

        comp = WnnGenerator.create_component_string(name, generics, ports)
        # print or save to file as needed
        print(f"-- Component for {name} from {vhdl_file} --\n{comp}\n")
        return comp

    @staticmethod
    def generate(
        input_num: int,
        output_num: int,
        output_dir: str,
        input_file_list: List[str]
    ):
        """
        1) Copies all input VHDL files into output_dir.
        2) Emits wnn.vhdl top-level stub.
        """
        os.makedirs(output_dir, exist_ok=True)
        # copy dependencies
        for path in input_file_list:
            if not os.path.exists(path):
                raise FileNotFoundError(path)
            shutil.copy(path, output_dir)

        # write top module
        top = f"""library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;

entity WNN is
    generic (
        INPUT_NUM  : integer := {input_num};
        OUTPUT_NUM : integer := {output_num}
    );
    port (
{WnnGenerator.create_inputs_outputs(input_num, output_num)}
    );
end entity WNN;

architecture Behavioral of WNN is
begin
    -- instantiate your LUTLayer, FixMultiAdder, GroupSum here
    -- example:
    LUT_inst: LUTLayer
        port map(data_in => data_in, data_out => data_out);
end architecture Behavioral;
"""
        with open(os.path.join(output_dir, "wnn.vhdl"), "w") as f:
            f.write(top)
        print(f"WNN top module written to {output_dir}/wnn.vhdl")

    @staticmethod
    def generate_tb(
        input_num: int,
        output_num: int,
        output_dir: str,
        dataset_file: str = "dataset.txt",
        pred_file: str = "predictions.txt"
    ):
        """
        Emits a testbench wnn_tb.vhdl in output_dir, wiring up:
        - signals of widths derived from input/output num
        - file I/O
        - a simple stimulus loop checking one-hot output
        """
        os.makedirs(output_dir, exist_ok=True)
        din_hi  = input_num  - 1
        dout_hi = output_num - 1
        rsum_hi = output_num* (input_num//output_num) - 1  # e.g. num of FixMultiAdder
        pred_hi = output_num - 1

        tb = f"""library ieee;
use ieee.std_logic_1164.all;
use ieee.numeric_std.all;
use std.textio.all;
use ieee.std_logic_textio.all;

entity WNN_tb is
end entity;

architecture Behavioral of WNN_tb is
  constant clk_period : time := 10 ns;

  signal clk        : std_logic := '0';
  signal reset      : std_logic := '0';
  signal data_in    : std_logic_vector({din_hi} downto 0) := (others => '0');
  signal data_out   : std_logic_vector({dout_hi} downto 0);
  signal R_all      : std_logic_vector({rsum_hi} downto 0);
  signal prediction : std_logic_vector({pred_hi} downto 0);

  file data_file : text open read_mode is "{dataset_file}";
  file pred_file : text open read_mode is "{pred_file}";

  component LUTLayer
    port(data_in  : in  std_logic_vector({din_hi} downto 0);
         data_out : out std_logic_vector({dout_hi} downto 0));
  end component;

  component FixMultiAdder_U_3_comb_uid2
    port(R  : out std_logic_vector({(pred_hi+1)*2 -1} downto 0);
         X0 : in  std_logic_vector(0 downto 0);
         X1 : in  std_logic_vector(0 downto 0);
         X2 : in  std_logic_vector(0 downto 0));
  end component;

  component GroupSum
    port(data_in  : in  std_logic_vector({rsum_hi} downto 0);
         data_out : out std_logic_vector({pred_hi} downto 0));
  end component;

begin
  UUT_LUTLayer: LUTLayer
    port map(data_in => data_in, data_out => data_out);

  -- instantiate your FixMultiAdder and GroupSum as needed

  clk_gen : process
  begin
    clk <= '0'; wait for clk_period/2;
    clk <= '1'; wait for clk_period/2;
  end process;

  stim_proc: process
    variable line_dat : line;
    variable line_prd : line;
    variable tmp_dat  : std_logic_vector({din_hi} downto 0);
    variable exp_int  : integer;
    variable exp_slv  : std_logic_vector({pred_hi} downto 0);
    variable got_int  : integer;
    variable total    : integer := 0;
    variable correct  : integer := 0;
    variable pct      : integer;
    variable idx      : integer := 0;
  begin
    reset <= '1'; wait for 2*clk_period;
    reset <= '0'; wait for clk_period;
    while not endfile(data_file) loop
      readline(data_file, line_dat); read(line_dat, tmp_dat); data_in <= tmp_dat;
      readline(pred_file, line_prd); read(line_prd, exp_int);
      exp_slv := std_logic_vector(to_unsigned(exp_int, {pred_hi+1}));
      wait until rising_edge(clk); wait for clk_period/4;

      got_int := -1;
      for i in prediction'range loop
        if prediction(i) = '1' then got_int := i; exit; end if;
      end loop;

      total := total + 1;
      if got_int = exp_int then
        correct := correct + 1;
        report \"Sample \" & integer'image(idx) & \": OK\" severity note;
      else
        report \"Sample \" & integer'image(idx) & \": FAIL got \" & integer'image(got_int)
               & \" exp \" & integer'image(exp_int) severity error;
      end if;
      idx := idx + 1;
    end loop;

    if total > 0 then pct := (correct * 100) / total; else pct := 0; end if;
    report \"Accuracy = \" & integer'image(pct) & \"%\" severity note;
    wait;
  end process;
end Behavioral;
"""
        with open(os.path.join(output_dir, "WNN_tb.vhdl"), "w") as f:
            f.write(tb)
        print(f"Testbench written to {output_dir}/WNN_tb.vhdl")


if __name__ == "__main__":
